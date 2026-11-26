#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/image_chat_hook.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace infernix::serve {
namespace {

std::string sse_error_event(const ApiError& error) {
    return "data: " + make_error_body(error) + "\n\n";
}

} // namespace

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    OpenAIChatRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens        = options_.default_max_tokens;
        const auto body                  = parse_json_body(req);
        request                          = parse_chat_completion_request(body, limits);
        request.generation.ngram_session = resolve_ngram_session(req, body, options_);
        request.model = resolve_openai_model(request.model, public_model_id_, options_.any_model_name);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }
    // Tool-call turns sent back without their reasoning: counted, and with --restore-tool-reasoning
    // given the reasoning this server generated for them.
    const ReasoningRestore reasoning =
        options_.restore_tool_reasoning
            ? reasoning_store_.restore(request.generation.messages)
            : ReasoningRestore{.missing = count_missing_reasoning(request.generation.messages)};
    metrics_.reasoning(reasoning.missing, reasoning.restored);
    const auto remember_reasoning = [this](const GenerationOutcome& outcome, const std::vector<std::string>& ids) {
        if (options_.restore_tool_reasoning && !outcome.tool_calls.empty()) {
            reasoning_store_.remember(ids, outcome.reasoning);
        }
    };

    const std::uint64_t req_id = ++request_seq_;
    // The lane's two reads of this process, which only a member of HttpServer can make: the attached
    // pipeline and the one door a picture goes through. Everything they are used for -- the tool, the
    // rounds, the witness line a chat draw is filed under -- is the lane's: serve/image_chat_hook.h.
    auto hook = make_chat_image_hook(
        *this, req, "openai_chat", req_id, options_,
        ChatImageDoors{
            [this] { return static_cast<void*>(images_); },
            [this](const auto& image_request, std::string_view base, std::string_view source,
                   std::string_view protocol, std::string correlation) {
                return run_image_job(image_request, base, source, protocol, std::move(correlation));
            }});
    inject_chat_image_tool(hook, request.generation);
    RequestLogMetadata metadata{.http_request_id        = res.get_header_value("x-request-id"),
                                .model                  = request.model,
                                .stream                 = request.stream,
                                .output_tokens_explicit = request.output_tokens_explicit,
                                .label                  = request_label(req)};
    // Every round of one request is one HTTP request: the same observation wiring, and the same
    // request log, so a round the server runs itself stays part of this request.
    // Captured by value because a streaming provider runs after this handler has returned.
    const bool stream_live_timings    = request.stream && request.timings_per_token;
    const bool stream_prompt_progress = request.stream && request.return_progress;
    const std::string http_request_id = metadata.http_request_id;
    const auto make_observation = [this, req_id, http_request_id, stream_live_timings,
                                   stream_prompt_progress] {
        return infernix::GenerationObservationOptions{
            .phase_timings   = true,
            .live_timings    = stream_live_timings,
            .prompt_progress = stream_prompt_progress,
            .scheduling      = scheduling_observer(req_id, http_request_id),
            .first_token     = first_token_observer(),
        };
    };
    PreparedRequest prepared;
    try {
        prepared = service_->prepare(
            request.generation,
            request.stream ? GenerationConsumerMode::Streaming : GenerationConsumerMode::Aggregate,
            make_observation(), [&req] { return client_disconnected(req); });
    } catch (const ApiException& exception) {
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, error));
        write_openai_error(res, error);
        return;
    }

    const OpenAIChatResponseIdentity identity = make_openai_chat_response_identity(request.model);
    metadata.response_id                      = identity.id;
    auto lifecycle                            = begin_request(make_request_log_context(
        req_id, "openai_chat_completions", request.generation, metadata, prepared));

    if (!request.stream) {
        GenerationOutcome outcome;
        try {
            outcome = service_->run(prepared, nullptr, [&req] { return client_disconnected(req); });
            if (chat_image_round_needed(hook, outcome)) {
                auto rounds = run_chat_image_rounds(
                    *hook, request.generation, std::move(outcome),
                    [&] {
                        // A new round takes over this request's generation, and with it the lifetime
                        // that keeps media alive until the response is written.
                        prepared = service_->prepare(request.generation,
                                                     GenerationConsumerMode::Aggregate,
                                                     make_observation(),
                                                     [&req] { return client_disconnected(req); });
                        return service_->run(prepared, nullptr,
                                             [&req] { return client_disconnected(req); });
                    });
                outcome           = std::move(rounds.answer);
                outcome.text     += chat_image_missing_url_suffix(*hook, outcome.text, rounds.urls);
            }
        } catch (const ApiException& exception) {
            lifecycle->failure(make_generation_request_failure(exception.error()));
            write_openai_error(res, exception.error());
            return;
        } catch (const std::exception& exception) {
            const RequestFailure failure =
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what());
            lifecycle->failure(failure);
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = exception.what();
            write_openai_error(res, error);
            return;
        }
        lifecycle->done(outcome);
        try {
            set_ngram_generation_header(res, outcome.ngram_archive);
            std::vector<std::string> call_ids;
            set_owned_json_content(res, make_chat_completion_response(identity, outcome, &call_ids),
                                   prepared.lifetime);
            remember_reasoning(outcome, call_ids);
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = exception.what();
            write_openai_error(res, error);
        }
        return;
    }

    // The SSE response cannot be taken back once committed: wait for the Engine's admission first,
    // so the queue's own answers (a queue timeout, an orderly stop) are a plain HTTP status here as
    // on the non-streaming branch; after it only mid-generation failures remain, which belong in
    // the stream.
    try {
        service_->admit(prepared, [&req] { return client_disconnected(req); });
    } catch (const ApiException& exception) {
        lifecycle->failure(make_generation_request_failure(exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
        return;
    }

    try {
        const bool return_progress   = request.return_progress;
        const bool timings_per_token = request.timings_per_token;
        // The provider runs after this handler returns, so the request its rounds extend has to
        // outlive the handler. Moving it here is free: the aggregate path returned already.
        auto generation              = std::make_shared<GenerationRequest>(std::move(request.generation));
        auto stream                  = std::make_shared<HttpGenerationStream>(std::move(prepared));
        auto encoder = std::make_shared<OpenAIChatStream>(identity, request.include_usage,
                                                          timings_per_token, return_progress,
                                                          options_.usage_chunk_choice);

        prepare_sse_response(res);
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, stream, encoder, lifecycle, generation, return_progress, timings_per_token,
             hook, make_observation,
             remember_reasoning](std::size_t, httplib::DataSink& sink) -> bool {
                if (stream->started.exchange(true, std::memory_order_acq_rel)) {
                    sink.done();
                    return true;
                }
                SseTransport transport(sink, stream->cancelled);
                const auto send_error = [&](const ApiError& error) {
                    try {
                        render_and_write(transport, [&] { return sse_error_event(error); });
                        sink.done();
                        return true;
                    } catch (const ClientDisconnected&) {
                        lifecycle->response_failure(
                            make_client_disconnected_failure(RequestFailurePhase::Transport));
                        return false;
                    } catch (const ResponseRenderFailure& exception) {
                        lifecycle->response_failure(make_internal_request_failure(
                            RequestFailurePhase::ResponseRender, exception.what()));
                        return false;
                    } catch (...) {
                        lifecycle->response_failure(make_internal_request_failure(
                            RequestFailurePhase::ResponseRender, "the error event could not be written"));
                        return false;
                    }
                };
                try {
                    render_and_write(transport, [&] { return encoder->start(); });
                } catch (const ClientDisconnected&) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                }

                // A tool-call this adapter cannot render ends the request's stream, not the server:
                // the Engine is asked to stop this generation and one terminal error event closes the
                // stream once the round unwinds.
                bool protocol_failed = false;
                std::string protocol_failure_detail;
                // `live` is whether this round's events reach the client. A round that only asked
                // for a picture carries nothing: its prose is not part of the answer, and its start,
                // prompt-progress and per-token timing would open a second generation inside the one
                // streamed message. Its text still reaches the client as the answer's tail.
                const auto run_stream_round = [&](PreparedRequest& slot, bool live) {
                    StreamSink output;
                    if (live) {
                        output.on_start = [&](const infernix::GenerationStart& start) {
                            encoder->note_start(start);
                            if (return_progress) {
                                render_and_write(transport,
                                                 [&] { return encoder->initial_prompt_progress(); });
                            }
                        };
                        if (return_progress) {
                            output.on_progress = [&](const infernix::PromptProgress& progress) {
                                render_and_write(transport,
                                                 [&] { return encoder->prompt_progress(progress); });
                            };
                        }
                        if (timings_per_token) {
                            output.on_timing =
                                [&](const infernix::GenerationTimingObservation& timing) {
                                    encoder->note_timing(timing);
                                };
                        }
                        output.on_content = [&](const std::string& text) {
                            render_and_write(transport,
                                             [&] { return encoder->content_delta(text); });
                        };
                        output.on_reasoning = [&](const std::string& text) {
                            render_and_write(transport,
                                             [&] { return encoder->reasoning_delta(text); });
                        };
                        // Every delta reaches the encoder, including the server's own picture call:
                        // the turn's result counts that call, and the encoder matches its streamed
                        // calls against the result by position. The encoder is what keeps it off the
                        // wire, and what gives the calls that do reach the client their indices.
                        output.on_tool_call = [&](const infernix::ToolCallDelta& delta) {
                            const OpenAIChatToolCallChunk rendered = encoder->tool_call_delta(delta);
                            if (!rendered.ok()) {
                                protocol_failed         = true;
                                protocol_failure_detail = describe_openai_chat_protocol_error(
                                    rendered.error);
                                return;
                            }
                            render_and_write(transport, [&] { return rendered.chunk; });
                        };
                    }
                    // A refused tool-call cancels this request's generation only.
                    output.is_cancelled = [&] { return transport.poll() || protocol_failed; };
                    return service_->run(slot, &output);
                };

                GenerationOutcome outcome;
                try {
                    outcome = run_stream_round(stream->prepared, true);
                    if (protocol_failed) {
                        // The stream is committed, so it ends the way OpenAI ends a failed stream: one
                        // error event and the connection closes. No fabricated finish chunk for a
                        // message that can no longer be valid, and no picture round after it.
                        lifecycle->failure(make_internal_request_failure(
                            RequestFailurePhase::Generation,
                            "invalid tool-call sequence: " + protocol_failure_detail));
                        ApiError error;
                        error.status  = 500;
                        error.type    = "api_error";
                        error.message = "the response stream was terminated because the server could "
                                        "not produce a valid tool-call sequence: " +
                                        protocol_failure_detail;
                        return send_error(error);
                    }
                    if (chat_image_round_needed(hook, outcome)) {
                        // The picture draws with the Engine loop stopped: the lane says what that
                        // silence carries on the wire.
                        // The ticker is also the stall watchdog's only observer: only this thread can see
                        // that the job stopped reporting, and only the lane knows when that means a wedge
                        // (serve/image_job_progress.h, armed by src/models/qwen_image/image_pipeline.cu).
                        HeartbeatTicker ticker(
                            transport,
                            [&hook]() -> std::string { return chat_image_progress_note(*hook); },
                            [&hook]() -> std::shared_ptr<ImageJobProgress> {
                                return chat_image_watch_progress(*hook);
                            });
                        auto rounds = run_chat_image_rounds(
                            *hook, *generation, std::move(outcome),
                            [&] {
                                stream->prepared =
                                    service_->prepare(*generation, GenerationConsumerMode::Streaming,
                                                      make_observation(),
                                                      [&] { return transport.poll(); });
                                return run_stream_round(stream->prepared, false);
                            });
                        outcome       = std::move(rounds.answer);
                        outcome.text += chat_image_missing_url_suffix(*hook, outcome.text, rounds.urls);
                    }
                } catch (const ClientDisconnected&) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                } catch (const ApiException& exception) {
                    lifecycle->failure(make_generation_request_failure(exception.error()));
                    return send_error(exception.error());
                } catch (const std::exception& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::Generation, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                }

                // The tail of a committed stream runs on the HTTP worker with no handler above it, so
                // everything from the lifecycle record to the last byte is one guarded block: a throw
                // here -- an aggregate that will not serialize, a chunk that will not encode, a socket
                // that closes on the final write -- ends this request's stream, never the process.
                lifecycle->done(outcome);
                try {
                    if (outcome.finish_reason == infernix::FinishReason::Cancelled ||
                        stream->cancelled.load(std::memory_order_acquire)) {
                        lifecycle->response_failure(
                            make_client_disconnected_failure(RequestFailurePhase::Transport));
                        return false;
                    }
                    std::vector<std::string> terminal = encoder->finish(outcome);
                    remember_reasoning(outcome, encoder->tool_call_ids());
                    if (auto comment = ngram_generation_comment(outcome.ngram_archive);
                        !comment.empty()) {
                        terminal.insert(terminal.begin(), std::move(comment));
                    }
                    transport.write(terminal);
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) {
                    lifecycle->response_failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const std::exception& exception) {
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "api_error";
                    error.message = "the response stream failed while it was being written: " +
                                    std::string(exception.what());
                    return send_error(error);
                } catch (...) {
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, "the response stream failed unexpectedly"));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "api_error";
                    error.message = "the response stream failed unexpectedly while it was being written";
                    return send_error(error);
                }
            },
            [stream, lifecycle](bool successful) {
                stream->cancelled.store(true, std::memory_order_release);
                if (!successful || !stream->started.load(std::memory_order_acquire)) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                }
            });
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::ResponseRender, exception.what()));
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
    }
}

} // namespace infernix::serve
