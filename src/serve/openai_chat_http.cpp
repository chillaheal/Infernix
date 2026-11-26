#include "serve/http_server.h"

#include "serve/http_stream_driver.h"
#include "serve/http_transport.h"
#include "serve/image_chat_hook.h" // the image lane's one attach point on this route
#include "serve/openai_chat.h"
#include "serve/openai_common.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <string_view> // the lane's typed doors hand their arguments over as views
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
        require_generation();
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
    // Assistant messages sent back without their reasoning: counted, and with --restore-reasoning given
    // the reasoning this server generated for them. The reply's reasoning is remembered under its tool-call
    // ids, or as an answer to this request's conversation (a continued final message is not a new answer).
    ConversationDigest conversation;
    const ReasoningRestore reasoning = reasoning_store_.restore(request.generation, conversation);
    metrics_.reasoning(reasoning);
    const bool new_answer = request.generation.continuation == infernix::PromptContinuationMode::NewAssistantTurn;
    const auto remember_reasoning = [this, conversation, new_answer](const GenerationOutcome& outcome,
                                                                     const std::vector<std::string>& ids) {
        if (!outcome.tool_calls.empty()) {
            reasoning_store_.remember_tool_calls(ids, outcome.reasoning);
        } else if (new_answer) {
            reasoning_store_.remember_answer(conversation, outcome.text, outcome.reasoning);
        }
    };

    const std::uint64_t req_id = ++request_seq_;
    // PORT(ported tree): re-adds the TOOL_INJECTION seam of FORK src/serve/openai_chat_http.cpp:58, which
    // the automated pass dropped: the hook, the tool it injects, and the observation the rounds share. What
    // the lane decides -- that the `generate_image` tool is offered, when a round is owed, what the drawn
    // picture's URL suffix says, what the SSE silence carries while a draw runs, and what a chat draw is
    // filed under in the request log -- lives in serve/image_chat_hook.cpp; this file keeps only its own
    // request, its `service_`, its re-prepare/re-run callbacks, its route label, and the two reads of its
    // private state that the lane cannot make (`images_`, `run_image_job`), handed over as the typed
    // ChatImageDoors. A serve with no --image-artifact gets a null hook, which is the lane's own answer that
    // it draws no picture: no tool injected, no round owed, no field written.
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
                                .label                  = request_label(req),
                                .reasoning              = reasoning};
    // PORT(ported tree): re-adds the shared observation of FORK src/serve/openai_chat_http.cpp:78, which the
    // rounds below re-prepare with: every round of one request is one HTTP request, with the same
    // observation wiring and the same request log, so a round the server runs itself stays part of this
    // request. Upstream tip builds the object inline at the first prepare and has added `.progress` to it
    // since the fork was written, so the fields are copied from upstream's own literal (including
    // `.live_timings`/`.prompt_progress`, which it derives from the request's stream flags) rather than from
    // the fork's; upstream's literal at the first prepare is left exactly as it stands. Everything it reads
    // is captured by value because a streaming provider runs after this handler has returned.
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
            .progress        = progress_observer(req_id),
        };
    };
    PreparedRequest prepared;
    try {
        infernix::GenerationObservationOptions observation{
            .phase_timings   = true,
            .live_timings    = request.stream && request.timings_per_token,
            .prompt_progress = request.stream && request.return_progress,
            .scheduling      = scheduling_observer(req_id, metadata.http_request_id),
            .first_token     = first_token_observer(),
            .progress        = progress_observer(req_id),
        };
        prepared = service_->prepare(
            request.generation,
            request.stream ? GenerationConsumerMode::Streaming : GenerationConsumerMode::Aggregate,
            std::move(observation), [&req] { return client_disconnected(req); });
    } catch (const ApiException& exception) {
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        const ApiError error = internal_server_error(exception);
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
            // PORT(ported tree): re-adds the aggregate picture rounds of FORK
            // src/serve/openai_chat_http.cpp:119 (the automated pass kept this one): the lane runs the calls
            // the model made to the server's own tool, and a new round takes over this request's generation
            // and its lifetime. The answer must carry the picture, so any URL it still does not quote is
            // appended by the lane. Null hook, null work.
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
            write_openai_error(res, internal_server_error(exception));
            return;
        }
        lifecycle->done(outcome);
        try {
            set_ngram_generation_header(res, outcome.ngram_archive);
            std::vector<std::string> call_ids;
            set_owned_json_content(res, make_chat_completion_response(identity, outcome, &call_ids),
                                   prepared.lifetime);
            // A cancelled answer is cut: its reasoning is not kept for the next turn to resend.
            if (outcome.finish_reason != infernix::FinishReason::Cancelled) { remember_reasoning(outcome, call_ids); }
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            write_openai_error(res, internal_server_error(exception));
        }
        return;
    }

    // The SSE response cannot be taken back once committed: wait for the Engine's admission, so the
    // queue's answers (a queue timeout, an orderly stop) are a plain HTTP status here as on the
    // non-streaming branch. Unlike Messages and Responses, Chat Completions has no error event clients
    // agree on: a queue timeout as one `data:` error frame after a 200 is recorded by some clients
    // (LiteLLM 1.98, raw HTTP) as a successful empty reply and retried at once (NInfer #420), so this
    // route does not start its stream early (kStreamAdmissionGrace).
    try {
        (void)service_->admit(prepared, [&req] { return client_disconnected(req); },
                              std::chrono::steady_clock::time_point::max());
    } catch (const ApiException& exception) {
        lifecycle->failure(make_generation_request_failure(exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
        write_openai_error(res, internal_server_error(exception));
        return;
    }

    try {
        const bool return_progress   = request.return_progress;
        const bool timings_per_token = request.timings_per_token;
        // PORT(ported tree): re-adds the owned request of FORK src/serve/openai_chat_http.cpp:195, which the
        // streaming rounds extend: the provider runs after this handler returns, so the request has to
        // outlive it. The aggregate branch above returned already, so moving it out is free.
        auto generation              = std::make_shared<GenerationRequest>(std::move(request.generation));
        auto stream                  = std::make_shared<HttpGenerationStream>(std::move(prepared));
        auto encoder = std::make_shared<OpenAIChatStream>(identity, request.include_usage,
                                                          timings_per_token, return_progress,
                                                          options_.usage_chunk_choice);

        prepare_sse_response(res);
        // PORT(ported tree): the capture gains the three things the picture rounds need to re-prepare and
        // re-run this request after the provider outlives the handler -- the request, the hook it draws
        // through, and the observation every round of this request shares -- exactly as in FORK
        // src/serve/openai_chat_http.cpp:204. A null hook carries nothing.
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, stream, encoder, lifecycle, return_progress, timings_per_token, generation, hook,
             make_observation,
             remember_reasoning](std::size_t, httplib::DataSink& sink) -> bool {
                if (stream->started.exchange(true, std::memory_order_acq_rel)) {
                    sink.done();
                    return true;
                }
                SseTransport transport(sink, stream->cancelled);
                transport.set_heartbeat([&] { return encoder->keep_alive(); });
                const auto send_error = [&](const ApiError& error) {
                    return end_stream_with_error(sink, *lifecycle, [&] {
                        render_and_write(transport, [&] { return sse_error_event(error); });
                    });
                };
                // The assistant-role chunk opens the stream with its first event (prompt progress, the
                // first output delta or the finish), not at admission, so a client's time to its first
                // chunk is the prefill's (as OpenAI's). Until then a heartbeat is an SSE comment.
                bool opened     = false;
                const auto open = [&] {
                    if (opened) { return; }
                    opened = true;
                    render_and_write(transport, [&] { return encoder->start(); });
                };

                GenerationOutcome outcome;
                try {
                    StreamSink output;
                    output.on_start = [&](const infernix::GenerationStart& start) {
                        encoder->note_start(start);
                        if (return_progress) {
                            open();
                            render_and_write(transport,
                                             [&] { return encoder->initial_prompt_progress(); });
                        }
                    };
                    if (return_progress) {
                        output.on_progress = [&](const infernix::PromptProgress& progress) {
                            open();
                            render_and_write(transport,
                                             [&] { return encoder->prompt_progress(progress); });
                        };
                    }
                    if (timings_per_token) {
                        output.on_timing = [&](const infernix::GenerationTimingObservation& timing) {
                            encoder->note_timing(timing);
                        };
                    }
                    output.on_content = [&](const std::string& text) {
                        open();
                        render_and_write(transport, [&] { return encoder->content_delta(text); });
                    };
                    output.on_reasoning = [&](const std::string& text) {
                        open();
                        render_and_write(transport, [&] { return encoder->reasoning_delta(text); });
                    };
                    output.on_tool_call = [&](const infernix::ToolCallDelta& delta) {
                        open();
                        render_and_write(transport, [&] { return encoder->tool_call_delta(delta); });
                    };
                    output.is_cancelled = [&] { return transport.poll(); };

                    outcome = service_->run(stream->prepared, &output);
                    // PORT(ported tree): re-adds the streamed picture rounds of FORK
                    // src/serve/openai_chat_http.cpp:324, which the automated pass dropped, in upstream tip's
                    // own position (inside this try, so a round's failure is upstream's own stream ending).
                    // A picture round streams nothing of its own: its start, prompt progress and per-token
                    // timings would open a second generation inside the one streamed message, and its prose
                    // reaches the client as the answer's tail, so the round runs on a sink that carries
                    // nothing but cancellation. The lane says what the silence while a picture is drawn
                    // carries on the wire: upstream tip replaced the fork's own HeartbeatTicker with this
                    // single heartbeat slot (serve/http_transport.h), so the note is installed for the
                    // rounds and upstream's own keep-alive is put back after them. Nothing here runs at all
                    // for a serve with no --image-artifact: a null hook needs no round.
                    if (chat_image_round_needed(hook, outcome)) {
                        // PORT(ported tree): re-adds the fork's ticker of FORK
                        // src/serve/openai_chat_http.cpp:327 (ours-only, docs/maintainer/
                        // image-integration-points.md). The draw runs with the Engine loop stopped, so
                        // upstream's poll()-driven heartbeat below cannot fire at all during it: this
                        // thread is the stream's only traffic while the picture is drawn. It carries the
                        // job's real percent as an SSE comment (the wording is the lane's) and it watches
                        // the job for a wedged draw, which nothing else can see from a blocked handler.
                        HeartbeatTicker ticker(
                            transport,
                            [&hook]() -> std::string { return chat_image_progress_note(*hook); },
                            [&hook]() -> std::shared_ptr<ImageJobProgress> {
                                return chat_image_watch_progress(*hook);
                            });
                        transport.set_heartbeat([&] {
                            const std::string note = chat_image_progress_note(*hook);
                            return note.empty() ? encoder->keep_alive() : note;
                        });
                        auto rounds = run_chat_image_rounds(
                            *hook, *generation, std::move(outcome),
                            [&] {
                                stream->prepared =
                                    service_->prepare(*generation, GenerationConsumerMode::Streaming,
                                                      make_observation(),
                                                      [&] { return transport.poll(); });
                                StreamSink quiet;
                                quiet.is_cancelled = [&] { return transport.poll(); };
                                return service_->run(stream->prepared, &quiet);
                            });
                        outcome       = std::move(rounds.answer);
                        outcome.text += chat_image_missing_url_suffix(*hook, outcome.text, rounds.urls);
                        transport.set_heartbeat([&] { return encoder->keep_alive(); });
                    }
                } catch (...) {
                    return end_failed_stream_generation(
                        *lifecycle, [](const ApiError& error) -> const ApiError& { return error; },
                        [](const std::exception& exception) { return internal_server_error(exception); },
                        send_error);
                }

                if (!record_stream_generation(*lifecycle, outcome, stream->cancelled)) { return false; }
                std::vector<std::string> terminal;
                try {
                    // An answer with no streamed delta (an empty one, or tool calls sent whole) opens here.
                    std::string opening;
                    if (!opened) {
                        opened  = true;
                        opening = encoder->start();
                    }
                    terminal = encoder->finish(outcome);
                    if (!opening.empty()) { terminal.insert(terminal.begin(), std::move(opening)); }
                    remember_reasoning(outcome, encoder->tool_call_ids());
                    if (auto comment = ngram_generation_comment(outcome.ngram_archive);
                        !comment.empty()) {
                        terminal.insert(terminal.begin(), std::move(comment));
                    }
                } catch (const std::exception& exception) {
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    return send_error(internal_server_error(exception));
                }
                return write_stream_terminal(transport, sink, *lifecycle, terminal);
            },
            stream_release(stream, lifecycle));
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::ResponseRender, exception.what()));
        write_openai_error(res, internal_server_error(exception));
    }
}

} // namespace infernix::serve
