#include "serve/http_server.h"

#include "serve/anthropic_messages.h"
#include "serve/chat_image_tool.h"
#include "serve/http_transport.h"
#include "serve/image_generation_service.h"
#include "serve/openai_common.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace infernix::serve {

void HttpServer::handle_count_tokens(const httplib::Request& req, httplib::Response& res) {
    const std::string request_id = res.get_header_value("request-id");
    try {
        const AnthropicCountTokensRequest request = parse_anthropic_count_tokens_request(
            parse_json_body(req), {.claude_code_prompt = options_.claude_code_prompt});
        const int input_tokens = service_->count_prompt_tokens(
            request.generation, [&req] { return client_disconnected(req); });
        res.set_content(make_anthropic_count_tokens_response(input_tokens), "application/json");
    } catch (const ApiException& exception) {
        write_anthropic_error(res, exception.error(), request_id);
    } catch (const std::exception& exception) {
        operational_log_.http_failure(
            "anthropic_count_tokens",
            make_internal_request_failure(RequestFailurePhase::Http, exception.what()), request_id);
        ApiError error;
        error.status  = 500;
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
    }
}

void HttpServer::handle_messages(const httplib::Request& req, httplib::Response& res) {
    const std::string request_id = res.get_header_value("request-id");

    AnthropicMessagesRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens        = options_.default_max_tokens;
        limits.claude_code_prompt        = options_.claude_code_prompt;
        const auto body                  = parse_json_body(req);
        request                          = parse_anthropic_messages_request(body, limits);
        request.generation.ngram_session = resolve_ngram_session(req, body, options_);
    } catch (const ApiException& exception) {
        write_anthropic_error(res, exception.error(), request_id);
        return;
    } catch (const std::exception& exception) {
        operational_log_.http_failure(
            "anthropic_messages",
            make_internal_request_failure(RequestFailurePhase::Http, exception.what()), request_id);
        ApiError error;
        error.status  = 500;
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
        return;
    }

    // With an image pipeline in this process the request also carries the server-executed image
    // tool, so the picture rounds below are the ones a client would otherwise have to run. The URL
    // base and the model id are read here because a streaming provider runs after this handler has
    // returned, and the generator is bound to this request's Host so the URL resolves for that client.
    const bool images_configured = images_ != nullptr;
    append_chat_image_tool(request.generation, images_configured);
    const std::string image_url_base =
        images_configured ? artifact_url_base(req, options_) : std::string();
    const std::string image_model = image_model_id();
    const ChatImageGenerator make_picture = [this, url_base = image_url_base](
                                                const OpenAIImagesRequest& image_request) {
        return images_->generate(image_request, unix_time_now(), url_base);
    };

    const std::uint64_t req_id = ++request_seq_;
    RequestLogMetadata metadata{.http_request_id        = request_id,
                                .model                  = request.model,
                                .stream                 = request.stream,
                                .output_tokens_explicit = request.output_tokens_explicit};
    // Every round of one request is one HTTP request: the same observation wiring, and the same
    // request log, so an image round is visible as part of this request rather than as a new one.
    // Captured by value because a streaming provider runs after this handler has returned.
    const auto make_observation = [this, req_id, http_request_id = metadata.http_request_id] {
        return infernix::GenerationObservationOptions{
            .phase_timings = true,
            .scheduling    = scheduling_observer(req_id, http_request_id),
            .first_token   = first_token_observer(),
        };
    };
    PreparedRequest prepared;
    try {
        prepared = service_->prepare(
            request.generation,
            request.stream ? GenerationConsumerMode::Streaming : GenerationConsumerMode::Aggregate,
            make_observation(), [&req] { return client_disconnected(req); });
    } catch (const ApiException& exception) {
        const ApiError error = normalize_anthropic_error(exception.error());
        record_request_rejected(make_request_rejection_log_context(
            req_id, "anthropic_messages", request.generation, metadata, error));
        write_anthropic_error(res, error, request_id);
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, "anthropic_messages", request.generation, metadata, error));
        write_anthropic_error(res, error, request_id);
        return;
    }

    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity(request_id, request.model);
    metadata.response_id   = identity.message_id;
    const int input_tokens = prepared.prompt_tokens;

    auto lifecycle = begin_request(make_request_log_context(
        req_id, "anthropic_messages", request.generation, metadata, prepared));

    if (!request.stream) {
        GenerationOutcome outcome;
        try {
            outcome = service_->run(prepared, nullptr, [&req] { return client_disconnected(req); });
            if (images_configured && outcome_is_chat_image_tool_round(outcome)) {
                ChatImageRounds rounds = run_chat_image_rounds(
                    request.generation, std::move(outcome), image_model, image_url_base,
                    unix_time_now(), make_picture,
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
                outcome.text     += missing_image_url_suffix(outcome.text, rounds.urls);
            }
        } catch (const ApiException& exception) {
            const ApiError error = normalize_anthropic_error(exception.error());
            lifecycle->failure(make_generation_request_failure(error));
            write_anthropic_error(res, error, request_id);
            return;
        } catch (const std::exception& exception) {
            lifecycle->failure(
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
            ApiError error;
            error.status  = 500;
            error.message = exception.what();
            write_anthropic_error(res, error, request_id);
            return;
        }
        lifecycle->done(outcome);
        try {
            set_ngram_generation_header(res, outcome.metrics.ngram_archive);
            set_owned_json_content(
                res, make_anthropic_messages_response(identity, outcome, request.hide_thinking),
                prepared.lifetime);
        } catch (const ApiException& exception) {
            const ApiError error = normalize_anthropic_error(exception.error());
            lifecycle->response_failure(
                make_request_failure(RequestFailurePhase::ResponseRender, error));
            write_anthropic_error(res, error, request_id);
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            ApiError error;
            error.status  = 500;
            error.message = exception.what();
            write_anthropic_error(res, error, request_id);
        }
        return;
    }

    try {
        // The provider runs after this handler returns, so the request its rounds extend has to
        // outlive the handler. Moving it here is free: the aggregate path returned already.
        auto generation = std::make_shared<GenerationRequest>(std::move(request.generation));
        auto stream     = std::make_shared<HttpGenerationStream>(std::move(prepared));
        auto encoder    = std::make_shared<AnthropicMessagesStream>(identity, input_tokens,
                                                                    request.hide_thinking);

        prepare_sse_response(res);
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, stream, encoder, lifecycle, generation, images_configured, image_model,
             image_url_base, make_picture,
             make_observation](std::size_t, httplib::DataSink& sink) -> bool {
                if (stream->started.exchange(true, std::memory_order_acq_rel)) {
                    sink.done();
                    return true;
                }
                SseTransport transport(sink, stream->cancelled);
                const auto send_error = [&](const ApiError& error) {
                    try {
                        if (!encoder->started()) {
                            render_and_write(transport, [&] { return encoder->start(); });
                        }
                        render_and_write(transport, [&] { return encoder->error(error); });
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
                    }
                };

                // `live` is whether this round's events reach the client. A round that only asked
                // for a picture carries nothing: its prose is not part of the answer, and its start
                // would open a second message inside the one streamed message. Its text still
                // reaches the client as the answer's tail.
                const auto run_stream_round = [&](PreparedRequest& slot, bool live) {
                    StreamSink output;
                    if (live) {
                        output.on_start = [&](const infernix::GenerationStart& start) {
                            render_and_write(transport, [&] { return encoder->start(start); });
                        };
                        output.on_reasoning = [&](const std::string& text) {
                            render_and_write(transport,
                                             [&] { return encoder->reasoning_delta(text); });
                        };
                        output.on_content = [&](const std::string& text) {
                            render_and_write(transport, [&] { return encoder->content_delta(text); });
                        };
                    }
                    output.is_cancelled = [&] { return transport.poll(); };
                    return service_->run(slot, &output);
                };

                GenerationOutcome outcome;
                try {
                    outcome = run_stream_round(stream->prepared, true);
                    if (images_configured && outcome_is_chat_image_tool_round(outcome)) {
                        // The picture draws with the Engine loop stopped: this ticker is the
                        // stream's only traffic while it runs, and it carries the job's real
                        // percent as an SSE comment.
                        HeartbeatTicker ticker(
                            transport, [this] -> std::string {
                                const auto progress =
                                    images_ != nullptr ? images_->last_job_progress() : nullptr;
                                if (progress == nullptr) { return {}; }
                                const int percent = progress->percent.load(std::memory_order_relaxed);
                                if (percent < 0) { return {}; }
                                return ": image " + std::to_string(percent) + "%\n\n";
                            });
                        ChatImageRounds rounds = run_chat_image_rounds(
                            *generation, std::move(outcome), image_model, image_url_base,
                            unix_time_now(), make_picture,
                            [&] {
                                stream->prepared =
                                    service_->prepare(*generation, GenerationConsumerMode::Streaming,
                                                      make_observation(),
                                                      [&] { return transport.poll(); });
                                return run_stream_round(stream->prepared, false);
                            });
                        outcome       = std::move(rounds.answer);
                        outcome.text += missing_image_url_suffix(outcome.text, rounds.urls);
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
                    error.message = exception.what();
                    return send_error(error);
                } catch (const ApiException& exception) {
                    const ApiError error = normalize_anthropic_error(exception.error());
                    lifecycle->failure(make_generation_request_failure(error));
                    return send_error(error);
                } catch (const std::exception& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::Generation, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.message = exception.what();
                    return send_error(error);
                }

                lifecycle->done(outcome);
                if (outcome.finish_reason == infernix::FinishReason::Cancelled ||
                    stream->cancelled.load(std::memory_order_acquire)) {
                    lifecycle->response_failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                }
                std::vector<std::string> terminal;
                try {
                    terminal = encoder->finish(outcome);
                    if (auto comment = ngram_generation_comment(outcome.metrics.ngram_archive);
                        !comment.empty()) {
                        terminal.insert(terminal.begin(), std::move(comment));
                    }
                } catch (const std::exception& exception) {
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.message = exception.what();
                    return send_error(error);
                }
                try {
                    transport.write(terminal);
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) {
                    lifecycle->response_failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
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
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
    }
}

} // namespace infernix::serve
