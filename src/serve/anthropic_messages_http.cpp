#include "serve/http_server.h"

#include "serve/anthropic_messages.h"
#include "serve/http_stream_driver.h"
#include "serve/http_transport.h"
#include "serve/image_chat_hook.h" // the image lane's one attach point on this route
#include "serve/server_tools.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace infernix::serve {
namespace {

// The 500 for an exception a request's handling did not expect, with its message (rendering names its Anthropic
// type, api_error).
ApiError anthropic_internal_error(const std::exception& exception) {
    ApiError error;
    error.status  = 500;
    error.message = exception.what();
    return error;
}

// One Messages response. A request that declares tool search tools may run several generations:
// each that ends in searches only is followed by the searches' results and another generation,
// until the model answers, calls a client tool, or a limit ends the response.
struct AnthropicResponseState {
    explicit AnthropicResponseState(AnthropicMessagesRequest parsed)
        : request(std::move(parsed)), executor(request), continuation(request.generation) {
        rounds.reserve(kMaximumServerToolRounds);
        round_results.reserve(kMaximumServerToolRounds);
    }

    AnthropicMessagesRequest request;
    AnthropicToolSearchExecutor executor;
    GenerationRequest continuation; // the request plus every completed round
    std::vector<ServerToolRound> rounds;
    std::vector<std::vector<std::optional<RequestJson>>> round_results;

    // The final generation and how the response ends.
    GenerationOutcome final_outcome;
    std::vector<ToolCall> final_calls;
    std::vector<std::optional<RequestJson>> final_results;
    std::optional<std::string_view> stop_reason;

    [[nodiscard]] AnthropicResponseSegment round_segment(std::size_t index) const {
        return AnthropicResponseSegment{.outcome        = &rounds[index].outcome,
                                        .calls          = rounds[index].calls,
                                        .search_results = round_results[index]};
    }

    [[nodiscard]] AnthropicResponseSegment final_segment() const {
        return AnthropicResponseSegment{
            .outcome = &final_outcome, .calls = final_calls, .search_results = final_results};
    }

    [[nodiscard]] std::vector<AnthropicResponseSegment> segments() const {
        std::vector<AnthropicResponseSegment> result;
        for (std::size_t index = 0; index < rounds.size(); ++index) {
            result.push_back(round_segment(index));
        }
        result.push_back(final_segment());
        return result;
    }

    [[nodiscard]] GenerationOutcome totals() const {
        return combine_round_outcomes(rounds, final_outcome);
    }
};

using ContinuationPreparer = std::function<PreparedRequest(const GenerationRequest&)>;
using SegmentWriter        = std::function<void(const AnthropicResponseSegment&)>;
// A generation's calls with wire identifiers: a stream's keep those it streamed them with.
using CallIdentifier = std::function<std::vector<ToolCall>(const GenerationOutcome&)>;

// Runs `prepared` and every continuation its searches lead to, leaving the final generation in
// `state`. `prepared` ends holding the last generation's request (and its ingress reservation).
void run_anthropic_response(GenerationBackend& service, AnthropicResponseState& state,
                            PreparedRequest& prepared, const StreamSink* sink,
                            const std::function<bool()>& is_cancelled,
                            const ContinuationPreparer& prepare_next,
                            const SegmentWriter& on_segment, const CallIdentifier& identify) {
    // The protocol's part of run_server_tool_rounds: the searches the executor knows run, any other call is the
    // client's, and each generation keeps its searches' result content for its segment.
    struct Protocol {
        AnthropicResponseState& state;
        const SegmentWriter& on_segment;
        const CallIdentifier& identify;
        std::vector<std::optional<RequestJson>> results; // the current generation's, by call

        ServerRoundCalls calls(const GenerationOutcome& outcome) {
            ServerRoundCalls out;
            out.calls = identify(outcome);
            results.assign(out.calls.size(), std::nullopt);
            if (outcome.finish_reason == infernix::FinishReason::Cancelled) { return out; }
            for (std::size_t index = 0; index < out.calls.size(); ++index) {
                if (state.executor.search_tool(out.calls[index].name) == nullptr) {
                    out.client_calls = true;
                    continue;
                }
                AnthropicToolSearchResult executed = state.executor.execute(out.calls[index]);
                results[index] = std::move(executed.content);
                out.results.push_back(std::move(executed.turn));
            }
            return out;
        }

        [[nodiscard]] bool limit_reached() const noexcept { return false; }

        void keep() {
            state.round_results.push_back(std::move(results));
            if (on_segment) { on_segment(state.round_segment(state.rounds.size() - 1U)); }
        }

        void end(ServerToolRound&& round, ServerRoundsEnd end) {
            state.final_outcome = std::move(round.outcome);
            state.final_calls   = std::move(round.calls);
            state.final_results = std::move(results);
            switch (end) {
            case ServerRoundsEnd::Finished: state.stop_reason = std::nullopt; break;
            case ServerRoundsEnd::RoundLimit: state.stop_reason = "pause_turn"; break;
            case ServerRoundsEnd::OutputLimit: state.stop_reason = "max_tokens"; break;
            case ServerRoundsEnd::ContextWindow: state.stop_reason = "model_context_window_exceeded"; break;
            }
        }
    } protocol{.state = state, .on_segment = on_segment, .identify = identify};
    run_server_tool_rounds(service, prepared, sink, is_cancelled, prepare_next, state.continuation, state.rounds,
                           state.executor.names(), protocol);
}

} // namespace

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
        write_anthropic_error(res, anthropic_internal_error(exception), request_id);
    }
}

void HttpServer::handle_messages(const httplib::Request& req, httplib::Response& res) {
    const std::string request_id = res.get_header_value("request-id");

    std::shared_ptr<AnthropicResponseState> state;
    try {
        require_generation();
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        limits.claude_code_prompt = options_.claude_code_prompt;
        const auto body           = parse_json_body(req);
        AnthropicMessagesRequest parsed = parse_anthropic_messages_request(body, limits);
        parsed.generation.ngram_session = resolve_ngram_session(req, body, options_);
        state = std::make_shared<AnthropicResponseState>(std::move(parsed));
    } catch (const ApiException& exception) {
        write_anthropic_error(res, exception.error(), request_id);
        return;
    } catch (const std::exception& exception) {
        operational_log_.http_failure(
            "anthropic_messages",
            make_internal_request_failure(RequestFailurePhase::Http, exception.what()), request_id);
        write_anthropic_error(res, anthropic_internal_error(exception), request_id);
        return;
    }
    const AnthropicMessagesRequest& request = state->request;

    const std::uint64_t req_id = ++request_seq_;
    // PORT(ported tree): re-adds the TOOL_INJECTION seam of FORK src/serve/anthropic_messages_http.cpp:201,
    // which the automated pass dropped: the hook, the tool it injects, and the observation the rounds share.
    // The lane owns the tool's text and every rule about it (serve/image_chat_hook.cpp); this file keeps its
    // own request, its `service_`, its re-prepare/re-run callbacks, its route label, and the two reads of its
    // private state the lane cannot make (`images_`, `run_image_job`), handed over as the typed
    // ChatImageDoors. Upstream tip reads the request through the const reference below, so the lane's two
    // calls name `state->request.generation`, which is the same object the fork's non-const reference was.
    auto hook = make_chat_image_hook(
        *this, req, "anthropic_messages", req_id, options_,
        ChatImageDoors{
            [this] { return static_cast<void*>(images_); },
            [this](const auto& image_request, std::string_view base, std::string_view source,
                   std::string_view protocol, std::string correlation) {
                return run_image_job(image_request, base, source, protocol, std::move(correlation));
            }});
    inject_chat_image_tool(hook, state->request.generation);
    RequestLogMetadata metadata{.http_request_id        = request_id,
                                .model                  = request.model,
                                .stream                 = request.stream,
                                .output_tokens_explicit = request.output_tokens_explicit,
                                .label                  = request_label(req)};
    const GenerationConsumerMode consumer_mode =
        request.stream ? GenerationConsumerMode::Streaming : GenerationConsumerMode::Aggregate;
    // PORT(ported tree): re-adds the shared observation of FORK src/serve/anthropic_messages_http.cpp:220,
    // which the rounds below re-prepare with: every round of one request is one HTTP request, with the same
    // observation wiring and the same request log, so a round the server runs itself stays part of this
    // request. Upstream tip passes the object as an inline literal at the first prepare and has added
    // `.progress` since the fork was written, so its fields are copied from upstream's own literal --
    // upstream's literal is left exactly as it stands -- and everything read is captured by value because a
    // streaming provider runs after this handler has returned.
    const std::string first_http_request_id = metadata.http_request_id;
    const auto make_observation             = [this, req_id, first_http_request_id] {
        return infernix::GenerationObservationOptions{
            .phase_timings = true,
            .scheduling    = scheduling_observer(req_id, first_http_request_id),
            .first_token   = first_token_observer(),
            .progress      = progress_observer(req_id),
        };
    };
    PreparedRequest prepared;
    try {
        prepared = service_->prepare(request.generation, consumer_mode,
                                     {.phase_timings = true,
                                      .scheduling    = scheduling_observer(req_id, metadata.http_request_id),
                                      .first_token   = first_token_observer(),
                                      .progress      = progress_observer(req_id)},
                                     [&req] { return client_disconnected(req); });
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
    // Continuations after a search resolve the same documents; the first generation's sources serve them all.
    const std::shared_ptr<const CitationRegistry> citations = prepared.citations;

    auto lifecycle = begin_request(make_request_log_context(
        req_id, "anthropic_messages", request.generation, metadata, prepared));

    // A continuation after a search keeps the request's ingress reservation and log identity; its
    // first token is not the response's.
    const std::string http_request_id = metadata.http_request_id;
    const auto continuation_preparer  = [this, consumer_mode, req_id,
                                         http_request_id](const std::function<bool()>& cancelled) {
        return ContinuationPreparer([this, consumer_mode, req_id, http_request_id,
                                     cancelled](const GenerationRequest& next) {
            return service_->prepare_continuation(
                next, consumer_mode,
                {.phase_timings = true,
                 .scheduling    = scheduling_observer(req_id, http_request_id),
                 .progress      = progress_observer(req_id)},
                cancelled);
        });
    };

    if (!request.stream) {
        const std::function<bool()> cancelled = [&req] { return client_disconnected(req); };
        try {
            run_anthropic_response(*service_, *state, prepared, nullptr, cancelled,
                                   continuation_preparer(cancelled), {},
                                   [&](const GenerationOutcome& outcome) {
                                       return identify_anthropic_tool_calls(outcome, request.tool_searches);
                                   });
            // PORT(ported tree): re-adds the aggregate picture rounds of FORK
            // src/serve/anthropic_messages_http.cpp:278 (the automated pass kept this one). The lane runs the
            // calls the model made to the server's own tool and the answer replaces itself, so the handler
            // re-derives its own call bookkeeping from the answer that comes back: the hidden image call of
            // the round before is no longer part of this one, and any call the answer now carries is a client
            // call (what the rounds were, and why they are needed, is in serve/image_chat_hook.cpp). Upstream
            // tip reads the request through a const reference, so the generation is named through `state`,
            // which is the same object the fork's non-const `request` referred to.
            if (chat_image_round_needed(hook, state->final_outcome)) {
                auto pictures = run_chat_image_rounds(
                    *hook, state->request.generation, std::move(state->final_outcome),
                    [&] {
                        // A new round takes over this request's generation, and with it the lifetime
                        // that keeps media alive until the response is written.
                        prepared = service_->prepare(state->request.generation, consumer_mode,
                                                     make_observation(),
                                                     [&req] { return client_disconnected(req); });
                        return service_->run(prepared, nullptr,
                                             [&req] { return client_disconnected(req); });
                    });
                state->final_outcome       = std::move(pictures.answer);
                state->final_outcome.text += chat_image_missing_url_suffix(
                    *hook, state->final_outcome.text, pictures.urls);
                // The segment's calls follow its generation, and any call the answer now carries is a
                // client call.
                state->final_calls   = identify_anthropic_tool_calls(state->final_outcome,
                                                                    request.tool_searches);
                state->final_results = std::vector<std::optional<RequestJson>>(
                    state->final_calls.size());
            }
        } catch (const ApiException& exception) {
            const ApiError error = normalize_anthropic_error(exception.error());
            lifecycle->failure(make_generation_request_failure(error));
            write_anthropic_error(res, error, request_id);
            return;
        } catch (const std::exception& exception) {
            lifecycle->failure(
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
            write_anthropic_error(res, anthropic_internal_error(exception), request_id);
            return;
        }
        const GenerationOutcome totals = state->totals();
        lifecycle->done(totals);
        if (totals.finish_reason == infernix::FinishReason::Cancelled) {
            // A message cannot carry a cancelled generation (the client left, or the server is stopping): the
            // cancellation error the server gives elsewhere, not a render failure counted as internal.
            ApiError error;
            error.status  = 499;
            error.type    = "request_cancelled";
            error.code    = "client_disconnected";
            error.message = "the request was cancelled before it finished";
            write_anthropic_error(res, error, request_id);
            return;
        }
        try {
            set_ngram_generation_header(res, state->final_outcome.ngram_archive);
            const std::vector<AnthropicResponseSegment> segments = state->segments();
            set_owned_json_content(res,
                                   request.prewarm
                                       ? make_anthropic_prewarm_response(identity, totals)
                                       : make_anthropic_messages_response(identity, segments, totals,
                                                                          state->stop_reason,
                                                                          request.hide_thinking, citations.get()),
                                   prepared.lifetime);
        } catch (const ApiException& exception) {
            const ApiError error = normalize_anthropic_error(exception.error());
            lifecycle->response_failure(
                make_request_failure(RequestFailurePhase::ResponseRender, error));
            write_anthropic_error(res, error, request_id);
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            write_anthropic_error(res, anthropic_internal_error(exception), request_id);
        }
        return;
    }

    // The event stream cannot be taken back once committed: wait up to kStreamAdmissionGrace for the
    // Engine's admission of the first generation, so the queue's prompt answers are a plain HTTP status
    // here as on the non-streaming branch. A request still queued then starts its stream: pings keep
    // the client while it waits, and a later queue timeout is the stream's error event, as a later
    // round's failure (after a server tool) is.
    try {
        (void)service_->admit(prepared, [&req] { return client_disconnected(req); },
                              std::chrono::steady_clock::now() + kStreamAdmissionGrace);
    } catch (const ApiException& exception) {
        const ApiError error = normalize_anthropic_error(exception.error());
        lifecycle->failure(make_generation_request_failure(error));
        write_anthropic_error(res, error, request_id);
        return;
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
        write_anthropic_error(res, anthropic_internal_error(exception), request_id);
        return;
    }

    try {
        auto stream  = std::make_shared<HttpGenerationStream>(std::move(prepared));
        // The compat mode reaches only this streaming encoder: the non-streaming branch and the
        // terminal usage here are untouched, and an error after admission but before the Engine's
        // start observation still starts the message with its provisional usage and then reports
        // the error event.
        auto encoder = std::make_shared<AnthropicMessagesStream>(
            identity, input_tokens, request.hide_thinking, request.tool_searches,
            options_.claude_code_usage_compat, citations);

        prepare_sse_response(res);
        // PORT(ported tree): the capture gains the two things the picture rounds need to re-prepare and
        // re-run this request after the provider outlives the handler -- the hook it draws through (the
        // request its rounds extend already lives in `state`, which is captured) and the observation every
        // round of this request shares -- exactly as in FORK src/serve/anthropic_messages_http.cpp:370. A
        // null hook carries nothing.
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, stream, encoder, lifecycle, state, hook,
             make_observation,
             continuation_preparer](std::size_t, httplib::DataSink& sink) -> bool {
                if (stream->started.exchange(true, std::memory_order_acq_rel)) {
                    sink.done();
                    return true;
                }
                SseTransport transport(sink, stream->cancelled);
                transport.set_heartbeat([&] { return encoder->ping(); });
                const auto send_error = [&](const ApiError& error) {
                    return end_stream_with_error(sink, *lifecycle, [&] {
                        if (!encoder->started()) {
                            render_and_write(transport, [&] { return encoder->start(); });
                        }
                        render_and_write(transport, [&] { return encoder->error(error); });
                    });
                };

                try {
                    StreamSink output;
                    // Only the first generation starts the message; later ones continue it.
                    output.on_start = [&](const infernix::GenerationStart& start) {
                        if (!encoder->started()) {
                            render_and_write(transport, [&] { return encoder->start(start); });
                        }
                    };
                    output.on_reasoning = [&](const std::string& text) {
                        render_and_write(transport, [&] { return encoder->reasoning_delta(text); });
                    };
                    output.on_content = [&](const std::string& text) {
                        render_and_write(transport, [&] { return encoder->content_delta(text); });
                    };
                    output.on_tool_call = [&](const infernix::ToolCallDelta& delta) {
                        render_and_write(transport, [&] { return encoder->tool_call_delta(delta); });
                    };
                    output.is_cancelled = [&] { return transport.poll(); };
                    const std::function<bool()> cancelled = [&] { return transport.poll(); };

                    run_anthropic_response(
                        *service_, *state, stream->prepared, &output, {},
                        continuation_preparer(cancelled),
                        [&](const AnthropicResponseSegment& segment) {
                            render_and_write(transport, [&] { return encoder->segment(segment); });
                        },
                        [&](const GenerationOutcome& outcome) { return encoder->identify_calls(outcome); });

                    // PORT(ported tree): re-adds the streamed picture rounds of FORK
                    // src/serve/anthropic_messages_http.cpp:466, which the automated pass dropped, in
                    // upstream tip's own position (inside this try, so a round's failure ends the stream the
                    // way upstream ends it). A picture round streams nothing of its own: its start would open
                    // a second message inside the streamed one and its text reaches the client as the answer's
                    // tail, so it runs on a sink that carries nothing but cancellation. The lane says what the
                    // silence while a picture is drawn carries on the wire. Two mechanisms, because they
                    // cover different spans: the lane's own HeartbeatTicker (serve/http_transport.h,
                    // re-added by hand verbatim from FORK src/serve/anthropic_messages_http.cpp:471) is the
                    // stream's only traffic while the draw itself blocks this thread -- upstream tip's
                    // poll()-driven heartbeat cannot run then -- and upstream tip's single heartbeat slot
                    // carries the same note through the rounds' re-prepare and re-run, where the Engine wait
                    // loop does call poll(), with upstream's own ping put back after them. The answer
                    // replaces itself, so the handler re-derives its own call bookkeeping from it: the hidden
                    // image call of the round before is no longer part of this one. Null hook, null work.
                    if (chat_image_round_needed(hook, state->final_outcome)) {
                        // The ticker is also the stall watchdog's only observer: only this thread can see
                        // that the job stopped reporting, and only the lane knows when that means a wedge
                        // (serve/image_job_progress.h, armed by src/models/qwen_image/image_pipeline.cu).
                        HeartbeatTicker ticker(
                            transport,
                            [&hook]() -> std::string { return chat_image_progress_note(*hook); },
                            [&hook]() -> std::shared_ptr<ImageJobProgress> {
                                return chat_image_watch_progress(*hook);
                            });
                        transport.set_heartbeat([&] {
                            const std::string note = chat_image_progress_note(*hook);
                            return note.empty() ? encoder->ping() : note;
                        });
                        auto pictures = run_chat_image_rounds(
                            *hook, state->request.generation, std::move(state->final_outcome),
                            [&] {
                                stream->prepared = service_->prepare(
                                    state->request.generation, GenerationConsumerMode::Streaming,
                                    make_observation(), [&] { return transport.poll(); });
                                StreamSink quiet;
                                quiet.is_cancelled = [&] { return transport.poll(); };
                                return service_->run(stream->prepared, &quiet);
                            });
                        state->final_outcome       = std::move(pictures.answer);
                        state->final_outcome.text += chat_image_missing_url_suffix(
                            *hook, state->final_outcome.text, pictures.urls);
                        state->final_calls   = encoder->identify_calls(state->final_outcome);
                        state->final_results = std::vector<std::optional<RequestJson>>(
                            state->final_calls.size());
                        transport.set_heartbeat([&] { return encoder->ping(); });
                    }
                } catch (...) {
                    return end_failed_stream_generation(*lifecycle, normalize_anthropic_error,
                                                        anthropic_internal_error, send_error);
                }

                const GenerationOutcome totals = state->totals();
                if (!record_stream_generation(*lifecycle, totals, stream->cancelled)) { return false; }
                std::vector<std::string> terminal;
                try {
                    terminal = encoder->finish(state->final_segment(), totals, state->stop_reason);
                    if (auto comment = ngram_generation_comment(state->final_outcome.ngram_archive);
                        !comment.empty()) {
                        terminal.insert(terminal.begin(), std::move(comment));
                    }
                } catch (const std::exception& exception) {
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    return send_error(anthropic_internal_error(exception));
                }
                return write_stream_terminal(transport, sink, *lifecycle, terminal);
            },
            stream_release(stream, lifecycle));
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::ResponseRender, exception.what()));
        write_anthropic_error(res, anthropic_internal_error(exception), request_id);
    }
}

} // namespace infernix::serve
