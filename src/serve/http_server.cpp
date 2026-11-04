#include "serve/http_server.h"

#include "infernix/ops/attn_dit_joint.h"
#include "product/logging/logging.h"
#include "product/speculative_options.h"
#include "serve/anthropic_messages.h"
#include "serve/chat_image_tool.h"
#include "serve/http_transport.h"
#include "serve/image_generation_service.h"
#include "serve/openai_common.h"
#include "serve/openai_images.h"
#include "serve/request_log.h"

#include <nlohmann/json.hpp>
#include <spdlog/logger.h>

#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace infernix::serve {
namespace {

void write_exception(httplib::Response& res, const std::exception& ex) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = ex.what();
    write_openai_error(res, error);
}

bool is_anthropic_path(std::string_view path) {
    return canonical_api_path(path).starts_with("/v1/messages");
}

bool is_openai_path(std::string_view path) {
    return path.starts_with("/v1/") && !is_anthropic_path(path);
}

void ensure_http_request_id(const httplib::Request& request, httplib::Response& response) {
    if (is_anthropic_path(request.path) && !response.has_header("request-id")) {
        response.set_header("request-id", new_anthropic_request_id());
    } else if (is_openai_path(request.path) && !response.has_header("x-request-id")) {
        response.set_header("x-request-id", new_openai_request_id());
    }
}

ThroughputReport make_throughput_report(const infernix::RuntimeStats& previous,
                                        const infernix::RuntimeStats& current,
                                        double interval_seconds) {
    return ThroughputReport{
        .interval_seconds = interval_seconds,
        .computed_prefill_tokens =
            current.computed_prefill_tokens - previous.computed_prefill_tokens,
        .committed_decode_tokens =
            current.committed_decode_tokens - previous.committed_decode_tokens,
        .decode_rounds     = current.decode_rounds - previous.decode_rounds,
        .decode_row_rounds = current.decode_row_rounds - previous.decode_row_rounds,
        .previous          = previous,
        .current           = current,
    };
}

bool report_has_activity(const ThroughputReport& report) {
    return report.computed_prefill_tokens != 0 || report.committed_decode_tokens != 0 ||
           report.decode_rounds != 0 || report.current.running_requests != 0 ||
           report.current.waiting_requests != 0 || report.current.paused_requests != 0 ||
           report.current.replaying_requests != 0 || report.current.materializing_requests != 0 ||
           report.current.terminal_pending_requests != 0 ||
           report.current.preemptions != report.previous.preemptions ||
           report.current.snapshot_restores != report.previous.snapshot_restores ||
           report.current.replay_restores != report.previous.replay_restores ||
           report.current.replayed_tokens != report.previous.replayed_tokens ||
           report.current.root_selections != report.previous.root_selections ||
           report.current.prefix_selections != report.previous.prefix_selections ||
           report.current.state_forks != report.previous.state_forks ||
           report.current.materialization_state_forks !=
               report.previous.materialization_state_forks ||
           report.current.state_restores != report.previous.state_restores ||
           report.current.partial_tail_cow_pages != report.previous.partial_tail_cow_pages ||
           report.current.device_state_occupied_slots !=
               report.previous.device_state_occupied_slots ||
           report.current.device_main_kv_occupied_pages !=
               report.previous.device_main_kv_occupied_pages ||
           report.current.device_backend_kv_occupied_pages !=
               report.previous.device_backend_kv_occupied_pages ||
           report.current.host_work.engine_boundary_ns !=
               report.previous.host_work.engine_boundary_ns ||
           report.current.host_work.program_submit_ns !=
               report.previous.host_work.program_submit_ns ||
           report.current.host_work.program_post_ns != report.previous.host_work.program_post_ns ||
           report.current.host_work.engine_commit_output_ns !=
               report.previous.host_work.engine_commit_output_ns ||
           report.current.host_work.engine_maintenance_ns !=
               report.previous.host_work.engine_maintenance_ns ||
           report.current.host_work.device_wait_ns != report.previous.host_work.device_wait_ns;
}

const char* endpoint_name(std::string_view request_path) noexcept {
    const std::string_view path = canonical_api_path(request_path);
    if (path == "/v1/chat/completions") { return "openai_chat_completions"; }
    if (path == "/v1/responses") { return "openai_responses"; }
    if (path == "/v1/responses/input_tokens") { return "openai_responses_input_tokens"; }
    if (path == "/v1/messages") { return "anthropic_messages"; }
    if (path == "/v1/messages/count_tokens") { return "anthropic_count_tokens"; }
    if (path == "/v1/decide" || path == "/v1/systemone") { return "decide"; }
    return "http_route";
}

std::string response_request_id(const httplib::Response& response) {
    if (response.has_header("x-request-id")) { return response.get_header_value("x-request-id"); }
    if (response.has_header("request-id")) { return response.get_header_value("request-id"); }
    return {};
}

} // namespace

void write_openai_error(httplib::Response& response, const ApiError& error) {
    response.status = error.status;
    if (error.retry_after_seconds > 0) {
        response.set_header("Retry-After", std::to_string(error.retry_after_seconds));
    }
    response.set_content(make_error_body(error), "application/json");
}

void write_anthropic_error(httplib::Response& response, const ApiError& api_error,
                           const std::string& request_id) {
    const ApiError error = normalize_anthropic_error(api_error);
    response.status      = error.status;
    response.headers.erase("request-id");
    response.set_header("request-id", request_id);
    if (error.retry_after_seconds > 0) {
        response.set_header("Retry-After", std::to_string(error.retry_after_seconds));
    }
    response.set_content(make_anthropic_error_body(error, request_id), "application/json");
}

httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response) {
    ensure_http_request_id(request, response);
    if (!response.body.empty()) { return httplib::Server::HandlerResponse::Unhandled; }

    ApiError error;
    if (response.status == 413) {
        error.status  = 413;
        error.type    = "invalid_request_error";
        error.code    = "request_too_large";
        error.message = "request body exceeds the configured payload limit of " +
                        std::to_string(options.max_request_bytes) + " bytes";
    } else if (response.status == 404 && is_anthropic_path(request.path)) {
        error.status  = 404;
        error.code    = "not_found";
        error.message = "requested Anthropic resource was not found";
    } else {
        return httplib::Server::HandlerResponse::Unhandled;
    }
    if (is_anthropic_path(request.path)) {
        write_anthropic_error(response, error, response.get_header_value("request-id"));
    } else {
        write_openai_error(response, error);
    }
    return httplib::Server::HandlerResponse::Handled;
}

bool matches_bearer_credential(std::string_view authorization, std::string_view api_key) noexcept {
    if (api_key.empty()) { return false; }
    const auto is_whitespace = [](char value) { return value == ' ' || value == '\t'; };
    const auto ascii_equal   = [](char lhs, char rhs) {
        if (lhs >= 'A' && lhs <= 'Z') { lhs = static_cast<char>(lhs - 'A' + 'a'); }
        if (rhs >= 'A' && rhs <= 'Z') { rhs = static_cast<char>(rhs - 'A' + 'a'); }
        return lhs == rhs;
    };

    std::size_t position = 0;
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    constexpr std::string_view scheme = "Bearer";
    if (authorization.size() - position < scheme.size()) { return false; }
    for (std::size_t index = 0; index < scheme.size(); ++index) {
        if (!ascii_equal(authorization[position + index], scheme[index])) { return false; }
    }
    position += scheme.size();
    if (position == authorization.size() || !is_whitespace(authorization[position])) {
        return false;
    }
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    std::size_t end = authorization.size();
    while (end > position && is_whitespace(authorization[end - 1])) { --end; }
    return authorization.substr(position, end - position) == api_key;
}

std::string api_route_pattern(std::string_view endpoint) {
    return "(?:/v1)?/v1" + std::string(endpoint);
}

std::string_view canonical_api_path(std::string_view path) noexcept {
    constexpr std::string_view kVersion = "/v1";
    if (path.starts_with("/v1/v1/")) { path.remove_prefix(kVersion.size()); }
    return path;
}

HttpServer::HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger,
                       std::shared_ptr<product::TerminalPanel> panel)
    : options_(std::move(options)), openai_responses_store_(options_.response_store_max_records,
                                                            options_.response_store_max_bytes),
      operational_log_(logger), request_jsonl_(options_.request_log_jsonl, options_.artifact_path,
                                               std::move(logger), options_.request_log_rotation) {
    if (options_.log_stats_panel && panel != nullptr && panel->enabled()) {
        console_stats_ = std::make_unique<ConsoleStatsPanel>(
            std::move(panel), options_.speculative.ngram_draft_tokens != 0 ||
                                  options_.speculative.ngram_draft_tokens_auto);
    }
    const std::size_t queued_requests =
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests;
    const std::size_t worker_count = queued_requests + 1;
    server_.new_task_queue         = [queued_requests, worker_count] {
        return new httplib::ThreadPool(worker_count, worker_count, queued_requests);
    };
    server_.set_socket_options(configure_http_server_socket);
    server_.set_payload_max_length(options_.max_request_bytes);
    register_routes();
}

HttpServer::RequestLifecycle::RequestLifecycle(HttpServer& owner, RequestLogContext context)
    : owner_(&owner), context_(std::move(context)) {
    owner_->record_request_start(context_);
}

bool HttpServer::RequestLifecycle::claim(State terminal) noexcept {
    State expected = State::Pending;
    return state_.compare_exchange_strong(expected, terminal, std::memory_order_acq_rel);
}

void HttpServer::RequestLifecycle::done(const GenerationOutcome& outcome) {
    if (claim(State::Done)) { owner_->record_request_done(context_, outcome); }
}

void HttpServer::RequestLifecycle::failure(const RequestFailure& failure) {
    if (claim(State::Error)) { owner_->record_request_failure(context_, failure); }
}

void HttpServer::RequestLifecycle::response_failure(const RequestFailure& failure) {
    owner_->record_response_failure(context_.id, failure);
}

std::shared_ptr<HttpServer::RequestLifecycle> HttpServer::begin_request(RequestLogContext context) {
    return std::make_shared<RequestLifecycle>(*this, std::move(context));
}

void HttpServer::record_request_start(const RequestLogContext& context) {
    request_jsonl_.write_request_start(context);
    operational_log_.request_start(context);
}

void HttpServer::record_request_rejected(const RequestRejectionLogContext& context) {
    metrics_.rejected();
    request_jsonl_.write_request_rejected(context);
    operational_log_.request_rejected(context);
    if (console_stats_) {
        console_stats_->request_rejected(
            make_request_failure(RequestFailurePhase::Prepare, context.error));
    }
}

infernix::GenerationFirstTokenObserver HttpServer::first_token_observer() {
    return [this](const infernix::GenerationFirstTokenObservation& observation) {
        metrics_.first_token(observation);
    };
}

infernix::GenerationSchedulingObserver HttpServer::scheduling_observer(std::uint64_t request_id,
                                                                     std::string http_request_id) {
    if (!request_jsonl_.enabled()) { return {}; }
    return [this, request_id, http_request_id = std::move(http_request_id)](
               const infernix::GenerationSchedulingObservation& observation) {
        request_jsonl_.write_request_scheduling(request_id, http_request_id, observation);
    };
}

void HttpServer::record_request_done(const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    metrics_.done(outcome);
    metrics_json_history_.done(context, outcome);
    request_jsonl_.write_request_done(context, outcome);
    operational_log_.request_done(context, outcome);
    if (console_stats_) { console_stats_->request_done(outcome); }
}

void HttpServer::record_request_failure(const RequestLogContext& context,
                                        const RequestFailure& failure) {
    const bool cancelled = failure.classification == RequestFailureClass::ClientDisconnected;
    metrics_.failed(cancelled);
    metrics_json_history_.failure(context, cancelled);
    request_jsonl_.write_request_error(context, failure.machine_message);
    operational_log_.request_failure(context, failure);
    if (console_stats_) { console_stats_->request_failure(failure); }
}

void HttpServer::record_response_failure(std::uint64_t request_id, const RequestFailure& failure) {
    metrics_.response_failed();
    operational_log_.response_failure(request_id, failure);
}

std::string HttpServer::metrics_json_body() const {
    MetricsJsonSnapshot snapshot;
    const infernix::LoadSummary load      = service_->load_summary();
    const infernix::EngineOptions& engine = service_->engine_options();
    const Metrics::RequestCounts counts   = metrics_.counts();

    snapshot.instance_id  = request_jsonl_.server_instance_id();
    snapshot.model        = public_model_id_;
    snapshot.architecture = load.architecture;
    snapshot.weights = load.weight_formats.empty() ? std::string{} : load.weight_formats.front();
    snapshot.kv      = infernix::kv_cache_cli_name(engine.kv_cache);
    snapshot.speculative_backend = product::speculative_backend_name(engine.speculative.backend);
    snapshot.time_unix           = unix_time_seconds();
    snapshot.start_time_unix     = metrics_.start_time_seconds();
    snapshot.max_context         = engine.max_context;
    snapshot.max_concurrency     = engine.max_concurrency;
    snapshot.stats               = service_->runtime_stats();
    snapshot.counts              = {.completed = counts.completed,
                                    .cancelled = counts.cancelled,
                                    .failed    = counts.failed,
                                    .rejected  = counts.rejected};
    snapshot.requests            = metrics_json_history_.rows();
    snapshot.rates_valid         = metrics_json_history_.rates(snapshot.decode_tokens_per_second,
                                                               snapshot.prefill_tokens_per_second);
    return render_metrics_json(snapshot);
}

void HttpServer::record_throughput(const ThroughputReport& report) {
    if (report.interval_seconds > 0.0) {
        metrics_json_history_.set_rates(
            static_cast<double>(report.committed_decode_tokens) / report.interval_seconds,
            static_cast<double>(report.computed_prefill_tokens) / report.interval_seconds);
    }
    request_jsonl_.write_throughput(report);
    operational_log_.throughput(report);
}

void HttpServer::run_stats_reporter() {
    using Clock                     = std::chrono::steady_clock;
    // The device's clocks, power, energy and throttle reasons beside each interval (NVML; every
    // sample is invalid without it).
    const infernix::GpuTelemetry telemetry(options_.device);
    infernix::GpuTelemetrySample gpu_previous = telemetry.sample();
    infernix::RuntimeStats previous   = service_->runtime_stats();
    Clock::time_point previous_time = Clock::now();
    const auto interval             = std::chrono::milliseconds(options_.log_stats_interval_ms);
    Clock::time_point next_deadline = previous_time + interval;
    // An exception leaving this thread would terminate the server: a failed report is noted on
    // stderr (once; the log writers may be what failed) and the next interval reports again.
    bool failure_noted       = false;
    const auto note_failure = [&failure_noted](const char* what) noexcept {
        if (failure_noted) { return; }
        failure_noted = true;
        std::fprintf(stderr, "[infernix-serve] throughput report failed: %s; reporting continues\n",
                     what);
    };

    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_until(lock, next_deadline, [this] { return stats_stopping_; })) {
                break;
            }
        }

        try {
            const infernix::RuntimeStats current = service_->runtime_stats();
            const Clock::time_point now        = Clock::now();
            ThroughputReport report            = make_throughput_report(
                previous, current, std::chrono::duration<double>(now - previous_time).count());
            report.gpu_previous = gpu_previous;
            report.gpu_current  = telemetry.sample();
            gpu_previous        = report.gpu_current;
            previous      = current;
            previous_time = now;
            if (report_has_activity(report)) { record_throughput(report); }
            if (console_stats_) { console_stats_->runtime(current); }
        } catch (const std::exception& error) {
            note_failure(error.what());
        } catch (...) {
            note_failure("unknown error");
        }
        next_deadline += interval;
        const Clock::time_point after_write = Clock::now();
        if (next_deadline <= after_write) { next_deadline = after_write + interval; }
    }

    try {
        const infernix::RuntimeStats current = service_->runtime_stats();
        const Clock::time_point now        = Clock::now();
        ThroughputReport tail              = make_throughput_report(
            previous, current, std::chrono::duration<double>(now - previous_time).count());
        tail.gpu_previous = gpu_previous;
        tail.gpu_current  = telemetry.sample();
        // The exact partial interval remains useful to measurement consumers. Pretty throughput is
        // a fixed-cadence operational record and deliberately has no irregular shutdown tail.
        tail.final_interval = true;
        request_jsonl_.write_throughput(tail);
    } catch (const std::exception& error) {
        note_failure(error.what());
    } catch (...) {
        note_failure("unknown error");
    }
}

void HttpServer::stop_stats_reporter() {
    if (!stats_thread_.joinable()) { return; }
    {
        std::lock_guard lock(stats_mutex_);
        stats_stopping_ = true;
    }
    stats_cv_.notify_one();
    stats_thread_.join();
}

void HttpServer::register_routes() {
    server_.set_error_handler([this](const httplib::Request& request, httplib::Response& response) {
        return handle_unrendered_http_error(options_, request, response);
    });
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Expose-Headers", "x-request-id, request-id"},
             {"Access-Control-Allow-Headers",
              "Authorization, Content-Type, X-API-Key, anthropic-version, anthropic-beta, "
              "anthropic-user-profile-id"},
             {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key.
        server_.Options(R"(.*)",
                        [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
    }

    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        ensure_http_request_id(req, res);
        if (options_.api_key.empty() || req.path == "/health" || req.method == "OPTIONS") {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        // Accept both the OpenAI-style bearer token and the Anthropic-style
        // x-api-key header so OpenAI clients and Claude Code (ANTHROPIC_API_KEY
        // -> x-api-key, ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer) both work.
        const bool bearer_ok =
            matches_bearer_credential(req.get_header_value("Authorization"), options_.api_key);
        const bool x_api_key_ok = req.get_header_value("x-api-key") == options_.api_key;
        if (!bearer_ok && !x_api_key_ok) {
            ApiError error;
            error.status  = 401;
            error.type    = "invalid_request_error";
            error.code    = "invalid_api_key";
            error.message = "missing or invalid API key";
            // Render the 401 in the shape the target endpoint speaks.
            if (is_anthropic_path(req.path)) {
                write_anthropic_error(res, error, res.get_header_value("request-id"));
            } else {
                write_openai_error(res, error);
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server_.set_exception_handler(
        [this](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
            ensure_http_request_id(req, res);
            try {
                std::rethrow_exception(ep);
            } catch (const ApiException& e) {
                if (e.error().status >= 500) {
                    operational_log_.http_failure(
                        endpoint_name(req.path),
                        make_request_failure(RequestFailurePhase::Http, e.error()),
                        response_request_id(res));
                }
                if (is_anthropic_path(req.path)) {
                    write_anthropic_error(res, e.error(), res.get_header_value("request-id"));
                } else {
                    write_openai_error(res, e.error());
                }
            } catch (const std::exception& e) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, e.what()),
                    response_request_id(res));
                if (is_anthropic_path(req.path)) {
                    ApiError error;
                    error.status  = 500;
                    error.message = e.what();
                    write_anthropic_error(res, error, res.get_header_value("request-id"));
                } else {
                    write_exception(res, e);
                }
            } catch (...) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, "unknown error"),
                    response_request_id(res));
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = "unknown error";
                if (is_anthropic_path(req.path)) {
                    write_anthropic_error(res, error, res.get_header_value("request-id"));
                } else {
                    write_openai_error(res, error);
                }
            }
        });

    server_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        const bool available = service_ != nullptr && service_->is_available();
        res.status           = available ? 200 : 503;
        nlohmann::json body  = {{"status", available ? "ok" : "unavailable"}};
        // The busy signal. A keepalive that only asks "does the chat model answer" cannot tell a
        // drawing engine from a dead one: a chat request queued behind a two-minute picture waits
        // through the whole interlude and times out, and killing the serve at that moment destroys a
        // job that was succeeding -- measured on 2026-11-01 22:30. Only this server knows whether a
        // picture is in flight, so it says so here, where a supervisor looks first. Omitted entirely
        // when no image pipeline was configured, rather than promising a zero nobody can distinguish
        // from an answer.
        if (images_ != nullptr) {
            const auto now = std::chrono::duration_cast<std::chrono::seconds>(
                                 std::chrono::system_clock::now().time_since_epoch())
                                 .count();
            body["image_jobs"] = {{"in_flight", images_->in_flight_jobs()},
                                  {"oldest_age_seconds",
                                   images_->oldest_in_flight_age_seconds(
                                       static_cast<std::int64_t>(now))}};
        }
        res.set_content(body.dump(), "application/json");
    });
    server_.Get(api_route_pattern("/models"),
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_models(req, res);
                });
    server_.Get("/metrics", [this](const httplib::Request& req, httplib::Response& res) {
        const MetricsFormat format =
            select_metrics_format(req.get_param_value("format"), req.get_header_value("Accept"));
        res.set_header("Cache-Control", "no-store");
        if (format == MetricsFormat::Invalid) {
            res.status = 400;
            res.set_content(
                R"({"error":{"code":"invalid_format","message":"format must be json or prometheus"}})",
                "application/json; charset=utf-8");
            return;
        }
        if (format == MetricsFormat::Json) {
            res.set_content(metrics_json_body(), "application/json; charset=utf-8");
            return;
        }
        res.set_content(metrics_.render(service_->runtime_stats(), service_->is_available()),
                        "text/plain; version=0.0.4; charset=utf-8");
    });
    server_.Get(api_route_pattern(R"(/models/(.+))"),
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_model(req, res);
                });
    server_.Post(api_route_pattern("/chat/completions"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_chat_completions(req, res);
                 });
    server_.Post(api_route_pattern("/images/generations"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_images_generations(req, res);
                 });
    // The OpenAI images/edits name, honored: same fields, and a reference that must be there. It
    // is a route of its own rather than a flag on generations because that is what makes forgetting
    // the picture a refusal instead of a text-to-image surprise.
    server_.Post(api_route_pattern("/images/edits"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_images_edits(req, res);
                 });
    // The finished picture, addressed by the artifact id the generation returned. The id is validated
    // against its stored shape before any path is built, so a request cannot name a file.
    server_.Get(api_route_pattern(R"(/images/([^/]+))"),
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_image_artifact(req, res);
                });
    server_.Get(api_route_pattern("/capabilities"),
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_capabilities(req, res);
                });
    server_.Post(api_route_pattern("/responses"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_responses(req, res);
                 });
    server_.Post(api_route_pattern("/responses/input_tokens"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_input_tokens(req, res);
                 });
    server_.Post(api_route_pattern("/responses/compact"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_compact(req, res);
                 });
    server_.Post(api_route_pattern(R"(/responses/([^/]+)/cancel)"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_cancel(req, res);
                 });
    server_.Get(api_route_pattern(R"(/responses/([^/]+)/input_items)"),
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_input_items(req, res);
                });
    server_.Get(api_route_pattern(R"(/responses/([^/]+))"),
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_get(req, res);
                });
    server_.Delete(api_route_pattern(R"(/responses/([^/]+))"),
                   [this](const httplib::Request& req, httplib::Response& res) {
                       handle_response_delete(req, res);
                   });
    server_.Post(api_route_pattern("/messages/count_tokens"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_count_tokens(req, res);
                 });
    server_.Post(api_route_pattern("/messages"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_messages(req, res);
                 });
    // /v1/systemone is TypeSafe Jev's name for the same request, so a Jev client needs only a URL.
    for (const char* endpoint : {"/decide", "/systemone"}) {
        server_.Post(api_route_pattern(endpoint),
                     [this](const httplib::Request& req, httplib::Response& res) {
                         handle_decide(req, res);
                     });
    }
}

std::string HttpServer::image_model_id() const {
    if (images_ == nullptr || !images_->is_configured()) { return {}; }
    return image_model_id_from_artifact(options_.image_artifact_path);
}

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_models_list(public_model_id_, unix_time_now(), options_.max_context,
                                     model_metadata_, chat_template_caps_, image_model_id()),
                    "application/json");
}

void HttpServer::handle_capabilities(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_capabilities_json(options_, public_model_id_, image_model_id()),
                    "application/json");
}

std::string make_capabilities_json(const ServeOptions& options, std::string_view chat_model_id,
                                   std::string_view image_model_id) {
    using Json = nlohmann::json;
    Json image = Json(nullptr);
    if (!image_model_id.empty()) {
        image = Json{{"model", image_model_id},
                     {"endpoint", "/v1/images/generations"},
                     {"artifact_endpoint", std::string(kImageArtifactRoutePrefix) + "<id>"},
                     // How to tell a drawing engine from a dead one without a private protocol. A
                     // supervisor that only asks whether the chat model answers reads a picture's
                     // queue as death and kills the job: /health names the two fields that say
                     // otherwise, so the client that reads only this document can hold off healing
                     // without knowing anything about interludes.
                     {"health",
                      Json{{"endpoint", "/health"},
                           {"busy_field", "image_jobs.in_flight"},
                           {"age_field", "image_jobs.oldest_age_seconds"},
                           {"meaning", "an in_flight above 0 means the card is drawing a picture, not "
                                       "hung: the chat routes resume once the interlude restores the LLM"}}},
                     {"response_formats", Json::array({"b64_json", "url"})},
                     {"dimensions",
                      Json{{"min", kMinimumImageDimension}, {"max", kMaximumImageDimension}}},
                     {"steps", Json{{"min", kMinimumImageSteps}, {"max", kMaximumImageSteps}}},
                     {"true_cfg_scale",
                      Json{{"min", kMinimumImageCFGScale}, {"max", kMaximumImageCFGScale}}},
                     // The other half of steering, named so a client that reads nothing but this
                     // document can find it: scale above 1 means a second DiT forward against this
                     // text, and without it the steer would silently mean nothing, so the pair is
                     // refused at parse (openai_images.cpp:321) instead of degraded in the engine.
                     {"negative_prompt",
                      Json{{"type", "string"},
                           {"max_bytes", kMaximumImagePromptBytes},
                           {"required_when", "true_cfg_scale above 1"},
                           {"omitted_with_scale_above_1",
                            "HTTP 400 image_negative_prompt_required"}}},
                     // Reference-image conditioning runs on the diffusion pipeline's own vision tower,
                     // which sees only the picture being edited; --vision is a separate surface.
                     // TRUE because the behavioural gate (tests/models/qwen_image/
                     // test_image_edit_behavior.cu) is green: it renders the M9 fixture twice, once per
                     // instruction, and grades the PICTURES -- not-a-copy (MAE/SSIM vs the reference),
                     // subject held, instruction happened. Advertising true is exactly the claim that
                     // gate exists to earn; it was false while every edit came back a copy.
                     {"reference_image", true},
                     // How the picture is sized, in the client's terms. Naming the three keys and the
                     // reference-following default here is what lets a client that reads nothing but
                     // this document know that an edit of a portrait should come back portrait: the
                     // canvas is the thing a fixed square default used to get wrong.
                     {"canvas", Json{{"keys", Json::array({"size", "wh_ratio", "ratio_follow"})},
                                     {"give_at_most_one", true},
                                     // What the card gets to decide about a canvas the client named. The
                                     // admission need is a function of the canvas, so the 503 that used to
                                     // answer "this size does not fit today" was a smaller picture away
                                     // from fitting -- and the client's own model found that size one round
                                     // and two minutes later. The server now walks its own 32-px grid down
                                     // (longer side first, so the chosen aspect survives) and draws the
                                     // largest canvas the card can hold, saying which one it drew. Only
                                     // when not even the floor fits does the 503 reach a client, with the
                                     // same two numbers it always carried.
                                     {"vram_downgrade",
                                      "a canvas this process cannot hold is drawn at the largest size that "
                                      "fits instead of refused; the answer then names both, in data[0].size "
                                      "(drawn) and data[0].size_asked (sent), and the same pair rides on "
                                      "image_tool_attempts[].result"},
                                     {"size", "WIDTHxHEIGHT, multiples of 32 in [32,2048]"},
                                     {"wh_ratio", "\"W:H\" held at the working area, e.g. \"16:9\""},
                                     {"ratio_follow", "\"<image1>\": the canvas keeps the reference's aspect"},
                                     // The one ceiling a canvas can break that is NOT about memory: the DiT
                                     // serves one attention sequence over the text rows and the picture
                                     // tokens together, and that range belongs to the model. It is published
                                     // here so a client sizes to it instead of discovering it as a 400 -- and
                                     // so nobody reaches for free VRAM to raise it, because freeing memory
                                     // changes nothing: 2048x2048 draws from words and is refused as an edit.
                                     {"joint_sequence",
                                      Json{{"ceiling_tokens", ops::kAttnDitJointMaxTokens},
                                           {"counts", "the canvas costs (W/16)x(H/16) tokens, every "
                                                      "reference triple its vision tokens, and every text "
                                                      "row one token; the sum must stay under the ceiling"},
                                           {"measured", "2048x2048 fits a draw from words alone; with one "
                                                        "reference the largest edit measured to draw here "
                                                        "is 1792x1792"},
                                           {"ask_the_server", "the refusal names the largest canvas that "
                                                              "fits this prompt and reference -- take that "
                                                              "one rather than shrinking by guess"}}},
                                     // The codes each rule above earns, beside the rule. A client that read
                                     // the rule and still guessed wrong is told WHICH rule it broke only if
                                     // the code is next to the text that describes it -- otherwise the
                                     // document teaches the format and the wire answers in a vocabulary the
                                     // client has to already know.
                                     {"refusals",
                                      Json{{"size_not_a_multiple_of_32_or_out_of_range",
                                           "HTTP 400 image_size_invalid"},
                                           {"wh_ratio_malformed_or_outside_the_supported_span",
                                            "HTTP 400 image_ratio_invalid"},
                                           {"ratio_follow_not_an_imageN_tag_naming_a_sent_reference",
                                            "HTTP 400 image_ratio_follow_unsupported"},
                                           {"more_than_one_of_the_three_keys",
                                            "HTTP 400 image_size_conflict: the keys are mutually "
                                            "exclusive, name at most one"}}},
                                     {"default_with_reference", "the reference's own aspect"},
                                     {"default_without_reference", "768x768"}}},
                     // What a prompt may be. The byte bound is what one request may carry; the whole
                     // text reaches the model, nothing is truncated, and the model's own limit is the
                     // joint sequence -- which depends on the canvas, so it is answered per request (400
                     // image_prompt_too_long names both numbers and the largest canvas that fits). A
                     // client that reads only this document
                     // can therefore write a long instruction without guessing whether it will survive.
                     {"prompt", Json{{"max_bytes", kMaximumImagePromptBytes},
                                     {"truncated", false},
                                     {"too_long", "HTTP 400 image_prompt_too_long, naming the joint "
                                                  "sequence, its ceiling and the largest canvas that "
                                                  "fits this prompt -- ask that one instead of guessing"}}},
                     // The rest of the refusals one of these two routes can answer, split the only way a
                     // client can act on: the 400s are the client's own request to fix, the 429 is worth
                     // retrying as it stands, and the 503 is a card that cannot hold this canvas -- which
                     // is a smaller picture away, not a reason to give up. Without this block a client has
                     // to have met the answer before to know which of the three it is holding.
                     {"refusals",
                      Json{{"steps_out_of_range", "HTTP 400 image_steps_invalid"},
                           {"true_cfg_scale_out_of_range", "HTTP 400 image_cfg_scale_invalid"},
                           {"n_other_than_one", "HTTP 400 image_n_not_supported"},
                           // The route takes the family's LIST form: one to kMaximumImageReferences
                           // pictures, each one conditioning the forward. The number is written from the
                           // constant the parser enforces, so the document can never advertise a cap the
                           // route does not have -- it used to promise "exactly one picture", which the
                           // parser has not enforced since the list form landed.
                           {"reference_count_out_of_range",
                            "HTTP 400 image_count_invalid: the reference takes between 1 and " +
                                std::to_string(kMaximumImageReferences) +
                                " base64 pictures, each of them conditioning the draw"},
                           {"reference_given_as_a_url",
                            "HTTP 400 image_url_not_supported: base64 only, never a URL"},
                           {"reference_data_uri_without_base64",
                            "HTTP 400 image_data_uri_not_base64: a data: URI is accepted only when it "
                            "carries base64"},
                           {"reference_over_the_decoding_budget", "HTTP 400 image_too_large"},
                           {"reference_not_a_decodable_png_or_jpeg", "HTTP 400 image_decode_failed"},
                           {"multipart_file_field_other_than_image",
                            "HTTP 400 image_unsupported_file_field"},
                           {"another_draw_holds_the_card",
                            "HTTP 429 image_interlude_in_flight: retry the same request"},
                           {"the_yield_cannot_cover_this_canvas",
                            "HTTP 503 image_vram_insufficient: the message names the bytes needed and the "
                            "bytes the model can yield, and a smaller size is the fix"},
                           // Three answers that come from outside the request, so nothing the client sent
                           // predicts them -- which is exactly when a name pays: an unknown id is final, an
                           // unconfigured pipeline is the operator's to fix, and a rewriter that answered
                           // badly is one switch away from being drawn without.
                           {"artifact_or_job_id_unknown",
                            "HTTP 404 image_not_found: no job and no picture carries this id, so polling it "
                            "again will not help"},
                           {"pipeline_not_configured",
                            "HTTP 503 image_generation_not_configured: this process has no image model, so "
                            "no request can be drawn until one is loaded"},
                           {"enhancer_answered_badly",
                            "HTTP 500 image_prompt_enhancer_failed: the prompt rewriter answered an error "
                            "status or something that is not JSON; the same request with prompt_enhance "
                            "false goes straight to the draw"},
                           // The rest are the answers that are nobody's request to fix, and the document
                           // gives them anyway because the one thing a caller can still do is choose
                           // BETWEEN them: wait, walk away, or stop spending the card on a retry that
                           // reproduces. Without these names 503 and 500 are one blob and a client cannot
                           // tell the canvas that wants to be smaller (above) from the engine that wants
                           // to be left alone, nor a draw that simply failed from a kernel that will refuse
                           // the same geometry forever.
                           {"the_card_is_not_healthy_right_now",
                            "HTTP 503 image_engine_degraded: the draw failed and the text model could not "
                            "be confirmed restored, so nothing about this request is wrong -- leave the "
                            "card alone and come back later"},
                           {"the_handover_failed_before_the_draw",
                            "HTTP 503 image_interlude_failed: the card could not be taken or given back "
                            "before the draw began, so the same request may go through once it is free"},
                           {"this_model_has_no_handover_to_give",
                            "HTTP 503 image_interlude_unsupported: the loaded image model supports no "
                            "interlude, so a request that needs one will never be served here and retrying "
                            "changes nothing"},
                           {"every_job_slot_is_busy",
                            "HTTP 503 image_store_active_full: this process holds no free job slot, which is "
                            "the store's own half of the overload policy -- the same request is worth "
                            "retrying"},
                           {"the_draw_simply_failed",
                            "HTTP 500 image_generation_failed: the draw ran, the text model was restored, "
                            "and the picture just did not come, so a retry is fair"},
                           {"a_weight_could_not_be_bound",
                            "HTTP 500 image_artifact_failed: a weight could not be bound or streamed, which "
                            "is the artifact lane rather than this request"},
                           {"a_shape_the_admission_missed",
                            "HTTP 500 image_request_unservable: a kernel refused the geometry it was handed, "
                            "which is the server's to fix -- the same request reproduces it, so do not "
                            "retry it as it stands"},
                           {"a_reference_that_disagrees_with_itself",
                            "HTTP 500 image_conditioning_inconsistent: a prepared reference carries a vision "
                            "token count that contradicts its own latent size, and the same picture "
                            "reproduces it"}}},
                     // A chat request to this process is given the tool below and the picture is
                     // drawn inside that same request, so a client needs no tool call of its own.
                     {"automatic_from_chat", true},
                     {"chat_tool", std::string(kChatImageToolName)},
                     {"chat_max_images_per_request", kMaximumChatImageGenerations},
                     // A picture a chat request attaches goes through the same parsing on all three
                     // protocols, and the one field of it this server cannot honour is a downscale: only
                     // auto and high exist here, so a client that sent "low" is told which field to drop
                     // instead of being left with a bare 400 and a picture it believes it sent.
                     {"chat_picture_detail",
                      "a picture part may ask for detail \"auto\" or \"high\"; anything else is HTTP 400 "
                      "image_detail_not_supported"},
                     // What became of each picture a chat request asked for, in the server's own words
                     // rather than the model's account of it. The model is shown the tool error and still
                     // answers the user in prose -- and invented a cause for it here (a VRAM figure, an
                     // unload it never attempted). This is the half a client could not get at all: the
                     // refusals left no trace in a log and none on the wire.
                     {"chat_tool_attempts",
                      Json{{"field", "image_tool_attempts"},
                           {"on", Json::array({"the chat.completion body of POST /v1/chat/completions",
                                               "the terminal chunk of a streamed chat completion, or its "
                                               "usage chunk when stream_options.include_usage is set",
                                               "the Anthropic message object, and the streamed message_delta",
                                               "the Responses response body, including the streamed "
                                               "response.completed"})},
                           {"present_when", "this request asked for at least one picture, and absent "
                                            "otherwise -- a plain chat answer stays the exact bytes it was "
                                            "before the field existed, so a client must not expect it"},
                           {"entries", "one per picture asked for, even when several tool calls shared a "
                                       "round: {round, outcome, tool_call_id, request, result|error}"},
                           {"outcome", Json{{"succeeded", "result carries id, url and "
                                                          "prompt_enhance_outcome",
                                             "refused", "the request never reached the model: error carries "
                                                       "the route's own code, message and http_status, which "
                                                       "is what to show the user",
                                             "failed", "the draw itself broke inside the server (HTTP 500 "
                                                       "from the pipeline), which a retry may clear"}}},
                           {"request", "what the tool call asked for: size, steps, image_count, and seed "
                                       "and prompt_enhance when the call set them"},
                           {"why", "a client can say which picture failed and why without taking the "
                                   "model's word for it. The same facts per picture also reach the "
                                   "operator's requests.jsonl as an image_job event, refused draws "
                                   "included, with the SAME outcome word for the same draw -- a 503 "
                                   "reads as refused in both, so a failure a client never reports is "
                                   "still readable afterwards, and never as a server that broke"}}},
                     // A chat edit goes through the very same parser as the routes, so its refusals ARE
                     // image_generation.refusals -- the tool hands the model back the route's own code, and
                     // the model retries smaller or waits. Only these two names come from the tool itself
                     // rather than from the route, and a client reading only this document is the one who
                     // needs to know which is which: the first is a per-request budget, the second means
                     // the arguments never were a request.
                     {"chat_tool_refusals",
                      Json{{"more_pictures_than_one_request_may_draw",
                            "image_generation_limit_reached: one chat request draws at most " +
                                std::to_string(kMaximumChatImageGenerations) + " pictures, the rest of the "
                                "calls are refused as tool errors"},
                           {"arguments_not_readable",
                            "image_arguments_invalid: the tool call is not a JSON object, so no route "
                            "refusal can apply to it"}}}};
    }
    // The edit surface, named on its own. image_generation says a picture can be drawn; this says
    // how to change one you already have, which is the question a fresh client actually asks. Every
    // field is the generations one except the reference, so the block points there instead of
    // restating geometry and step bounds a change would then have to make twice.
    // Whether this process was started with the enhancer's sidecar address. "Configured" is the honest
    // word the document can answer without asking the sidecar: a capabilities request is a cheap read,
    // and a health probe of another process does not belong in it. A configured sidecar that has since
    // stopped answering still answers the request with the documented 503 below, so a client that acts
    // on this field and gets that answer has lost nothing but a round trip.
    const bool enhancer_configured = !options.image_prompt_enhancer_url.empty();
    Json edit = Json(nullptr);
    if (!image_model_id.empty()) {
        edit = Json{{"model", image_model_id},
                     {"endpoint", "/v1/images/edits"},
                     {"fields", "every field /v1/images/generations takes"},
                     {"content_types", Json::array({"application/json", "multipart/form-data"})},
                     {"reference_required", true},
                     {"reference_max", kMaximumImageReferences},
                     {"reference_formats",
                      Json::array({"base64 in the image field (a data: URI prefix is accepted)",
                                   "a multipart file part named image",
                                   "a multipart text field named image holding the base64"})},
                     {"accepts_url_reference", false},
                     // The chat route reaches the same edit with no wire work at all: the tool the server
                     // injects into every chat request takes these forms of image, one of them or a list of
                     // up to reference_max -- the same cap this block states two lines above, because the
                     // chat tool resolves each element and hands the route the list the route already took.
                     // The http routes test reads the tool's own schema for this, so the pair cannot drift.
                     {"chat_tool_image_argument",
                      Json::array({"a job id img-<created>-<sequence> this server drew",
                                   "that picture's own URL",
                                   "attachment:N of the chat request",
                                   "a JSON list of 1..reference_max of those forms, which is what lets an "
                                   "instruction address them as <image1>, <image2>, ... in the order given"})},
                     {"missing_reference",
                      "HTTP 400 image_reference_required -- this route never draws a picture "
                      "nobody attached"},
                     {"mask", "not supported: HTTP 400 image_mask_not_supported"},
                     {"prompt_enhancer",
                      Json{{"available", enhancer_configured},
                           {"model", std::string(kImagePromptEnhancerModel)},
                           {"field", "prompt_enhance"},
                           // Which route the field belongs to, said out loud. The tool this server injects
                           // into every chat request -- the path a fresh client reaches for first, because
                           // it needs no base64 and no second call -- carries the same argument: its
                           // arguments go through the images route's own parser, so an edit drawn inside a
                           // chat request can be rewritten too. The document and the tool schema are held
                           // to each other by the http routes test, so neither can advertise a switch the
                           // other does not have.
                           {"via", "the images routes (/v1/images/generations and /v1/images/edits) and the "
                                   "chat tool this server injects into every chat request: put prompt_enhance "
                                   "in the tool call and it reaches the same parser the routes use. Absent is "
                                   "no rewrite, so a chat edit is drawn from the words exactly as written "
                                   "unless the call asks"},
                           {"values", Json::array({"true", "false", "auto"})},
                           {"when_auto",
                            enhancer_configured
                                ? "the server decides per request: the rewrite runs when the request "
                                  "carries a reference picture and the instruction is not already a "
                                  "rewriter's own answer (one that addresses its inputs as <imageN>), and "
                                  "is skipped otherwise -- and also when no rewriter is reachable or the "
                                  "card cannot be had for one, which for auto is a skip and never a 503, "
                                  "because auto asked the server to choose"
                                : "the picture is drawn from the instruction exactly as written: this "
                                  "build has no rewriter to run, and auto does not refuse over it -- only "
                                  "true gets the 503"},
                           {"when_true",
                            enhancer_configured
                                ? "the server rewrites the instruction with the enhancer before drawing; its "
                                  "rewritten_prompt is what reaches the model, and its wh_ratio or ratio_follow "
                                  "decides the canvas only where this request named no size of its own. HTTP "
                                  "503 image_prompt_enhancer_unavailable if the sidecar has stopped answering, "
                                  "500 image_prompt_enhancer_bad_answer if its answer breaks the contract below"
                                : "HTTP 503 image_prompt_enhancer_unavailable -- no rewriter is running in this "
                                  "build (serve --image-prompt-enhancer-url), so the client is the rewriter"},
                           {"answers",
                            "rewritten_prompt, wh_ratio, ratio_follow -- the two size fields are the "
                            "same ones the images routes take, mutually exclusive, and rewritten_prompt carries no "
                            "resolution or ratio of its own"},
                           // The answer to when_auto, per job. The strings come from the code that writes
                           // them (to_string) rather than from a list here, because a capability document
                           // that keeps its own copy is a document that can advertise an outcome the
                           // server never produces.
                           {"reported_as",
                            Json{{"field", "prompt_enhance_outcome"},
                                 {"on", Json::array({"the successful POST response body",
                                                     "the job's status object, which GET /v1/images/<id> "
                                                     "serves while no picture file exists (an in-flight job, "
                                                     "or one that failed)",
                                                     "the stored record"})},
                                 {"values", Json::array({to_string(EnhancerOutcome::Off),
                                                          to_string(EnhancerOutcome::SkippedNoReference),
                                                          to_string(EnhancerOutcome::SkippedAlreadyRewritten),
                                                          to_string(EnhancerOutcome::SkippedNoRewriter),
                                                          to_string(EnhancerOutcome::SkippedNoDevice),
                                                          to_string(EnhancerOutcome::Applied)})},
                                 {"meaning", "which decision this job got; every skip_* value is legitimate for "
                                             "auto, and only skipped_no_rewriter and skipped_no_device point at "
                                             "the deployment rather than at the request"}}}}},
                     // Not just true: auto rewrites too, whenever its own rule is met, and a client told
                     // only about true pays the rewriter on requests that do not need it -- or believes a
                     // short instruction is impossible under auto. Absent is spelled out because the
                     // request's own default is the third answer and the client cannot guess it.
                     {"short_instruction",
                      enhancer_configured
                          ? "acceptable when prompt_enhance is true, and also when it is the string auto and "
                            "the request carries a reference picture the instruction does not already address "
                            "as <imageN>: the server rewrites it first. Absent or false is no rewrite at all "
                            "(the request default is false), so on those requests the words reach the model "
                            "exactly as written and the instruction below is what carries the edit"
                          : "not sufficient in this build: nothing between the client and the model rewrites it, "
                            "so a one-phrase instruction reaches the diffusion model exactly as written"},
                     {"instruction",
                      Json::array({
                          "lead with the operation, not a description of the finished picture",
                          "name the target concretely, and push the named attribute to a strong and unmistakable "
                              "degree",
                          "hold everything else at input fidelity: say what stays by type, position and role, never "
                              "by describing its appearance",
                          "one blanket preservation clause beats a checklist -- describing the things you meant to keep "
                              "is what makes them drift",
                          "point identity at the reference image instead of describing features in words",
                          "with more than one reference, address them as <image1>, <image2>, ... and say which one is "
                              "the canvas and which supply material; with one reference, refer to it naturally",
                          "put no resolution or aspect ratio in the instruction: size travels only in size, wh_ratio "
                              "or ratio_follow",
                          "commit to the exact characters of any text that will be rendered into the picture, in the "
                              "image's own dominant language",
                          "state what must be preserved affirmatively, not as a prohibition",
                          "describe only what is visibly there -- a preserved element named at a higher abstraction "
                              "is safer than an invented specific",
                          "ask for nothing that was not requested, and say enough about any newly exposed region that "
                              "the result stays physically coherent",
                      })},
                     {"canvas_default", "the reference's own aspect (see image_generation.canvas)"}};
    }
    // The rewriter is not an edit-only argument: prompt_enhance is parsed by the images routes'
    // shared parser, so a draw with no reference takes it too. A client that reads only
    // image_generation could not find the switch at all -- it had to guess that a field named under
    // image_edit also exists one level up. Copied out of the edit block rather than written twice,
    // so the two blocks cannot advertise different rules for the same argument.
    if (!image.is_null() && !edit.is_null()) { image["prompt_enhancer"] = edit["prompt_enhancer"]; }
    const Json payload = {
        {"object", "infernix.capabilities"},
        {"chat_model", chat_model_id},
        {"vision_input", options.enable_vision},
        {"image_generation", std::move(image)},
        {"image_edit", std::move(edit)},
        // The decide route is the third surface a client with no files can post to, and the document
        // said nothing about it. It shares no prefix with the image_ and canvas_ codes, so the
        // coverage probe could not even see it, and the route answers a caller's failure THREE ways
        // (decide_http.cpp:202-217): a 400 for a malformed question, an error object riding inside
        // the 200 as that question's answer, and an engine fault raised per question. A client that
        // read those as one category retries a request it has to rewrite, or trusts a run that never
        // produced an answer. Every entry below is copied from the call that answers it.
        {"decide", Json{{"endpoint", "/v1/decide"},
                        {"alias", "POST /v1/systemone is the same request under the other name: one "
                                  "handler answers both, so a client written for that API changes only "
                                  "the URL"},
                        {"request", Json{
                            {"model", "optional string, one of the ids /v1/models lists"},
                            {"state", "required: the evidence every question is answered over, either a "
                                      "JSON value or an array of content parts. A point or a box needs "
                                      "exactly one image in it, since both are scaled by its sides. "
                                      "Omitting it is the one 400 that carries no code at all -- match it "
                                      "on its param instead"},
                            {"questions", "required: a non-empty object of question id to question. The "
                                          "answers come back under those same ids"},
                            {"question_fields", Json::array({"type", "instructions", "question", "criteria",
                                                             "options", "digits"})},
                            {"question_types", Json::array({"noul", "choice", "score", "number", "scalar",
                                                            "point", "box"})},
                            {"options", "at most 256 of them, and in practice fewer: a question with more "
                                        "options than the loaded tokenizer can name as single, distinct "
                                        "tokens is refused rather than served with two options sharing a "
                                        "token"},
                            {"digits", "the field width for number, point and box (default 3, at most 6) "
                                       "and the ceiling for scalar (default 8, at most 15); outside those "
                                       "it is the digits_out_of_range 400. point and box are answered in "
                                       "the submitted picture's own pixels"},
                            {"thinking", "enable_thinking is refused: a decision is read at the position "
                                         "the prompt ends, where a thinking prompt would open a reasoning "
                                         "block instead of an answer"},
                        }},
                        {"answers", "an object keyed by each question's id. EVERY question gets an entry, "
                                    "an answer or an error object, never a missing key, so a caller can "
                                    "walk the request it sent"},
                        {"answer", Json{
                            {"type", "one of question_types, and the object's discriminator"},
                            {"answer_mass", "on every answer: the declared options' share of the WHOLE "
                                            "next-token distribution, clamped to 0..1. It is the number a "
                                            "tolerance is tuned against, and it is NOT the probability of "
                                            "any one option. A report, not a gate -- nothing in the server "
                                            "compares it against a threshold, so no value of it is a refusal "
                                            "and no floor is published for it"},
                            {"confidence", "choice and score: the top probability, which is the abstention "
                                           "signal. A low one is a question to restate, not a failure"},
                            {"uncertainty", "number, scalar, point and box: how wide the digit steps left "
                                            "the value, in the unit that field is reported in"},
                        }},
                        {"error_answer", "a question that failed answers as an object of type error with "
                                         "code and message, riding INSIDE the 200 as that question's entry. "
                                         "Its status only becomes the response's when no question in the "
                                         "request got an answer at all"},
                        {"usage", Json::array({"input_tokens", "output_tokens", "cached_input_tokens",
                                               "computed_prefill_tokens"})},
                        {"refusals", Json{
                            // Named at questions.<id>, about the request, fixable without the server.
                            {"bad_request_400", Json{
                                {"thinking_unsupported", "enable_thinking was set on a decision"},
                                {"malformed_criteria", "criteria is not what the question's type asks for: "
                                                       "noul wants an object of true and false, choice an "
                                                       "object of option to description, score an array of "
                                                       "levels"},
                                {"missing_criteria", "a choice came with no criteria, which is where its "
                                                    "options live"},
                                {"no_options", "a choice declared no options at all"},
                                {"unclean_option", "an option has a blank name"},
                                {"too_few_levels", "a score declared fewer than two levels"},
                                {"malformed_question", "the question is not an object, or its type is not a "
                                                      "string"},
                                {"question_type_unsupported", "the type is a known question kind this "
                                                             "endpoint does not serve"},
                                {"unsupported_field", "the question carries a field the route does not "
                                                     "know; only question_fields are read"},
                                {"empty_instructions", "instructions is missing or empty; it must be a "
                                                      "non-empty string, object or array"},
                                {"criteria_unsupported", "a generated kind (number, scalar, point, box) was "
                                                        "given criteria, which it never reads"},
                                {"digits_unsupported", "digits was given to a question that reads one "
                                                      "position, so nothing can honour it"},
                                {"digits_out_of_range", "digits is not an integer in the served range"},
                                {"too_many_options", "more options than this model's labels can name"},
                                {"too_many_steps", "the question needs more generated steps than are served "
                                                   "-- its digits or its axes are too wide, not the "
                                                   "server's fault"},
                                {"no_questions", "questions is missing, empty, or not an object of id to "
                                                "question"},
                                {"spatial_needs_one_image", "a point or box was asked over a state that "
                                                           "does not carry exactly one image"},
                            }},
                            // The request was well formed and the run went off its own token schedule. The
                            // 200 is the status; these are the answer. Retrying the identical bytes gets
                            // the same error, so restate the question or narrow its digits.
                            {"in_answer_200", Json{
                                {"off_schedule", "the run is longer than its schedule, or a step committed "
                                                "a token that step does not permit"},
                                {"run_cut_short", "the engine stopped before the run closed its object or "
                                                 "reached its cap, so its digits are the front of a number "
                                                 "and not a number"},
                                {"too_many_digits", "the run spells more digits than the question allowed"},
                                {"malformed_scalar", "the digits the run spelled do not read as a number"},
                                {"image_size_unknown", "the submitted picture's size is unknown, so a point "
                                                      "or box has nothing to scale against"},
                            }},
                            // The engine's own fault, carried per question: the same request is worth
                            // sending again. cancelled is 499 and belongs to the client hanging up, so it
                            // is not a retry signal.
                            {"run_499_or_500", Json{
                                {"cancelled", "HTTP 499: the client disconnected, or the engine's finish "
                                              "reason was cancelled. Not a server fault and not a retry "
                                              "signal"},
                                {"no_trace", "HTTP 500: the engine returned an incomplete trace for a "
                                             "generated question"},
                                {"no_readout", "HTTP 500: the engine returned no readout, or one whose "
                                               "length does not match the question's options"},
                                {"tokenizer_unsupported", "HTTP 500: this model's tokenizer spells a label "
                                                          "as more than one token, and a constrained step "
                                                          "names single tokens"},
                                {"internal_error", "HTTP 500: anything unexpected while answering that "
                                                   "question"},
                            }},
                        }}}},
        {"endpoints", Json{{"chat_completions", "/v1/chat/completions"},
                           {"responses", "/v1/responses"},
                           {"messages", "/v1/messages"},
                           {"images_generations", "/v1/images/generations"},
                           {"images_edits", "/v1/images/edits"}}}};
    return payload.dump();
}

void HttpServer::handle_image_artifact(const httplib::Request& req, httplib::Response& res) const {
    const std::string id             = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    const std::filesystem::path path = generated_image_path(id);
    std::ifstream input(path, std::ios::binary);
    if (!path.empty() && input) {
        std::string body{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        if (input.bad()) { body.clear(); }
        if (!body.empty()) {
            res.set_header("Cache-Control", "no-store");
            res.set_content(std::move(body), "image/png");
            return;
        }
    }
    // No picture file yet (or never): a stored job still answers with its status object, so a
    // client polling an id can watch an in-flight generation instead of seeing 404.
    if (images_ != nullptr) {
        if (const auto status = images_->job_status_json(id); !status.is_null()) {
            res.set_header("Cache-Control", "no-store");
            res.set_content(status.dump(), "application/json");
            return;
        }
    }
    write_openai_error(res, ApiError{.status  = 404,
                                     .type    = "invalid_request_error",
                                     .message = "image '" + id + "' not found",
                                     .param   = "id",
                                     .code    = "image_not_found"});
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    if (const std::string image_id = image_model_id(); !image_id.empty() && id == image_id) {
        res.set_content(make_image_model_object(image_id, unix_time_now()), "application/json");
        return;
    }
    if (id != public_model_id_) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + id + "' not found";
        write_openai_error(res, error);
        return;
    }
    res.set_content(make_model_object(public_model_id_, unix_time_now(), options_.max_context,
                                      model_metadata_, chat_template_caps_),
                    "application/json");
}

bool HttpServer::bind() { return server_.bind_to_port(options_.host, options_.port); }

void HttpServer::attach(GenerationService& service) {
    if (service_ != nullptr) {
        throw std::logic_error("HTTP generation service is already attached");
    }
    const infernix::LoadSummary load = service.load_summary();
    // memory_summary waits for the completed worker boundary, including warmup publication. The
    // execution maxima restart here, so /metrics reports serving's, not the warm-up's.
    const infernix::MemorySummary memory  = service.memory_summary();
    service.reset_execution_maxima();
    const infernix::RuntimeStats baseline = service.runtime_stats();
    public_model_id_                    = resolve_public_model_id(options_, load.model_name);
    model_metadata_                     = service.model_metadata();
    chat_template_caps_ = probe_chat_template_caps([&service](const GenerationRequest& request) {
        return service.count_prompt_tokens(request);
    });
    decide_labels_                      = build_decide_alphabet(
        [&service](std::string_view text) { return service.tokenize_text(text); });
    service_                            = &service;
    metrics_.configure(public_model_id_, service.engine_options(), memory, baseline);
    request_jsonl_.write_server_start(options_, service.engine_options(),
                                      service.sampling_defaults(), public_model_id_, load, memory);
}

void HttpServer::attach_images(ImageGenerationService& images) {
    if (images_ != nullptr) {
        throw std::logic_error("HTTP image generation service is already attached");
    }
    images_ = &images;
}

bool HttpServer::listen() {
    if (service_ == nullptr) { throw std::logic_error("HTTP generation service is not attached"); }
    if (public_model_id_.empty()) {
        throw std::logic_error("HTTP public model id is not resolved");
    }
    if (console_stats_) { console_stats_->show(); }
    if (options_.log_stats_interval_ms != 0) {
        stats_stopping_ = false;
        stats_thread_   = std::thread([this] { run_stats_reporter(); });
    }
    watch_stopping_ = false;
    watch_thread_   = std::thread([this] { run_engine_watch(); });
    try {
        const bool result = server_.listen_after_bind();
        stop_engine_watch();
        stop_stats_reporter();
        // A stop for an Engine failure is not a listen failure.
        return result || engine_failed_.load();
    } catch (...) {
        stop_engine_watch();
        stop_stats_reporter();
        throw;
    }
}

void HttpServer::run_engine_watch() {
    for (;;) {
        {
            std::unique_lock lock(watch_mutex_);
            if (watch_cv_.wait_for(lock, kEngineWatchInterval, [this] { return watch_stopping_; })) {
                return;
            }
        }
        // The failed Engine has already answered every request with an error; nothing more can
        // be served, so the accept loop ends and the process exits for a launcher to restart.
        if (service_->has_failed()) {
            engine_failed_.store(true);
            server_.stop();
            return;
        }
    }
}

void HttpServer::stop_engine_watch() {
    if (!watch_thread_.joinable()) { return; }
    {
        std::lock_guard lock(watch_mutex_);
        watch_stopping_ = true;
    }
    watch_cv_.notify_one();
    watch_thread_.join();
}

void HttpServer::stop() {
    server_.stop();
    if (service_ != nullptr) { service_->stop(); }
}

} // namespace infernix::serve
