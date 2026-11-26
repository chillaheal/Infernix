#include "serve/http_server.h"

#include "product/logging/logging.h"
#include "product/speculative_options.h"
#include "serve/anthropic_messages.h"
#include "serve/http_transport.h"
#include "serve/image_capabilities.h"
#include "serve/image_generation_service.h"
#include "serve/openai_common.h"
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
    Json payload = {
        {"object", "infernix.capabilities"},
        {"chat_model", chat_model_id},
        {"vision_input", options.enable_vision},
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
                           {"messages", "/v1/messages"}}}};
    add_image_capabilities(payload, options, image_model_id); // the image lane's own blocks: serve/image_capabilities.cpp
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
