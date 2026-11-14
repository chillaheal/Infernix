#pragma once

// Optional full-precision JSONL measurement log. Human operational rendering is owned separately
// by operational_log.* and never consumes these serialized records.

#include "serve/request_events.h"
#include "serve/serve_options.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace spdlog {
class logger;
}

namespace infernix::serve {

inline constexpr int kRequestLogSchemaVersion        = 31;
inline constexpr const char* kRequestLogArtifactType = "infernix_serve_request_log";

struct ServerLogEnvironment {
    int device = 0;
    std::string gpu_name;
    std::string gpu_uuid;
    std::uint64_t total_device_memory_bytes = 0;
    int compute_capability_major            = 0;
    int compute_capability_minor            = 0;
    std::string cuda_compile_version;
    std::string cuda_runtime_version;
    std::string cuda_driver_version;
};

// Pure JSON formatters are public to repository tests. Each return value is one complete JSON
// object without a trailing newline.
std::string format_server_start_json(
    const std::string& server_instance_id, std::uint64_t timestamp_unix_ms,
    const ServeOptions& options, const infernix::EngineOptions& engine_options,
    const infernix::ModelSamplingDefaults& sampling_defaults, const std::string& public_model_id,
    const infernix::LoadSummary& load, const infernix::MemorySummary& memory,
    const ServerLogEnvironment& environment, std::optional<std::uint64_t> artifact_size_bytes);
std::string format_request_start_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp_unix_ms,
                                      const RequestLogContext& context);
std::string
format_request_scheduling_json(const std::string& server_instance_id,
                               std::uint64_t timestamp_unix_ms, std::uint64_t request_id,
                               const std::string& http_request_id,
                               const infernix::GenerationSchedulingObservation& observation);
std::string format_request_rejected_json(const std::string& server_instance_id,
                                         std::uint64_t timestamp_unix_ms,
                                         const RequestRejectionLogContext& context);
std::string format_request_done_json(const std::string& server_instance_id,
                                     std::uint64_t timestamp_unix_ms,
                                     const RequestLogContext& context,
                                     const GenerationOutcome& outcome);
std::string format_request_error_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp_unix_ms,
                                      const RequestLogContext& context, const std::string& message);
std::string format_throughput_json(const std::string& server_instance_id,
                                   std::uint64_t timestamp_unix_ms, const ThroughputReport& report);

// One picture, however it ended and whoever asked for it. The image route was the one piece of this
// server an operator could not look up afterwards: `requests.jsonl` recorded chat generations only, so
// when a client's draw was refused at 22:41 nothing in the process remembered the refusal, and the
// client's own model filled the gap with an invented cause. This record shares the file's artifact type
// so every existing reader still parses the file, and is named by `event` so it can be filtered alone.
//
// It carries what was asked and what was decided -- never the pictures: a reference is up to
// kMaximumImageInputBytes of base64 and is the user's photograph, so only its byte SIZE is recorded,
// and the prompt is recorded as its length for the same reason a chat request's prompt tokens are.
struct ImageJobLogRecord {
    std::string source;      // "images_route" or "chat_tool": who asked the service to draw
    std::string protocol;    // the endpoint that served it, for filtering
    std::string correlation; // the outer request/response id, when the caller has one
    std::string model;
    std::string size;        // "WIDTHxHEIGHT" as the parser settled it
    // The canvas the caller asked for, filled ONLY when the server drew a different one. The admission
    // need is a function of the canvas, so a card that cannot hold the asked size is a size away from
    // holding the job, and the service then draws the largest that fits (image_generation_service.cpp).
    // A line that showed only the drawn canvas would make that swap invisible -- the same reason this
    // record carries `prompt_enhance` next to `prompt_enhance_outcome`.
    std::string size_asked;
    std::uint32_t steps       = 0;
    std::optional<std::uint64_t> seed;
    std::size_t image_count = 0;
    std::vector<std::size_t> image_bytes;  // one entry per reference, sizes only
    std::string prompt_enhance;            // "off", "true" or "auto" as asked
    std::size_t prompt_bytes = 0;
    std::string outcome;     // "succeeded", "refused", "failed" or "not_configured"
    std::string stage;       // where it ended, when the site knows: "parse", "configured", "service"
    std::string code;        // the route's own error code on any failure
    std::string message;     // and its message, which is this server's own text
    int http_status     = 0; // the status the caller was given, 0 when none applied
    std::string job_id;      // the stored job id, once one exists
    std::string enhancer_outcome; // the rewriter's own reported outcome on a success
    std::int64_t duration_ms = -1; // wall clock of the attempt; -1 when not measured
};

std::string format_image_job_json(const std::string& server_instance_id, std::uint64_t timestamp,
                                  const ImageJobLogRecord& record);

ServerLogEnvironment query_server_log_environment(int device);

// Opens in append mode so one campaign file can contain multiple independently started MTP/model
// blocks. Every line carries server_instance_id because request ids restart at one per process.
// With rotation, the file that reaches `max_bytes` becomes `PATH.1` (older files shift to
// `PATH.2` .. `PATH.keep`, the oldest is deleted) and a new `PATH` starts with a copy of this
// server's `server_start` record, so every retained file can be read on its own.
class JsonlRequestLog {
public:
    explicit JsonlRequestLog(const std::string& path,
                             const std::string& protected_artifact_path = {},
                             std::shared_ptr<spdlog::logger> logger     = {},
                             RequestLogRotation rotation                = {});

    JsonlRequestLog(const JsonlRequestLog&)            = delete;
    JsonlRequestLog& operator=(const JsonlRequestLog&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    [[nodiscard]] const std::string& server_instance_id() const noexcept {
        return server_instance_id_;
    }

    void write_server_start(const ServeOptions& options,
                            const infernix::EngineOptions& engine_options,
                            const infernix::ModelSamplingDefaults& sampling_defaults,
                            const std::string& public_model_id, const infernix::LoadSummary& load,
                            const infernix::MemorySummary& memory);
    void write_request_start(const RequestLogContext& context);
    void write_request_scheduling(std::uint64_t request_id, const std::string& http_request_id,
                                  const infernix::GenerationSchedulingObservation& observation);
    void write_request_rejected(const RequestRejectionLogContext& context);
    void write_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void write_request_error(const RequestLogContext& context, const std::string& message);
    // One line per picture this process was asked to draw, from either door. Like every writer here it
    // is a no-op when the log is off, and it can never fail the draw it is describing.
    void write_image_job(const ImageJobLogRecord& record);
    void write_throughput(const ThroughputReport& report);

private:
    void append(std::string record);
    // Caller holds mutex_. Returns a warning to log after the lock is released, or empty.
    [[nodiscard]] std::string rotate_locked();
    [[nodiscard]] std::string rotated_path(std::uint32_t index) const;

    std::string path_;
    std::string server_instance_id_;
    std::ofstream output_;
    std::mutex mutex_;
    std::shared_ptr<spdlog::logger> logger_;
    RequestLogRotation rotation_;
    std::string server_start_record_;
    std::uint64_t written_bytes_ = 0; // bytes in the current file
    bool enabled_                = false;
    bool failed_                 = false;
};

} // namespace infernix::serve
