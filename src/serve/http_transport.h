#pragma once

#include "serve/generation_service.h"
#include "serve/image_job_progress.h"
#include "serve/request_json.h"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace infernix::serve {

[[nodiscard]] NgramSessionHints resolve_ngram_session(const httplib::Request& request,
                                                      const RequestJson& body,
                                                      const ServeOptions& options);
void set_ngram_generation_header(httplib::Response& response, const NgramArchiveStats& stats);
// The request log's label from one X-Infernix-Request-Label header of 1-128 printable ASCII
// characters (a benchmark marks its traffic with it); empty when absent or malformed.
[[nodiscard]] std::string request_label(const httplib::Request& request);
[[nodiscard]] std::string ngram_generation_comment(const NgramArchiveStats& stats);
class ResponseRenderFailure final : public std::runtime_error {
public:
    explicit ResponseRenderFailure(const std::string& message) : std::runtime_error(message) {}
};

struct HttpGenerationStream {
    explicit HttpGenerationStream(PreparedRequest request) : prepared(std::move(request)) {}

    PreparedRequest prepared;
    std::atomic<bool> cancelled{false};
    std::atomic<bool> started{false};
};

class SseTransport final {
public:
    using Clock = std::chrono::steady_clock;

    static constexpr std::chrono::seconds kHeartbeatInterval{5};
    static constexpr std::string_view kHeartbeatComment = ": keep-alive\n\n";

    explicit SseTransport(httplib::DataSink& sink, std::atomic<bool>& cancelled,
                          Clock::duration heartbeat_interval = kHeartbeatInterval,
                          Clock::time_point now              = Clock::now());

    // The transport is shared between the Engine wait loop and a HeartbeatTicker while a
    // server-executed image round blocks the handler: every sink write and the quiet-interval
    // clock are serialized under one mutex.
    void write(std::string_view item, Clock::time_point now = Clock::now());
    void write(const std::vector<std::string>& items, Clock::time_point now = Clock::now());

    // Called by the Engine wait loop. Besides observing an already-closed socket, a quiet stream
    // periodically writes an SSE comment so TCP_USER_TIMEOUT has traffic with which to detect an
    // unacknowledged peer. A failed probe marks the request for cancellation.
    [[nodiscard]] bool poll(Clock::time_point now = Clock::now());

    // The ticker's quiet-interval heartbeat: the comment half of poll() without the socket's
    // writability probe (a ticker running while the Engine loop is idle must not cancel on a
    // transient !is_writable). A failed write marks the request cancelled, like poll().
    void beat(Clock::time_point now = Clock::now());

    [[nodiscard]] Clock::duration heartbeat_interval() const noexcept { return heartbeat_interval_; }
    [[nodiscard]] bool is_cancelled() const noexcept {
        return cancelled_.load(std::memory_order_acquire);
    }

private:
    bool mark_cancelled() noexcept;

    httplib::DataSink& sink_;
    std::atomic<bool>& cancelled_;
    Clock::duration heartbeat_interval_;
    Clock::time_point last_write_;
    std::mutex write_mutex_;
};

// Beats `transport` from its own thread while the handler is blocked in a server-executed image
// round: every interval it writes next_comment()'s text when nonempty (an ": image 42%" progress
// comment), else the transport's standard quiet-interval beat. Destruction stops and joins the
// thread. A write failure marks the transport's request cancelled, exactly as poll() does.
// When `watch` names a job's progress, the ticker is also its stall watchdog: a state whose
// last_progress stands still past `stall_bound` gets `stalled` set, and the pipeline's next step
// boundary throws instead of letting a wedged job hold the interlude's VRAM forever.
class HeartbeatTicker final {
public:
    static constexpr std::chrono::seconds kDefaultStallBound{60};

    HeartbeatTicker(SseTransport& transport, std::function<std::string()> next_comment,
                    std::function<std::shared_ptr<ImageJobProgress>()> watch = {},
                    std::chrono::milliseconds stall_bound =
                        std::chrono::duration_cast<std::chrono::milliseconds>(kDefaultStallBound),
                    std::function<SseTransport::Clock::time_point()> now = SseTransport::Clock::now);
    ~HeartbeatTicker();

    HeartbeatTicker(const HeartbeatTicker&)            = delete;
    HeartbeatTicker& operator=(const HeartbeatTicker&) = delete;

private:
    void run();

    SseTransport& transport_;
    std::function<std::string()> next_comment_;
    std::function<std::shared_ptr<ImageJobProgress>()> watch_;
    std::chrono::nanoseconds stall_bound_;
    std::function<SseTransport::Clock::time_point()> now_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool stopping_ = false;
    std::thread thread_;
};

template <class Render>
void render_and_write(SseTransport& transport, Render&& render) {
    try {
        auto payload = std::forward<Render>(render)();
        transport.write(payload);
    } catch (const ClientDisconnected&) { throw; } catch (const ResponseRenderFailure&) {
        throw;
    } catch (const std::exception& exception) { throw ResponseRenderFailure(exception.what()); }
}

RequestJson parse_json_body(const httplib::Request& request);

// The absolute prefix this request's generated-image artifacts are reachable at. Every route that
// hands out an image URL builds it here, so a chat answer and an images response name the same GET.
[[nodiscard]] std::string artifact_url_base(const httplib::Request& request,
                                           const ServeOptions& options);
[[nodiscard]] bool client_disconnected(const httplib::Request& request);

void prepare_sse_response(httplib::Response& response);
void configure_http_server_socket(socket_t socket) noexcept;
void set_owned_json_content(httplib::Response& response, std::string body,
                            std::shared_ptr<RequestLifetime> lifetime);

} // namespace infernix::serve
