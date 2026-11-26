#pragma once

#include "serve/generation_service.h"
// PORT(ported tree): the image lane's own include, re-added by hand (ours-only, docs/maintainer/
// image-integration-points.md) -- HeartbeatTicker's stall watchdog reads the job's progress state.
#include "serve/image_job_progress.h"
#include "serve/request_json.h"

#include <httplib.h>

#include <atomic>
#include <chrono>
// PORT(ported tree): the image lane's own includes for HeartbeatTicker's thread (ours-only,
// docs/maintainer/image-integration-points.md).
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
// PORT(ported tree): the lane's own include, re-added by hand (ours-only, docs/maintainer/
// image-integration-points.md). SseTransport serializes its sink writes under a mutex so the image
// round ticker may write while the handler thread is blocked in a draw.
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
// PORT(ported tree): HeartbeatTicker's thread (ours-only, docs/maintainer/image-integration-points.md).
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

// How long the Messages and Responses streaming routes wait for the Engine's admission before they
// commit their response anyway. Within it the queue's own answers (a queue timeout, an orderly stop)
// are a plain HTTP status. A request still queued then gets its headers and the stream's heartbeats
// while it waits, so a client that gives up on a silent connection (one agent client closed every
// request queued for 120 s, 2026-11-14) keeps it; the queue's later answers arrive as the protocol's
// own error event. Chat Completions waits for admission instead (openai_chat_http.cpp).
inline constexpr std::chrono::seconds kStreamAdmissionGrace{5};

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

    // PORT(ported tree): the lane's half of this class, re-added by hand (ours-only,
    // docs/maintainer/image-integration-points.md). Upstream tip has no lock here because at its base
    // nothing ever wrote this sink from a second thread. The image lane does: while a server-executed
    // picture round blocks the handler (60-330 s of Engine silence), the lane's ticker thread keeps the
    // stream alive. Every sink write and the quiet-interval clock are therefore serialized under one
    // mutex, which is the object that owns both (this transport), not something a foreign thread may
    // bolt on from outside. Same text as the fork's src/serve/http_transport.h:57-61.
    void write(std::string_view item, Clock::time_point now = Clock::now());
    void write(const std::vector<std::string>& items, Clock::time_point now = Clock::now());

    // Called by the Engine wait loop. Besides observing an already-closed socket, a quiet stream
    // periodically writes a heartbeat so TCP_USER_TIMEOUT has traffic with which to detect an
    // unacknowledged peer, and so a client whose idle timer counts events (Codex's restarts only on an
    // event; a comment is no event) keeps the stream through a long prefill. A failed probe marks the
    // request for cancellation.
    [[nodiscard]] bool poll(Clock::time_point now = Clock::now());

    // The protocol's own event for a quiet stream (candidates UE-05): a Chat chunk with an empty delta, a
    // repeated `response.in_progress`, Anthropic's `ping`. It runs on the thread that writes the stream's
    // events, so it may number them; an empty result (an event not yet allowed) writes the SSE comment.
    void set_heartbeat(std::function<std::string()> event) { heartbeat_event_ = std::move(event); }

    // PORT(ported tree): re-added by hand, verbatim from the fork's src/serve/http_transport.h:68-76
    // (ours-only, docs/maintainer/image-integration-points.md). The ticker's quiet-interval heartbeat:
    // the comment half of poll() WITHOUT the socket's writability probe -- a ticker running while the
    // Engine loop is idle must not cancel the request on a transient !is_writable. A failed write marks
    // the request cancelled, exactly as poll() does.
    void beat(Clock::time_point now = Clock::now());

    [[nodiscard]] bool is_cancelled() const noexcept {
        return cancelled_.load(std::memory_order_acquire);
    }

    // How often the ticker beats: the transport's own quiet-interval bound, so a slower or faster
    // heartbeat configuration moves the ticker with it instead of disagreeing with it.
    [[nodiscard]] Clock::duration heartbeat_interval() const noexcept { return heartbeat_interval_; }

private:
    bool mark_cancelled() noexcept;

    // The sink write and the quiet-interval clock, under the caller's hold of write_mutex_. Every
    // public entry above takes that lock and then calls this: the lock is never taken twice, so no
    // public path may reach another public write while holding it (upstream's heartbeat_event_ is a
    // pure render and does not). True when the sink refused the item and the request was marked
    // cancelled -- which is what upstream's poll() returns on a failed heartbeat, and what its
    // tests/test_http_transport.cpp asserts.
    bool write_locked(std::string_view item, Clock::time_point now);

    httplib::DataSink& sink_;
    std::atomic<bool>& cancelled_;
    Clock::duration heartbeat_interval_;
    Clock::time_point last_write_;
    std::function<std::string()> heartbeat_event_;
    std::mutex write_mutex_;
};

// PORT(ported tree): the image lane's own ticker, re-added by hand verbatim from the fork's
// src/serve/http_transport.h:88-121 (ours-only, docs/maintainer/image-integration-points.md). Upstream
// tip deleted it along with the transport's lock: at its base nothing blocks a streaming handler for
// minutes, so its poll()-driven heartbeat (above) is the whole of its liveness. A server-executed
// picture round does block one for 60-330 s with the Engine loop stopped, so no poll runs and the
// stream would carry nothing -- a client with an idle timer sits through the whole draw. The lane owns
// the rule of what that silence carries on the wire (serve/chat_image_tool.cpp's progress note); the
// transport owns the sink, its quiet-interval clock and the lock this thread needs to reach them.
// When `watch` names a job's progress, the ticker is also its stall watchdog -- see
// serve/image_job_progress.h for what it may and may not flag.
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

// PORT(ported tree): the image lane's artifact URL builder, re-added by hand (ours-only,
// docs/maintainer/image-integration-points.md): every route that hands out an image URL builds it
// here, so a chat answer and an images response name the same GET.
// The absolute prefix this request's generated-image artifacts are reachable at.
[[nodiscard]] std::string artifact_url_base(const httplib::Request& request,
                                           const ServeOptions& options);
[[nodiscard]] bool client_disconnected(const httplib::Request& request);

void prepare_sse_response(httplib::Response& response);
// The listener's options: its port is exclusively its own (no second server binds beside it), and the
// connections it accepts inherit TCP keepalive and a bound on unacknowledged writes.
void configure_http_server_socket(socket_t socket) noexcept;
// Why the listener cannot bind `host`:`port`: the error a socket with the listener's options gets
// binding it now (cpp-httplib's bind keeps no error; 0 when none does), and what it means for the user.
[[nodiscard]] int probe_bind_error(const std::string& host, int port);
[[nodiscard]] std::string bind_failure_reason(int error);
void set_owned_json_content(httplib::Response& response, std::string body,
                            std::shared_ptr<RequestLifetime> lifetime);

} // namespace infernix::serve
