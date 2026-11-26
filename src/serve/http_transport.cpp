#include "serve/http_transport.h"

#include "serve/openai_images.h" // PORT(ported tree): kImageArtifactRoutePrefix, the artifact URL the image lane hands out
#include "serve/request_validation.h"
#include "text/json_input.h"

#if defined(__linux__)
#    include <netdb.h>
#    include <netinet/tcp.h>
#    include <sys/socket.h>
#    include <unistd.h>
#elif defined(_WIN32)
#    include <winsock2.h>
#    include <ws2tcpip.h>
#endif

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace infernix::serve {
namespace {

bool has_ngram_generation(const NgramArchiveStats& stats) noexcept {
    return stats.enabled && stats.bound && stats.published && stats.generation != 0;
}

#if defined(__linux__) || defined(_WIN32)
constexpr int kKeepAliveIdleSeconds                = 10;
constexpr int kKeepAliveIntervalSeconds            = 3;
constexpr int kKeepAliveProbeCount                 = 3;
constexpr unsigned int kTcpUserTimeoutMilliseconds = 15000;

template <class T>
void set_socket_option(socket_t socket, int level, int option, const T& value) noexcept {
    (void)::setsockopt(socket, level, option, reinterpret_cast<const char*>(&value),
                       static_cast<int>(sizeof(value)));
}
#endif

} // namespace

void set_ngram_generation_header(httplib::Response& response, const NgramArchiveStats& stats) {
    if (has_ngram_generation(stats)) {
        response.set_header("X-Infernix-Draft-Generation", std::to_string(stats.generation));
    }
}

std::string ngram_generation_comment(const NgramArchiveStats& stats) {
    if (!has_ngram_generation(stats)) { return {}; }
    return ": infernix-draft-generation: " + std::to_string(stats.generation) + "\n\n";
}

RequestJson parse_json_body(const httplib::Request& request) {
    text::ParsedJsonNumbers parsed;
    try {
        parsed = text::parse_json_numbers(request.body);
    } catch (const text::JsonNestingError& error) {
        bad_request(std::string("request body ") + error.what());
    } catch (const std::exception&) { bad_request("request body is not valid JSON"); }
    validate_schema_number_input(parsed);
    return std::move(parsed.value);
}

NgramSessionHints resolve_ngram_session(const httplib::Request& request, const RequestJson& body,
                                        const ServeOptions& options) {
    if (options.speculative.ngram_archive_bytes == 0) { return {}; }
    auto valid = [](std::string_view id) {
        // Leaves room for the identity namespace; the archive also bounds the complete key.
        return !id.empty() && id.size() <= 240 &&
               std::none_of(id.begin(), id.end(),
                            [](unsigned char c) { return c <= 32 || c == 127; });
    };
    auto field = [](const RequestJson& object, const char* key) -> std::string_view {
        const auto it = object.find(key);
        return it != object.end() && it->is_string() ? it->get_ref<const std::string&>()
                                                     : std::string_view{};
    };
    NgramSessionHints hints;
    if (request.has_header("x-infernix-draft-session")) {
        const auto key = request.get_header_value("x-infernix-draft-session");
        if (request.get_header_value_count("x-infernix-draft-session") != 1 || !valid(key)) {
            return {};
        }
        hints.key = "explicit:" + key;
        if (request.has_header("x-infernix-draft-reset")) {
            if (request.get_header_value_count("x-infernix-draft-reset") != 1 ||
                request.get_header_value("x-infernix-draft-reset") != "1") {
                return {};
            }
            hints.reset = true;
        }
        const bool parent     = request.has_header("x-infernix-draft-parent");
        const bool generation = request.has_header("x-infernix-draft-generation");
        if (parent != generation || (parent && hints.reset)) { return {}; }
        if (parent) {
            const auto name   = request.get_header_value("x-infernix-draft-parent");
            const auto number = request.get_header_value("x-infernix-draft-generation");
            if (!valid(name) || request.get_header_value_count("x-infernix-draft-parent") != 1 ||
                request.get_header_value_count("x-infernix-draft-generation") != 1) {
                return {};
            }
            const auto [end, error] = std::from_chars(number.data(), number.data() + number.size(),
                                                      hints.parent_generation);
            if (error != std::errc{} || end != number.data() + number.size() ||
                hints.parent_generation == 0) {
                return {};
            }
            hints.parent = "explicit:" + name;
        }
        return hints;
    }
    // Separate local opt-in includes store:false. Only verified conversation IDs
    // are recognized; an API key, cache prefix or device ID is not an identity.
    if (!options.ngram_native_sessions) { return {}; }
    bool ambiguous = false;
    auto select    = [&](std::string_view prefix, std::string_view id) {
        if (!valid(id) || !hints.key.empty()) {
            ambiguous = true;
            return;
        }
        hints.key = std::string(prefix) + std::string(id);
    };
    if (request.has_header("x-session-affinity")) {
        if (request.get_header_value_count("x-session-affinity") != 1) { return {}; }
        select("kilo:", request.get_header_value("x-session-affinity"));
    }
    if (const auto metadata = body.find("client_metadata");
        metadata != body.end() && metadata->is_object()) {
        const auto thread  = field(*metadata, "thread_id");
        const auto session = field(*metadata, "session_id");
        if (!thread.empty()) {
            if (!session.empty() && thread != session) { return {}; }
            select("codex:", thread);
        }
    }
    if (const auto metadata = body.find("metadata");
        metadata != body.end() && metadata->is_object()) {
        const auto user = field(*metadata, "user_id");
        if (!user.empty() && user.size() <= 4096) {
            const auto parsed = RequestJson::parse(user, nullptr, false);
            if (parsed.is_object()) {
                const auto session = field(parsed, "session_id");
                if (!session.empty()) { select("claude:", session); }
            }
        }
    }
    if (request.has_header("x-infernix-draft-parent") ||
        request.has_header("x-infernix-draft-generation")) {
        return {};
    }
    if (request.has_header("x-infernix-draft-reset")) {
        if (request.get_header_value_count("x-infernix-draft-reset") != 1 ||
            request.get_header_value("x-infernix-draft-reset") != "1") {
            return {};
        }
        hints.reset = true;
    }
    return ambiguous ? NgramSessionHints{} : hints;
}

std::string request_label(const httplib::Request& request) {
    constexpr const char* kHeader = "x-infernix-request-label";
    if (request.get_header_value_count(kHeader) != 1) { return {}; }
    std::string label = request.get_header_value(kHeader);
    const bool valid  = !label.empty() && label.size() <= 128 &&
                       std::all_of(label.begin(), label.end(),
                                   [](unsigned char c) { return c >= 32 && c < 127; });
    return valid ? label : std::string{};
}

bool client_disconnected(const httplib::Request& request) { return request.is_connection_closed(); }

void prepare_sse_response(httplib::Response& response) {
    response.set_header("Cache-Control", "no-cache");
    response.set_header("X-Accel-Buffering", "no");
}

SseTransport::SseTransport(httplib::DataSink& sink, std::atomic<bool>& cancelled,
                           Clock::duration heartbeat_interval, Clock::time_point now)
    : sink_(sink), cancelled_(cancelled), heartbeat_interval_(heartbeat_interval),
      last_write_(now) {
    if (heartbeat_interval_ <= Clock::duration::zero()) {
        throw std::invalid_argument("SSE heartbeat interval must be positive");
    }
}

bool SseTransport::mark_cancelled() noexcept {
    cancelled_.store(true, std::memory_order_release);
    return true;
}

// PORT(ported tree): the lane's half of this file, re-added by hand (ours-only, docs/maintainer/
// image-integration-points.md). Upstream tip writes the sink unlocked because only the handler thread
// ever does; the image lane's ticker thread writes it while the handler is blocked in a picture round,
// so sink_ and last_write_ are serialized under write_mutex_. The two public writes and poll() take the
// lock and delegate to write_locked(), which never reaches back into a public path. The fork's
// src/serve/http_transport.cpp:189-216 has this same shape without upstream tip's heartbeat_event_.
bool SseTransport::write_locked(std::string_view item, Clock::time_point now) {
    if (!sink_.write(item.data(), item.size())) {
        mark_cancelled();
        return true;
    }
    last_write_ = now;
    return false;
}

void SseTransport::write(std::string_view item, Clock::time_point now) {
    const std::lock_guard<std::mutex> lock(write_mutex_);
    if (cancelled_.load(std::memory_order_acquire)) { throw ClientDisconnected(); }
    write_locked(item, now);
    if (cancelled_.load(std::memory_order_acquire)) { throw ClientDisconnected(); }
}

void SseTransport::write(const std::vector<std::string>& items, Clock::time_point now) {
    const std::lock_guard<std::mutex> lock(write_mutex_);
    if (cancelled_.load(std::memory_order_acquire)) { throw ClientDisconnected(); }
    for (const std::string& item : items) {
        write_locked(item, now);
        if (cancelled_.load(std::memory_order_acquire)) { throw ClientDisconnected(); }
    }
}

void SseTransport::beat(Clock::time_point now) {
    if (cancelled_.load(std::memory_order_acquire)) { return; }
    const std::lock_guard<std::mutex> lock(write_mutex_);
    if (now - last_write_ < heartbeat_interval_) { return; }
    write_locked(kHeartbeatComment, now);
}

bool SseTransport::poll(Clock::time_point now) {
    if (cancelled_.load(std::memory_order_acquire)) { return true; }
    // The writability probe stays outside the lock: it asks the socket, not the sink's write state,
    // and it is the one thing the lane's ticker must NOT do (a transient !is_writable would cancel a
    // live image round). Everything after it touches sink_ and last_write_ and is serialized.
    if (sink_.is_writable && !sink_.is_writable()) { return mark_cancelled(); }
    const std::lock_guard<std::mutex> lock(write_mutex_);
    if (now - last_write_ < heartbeat_interval_) { return false; }
    const std::string event = heartbeat_event_ ? heartbeat_event_() : std::string();
    const std::string_view heartbeat = event.empty() ? kHeartbeatComment : std::string_view(event);
    // A heartbeat whose write the socket refused marks the request cancelled and poll() says so, exactly
    // as upstream tip's does; that is the whole of upstream's mid-stream disconnect detection here.
    return write_locked(heartbeat, now);
}

// PORT(ported tree): the image lane's ticker, re-added by hand verbatim from the fork's
// src/serve/http_transport.cpp:235-287 (ours-only, docs/maintainer/image-integration-points.md), with
// one lane-side correction noted on the watchdog below.
HeartbeatTicker::HeartbeatTicker(SseTransport& transport, std::function<std::string()> next_comment,
                                 std::function<std::shared_ptr<ImageJobProgress>()> watch,
                                 std::chrono::milliseconds stall_bound,
                                 std::function<SseTransport::Clock::time_point()> now)
    : transport_(transport), next_comment_(std::move(next_comment)), watch_(std::move(watch)),
      stall_bound_(stall_bound), now_(std::move(now)), thread_([this] { run(); }) {}

HeartbeatTicker::~HeartbeatTicker() {
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    condition_.notify_all();
    thread_.join();
}

void HeartbeatTicker::run() {
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (condition_.wait_for(lock, transport_.heartbeat_interval(),
                                    [this] { return stopping_; })) {
                return;
            }
        }
        // Not poll(): its heartbeat half would write a second comment, and its writability
        // probe cancels on a transient !is_writable, which must not abort an image round.
        // Dead peers surface through the write below.
        if (transport_.is_cancelled()) { return; }
        if (watch_) {
            // The stall watchdog: the ticker is the only observer of a job whose pipeline has
            // wedged; setting `stalled` makes the job throw at its next step boundary.
            //
            // PORT(ported tree): the one line the fork never had. It checked elapsed time against a
            // clock stamped when the job was CONSTRUCTED, and no handler ever passed `watch`, so the
            // mechanism never ran. Armed that way it would be wrong: before the first DiT step the job
            // legitimately spends minutes in the prompt enhancer's sidecar call, the weight upload and
            // the text tower, none of which advances the DiT's units. It therefore watches only a job
            // whose DiT phase has announced itself (serve/image_job_progress.h's `armed`, set by the
            // pipeline as it enters the loop), which is the only span where standing still means
            // something. The heartbeat itself never re-stamps that clock: keeping the socket alive is
            // not evidence the draw progressed.
            if (const auto progress = watch_()) {
                if (progress->armed.load(std::memory_order_relaxed)) {
                    const auto elapsed_ns =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            now_().time_since_epoch())
                            .count() -
                        progress->last_progress.load(std::memory_order_relaxed);
                    if (elapsed_ns > stall_bound_.count()) {
                        progress->stalled.store(true, std::memory_order_relaxed);
                    }
                }
            }
        }
        const std::string comment = next_comment_ ? next_comment_() : std::string();
        try {
            // One comment per interval: the progress comment replaces the standard beat, it
            // does not join it.
            transport_.write(comment.empty() ? std::string(SseTransport::kHeartbeatComment)
                                             : comment);
        } catch (const ClientDisconnected&) {
            return;
        }
    }
}

// The listener's port is its own: no second server binds beside it (candidates SR-09). cpp-httplib's
// default options would allow that (SO_REUSEPORT on Linux; SO_REUSEADDR on Windows, where any socket
// may then bind an address in use), and the kernel would split or misroute the connections: an old
// build answering a timed A/B, or every local client reaching another program.
void configure_http_server_socket(socket_t socket) noexcept {
#if defined(__linux__)
    const int enabled = 1;
    // SO_REUSEADDR alone binds over the TIME_WAIT remnants of an earlier server's connections,
    // as servers that set just SO_REUSEADDR (llama.cpp, vLLM, uvicorn) do (upstream #341), but
    // never beside a listener. SO_REUSEPORT would let a second listener share the port.
    set_socket_option(socket, SOL_SOCKET, SO_REUSEADDR, enabled);
    set_socket_option(socket, SOL_SOCKET, SO_KEEPALIVE, enabled);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPIDLE, kKeepAliveIdleSeconds);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPINTVL, kKeepAliveIntervalSeconds);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPCNT, kKeepAliveProbeCount);
    set_socket_option(socket, IPPROTO_TCP, TCP_USER_TIMEOUT, kTcpUserTimeoutMilliseconds);
#elif defined(_WIN32)
    // Only SO_EXCLUSIVEADDRUSE refuses a bind beside another listener on Windows, including a
    // wildcard bind beside another program's 127.0.0.1 listener (Strata #4255: local clients then
    // reached that program while the server reported ready). The same liveness as on Linux:
    // keepalive probes find a silently dead peer, and TCP_MAXRTMS (Windows' TCP_USER_TIMEOUT)
    // bounds how long unacknowledged stream writes are retried, so the SSE liveness gate sees a
    // failed write instead of the retransmission window.
    const DWORD enabled = 1;
    set_socket_option(socket, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, enabled);
    set_socket_option(socket, SOL_SOCKET, SO_KEEPALIVE, enabled);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPIDLE, static_cast<DWORD>(kKeepAliveIdleSeconds));
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPINTVL,
                      static_cast<DWORD>(kKeepAliveIntervalSeconds));
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPCNT, static_cast<DWORD>(kKeepAliveProbeCount));
    set_socket_option(socket, IPPROTO_TCP, TCP_MAXRTMS,
                      static_cast<DWORD>(kTcpUserTimeoutMilliseconds));
#else
    httplib::set_socket_opt(socket, SOL_SOCKET, SO_REUSEADDR, 1);
#endif
}

int probe_bind_error(const std::string& host, int port) {
    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags    = AI_PASSIVE;
    addrinfo* found   = nullptr;
    const std::string service = std::to_string(port);
    if (::getaddrinfo(host.empty() ? nullptr : host.c_str(), service.c_str(), &hints, &found) != 0) { return 0; }
    int error = 0;
    for (const addrinfo* at = found; at != nullptr && error == 0; at = at->ai_next) {
        const socket_t probe = ::socket(at->ai_family, at->ai_socktype, at->ai_protocol);
        if (probe == INVALID_SOCKET) { continue; }
        configure_http_server_socket(probe);
        if (::bind(probe, at->ai_addr, static_cast<int>(at->ai_addrlen)) != 0) {
#if defined(_WIN32)
            error = ::WSAGetLastError();
#else
            error = errno;
#endif
        }
#if defined(_WIN32)
        ::closesocket(probe);
#else
        ::close(probe);
#endif
    }
    ::freeaddrinfo(found);
    return error;
}

std::string bind_failure_reason(int error) {
    if (error == 0) { return "the operating system gave no reason"; }
#if defined(_WIN32)
    if (error == WSAEADDRINUSE) {
        return "the port is in use (another server, an earlier infernix-serve say, holds it): stop it or choose "
               "another --port";
    }
    if (error == WSAEACCES) {
        return "Windows refused the port: it may be in use by another server, or in a range Windows reserves "
               "(netsh int ipv4 show excludedportrange protocol=tcp); choose another --port";
    }
    return std::system_category().message(error);
#else
    if (error == EADDRINUSE) {
        return "the port is in use (another server, an earlier infernix-serve say, holds it): stop it or choose "
               "another --port";
    }
    return std::generic_category().message(error);
#endif
}

void set_owned_json_content(httplib::Response& response, std::string body,
                            std::shared_ptr<RequestLifetime> lifetime) {
    response.set_content(std::move(body), "application/json");
    response.user_data.set("infernix.request_lifetime", std::move(lifetime));
}

// The Host header is the address the client actually used, so a renamed, port-forwarded or
// containerized deployment hands back a URL that resolves from where that client is; only a request
// with no Host falls back to the bound address. The scheme is the deployment's, which is plain HTTP:
// this server has no TLS listener.
std::string artifact_url_base(const httplib::Request& request, const ServeOptions& options) {
    std::string host(request.get_header_value("Host"));
    if (host.empty()) { host = options.host + ":" + std::to_string(options.port); }
    return "http://" + host + std::string(kImageArtifactRoutePrefix);
}

} // namespace infernix::serve
