#pragma once

// The shared progress state of one in-flight image job. The job thread is its only writer (the
// pipeline's step hook); the streaming handler's HeartbeatTicker and the /v1/images/<id> route are
// lock-free readers. The ticker's stall watchdog is the only writer of `stalled`: a job whose
// percent and last_progress stand still past the stall bound is failed at the pipeline's next
// step boundary, which is the only point where unwinding is safe (the interlude still owns the
// yielded VRAM). It measures only a job that has called arm() -- see `armed` below for why the span
// before the first DiT step is not a span whose silence means a wedged draw.

#include <atomic>
#include <chrono>

#include <cmath>

namespace infernix::serve {

// steady_clock::now() as nanoseconds since the epoch, the unit `last_progress` carries.
[[nodiscard]] inline long long progress_clock_now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct ImageJobProgress {
    // -1 until the pipeline reports its first step; then a monotone 0-100.
    std::atomic<int> percent{-1};
    // steady_clock::now().time_since_epoch().count(), re-stamped by every advance().
    std::atomic<long long> last_progress{
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count()};
    std::atomic<bool> stalled{false};
    // The gate on the watchdog: `last_progress` is stamped when the job is constructed, and the span
    // before the first DiT step is not a span where standing still means a wedged draw -- the job may be
    // in the prompt enhancer's sidecar call, the weight upload or the text tower, any of which
    // legitimately runs past the stall bound. So the ticker measures only from arm(), which the pipeline
    // calls as it enters the loop, and arm() re-stamps the clock so the first step has its whole bound.
    // Sticky in both directions: a later advance() must not clear a stall, and nothing disarms one.
    std::atomic<bool> armed{false};

    // Begins the monitored phase: stamps the clock and lets the watchdog measure this job.
    void arm(long long at_ns = progress_clock_now()) {
        last_progress.store(at_ns, std::memory_order_relaxed);
        armed.store(true, std::memory_order_relaxed);
    }

    // Records one step. Returns false once the watchdog has flagged the job: the pipeline hook
    // throws at that point rather than continuing a wedged generation. `at_ns` lets a test move
    // the stamp without moving the real clock. Every step re-stamps it, whether or not the
    // displayed percent moved: an unchanged integer percent is still liveness.
    bool advance(int pct, long long at_ns = progress_clock_now()) {
        percent.store(pct, std::memory_order_relaxed);
        last_progress.store(at_ns, std::memory_order_relaxed);
        return !stalled.load(std::memory_order_relaxed);
    }
};

} // namespace infernix::serve
