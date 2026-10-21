#pragma once

// The shared progress state of one in-flight image job. The job thread is its only writer (the
// pipeline's step hook); the streaming handler's HeartbeatTicker and the /v1/images/<id> route are
// lock-free readers. The ticker's stall watchdog is the only writer of `stalled`: a job whose
// percent and last_progress stand still past the stall bound is failed at the pipeline's next
// step boundary, which is the only point where unwinding is safe (the interlude still owns the
// yielded VRAM).

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

    // Records one step. Returns false once the watchdog has flagged the job: the pipeline hook
    // throws at that point rather than continuing a wedged generation. `at_ns` lets a test move
    // the stamp without moving the real clock.
    bool advance(int pct, long long at_ns = progress_clock_now()) {
        percent.store(pct, std::memory_order_relaxed);
        last_progress.store(at_ns, std::memory_order_relaxed);
        return !stalled.load(std::memory_order_relaxed);
    }
};

} // namespace infernix::serve
