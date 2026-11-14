// Task 6 (image fail-closed): the interlude's stream and the stream-ordered pool it allocates from
// belong to the ENGINE's device, and the pool's release threshold is a setting the image lane borrows
// and gives back -- not its to keep.
//
// WHAT THIS TEST ACTUALLY GATES (the honest list -- the last line repeats it)
//   1. The lane's PRODUCTION ImageStreamScope takes its device and its stream from the stream it was
//      handed, lowers the default pool's release threshold for the region, and writes the value it found
//      back when the region ends. These gates have teeth on every host: a scope that invented a device,
//      swapped the stream, or left the threshold lowered fails them.
//   2. A LEGACY region (the null stream the text and vision towers run on) touches neither that pool,
//      nor this thread's device, nor a sticky CUDA error that belongs to somebody else. Teeth on every
//      host: the region is its own block, so the pool, the thread and the planted sticky error are all
//      measured on BOTH SIDES of the destructor's close() -- and a blanket cudaGetLastError() in that
//      path would consume the planted error and fail the gate.
//   3. The scope binds the calling thread to the stream's device and puts it back afterwards. These two
//      gates need a SECOND CUDA device to be falsifiable: the thread is deliberately parked on a device
//      other than the stream's before the scope is constructed. On the certified single-GPU host (one
//      RTX 5090; AGENTS.md: "one GPU, one resident model") there is nothing else to stand on, so those
//      gates print "skip:" and are NOT counted as covered.
//   4. The stream the INTERLUDE delivers lives on the engine's device. That is a production line this
//      test cannot reach (Program::run_image_interlude needs a live, quiesced Program), and asserting it
//      by re-running the creation sequence here would only test CUDA against CUDA. The guard is
//      tests/models/qwen4_exp/test_image_interlude_prod_real.cu, which calls cudaStreamGetDevice on the
//      stream Engine::run_image_interlude hands to the job and compares it with the device the Engine was
//      built with. What is checked below is the test's own copy of that sequence, labelled as a probe.
//   5. The test's OWN hygiene: the device-global sentinel it writes must be undone before the verdict is
//      decided, so a leak fails the run instead of printing FAIL after the PASS line. Falsifiable on
//      every host; kept OUT of the gated tally, because it is not a guarantee of the image lane.
//
// The release threshold the test guards is a value THIS test writes. The engine never sets it today:
// the only cudaMemPoolAttrReleaseThreshold call anywhere under src/ was the image lane's own, so
// nothing else in the tree reads it either. It is therefore the test's stand-in for "global device
// state that is somebody else's to leave alone", not a claim about a number the engine configures.
//
// WHY IT REFUSES TO RUN BESIDE A SERVE
// Every gate here reads and writes DEVICE-GLOBAL state (a device's default memory pool) and a CUDA
// context, beside a live serve that would disturb serving and measure a card someone else holds. So a
// live infernix-serve.exe process makes this test return 77 BEFORE any CUDA call -- not even a device
// context is created -- and so does a machine with no CUDA device or a platform the serve probe has no
// implementation for. It is registered as a "_real" test, which tests/CMakeLists.txt gives RUN_SERIAL,
// the "gpu;real" labels and SKIP_RETURN_CODE 77, so the sanctioned host-test wrapper never touches the
// card; the run happens in the one offline window.

// The include order follows the one proven host precedent in this tree,
// tests/models/qwen4_exp/test_forward_real.cpp: the project's own headers, then CUDA, then the C++
// standard headers, and the Windows platform headers LAST -- <windows.h> ahead of the project headers
// leaves their standard-library types unresolvable to cl.exe.
//
// "artifact/binder.h" comes first because the lane's header declares its upload helpers over
// artifact::ParameterReference (src/artifact/binder.h) and relies on the consumer to have that
// declaration in place, exactly as tests/models/qwen_image/test_image_edit_behavior.cu does.
#include "artifact/binder.h"
#include "models/qwen_image/image_device_util.h"

#include "infernix/types.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cwchar>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tlhelp32.h>
#endif

namespace qi = infernix::models::qwen_image;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", what.c_str());
    }
}

[[noreturn]] void die(const std::string& what) { throw std::runtime_error(what); }

cudaError_t ok_or(cudaError_t error, const std::string& what) {
    if (error != cudaSuccess) { die(what + ": " + cudaGetErrorString(error)); }
    return error;
}

// The gates that COULD have failed on this host, and the ones that could not. A gate that cannot fail on
// the certified target is named, never counted: a green run must not read as coverage it did not get.
int g_gates_gated   = 0;
int g_gates_skipped = 0;

void gated(const std::string& what) {
    ++g_gates_gated;
    std::printf("gated: %s\n", what.c_str());
}

void cannot_fail(const std::string& what, const std::string& why) {
    ++g_gates_skipped;
    std::printf("skip: %s -- cannot fail on this host: %s\n", what.c_str(), why.c_str());
}

// Something this test cannot reach, and where the real guard for it lives. Neither coverage nor a skip.
void note(const std::string& what) { std::printf("note: %s\n", what.c_str()); }

// The thread's current device, which is what the gates must leave untouched.
int current_device_or_fail() {
    int device = -1;
    ok_or(cudaGetDevice(&device), "cudaGetDevice");
    return device;
}

// Every pool access below binds this thread to the pool's OWN device first and puts it back afterwards,
// which is the ordering the production scope itself uses: cudaSetDevice(device_) at
// src/models/qwen_image/image_device_util.h:198 before cudaDeviceGetDefaultMemPool at :202 and the
// threshold write at :141, with the thread restored only after the pool work (:151-159). WHY, rather
// than the alternative of reading the threshold while the thread happens to stand elsewhere: the CUDA
// 13.4 runtime does not document cudaMemPoolGetAttribute / cudaMemPoolSetAttribute tolerating a calling
// thread whose current device differs from the pool's, and cudaDeviceGetDefaultMemPool names a device the
// thread may not be on (the official pages docs.nvidia.com/cuda/cuda-runtime-api/
// group__CUDART__MEMPOOL.html and the 12.4.1 archive both answered 404 when this file was written, so no
// spec text settles it). A test must not rest a verdict on an undocumented tolerance: on a host with two
// or more devices a mismatch could make a CORRECT implementation refuse, and the gate would then fail for
// the wrong reason. Binding is plainly allowed either way, so the test binds. On the certified single-GPU
// host (one RTX 5090) previous == the pool's device, every bind here is a no-op, and nothing about the
// run changes.
class DeviceBind {
public:
    explicit DeviceBind(int device) : device_(device) {
        thread_known_ = cudaGetDevice(&previous_) == cudaSuccess;
        if (!thread_known_) {
            // This helper's own failed read: its sticky flag is ours to consume, and it is reported.
            (void)cudaGetLastError();
            ++failures;
            std::printf("FAIL: cudaGetDevice (before binding to the default pool of device %d)\n", device_);
            return;
        }
        if (const cudaError_t e = cudaSetDevice(device_); e != cudaSuccess) {
            ours_ = true;
            die("cudaSetDevice (to the default pool of device " + std::to_string(device_) + "): " +
                cudaGetErrorString(e));
        }
        bound_ = true;
    }
    ~DeviceBind() {
        if (bound_ && thread_known_ && previous_ != device_) {
            if (const cudaError_t e = cudaSetDevice(previous_); e != cudaSuccess) {
                ours_ = true;
                ++failures;
                std::printf("FAIL: cudaSetDevice (this thread back to device %d): %s\n", previous_,
                            cudaGetErrorString(e));
            }
        }
        if (ours_) { (void)cudaGetLastError(); } // only this helper's own refusals, each reported above
    }
    DeviceBind(const DeviceBind&)            = delete;
    DeviceBind& operator=(const DeviceBind&) = delete;

private:
    int  device_;
    int  previous_     = -1;
    bool thread_known_ = false;
    bool bound_        = false;
    bool ours_         = false;
};

// The default pool of ONE EXPLICIT device, never of "the current one": the test's device is the
// engine's, and the pool that matters is that device's. The caller owns the binding; see DeviceBind.
cudaMemPool_t default_pool_of(int device) {
    cudaMemPool_t pool = nullptr;
    ok_or(cudaDeviceGetDefaultMemPool(&pool, device), "cudaDeviceGetDefaultMemPool (device " + std::to_string(device) + ")");
    if (pool == nullptr) { die("cudaDeviceGetDefaultMemPool returned a null pool for device " + std::to_string(device)); }
    return pool;
}

std::uint64_t read_release_threshold(int device) {
    const DeviceBind  bind(device);
    std::uint64_t     threshold = 0;
    ok_or(cudaMemPoolGetAttribute(default_pool_of(device), cudaMemPoolAttrReleaseThreshold, &threshold),
          "cudaMemPoolGetAttribute (release threshold)");
    return threshold;
}

void write_release_threshold(int device, std::uint64_t threshold) {
    const DeviceBind bind(device);
    ok_or(cudaMemPoolSetAttribute(default_pool_of(device), cudaMemPoolAttrReleaseThreshold, &threshold),
          "cudaMemPoolSetAttribute (release threshold)");
}

// ---- the serve probe -------------------------------------------------------------
//
// The exact executable name of the live serving process, copied from `tasklist` on this machine.
#if defined(_WIN32)
struct ServeProbe {
    bool        running = false;
    const char* reason  = "";
};

[[nodiscard]] ServeProbe serve_state() {
    const HANDLE snapshot = ::CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        // Cannot tell whether the card is shared: fail closed rather than write a shared card's pools.
        return {/*running=*/true, "the process snapshot failed, so the test cannot prove no serve holds the card"};
    }
    PROCESSENTRY32W entry{};
    entry.dwSize = sizeof(entry);
    bool running = false;
    for (BOOL more = ::Process32FirstW(snapshot, &entry); more == TRUE; more = ::Process32NextW(snapshot, &entry)) {
        if (_wcsicmp(entry.szExeFile, L"infernix-serve.exe") == 0) { running = true; break; }
    }
    ::CloseHandle(snapshot);
    return {running, running ? "an infernix-serve.exe process holds the card" : ""};
}
#else
// The probe reads the Windows process table. Where that is unavailable the test cannot tell whether a
// serve holds the card, so it fails closed (skips) instead of writing a shared device's pool state.
struct ServeProbe {
    bool        running = true;
    const char* reason  = "no serve probe exists for this platform, so the card cannot be proven free";
};
[[nodiscard]] ServeProbe serve_state() { return ServeProbe{}; }
#endif

// ---- the interlude's creation sequence, as far as a test can reach it -------------
//
// Program::run_image_interlude needs a live, quiesced LLM Program to reach, so this reproduces its two
// device calls in their order -- bind this thread to the engine's device, cudaStreamCreate (a stream
// takes the CALLING thread's current device), put the thread back -- parks this thread on
// `thread_park_device` (a DIFFERENT device whenever the host has one, so the production scope below has
// to move the thread rather than find it already in place), and then holds the lane's production
// ImageStreamScope over that stream. Everything asserted is production state read through production
// APIs, and the scope is the production object.
struct TestInterlude {
    TestInterlude(int engine_device, int thread_park_device)
        : engine_device_(engine_device), thread_park_device_(thread_park_device) {
        thread_home_      = current_device_or_fail(); // where the test found this thread, before any parking
        const int thread_device = thread_home_;
        ok_or(cudaSetDevice(engine_device_), "cudaSetDevice (bind the calling thread to the engine's device)");
        if (const cudaError_t created = cudaStreamCreate(&lane_stream); created != cudaSuccess) {
            (void)cudaSetDevice(thread_device);
            (void)cudaGetLastError(); // the refusal of this test's own call, already in the text thrown
            die("cudaStreamCreate (the interlude's stream): " + std::string(cudaGetErrorString(created)));
        }
        ok_or(cudaSetDevice(thread_device), "cudaSetDevice (put the calling thread back)");
        ok_or(cudaSetDevice(thread_park_device_), "cudaSetDevice (park this thread away from the stream's device)");
        try {
            scope = std::make_unique<qi::ImageStreamScope>(lane_stream);
        } catch (...) {
            destroy_stream();
            throw;
        }
    }
    ~TestInterlude() {
        close();
        destroy_stream();
    }
    TestInterlude(const TestInterlude&)            = delete;
    TestInterlude& operator=(const TestInterlude&) = delete;

    // The lane's production scope ends here, exactly as the interlude ends its region -- and its verdict
    // is kept, because "everything borrowed came back" is what this test exists to prove. (The
    // production scope reports a failed restoration instead of throwing from a destructor that is
    // already unwinding, so a test that only watched for an exception would miss it.)
    void close() {
        if (scope != nullptr) {
            restoration = scope->close();
            report      = scope->last_report();
            scope.reset();
        }
    }
    [[nodiscard]] cudaStream_t stream() const noexcept { return lane_stream; }
    [[nodiscard]] int thread_park_device() const noexcept { return thread_park_device_; }

    cudaStream_t                          lane_stream = nullptr;
    std::unique_ptr<qi::ImageStreamScope> scope;
    std::string                           restoration;
    std::string                           report;

private:
    // The teardown Program::run_image_interlude applies to the same object: bind to the stream's device,
    // destroy, put this thread back -- and say so when any of it could not be done, rather than ignoring
    // a leak of the test's own.
    void destroy_stream() {
        if (lane_stream == nullptr) { return; }
        // The thread goes back where the TEST found it, not merely where it happens to stand now -- the
        // same rule the production hand-back and Program::run_image_interlude's teardown follow. Without
        // it the test's own teardown would leave this thread parked on the second device it deliberately
        // moved to, and the hygiene assertion at the end of main() would fail there for a reason that has
        // nothing to do with the lane. (The production restore is gated earlier, right after close() and
        // before this function ever runs.)
        const bool thread_known = thread_home_ >= 0;
        const cudaError_t bound     = cudaSetDevice(engine_device_);
        const cudaError_t destroyed = cudaStreamDestroy(lane_stream);
        cudaError_t       restored  = cudaSuccess;
        if (bound == cudaSuccess && thread_known) { restored = cudaSetDevice(thread_home_); }
        // Consume the sticky flag ONLY when a call of this function's own refused, and report each such
        // refusal. Clearing it whenever the thread merely happened to move (an earlier version of this
        // line) silently ate a pending foreign error on every host with two devices, which is exactly the
        // discipline the production lane had to be fixed to (src/models/qwen_image/image_device_util.h:168
        // clears only when self_produced_).
        bool ours = !thread_known;
        if (!thread_known) { std::printf("FAIL: the test never recorded the thread's own home device\n"); }
        if (bound != cudaSuccess) {
            ours = true;
            std::printf("FAIL: cudaSetDevice (the stream's device %d): %s\n", engine_device_,
                        cudaGetErrorString(bound));
        }
        if (destroyed != cudaSuccess) {
            ours = true;
            std::printf("FAIL: cudaStreamDestroy (the test's copy of the interlude stream): %s\n",
                        cudaGetErrorString(destroyed));
        }
        if (restored != cudaSuccess) {
            ours = true;
            std::printf("FAIL: cudaSetDevice (this thread back to the device the test found it on, %d): %s\n",
                        thread_home_, cudaGetErrorString(restored));
        }
        if (ours) {
            ++failures;
            (void)cudaGetLastError(); // this function's own refusals, every one reported above
        }
        lane_stream = nullptr;
    }

    int engine_device_     = -1;
    int thread_park_device_ = -1;
    int thread_home_        = -1; // this thread's device when the test found it, read before any parking
};

[[nodiscard]] TestInterlude open_test_interlude_scope(int engine_device, int thread_park_device) {
    return TestInterlude(engine_device, thread_park_device);
}

// Undo what this test wrote to a device-global pool, whatever happens to the block below: leaving the
// sentinel behind on an exception path would hand the LLM lane a pool setting the test made up.
//
// The undo is ATTEMPTED by the destructor (so an exception path never leaves the sentinel behind) but
// ASSERTED by the explicit call in the linear path, before the verdict is decided: a guard that only
// reports from a destructor runs after the exit code is settled and can print FAIL next to "PASS" without
// changing either. See main(), "the test's own hygiene".
class SentinelRestore {
public:
    SentinelRestore(int device, std::uint64_t as_the_llm_left_it)
        : device_(device), as_the_llm_left_it_(as_the_llm_left_it) {}
    SentinelRestore(const SentinelRestore&)            = delete;
    SentinelRestore& operator=(const SentinelRestore&) = delete;

    // Writes the value the LLM lane left behind back over the test's sentinel. Idempotent; never throws,
    // so it is safe from an unwinding destructor. Reports nothing -- the caller's check() does, with
    // reason() -- and returns whether the device really holds the original value again.
    bool undo() {
        if (done_) { return undone_; }
        done_ = true;
        try {
            const DeviceBind  bind(device_);
            cudaMemPool_t     pool = default_pool_of(device_);
            ok_or(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &as_the_llm_left_it_),
                  "cudaMemPoolSetAttribute (undo the sentinel release threshold on device " +
                      std::to_string(device_) + ")");
            undone_ = true;
        } catch (const std::exception& failure) {
            reason_ = failure.what();
            (void)cudaGetLastError(); // the refusal that raised this is the guard's own, and it is kept
        }
        return undone_;
    }
    [[nodiscard]] bool                undone() const noexcept { return undone_; }
    [[nodiscard]] const std::string&  reason() const noexcept { return reason_; }

    ~SentinelRestore() { (void)undo(); }

private:
    int           device_;
    std::uint64_t as_the_llm_left_it_;
    bool          done_   = false;
    bool          undone_ = false;
    std::string   reason_;
};

} // namespace

int main() {
    const ServeProbe serve = serve_state();
    if (serve.running) {
        std::printf("SKIP: %s\n", serve.reason);
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0) {
        (void)cudaGetLastError();
        std::printf("SKIP: no CUDA device is present, so there is no default pool to gate\n");
        return 77;
    }

    // The engine's device, read from the public option the Engine builds its DeviceContext from
    // (src/runtime/engine/engine.cpp: initialize_device() constructs DeviceContext(options.device),
    // and Program::run_image_interlude creates the stream on impl_->device_.device).
    const int engine_device = infernix::EngineOptions{}.device;

    // The device this thread stands on while the lane's scope runs, chosen by ENUMERATION rather than by
    // an index so that no device literal appears at any stream or pool site. With a second device it is
    // deliberately NOT the stream's: that is what makes a missing bind or a missing restore observable.
    int thread_park_device = engine_device;
    for (int candidate = 0; candidate != devices; ++candidate) {
        if (candidate != engine_device) {
            thread_park_device = candidate;
            break;
        }
    }
    const bool have_second_device = thread_park_device != engine_device;

    try {
        const int             thread_before      = current_device_or_fail();
        const std::uint64_t   as_the_llm_left_it = read_release_threshold(engine_device);
        SentinelRestore       sentinel{engine_device, as_the_llm_left_it};

        // The value the image lane must hand back, written by this test alone. 4 MiB, non-zero and
        // distinct from the CUDA default, so a lane that leaves 0 behind cannot pass.
        constexpr std::uint64_t kThresholdBefore = 4ULL << 20;
        write_release_threshold(engine_device, kThresholdBefore);
        check(read_release_threshold(engine_device) == kThresholdBefore,
              "the sentinel threshold landed on the device");

        // ---- gate 2: the LEGACY (null-stream) route borrows nothing -----------------
        //
        // Measured on BOTH SIDES of close(): the region is its own block, so the destructor's hand-back
        // runs at its closing brace and every assertion below that brace tests it. Asserting only inside
        // the region tested the constructor -- which is how the first version of this gate came to be
        // vacuous against the sticky-error defect it was written for.
        {
            const int     thread_before_legacy = current_device_or_fail();
            cudaError_t   foreign_sticky       = cudaSuccess; // the planted error, if the plant worked
            bool          have_foreign         = false;
            {
                // Declared first so it is destroyed last: this destructor IS the close() the gate measures.
                qi::ImageStreamScope legacy_scope(nullptr);
                check(legacy_scope.stream() == nullptr,
                      "a legacy scope hands back the null stream it was given, claiming none");
                check(read_release_threshold(engine_device) == kThresholdBefore,
                      "a legacy region leaves the default pool's release threshold untouched while it runs");
                check(current_device_or_fail() == thread_before_legacy,
                      "a legacy region claims no device for this thread while it runs");
                // Plant a sticky error that is NOT the lane's, INSIDE the region and after the reads
                // above: the towers' own CUDA_CHECK(cudaGetLastError()) idiom
                // (src/ops/dit_adaln/launch.cu, src/core/block_copy.cu, src/core/device.cu) depends on it
                // surviving a legacy region, and a blanket clear in close() would consume it. The plant is
                // deliberately the last thing before the brace, so no query of this test's own runs while
                // the flag is pending and none of them can be affected by what a runtime call does with
                // somebody else's sticky error.
                const cudaError_t foreign = cudaEventDestroy(nullptr); // deliberately invalid
                const bool can_probe_stickiness =
                    foreign != cudaSuccess && cudaPeekAtLastError() == foreign; // peek: does not clear
                // The premise, not a gate: without a planted foreign error there is nothing to preserve,
                // and the gate below is then named as unfalsifiable here rather than counted.
                if (!can_probe_stickiness) {
                    cannot_fail("a legacy (null-stream) region borrows no pool, no device and no sticky error",
                                "the runtime raised no sticky error to plant");
                }
                if (can_probe_stickiness) {
                    // Asserted below the brace, so the sticky check reads the flag AFTER close() ran.
                    foreign_sticky   = foreign;
                    have_foreign     = true;
                } else {
                    (void)cudaGetLastError(); // nothing to test, and no flag left behind for the next gate
                }
            }
            // ---- after close() ----------------------------------------------------
            if (have_foreign) {
                check(cudaGetLastError() == foreign_sticky,
                      "a legacy region did not consume a sticky error that was not its own");
                gated("a legacy (null-stream) region borrows no pool, no device and no sticky error");
            }
            check(read_release_threshold(engine_device) == kThresholdBefore,
                  "a legacy region left the default pool's release threshold untouched through close() too");
            check(current_device_or_fail() == thread_before_legacy,
                  "a legacy region gave this thread back the device it stood on, having claimed none");
        }

        // ---- gates 1, 3 and 4: the production scope over the interlude's stream ------
        {
            auto scope            = open_test_interlude_scope(engine_device, thread_park_device);
            int  stream_device    = -1;
            ok_or(cudaStreamGetDevice(scope.stream(), &stream_device), "cudaStreamGetDevice");

            // Gate 1, falsifiable on every host: the scope reads its device and its stream from the
            // stream it was handed -- production reads, no test copy of the logic.
            check(scope.scope->device() == stream_device,
                  "the production scope takes its device from the stream it was handed");
            check(scope.scope->stream() == scope.lane_stream,
                  "the production scope keeps the very stream it was handed");
            gated("the scope takes its device and its stream from the stream it was handed");

            check(stream_device == engine_device,
                  "the test's own copy of the interlude's creation sequence put the stream on the engine's device");
            note("that the INTERLUDE's delivered stream is the engine's device is a production line no test here "
                 "can reach; the guard is tests/models/qwen4_exp/test_image_interlude_prod_real.cu, which calls "
                 "cudaStreamGetDevice on the stream Engine::run_image_interlude hands to the job");

            if (have_second_device) {
                check(current_device_or_fail() == engine_device,
                      "the lane's region binds this thread to the stream's device, where its synchronous uploads order");
                gated("the scope binds the calling thread to the stream's device");
            } else {
                cannot_fail("the scope binds the calling thread to the stream's device",
                            "this host has one CUDA device, so the bound device is indistinguishable from the "
                            "thread's default");
            }

            check(read_release_threshold(engine_device) == 0,
                  "the lane's region releases retired buffers to the driver while it holds the stream");

            scope.close();
            check(scope.restoration.empty(),
                  "the production scope reported no failure handing the pool and the thread back");
            check(scope.report.empty(), "the production scope recorded nothing it failed to hand back");
            // close() has by now put this thread back on the device the test parked it on, which on a host
            // with a second CUDA device is NOT the pool's. So this read must name the device and bind to
            // it (DeviceBind) rather than leave the thread elsewhere and hope the pool call tolerates it
            // -- that is the choice recorded in
            // the DeviceBind comment above, and the reason it is made here rather than by reading the
            // threshold while still bound: the assertion is about what close() LEFT on the device, which
            // can only be measured after close() returned the thread.
            check(read_release_threshold(engine_device) == kThresholdBefore,
                  "the default pool's release threshold came back as it was");
            gated("the default pool's release threshold is borrowed for the region and handed back exactly");

            if (have_second_device) {
                check(current_device_or_fail() == scope.thread_park_device(),
                      "close() put the calling thread back on the device it was found on");
                gated("close() restores the calling thread's device");
            } else {
                cannot_fail("close() restores the calling thread's device",
                            "this host has one CUDA device, so the thread cannot be parked anywhere else to prove "
                            "a missing restore");
            }
        }
        // Nobody -- lane or test -- may leave this thread on a device it was not found on. The test moved
        // it on purpose (TestInterlude parked it away from the stream's device so that a missing
        // production restore would show), and its own teardown brings it home (destroy_stream restores to
        // the device the test recorded at construction), so this assertion still has teeth on a host with
        // two CUDA devices: a teardown that forgot to bring the worker back fails it. The PRODUCTION
        // restore is gated earlier, right after close() and before that teardown runs.
        check(current_device_or_fail() == thread_before,
              "the test's own thread is where the test left it");

        // ---- the test's own hygiene: asserted HERE, not from a destructor -----------------
        // The sentinel this test wrote to the device's default pool must be gone before the verdict is
        // decided. The guard's destructor still attempts the undo (an exception path must not leave the
        // LLM lane with an invented pool setting), but a destructor runs after the exit code is settled,
        // so it could print FAIL next to "IMAGE_STREAM_DEVICE_REAL: PASS" and exit 0. undo() never throws
        // and reports nothing; check() below owns the verdict, with reason() in the message.
        sentinel.undo();
        std::string undo_what = "the sentinel release threshold this test wrote came back off the device";
        if (!sentinel.undone() && !sentinel.reason().empty()) { undo_what += ": " + sentinel.reason(); }
        check(sentinel.undone(), undo_what);

        if (failures != 0) {
            std::printf("%d check(s) failed\n", failures);
            return 1;
        }
        std::printf("guarantees gated on this host: %d, not falsifiable here (skipped): %d of %d\n", g_gates_gated,
                    g_gates_skipped, g_gates_gated + g_gates_skipped);
        if (g_gates_gated == 0) {
            std::printf("NOTE: this run checked no guarantee that could fail on this host\n");
        }
        std::printf("IMAGE_STREAM_DEVICE_REAL: PASS\n");
        return 0;
    } catch (const std::exception& failure) {
        std::printf("FAIL: %s\n", failure.what());
        return 1;
    }
}
