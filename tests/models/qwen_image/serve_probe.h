// Does a live serve already hold the card? The GPU suites that write DEVICE-GLOBAL state (a device's
// default memory pool, a CUDA context, a whole-model resident set) must refuse BEFORE any CUDA call when
// an infernix-serve.exe owns the card: the sanctioned harness certifies the image lane in one offline
// window, and ctest records a 77 exit as a pass, so a suite that took the card beside a live serve
// looked exactly like a legitimate skip while disturbing serving and measuring a card someone else
// holds. That is why the suites that are not named "_real" (so tests/CMakeLists.txt:21-26 grants them no
// RUN_SERIAL and no "gpu;real" label) call this themselves and exit 77 whenever it reports a serve.
//
// The probe is the one this branch introduced in tests/models/qwen_image/test_image_stream_device_real.cpp
// (its serve_state()), kept here verbatim so a second copy cannot drift from it: it reads the Windows
// process table for the exact executable name of the live serving process. Where the process table cannot
// be read -- no snapshot, or a platform without a probe -- it fails closed with running == true, so the
// caller skips rather than write a shared card's state.
//
// It lives in its own host translation unit (serve_probe_host.cpp), which each suite that needs it lists
// beside its own source, so no CUDA translation unit has to carry <windows.h> through nvcc.

#pragma once

namespace infernix::tests {

struct ServeProbe {
    bool        running = false;
    const char* reason  = "";
};

// Reads only the OS process table: never a CUDA call, never a device context, so a suite may gate on it
// as the first thing it does.
[[nodiscard]] ServeProbe serve_state();

} // namespace infernix::tests
