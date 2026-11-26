// The serve probe's implementation, in a host translation unit so the suites that need it never carry
// <windows.h> through nvcc (the include-order rule of
// tests/models/qwen_image/test_image_stream_device_real.cpp:44-47: <windows.h> ahead of the project
// headers leaves their standard-library types unresolvable to cl.exe -- and a CUDA lane has no business
// parsing the Windows headers at all).
//
// The body is copied from that suite's own serve_state(), which is where this branch introduced it, so
// the two agree on the one thing that matters: the exact executable name of the live serving process,
// taken from `tasklist` on this machine.

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tlhelp32.h>
#endif

#include <cwchar>

#include "models/qwen_image/serve_probe.h"

namespace infernix::tests {

#if defined(_WIN32)

ServeProbe serve_state() {
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

// The probe reads the Windows process table. Where that is unavailable the caller cannot tell whether a
// serve holds the card, so it fails closed (skips) instead of writing a shared device's pool state.
ServeProbe serve_state() { return {/*running=*/true, "no serve probe exists for this platform, so the card cannot be proven free"}; }

#endif

} // namespace infernix::tests
