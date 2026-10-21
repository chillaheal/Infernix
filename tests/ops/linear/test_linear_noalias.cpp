// ops::linear no-alias contract: `out` must not overlap `x` (in-place is unsupported). The cross-
// threadblock write-after-read hazard that in-place aliasing produces (observed at M=K=4096, N=11 as
// silent run-to-run non-determinism) is stopped by a host range-overlap check in
// validate_linear_semantics, which fires BEFORE dispatch_linear and before any CUDA call. This test
// protects that contract: an aliasing x/out must be refused with std::invalid_argument. Host-only --
// it never runs a kernel, so it runs on a device-less host too (no 77 skip, no cuda_available gate).

#include "core/tensor.h"
#include "core/weight.h"
#include "infernix/ops/linear.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace {

using namespace infernix;

alignas(16) std::uint8_t g_buf[4096];

bool expect_alias_rejected(const char* label, const Tensor& x, Tensor& out) {
    Weight w;
    w.qtype = QType::BF16;
    w.n = out.ne[0];
    w.k = x.ne[0];
    try {
        ops::linear(x, w, out, nullptr);
        std::printf("  %-16s FAIL  (no throw -- aliasing x/out was accepted)\n", label);
        return false;
    } catch (const std::invalid_argument& e) {
        std::printf("  %-16s OK  %s\n", label, e.what());
        return true;
    } catch (const std::exception& e) {
        std::printf("  %-16s FAIL  (unexpected exception: %s)\n", label, e.what());
        return false;
    }
}

} // namespace

int main() {
    // K x T input, N x T output (both [K|N, T, 1, 1]); T keeps a row longer than the 16-byte overlap
    // offset so the partial-overlap case is genuinely inside x's byte range.
    const std::int32_t K = 64, T = 8, N = 32;

    Tensor x(g_buf, DType::BF16, {K, T, 1, 1});
    Tensor out_full(g_buf, DType::BF16, {N, T, 1, 1});         // out == x (full alias)
    Tensor out_partial(g_buf + 16, DType::BF16, {N, T, 1, 1}); // 16-aligned partial overlap inside x

    int failures = 0;
    if (!expect_alias_rejected("in-place (out==x)", x, out_full)) {
        ++failures;
    }
    if (!expect_alias_rejected("partial overlap", x, out_partial)) {
        ++failures;
    }

    if (failures) {
        std::printf("LINEAR_NOALIAS_TEST: FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("LINEAR_NOALIAS_TEST: PASS (no-alias contract enforced: out==x and partial overlap rejected)\n");
    return 0;
}
