// CPU-only coexistence-feasibility bound for the Qwen-Image-2.1 DiT image lane (task #13 step-4
// admission). Increment 3a proved the HOST-staging bound (peak = one weight object). This closes the
// DEVICE side: how much device memory the image lane must hold to run ONE diffusion step, per token
// count, and whether that fits inside a single increment-2 yield. The lane streams block weights
// (one block resident at a time), so the device peak is the BF16 activation workspace for S tokens
// plus the seven FP8 RowScale projections of one block plus the per-run shared tables -- never all 32
// blocks' weights.
//
// The workspace enumeration is taken from the qualified production-route test
// (test_image_depth_fp8_prod_cuda.cu): 16 activation buffers of S*D and 3 of S*MLP, all BF16, plus the
// shared modulation [2,4D], rotary [S,128], and affine ones/zeros [D]. D=4096, MLP=12288, 32 heads of
// 128. The gate is an EXTERNAL coexistence constraint, not an implementation mirror: every supported
// resolution's device peak must fit within the VRAM one increment-2 yield frees (measured 8192 MiB),
// so admission can run an image job against the live server with a single yield. Independent of any
// oracle .npy or artifact -- pure published-dimension arithmetic; always runs.
#include <cstdint>
#include <cstdio>

namespace {

constexpr std::uint64_t D = 4096;        // hidden size
constexpr std::uint64_t MLP = 12288;     // mlp inner dim per block
constexpr std::uint64_t BF16 = 2;
constexpr std::uint64_t FP8 = 1;
constexpr std::uint64_t ALIGN = 256;     // RowScale code->scale pad (matches the converter/framing)

std::uint64_t align_up256(std::uint64_t v) { return (v + (ALIGN - 1)) & ~(ALIGN - 1); }

// Device bytes held for one row-scaled FP8 E4M3 [n,k] weight: codes, pad-to-256, then n BF16 row scales.
std::uint64_t fp8_weight_bytes(std::uint64_t n, std::uint64_t k) {
    const std::uint64_t codes = n * k;
    return align_up256(codes) + n * BF16;
}

// One DiT block's seven FP8 projections resident at once (single-streamed-block device weight cost).
std::uint64_t one_block_weight_bytes() {
    std::uint64_t b = 0;
    b += 4 * fp8_weight_bytes(D, D);            // attention q, k, v, out
    b += 2 * fp8_weight_bytes(MLP, D);          // mlp proj, gate
    b += fp8_weight_bytes(D, MLP);              // mlp out
    return b;
}

// Activation workspace for S image+text tokens: the 16 S*D and 3 S*MLP BF16 buffers of the block run.
std::uint64_t workspace_bytes(std::uint64_t S) {
    return (16 * S * D + 3 * S * MLP) * BF16;
}

// Per-run shared tables: modulation [2,4D], rotary [S,128], affine ones [D], zeros [D] -- all BF16.
std::uint64_t shared_bytes(std::uint64_t S) {
    return (2 * 4 * D + S * 128 + 2 * D) * BF16;
}

// Total device peak for one diffusion step at S tokens: streamed block weights + workspace + shared.
std::uint64_t device_peak(std::uint64_t S) {
    return one_block_weight_bytes() + workspace_bytes(S) + shared_bytes(S);
}

} // namespace

int main() {
    // One increment-2 yield frees 8192 MiB toward the image job (image_yield_test.out.log). The image
    // lane's whole device peak per step must fit inside a single such yield at every supported size.
    const std::uint64_t YIELD_BYTES = 8192ULL * 1024ULL * 1024ULL;
    // Representative token counts: the T=72 oracle probe, a 256x256 image (256 image + 11 text = 267),
    // and large canvases up to 4096 joint tokens.
    const std::uint64_t sizes[] = {72, 267, 1024, 2048, 4096};

    bool ok = true;
    std::printf("image-lane device peak vs one yield (%llu MiB):\n",
                (unsigned long long)(YIELD_BYTES / (1024ULL * 1024ULL)));
    std::printf("  one streamed block weight = %llu MiB\n",
                (unsigned long long)(one_block_weight_bytes() / (1024ULL * 1024ULL)));
    std::uint64_t prev = 0;
    for (std::uint64_t S : sizes) {
        const std::uint64_t peak = device_peak(S);
        const bool fits = peak < YIELD_BYTES;
        const bool grows = (S == sizes[0]) || (peak > prev); // linear, monotonic in tokens
        ok = ok && fits && grows;
        std::printf("  S=%-5llu device_peak=%-8llu MiB  fits_one_yield=%s\n", (unsigned long long)S,
                    (unsigned long long)(peak / (1024ULL * 1024ULL)), (fits ? "yes" : "NO"));
        prev = peak;
    }

    // Streaming keeps ONE block (208 MiB) resident instead of all 32 (6658 MiB): a large reduction in
    // device weight residency. Even a 4096-token step stays near 1 GiB -- well under one yield -- so the
    // image lane can share the GPU with the live LLM under a single increment-2 yield.
    const std::uint64_t all_blocks = 32ULL * one_block_weight_bytes();
    const bool streamed_reduces = all_blocks > one_block_weight_bytes() &&
                                  device_peak(4096) < all_blocks &&
                                  device_peak(4096) < YIELD_BYTES;
    ok = ok && streamed_reduces;
    std::printf("  all 32 blocks resident = %llu MiB; streaming keeps 4096-token device peak = %llu MiB (fits one yield) = %s\n",
                (unsigned long long)(all_blocks / (1024ULL * 1024ULL)),
                (unsigned long long)(device_peak(4096) / (1024ULL * 1024ULL)),
                (streamed_reduces ? "yes" : "NO"));

    std::printf(ok ? "IMAGE_BUDGET_HOST_TEST: PASS\n" : "IMAGE_BUDGET_HOST_TEST: FAIL\n");
    std::fflush(stdout);
    return ok ? 0 : 1;
}
