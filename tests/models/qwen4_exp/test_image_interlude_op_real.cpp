// C4 (LLM<->image coexistence): the first real image op on the LLM Program's YIELDED VRAM, in-process.
//
// C1/C2 (test_image_yield_real.cpp) proved the Program can quiesce and yield physical VRAM
// (cuMemUnmap of the expert-frame + KV pools) and that giving it back restores the pools so a later
// generation is undisturbed. This is the missing link: an ACTUAL image kernel runs in that freed
// space -- the Qwen-Image-2.1 DiT's row-scaled FP8 [4096,4096] projection (q/k/v/out), its real weight
// streamed object-by-object from the image artifact through the bounded pinned HostContextArena
// (peak staging == this single object, never the sum), H2D'd into a device buffer allocated in the
// freed region, and run with ops::linear on the drained device while the LLM sits Quiesced/Leased.
// The LLM then resumes undisturbed (post-interlude ids == baseline) -- the coexistence gate.
//
// Self-verifying: a CPU reference dequantizes the SAME E4M3 codes + row scales the kernel reads
// (e4m3_value) and does the dot products in double, so no external oracle is needed. Needs both
// artifacts: INFERNIX_QWEN4_ARTIFACT (the LLM to yield from) and INFERNIX_TEST_ARTIFACT (the
// Qwen-Image-2.1-FP8 artifact to stream a real DiT weight from). Skips (77) without either.
//
//   INFERNIX_QWEN4_ARTIFACT=part1.infernix [INFERNIX_QWEN4_NGRAM=..ngram] \
//   INFERNIX_TEST_ARTIFACT=out/Qwen-Image-2.1-FP8.infernix infernix_qwen4_exp_image_interlude_op_real_test

#include "artifact/reader.h"
#include "core/host_context_arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "core/vram_budget.h"
#include "infernix/engine.h"
#include "infernix/ops/linear.h"
#include "models/qwen4_exp/program/program.h"
#include "runtime/expert_offload/vram_monitor.h"
#include "ops/common/canonical_math.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::artifact;
namespace ops = infernix::ops;

constexpr std::uint64_t kGiB = 1ULL << 30;
constexpr int D = 4096;  // DiT hidden dim; [D,D] is a registered FP8 linear shape (q/k/v/out)
constexpr int T = 32;    // tokens: a clean GEMM width inside the qualified image-test regime (T>=72)

std::uint64_t device_free_bytes() {
    std::size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { throw std::runtime_error("cudaMemGetInfo failed"); }
    return free_b;
}

std::uint16_t to_bf16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    if ((x & 0x7fffffffU) > 0x7f800000U) return static_cast<std::uint16_t>((x >> 16) | 0x0040U);
    const std::uint32_t lsb = (x >> 16) & 1U;
    x += 0x7fffU + lsb;
    return static_cast<std::uint16_t>(x >> 16);
}
float from_bf16(std::uint16_t h) {
    std::uint32_t x = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

infernix::PromptInput story() {
    infernix::PromptInput input;
    infernix::ChatMessage message;
    message.role = infernix::ChatRole::User;
    message.parts.push_back({.kind = infernix::MessagePartKind::Text,
                             .text = "Write a long story about a lighthouse keeper who befriends a seagull."});
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

infernix::RequestOptions greedy(std::uint32_t tokens) {
    infernix::RequestOptions request;
    request.execution.requested_output_tokens = tokens;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    return request;
}

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) { ++failures; }
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* qwen4 = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    const char* image_artifact = std::getenv("INFERNIX_TEST_ARTIFACT");
    if (qwen4 == nullptr || image_artifact == nullptr) {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT and INFERNIX_TEST_ARTIFACT\n");
        return 77;
    }
    int result = 1;
    try {
        auto real = infernix::open_vram_budget_source(0);
        infernix::testing::set_vram_budget_source([&] { return real->query(); });
        infernix::runtime::expert_offload::testing::set_vram_grow_delay(3600.0);

        infernix::EngineOptions options;
        options.artifact_path         = qwen4;
        options.context_cache.enabled = false;
        options.enable_vision         = false;
        options.max_context           = 4096;
        options.kv_capacity           = infernix::KvCapacityPolicy::explicit_capacity(4096);
        options.kv_cache              = infernix::KvCacheStorage::Int8Group64;
        options.expert_ram_bytes      = 8 * kGiB;
        if (const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM")) { options.ngram_volume_path = ngram; }

        infernix::Engine engine(options);
        check(infernix::models::qwen4_exp::testing::has_image_yield_program(), "a Program is loaded to yield from");

        const auto baseline = engine.generate(engine.prepare(story()), greedy(200)).generated_token_ids;
        check(baseline.size() == 200, "baseline generated 200 tokens");

        // The image artifact: the first DiT [D,D] row-scaled-FP8 projection (q/k/v/out).
        const Reader reader(image_artifact);
        const Directory& dir = reader.directory();
        std::size_t wi = 0;
        bool have_weight = false;
        for (std::size_t i = 0; i < dir.objects.size(); ++i) {
            const auto* t = std::get_if<TensorObject>(&dir.objects[i]);
            if (t != nullptr && t->format == "fp8_e4m3fn_row_bf16" && t->shape.size() == 2 &&
                t->shape[0] == D && t->shape[1] == D) { wi = i; have_weight = true; break; }
        }
        if (!have_weight) { std::printf("SKIP: no [%d,%d] FP8 DiT weight in the image artifact\n", D, D); return 77; }
        reader.validate_object({wi});
        const WeightGeometry& g = reader.geometry({wi});
        const std::uint64_t len = object_bytes(dir.objects[wi]);
        const std::uint64_t off = object_offset(dir.objects[wi]);
        check(g.code_bytes == static_cast<std::uint64_t>(D) * D, "geometry code plane is the [D,D] code buffer");
        check(g.scale_offset + g.scale_bytes == len, "geometry spans the whole object");

        // Interlude: close the door, drain every LLM stream, THEN yield its physical VRAM.
        const std::uint64_t free_full = device_free_bytes();
        const std::uint64_t epoch     = infernix::models::qwen4_exp::testing::begin_image_quiesce();
        check(epoch != 0, "begin_image_quiesce returned an execution epoch");
        check(infernix::models::qwen4_exp::testing::image_interlude_state() == 2, "the Program is Quiesced");

        std::uintptr_t lease = 0;
        std::uint64_t freed  = 0;
        try {
            freed = infernix::models::qwen4_exp::testing::yield_image_vram(8 * kGiB, 0, lease);
            check(freed > 0, "the yield freed " + std::to_string(freed >> 20) + " MiB toward the image job");
            check(lease != 0, "the yield returned a lease to restore");
            const std::uint64_t free_peak = device_free_bytes();
            check(free_peak - free_full > kGiB / 2, "driver free VRAM rose across the yield (the unmap is physical)");

            // Stream the real DiT weight through the bounded pinned pool (peak staging == this object).
            HostContextArena arena(static_cast<std::size_t>((len + 255ULL) & ~255ULL), 256);
            auto staged = arena.allocate(len);
            check(staged.has_value(), "the pinned pool holds one [D,D] weight object");
            reader.read_into(off, std::span<std::byte>(staged->data(), len));

            // The image buffers are allocated in the freed region (the LLM no longer maps that VRAM).
            const std::uint64_t free_before_op = device_free_bytes();
            void* d_w = nullptr, *d_in = nullptr, *d_out = nullptr;
            const cudaError_t mw = cudaMalloc(&d_w, len);
            const cudaError_t mi = cudaMalloc(&d_in, static_cast<std::size_t>(D) * T * 2);
            const cudaError_t mo = cudaMalloc(&d_out, static_cast<std::size_t>(D) * T * 2);
            check(mw == cudaSuccess && mi == cudaSuccess && mo == cudaSuccess, "image buffers allocated in the freed VRAM");
            const std::uint64_t free_with_bufs = device_free_bytes();
            const std::uint64_t consumed = free_before_op > free_with_bufs ? free_before_op - free_with_bufs : 0;
            std::printf("  image buffers consumed %llu MiB of the freed VRAM\n",
                        static_cast<unsigned long long>(consumed >> 20));
            check(consumed >= 12ULL * 1024 * 1024, "the weight landed in the freed region (VRAM actually used)");

            check(cudaMemcpy(d_w, staged->data(), len, cudaMemcpyHostToDevice) == cudaSuccess,
                  "weight streamed into the freed device region");

            // Synthetic deterministic input (dim 0 = D fastest), BF16-represented for both device and reference.
            std::vector<std::uint16_t> in_bits(static_cast<std::size_t>(D) * T);
            std::vector<double> xref(static_cast<std::size_t>(D) * T);
            for (int t = 0; t < T; ++t) {
                for (int r = 0; r < D; ++r) {
                    const std::uint16_t h = to_bf16(std::sin(0.001F * r + 0.1F * t));
                    in_bits[static_cast<std::size_t>(t) * D + r] = h;
                    xref[static_cast<std::size_t>(t) * D + r]    = static_cast<double>(from_bf16(h));
                }
            }
            check(cudaMemcpy(d_in, in_bits.data(), in_bits.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess,
                  "input staged into the freed device region");

            const std::uint8_t* codes  = reinterpret_cast<const std::uint8_t*>(staged->data());
            const std::uint16_t* scales = reinterpret_cast<const std::uint16_t*>(staged->data() + g.scale_offset);

            // The image op, on the drained device (the LLM's streams are quiesced, so stream 0 is free).
            Tensor x(d_in, DType::BF16, {D, T, 1, 1});
            Tensor out(d_out, DType::BF16, {D, T, 1, 1});
            Weight w;
            w.payload          = d_w;
            w.payload_bytes    = len;
            w.qtype            = infernix::QType::FP8_E4M3FN_ROW_BF16;
            w.shape[0] = D; w.shape[1] = D; w.shape[2] = 1; w.shape[3] = 1;
            w.padded_shape[0] = D; w.padded_shape[1] = D; w.padded_shape[2] = 1; w.padded_shape[3] = 1;
            w.ndim             = 2;
            w.qdata            = d_w;
            w.qhigh            = nullptr;
            w.high_plane_bytes = 0;
            w.scales           = static_cast<const std::uint8_t*>(d_w) + g.scale_offset;
            w.n = D; w.k = D; w.group = D; w.group_size = static_cast<std::uint32_t>(D);
            w.layout            = infernix::QuantLayout::RowScale;
            w.scale_dtype       = DType::BF16;
            w.scale_ne[0] = D; w.scale_ne[1] = 1; w.scale_ne[2] = 1; w.scale_ne[3] = 1;
            const std::int64_t ss = static_cast<std::int64_t>(D) * 2;
            w.scale_nb[0] = 2; w.scale_nb[1] = ss; w.scale_nb[2] = ss; w.scale_nb[3] = ss;
            ops::linear(x, w, out, 0);
            check(cudaStreamSynchronize(0) == cudaSuccess, "image op completed on the drained device");

            std::vector<std::uint16_t> out_bits(static_cast<std::size_t>(D) * T);
            check(cudaMemcpy(out_bits.data(), d_out, out_bits.size() * 2, cudaMemcpyDeviceToHost) == cudaSuccess,
                  "device output read back");

            // CPU reference on the SAME bytes the kernel read: dequant each weight row (E4M3 + row scale)
            // and dot with every input column in double. A real H2D/wrong-weight bug lands here (rel_fro ~ 1).
            double num = 0.0, den = 0.0;
            std::vector<double> row(static_cast<std::size_t>(D));
            for (int i = 0; i < D; ++i) {
                const double scale = from_bf16(scales[i]);
                for (int j = 0; j < D; ++j) {
                    const std::uint8_t c  = codes[static_cast<std::size_t>(i) * D + j];
                    const float mag       = ops::canon::e4m3_value(c & 0x7FU);
                    row[static_cast<std::size_t>(j)] = (c & 0x80U ? -mag : mag) * scale;
                }
                for (int t = 0; t < T; ++t) {
                    double acc = 0.0;
                    const double* col = &xref[static_cast<std::size_t>(t) * D];
                    for (int j = 0; j < D; ++j) acc += row[static_cast<std::size_t>(j)] * col[j];
                    const double gv = from_bf16(out_bits[static_cast<std::size_t>(t) * D + i]);
                    const double diff = gv - acc;
                    num += diff * diff;
                    den += acc * acc;
                }
            }
            const double rel_fro = std::sqrt(num / (den + 1e-12));
            std::printf("  ops::linear [D=%d,T=%d] on freed VRAM: rel_fro=%.3e\n", D, T, rel_fro);
            check(den > 0.0, "the reference is non-degenerate");
            check(rel_fro < 0.01, "image op on the LLM's yielded VRAM matches the CPU dequant reference");

            cudaFree(d_w);
            cudaFree(d_in);
            cudaFree(d_out);
            (void)staged->release();

            infernix::models::qwen4_exp::testing::release_image_vram(lease);  // give the LLM its VRAM back
        } catch (...) {
            infernix::models::qwen4_exp::testing::release_image_vram(lease);  // always give it back
            infernix::models::qwen4_exp::testing::end_image_quiesce();        // recover the door on any failure
            throw;
        }
        infernix::models::qwen4_exp::testing::end_image_quiesce();
        check(infernix::models::qwen4_exp::testing::image_interlude_state() == 0, "the Program returned to Running");
        check(infernix::models::qwen4_exp::testing::image_interlude_submit_open(), "the LLM submit door reopened");
        const auto after = engine.generate(engine.prepare(story()), greedy(200)).generated_token_ids;
        check(after == baseline, "LLM generation is undisturbed after the in-process image op");

        infernix::testing::set_vram_budget_source({});
        result = failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(result == 0 ? "qwen4_exp image-interlude op checks passed\n" : "FAIL: %d checks failed\n", failures);
    return result;
}
