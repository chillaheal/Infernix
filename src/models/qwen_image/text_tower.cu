#include "models/qwen_image/text_tower.h"

#include "artifact/reader.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "infernix/ops/attn_dit_joint.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/residual_add.h"
#include "infernix/ops/rmsnorm.h"
#include "infernix/ops/rope.h"
#include "infernix/ops/silu_mul.h"
#include "models/qwen_image/image_device_util.h"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <vector>

namespace infernix::models::qwen_image {

namespace {

using infernix::DType;
using infernix::Tensor;
namespace artifact = infernix::artifact;
namespace ops      = infernix::ops;

// GQA -> MHA shape adapter (the qualified test's kernel): replicate a BF16 [HD, HKV, S] (dim0
// fastest) into [HD, HQ, S] where query-head h reads kv-head h / GROUP. Pure layout copy.
__global__ void replicate_kv_kernel(const std::uint16_t* __restrict__ src, std::uint16_t* __restrict__ dst,
                                    int S, int HD, int HQ, int HKV, int GROUP) {
    const int total = HD * HQ * S;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const int d = i % HD;
        const int t = i / HD;          // dst [HD, HQ, S] dim0 fastest: t = h + HQ * s
        const int h = t % HQ;
        const int s = t / HQ;
        const int j = h / GROUP;
        dst[i] = src[d + HD * j + HD * HKV * s];
    }
}

} // namespace

std::vector<std::uint16_t> run_text_tower(artifact::Reader& reader, const TextTowerWeights& tower,
                                          const std::vector<std::byte>& embed_table,
                                          const std::vector<int>& token_ids, std::size_t drop_rows) {
    ImageStreamScope legacy_scope(nullptr); // the tower's consumers run on the default stream
    const TextTowerDims& dims = tower.dims;
    const int D     = static_cast<int>(dims.hidden);
    const int I     = static_cast<int>(dims.intermediate);
    const int HQ    = static_cast<int>(dims.heads);
    const int HKV   = static_cast<int>(dims.kv_heads);
    const int HD    = static_cast<int>(dims.head_dim);
    const int KVW   = HKV * HD;
    const int GROUP = HQ / HKV;
    const float eps   = dims.eps;
    const float theta = dims.rope_theta;
    if (D == 0 || I == 0 || HQ == 0 || HKV == 0 || HD == 0 || HQ % HKV != 0 || theta <= 0.0F ||
        tower.layers.empty() || tower.layers.size() != dims.layers) {
        throw std::runtime_error("text tower: dims are unusable or layers differ from the config");
    }
    const std::size_t S    = token_ids.size();
    const std::size_t vocab = dims.vocab;
    if (S == 0 || drop_rows >= S || embed_table.size() != 2ULL * vocab * D) {
        throw std::runtime_error("text tower: ids or embed table do not match the model dims");
    }

    // Host gather: templated ids -> embedding rows, token-major flat ([S, D] == dim0-fastest [D, S]).
    const std::size_t sD  = S * D;
    const std::size_t sKV = S * static_cast<std::size_t>(KVW);
    const std::size_t sI  = S * static_cast<std::size_t>(I);
    std::vector<std::uint16_t> xbits(sD);
    {
        const auto* table = reinterpret_cast<const std::uint16_t*>(embed_table.data());
        for (std::size_t s = 0; s < S; ++s) {
            const int id = token_ids[s];
            if (id < 0 || static_cast<std::size_t>(id) >= vocab) {
                throw std::runtime_error("text tower: token id is outside the embedding table");
            }
            std::memcpy(xbits.data() + s * D, table + static_cast<std::size_t>(id) * D, sizeof(std::uint16_t) * D);
        }
    }

    // ---- all tower device memory lives in Bufs; RAII frees it before every return ----
    Buf x, dpos;
    x.alloc(sD * 2);
    h2d(x.d, xbits.data(), sD * 2);
    {
        std::vector<int> pos(S);
        for (std::size_t s = 0; s < S; ++s) pos[s] = static_cast<int>(s);
        dpos.alloc(S * sizeof(int));
        h2d(dpos.d, pos.data(), S * sizeof(int));
    }
    const Tensor posv(dpos.d, DType::I32, {static_cast<int>(S)});
    const ops::PreparedRope prepared = ops::prepare_rope(HD, theta);
    DeviceContext device;
    const auto execution = device.execution_view().on_stream(nullptr);

    Buf dh, dq, dk, dv, dqn, dkn, dk32, dv32, dattn, dop, dh2, dgate, dup, dhidden, ddown;
    dh.alloc(sD * 2);  dq.alloc(sD * 2);   dk.alloc(sKV * 2);  dv.alloc(sKV * 2);
    dqn.alloc(sD * 2); dkn.alloc(sKV * 2); dk32.alloc(sD * 2); dv32.alloc(sD * 2);
    dattn.alloc(sD * 2); dop.alloc(sD * 2); dh2.alloc(sD * 2);
    dgate.alloc(sI * 2); dup.alloc(sI * 2); dhidden.alloc(sI * 2); ddown.alloc(sD * 2);

    const int Si = static_cast<int>(S);
    const int grid = 128, block = 256;
    for (const auto& layer : tower.layers) {
        // One layer's roles stream through: uploaded per layer, freed at the end of the iteration.
        Uploaded wq = upload_ref(reader, layer.q);
        Uploaded wk = upload_ref(reader, layer.k);
        Uploaded wv = upload_ref(reader, layer.v);
        Uploaded wo = upload_ref(reader, layer.attention_output);
        Uploaded wqnorm = upload_ref(reader, layer.query_norm);
        Uploaded wknorm = upload_ref(reader, layer.key_norm);
        Uploaded wgate = upload_ref(reader, layer.mlp_gate);
        Uploaded wup   = upload_ref(reader, layer.mlp_up);
        Uploaded wdown = upload_ref(reader, layer.mlp_down);
        Uploaded wiln = upload_ref(reader, layer.input_norm);
        Uploaded wpln = upload_ref(reader, layer.post_attention_norm);

        Tensor xin = view(x.d, D, Si);
        Tensor iln = view(wiln.buf.d, D, 1, 1), pln = view(wpln.buf.d, D, 1, 1);
        Tensor qnw = view(wqnorm.buf.d, HD, 1, 1), knw = view(wknorm.buf.d, HD, 1, 1);
        Tensor h = view(dh.d, D, Si);
        ops::rmsnorm(xin, iln, eps, /*unit_offset=*/false, h, 0);
        Tensor q = view(dq.d, D, Si), k = view(dk.d, KVW, Si), v = view(dv.d, KVW, Si);
        ops::linear(h, wq.w, q, 0);
        ops::linear(h, wk.w, k, 0);
        ops::linear(h, wv.w, v, 0);
        Tensor qh = view(dq.d, HD, HQ * Si), qhn = view(dqn.d, HD, HQ * Si);
        Tensor kh = view(dk.d, HD, HKV * Si), khn = view(dkn.d, HD, HKV * Si);
        ops::rmsnorm(qh, qnw, eps, /*unit_offset=*/false, qhn, 0);
        ops::rmsnorm(kh, knw, eps, /*unit_offset=*/false, khn, 0);
        Tensor q3 = view(dqn.d, HD, HQ, Si), k3 = view(dkn.d, HD, HKV, Si);
        ops::rope(posv, prepared, q3, k3, execution);
        replicate_kv_kernel<<<grid, block, 0, 0>>>(static_cast<const std::uint16_t*>(dkn.d),
                                                   static_cast<std::uint16_t*>(dk32.d), Si, HD, HQ, HKV, GROUP);
        replicate_kv_kernel<<<grid, block, 0, 0>>>(static_cast<const std::uint16_t*>(dv.d),
                                                   static_cast<std::uint16_t*>(dv32.d), Si, HD, HQ, HKV, GROUP);
        Tensor q3r = view(dqn.d, HD, HQ, Si), k32 = view(dk32.d, HD, HQ, Si), v32 = view(dv32.d, HD, HQ, Si);
        Tensor attn3 = view(dattn.d, HD, HQ, Si);
        ops::attn_dit_joint(q3r, k32, v32, Si, attn3, 0);
        Tensor attn2 = view(dattn.d, D, Si), o = view(dop.d, D, Si);
        ops::linear(attn2, wo.w, o, 0);
        ops::residual_add(o, xin, 0);
        Tensor h2 = view(dh2.d, D, Si);
        ops::rmsnorm(xin, pln, eps, /*unit_offset=*/false, h2, 0);
        Tensor gate = view(dgate.d, I, Si), up = view(dup.d, I, Si), hidden = view(dhidden.d, I, Si);
        ops::linear(h2, wgate.w, gate, 0);
        ops::linear(h2, wup.w, up, 0);
        ops::silu_mul(gate, up, hidden, 0);
        Tensor down = view(ddown.d, D, Si);
        ops::linear(hidden, wdown.w, down, 0);
        ops::residual_add(down, xin, 0);
        cuda_ok(cudaGetLastError(), "text tower layer launch");
    }

    // The rolling residual now holds the pre-final-RMSNorm hidden [D, S] (dim0 fastest == token-major
    // flat). Synchronous D2H orders against the device work; rows [drop_rows, S) are the kept prompt
    // conditioning (single all-valid entry: build_prompt_embeds would return exactly these rows).
    std::vector<std::uint16_t> hidden_bits(sD);
    cuda_ok(cudaMemcpy(hidden_bits.data(), x.d, sD * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
            "D2H (text tower hidden)");
    return std::vector<std::uint16_t>(hidden_bits.begin() + static_cast<std::ptrdiff_t>(drop_rows) * D,
                                      hidden_bits.end());
}

} // namespace infernix::models::qwen_image
