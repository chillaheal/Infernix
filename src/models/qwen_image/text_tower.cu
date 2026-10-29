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

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <memory>
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

// x[rows[r], d] += add[r, d] -- the deepstack add touches ONLY the image rows, which
// ops::residual_add (same-shaped, in place) cannot express. The BF16 add is the residual_add
// kernel's idiom: widen both terms, add in FP32, round back once.
__global__ void deepstack_add_rows_kernel(const std::uint16_t* __restrict__ add, std::uint16_t* __restrict__ x,
                                          const int* __restrict__ rows, int D, int n_rows) {
    const auto* a  = reinterpret_cast<const __nv_bfloat16*>(add);
    auto* xd       = reinterpret_cast<__nv_bfloat16*>(x);
    const int total = D * n_rows;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const int d = i % D;
        const int r = i / D;
        const std::size_t dst = static_cast<std::size_t>(rows[r]) * static_cast<std::size_t>(D) + d;
        xd[dst] = __float2bfloat16_rn(__bfloat162float(xd[dst]) + __bfloat162float(a[i]));
    }
}

const TextTowerDims& checked_dims(const TextTowerWeights& tower) {
    const TextTowerDims& dims = tower.dims;
    if (dims.hidden == 0 || dims.intermediate == 0 || dims.heads == 0 || dims.kv_heads == 0 ||
        dims.head_dim == 0 || dims.heads % dims.kv_heads != 0 || dims.rope_theta <= 0.0F ||
        tower.layers.empty() || tower.layers.size() != dims.layers) {
        throw std::runtime_error("text tower: dims are unusable or layers differ from the config");
    }
    return dims;
}

// Host gather: templated ids -> embedding rows, token-major flat ([S, D] == dim0-fastest [D, S]).
std::vector<std::uint16_t> gather_inputs(const TextTowerWeights& tower, const std::vector<std::byte>& embed_table,
                                         const std::vector<int>& token_ids) {
    const TextTowerDims& dims  = checked_dims(tower);
    const std::size_t S        = token_ids.size();
    const std::size_t D        = dims.hidden;
    const std::size_t vocab    = dims.vocab;
    if (S == 0 || embed_table.size() != 2ULL * vocab * D) {
        throw std::runtime_error("text tower: ids or embed table do not match the model dims");
    }
    std::vector<std::uint16_t> xbits(S * D);
    const auto* table = reinterpret_cast<const std::uint16_t*>(embed_table.data());
    for (std::size_t s = 0; s < S; ++s) {
        const int id = token_ids[s];
        if (id < 0 || static_cast<std::size_t>(id) >= vocab) {
            throw std::runtime_error("text tower: token id is outside the embedding table");
        }
        std::memcpy(xbits.data() + s * D, table + static_cast<std::size_t>(id) * D, sizeof(std::uint16_t) * D);
    }
    return xbits;
}

// The shared streamed 36-layer body, so both entries run the identical forward (the text-only entry
// is the certified regression, the conditioned entry only changes the inputs and adds the deepstack
// rows). `xbits` is the gathered -- and for the conditioned entry already scattered -- inputs_embeds.
// When `pad_rows` and `deepstack` are given, deepstack[i] is added to those rows after decoder layer
// i: the reference adds the deepstack embeds by LIST POSITION, not by the vision tap index
// (modeling_qwen3_vl.py:847-852). Returns the full [S*D] pre-final-RMSNorm hidden bits.
std::vector<std::uint16_t> tower_forward(artifact::Reader& reader, const TextTowerWeights& tower,
                                         const std::vector<std::uint16_t>& xbits, const std::vector<int>* pad_rows,
                                         const std::vector<std::vector<std::uint16_t>>* deepstack) {
    ImageStreamScope legacy_scope(nullptr); // the tower's consumers run on the default stream
    const TextTowerDims& dims = checked_dims(tower);
    const int D     = static_cast<int>(dims.hidden);
    const int I     = static_cast<int>(dims.intermediate);
    const int HQ    = static_cast<int>(dims.heads);
    const int HKV   = static_cast<int>(dims.kv_heads);
    const int HD    = static_cast<int>(dims.head_dim);
    const int KVW   = HKV * HD;
    const int GROUP = HQ / HKV;
    const float eps   = dims.eps;
    const float theta = dims.rope_theta;

    const std::size_t sD = xbits.size();
    if (sD == 0 || sD % static_cast<std::size_t>(D) != 0) {
        throw std::runtime_error("text tower: inputs_embeds do not match the model dims");
    }
    const std::size_t S   = sD / static_cast<std::size_t>(D);
    const std::size_t sKV = S * static_cast<std::size_t>(KVW);
    const std::size_t sI  = S * static_cast<std::size_t>(I);

    // ---- all tower device memory lives in Bufs; RAII frees it before every return ----
    Buf x, dpos, drows;
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

    // The deepstack rows are prompt-sized and tiny next to the activations; they upload ONCE and are
    // reused by all three add layers. Unique_ptrs because Buf owns a raw pointer: a vector of Bufs
    // would hand that pointer to two owners when it grows.
    std::vector<std::unique_ptr<Buf>> deepstack_device;
    if (pad_rows != nullptr) {
        drows.alloc(pad_rows->size() * sizeof(int));
        h2d(drows.d, pad_rows->data(), pad_rows->size() * sizeof(int));
    }
    if (deepstack != nullptr) {
        deepstack_device.reserve(deepstack->size());
        for (const auto& rows_bits : *deepstack) {
            auto buf = std::make_unique<Buf>();
            buf->alloc(rows_bits.size() * sizeof(std::uint16_t));
            h2d(buf->d, rows_bits.data(), rows_bits.size() * sizeof(std::uint16_t));
            deepstack_device.push_back(std::move(buf));
        }
    }

    Buf dh, dq, dk, dv, dqn, dkn, dk32, dv32, dattn, dop, dh2, dgate, dup, dhidden, ddown;
    dh.alloc(sD * 2);  dq.alloc(sD * 2);   dk.alloc(sKV * 2);  dv.alloc(sKV * 2);
    dqn.alloc(sD * 2); dkn.alloc(sKV * 2); dk32.alloc(sD * 2); dv32.alloc(sD * 2);
    dattn.alloc(sD * 2); dop.alloc(sD * 2); dh2.alloc(sD * 2);
    dgate.alloc(sI * 2); dup.alloc(sI * 2); dhidden.alloc(sI * 2); ddown.alloc(sD * 2);

    const int Si = static_cast<int>(S);
    const int grid = 128, block = 256;
    for (std::size_t layer_index = 0; layer_index < tower.layers.size(); ++layer_index) {
        const auto& layer = tower.layers[layer_index];
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
        if (pad_rows != nullptr && layer_index < deepstack_device.size()) {
            deepstack_add_rows_kernel<<<grid, block, 0, 0>>>(static_cast<const std::uint16_t*>(deepstack_device[layer_index]->d),
                                                             static_cast<std::uint16_t*>(x.d),
                                                             static_cast<const int*>(drows.d), D,
                                                             static_cast<int>(pad_rows->size()));
        }
        cuda_ok(cudaGetLastError(), "text tower layer launch");
    }

    // The rolling residual now holds the pre-final-RMSNorm hidden [D, S] (dim0 fastest == token-major
    // flat). Synchronous D2H orders against the device work.
    std::vector<std::uint16_t> hidden_bits(sD);
    cuda_ok(cudaMemcpy(hidden_bits.data(), x.d, sD * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
            "D2H (text tower hidden)");
    return hidden_bits;
}

} // namespace

std::vector<std::uint16_t> run_text_tower(artifact::Reader& reader, const TextTowerWeights& tower,
                                          const std::vector<std::byte>& embed_table,
                                          const std::vector<int>& token_ids, std::size_t drop_rows) {
    const TextTowerDims& dims = checked_dims(tower);
    if (token_ids.empty() || drop_rows >= token_ids.size()) {
        throw std::runtime_error("text tower: ids or embed table do not match the model dims");
    }
    std::vector<std::uint16_t> xbits = gather_inputs(tower, embed_table, token_ids);
    std::vector<std::uint16_t> hidden = tower_forward(reader, tower, xbits, nullptr, nullptr);
    // Rows [drop_rows, S) are the kept prompt conditioning (single all-valid entry:
    // build_prompt_embeds would return exactly these rows).
    return std::vector<std::uint16_t>(hidden.begin() + static_cast<std::ptrdiff_t>(drop_rows) * dims.hidden,
                                      hidden.end());
}

ConditionedTextTowerOutput
run_conditioned_text_tower(artifact::Reader& reader, const TextTowerWeights& tower,
                           const std::vector<std::byte>& embed_table, const std::vector<int>& token_ids,
                           std::size_t drop_rows, const ConditionedVision& vision) {
    const TextTowerDims& dims = checked_dims(tower);
    const std::size_t D = dims.hidden;
    const std::size_t S = token_ids.size();
    if (S == 0 || drop_rows >= S) {
        throw std::runtime_error("text tower: ids or embed table do not match the model dims");
    }
    std::vector<int> pad_rows;
    for (std::size_t s = 0; s < S; ++s) {
        if (token_ids[s] >= 0 && static_cast<std::uint32_t>(token_ids[s]) == kImagePadTokenId) {
            pad_rows.push_back(static_cast<int>(s));
        }
    }
    // Both sides carry BF16 BITS -- one uint16 per element, see VisionTowerOutput -- so the sizes to
    // compare are ELEMENT counts (rows * hidden), not byte counts: a byte-sized expectation here
    // rejects every correct vision output at exactly half the length it names.
    const std::size_t rows_elements = pad_rows.size() * D;
    if (pad_rows.empty() || vision.merger_rows.size() != rows_elements ||
        vision.deepstack_rows.size() > tower.layers.size()) {
        throw std::runtime_error("text tower: vision rows do not cover the image pad tokens");
    }
    for (const auto& add : vision.deepstack_rows) {
        if (add.size() != rows_elements) {
            throw std::runtime_error("text tower: vision rows do not cover the image pad tokens");
        }
    }

    std::vector<std::uint16_t> xbits = gather_inputs(tower, embed_table, token_ids);
    for (std::size_t r = 0; r < pad_rows.size(); ++r) {
        std::memcpy(xbits.data() + static_cast<std::size_t>(pad_rows[r]) * D,
                    vision.merger_rows.data() + r * D, sizeof(std::uint16_t) * D);
    }
    std::vector<std::uint16_t> hidden = tower_forward(reader, tower, xbits, &pad_rows, &vision.deepstack_rows);

    ConditionedTextTowerOutput out;
    out.prompt_embeds.assign(hidden.begin() + static_cast<std::ptrdiff_t>(drop_rows) * D, hidden.end());
    out.image_pad_mask.assign(S - drop_rows, 0U);
    for (std::size_t s = drop_rows; s < S; ++s) {
        out.image_pad_mask[s - drop_rows] =
            (token_ids[s] >= 0 && static_cast<std::uint32_t>(token_ids[s]) == kImagePadTokenId) ? 1U : 0U;
    }
    return out;
}

} // namespace infernix::models::qwen_image
