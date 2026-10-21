#include "models/qwen_image/dit_block.h"

#include "core/tensor.h"

#include "infernix/ops/linear.h"
#include "infernix/ops/layer_norm.h"
#include "infernix/ops/rmsnorm.h"
#include "infernix/ops/silu_mul.h"
#include "infernix/ops/dit_adaln.h"
#include "infernix/ops/rope_c64.h"
#include "infernix/ops/attn_dit_joint.h"

namespace infernix::models::qwen_image {
namespace {

using infernix::Tensor;
using infernix::DType;
namespace ops = infernix::ops;

// Build a BF16 Tensor view over a device buffer with the given (up to 4) dimensions.
Tensor view(void* p, int a, int b, int c = 1, int dd = 1) { return Tensor(p, DType::BF16, {a, b, c, dd}); }

} // namespace

void run_block(void* din, void* dout,
               const infernix::Weight& Wq, const infernix::Weight& Wk, const infernix::Weight& Wv,
               const infernix::Weight& Wout, const infernix::Weight& Wproj,
               const infernix::Weight& Wgate, const infernix::Weight& Wmlpout,
               void* d_nq, void* d_nk,
               void* dmod, void* drot, void* d_ones, void* d_zeros, DiTWorkspace& ws,
               int Si, int Di, int Hi, int Mi, int tp, float eps) {
    Tensor nq = view(d_nq, 128, 1, 1), nk = view(d_nk, 128, 1, 1);
    Tensor onev = view(d_ones, Di, 1, 1), zerov = view(d_zeros, Di, 1, 1);
    Tensor modv = view(dmod, 4 * Di, 2);   // [4D,2] table, row selected by text_prefix (s<tp -> text row)
    Tensor rotv = view(drot, 128, 1, Si);  // [128,1,S] interleaved cis pairs

    Tensor x_in = view(din, Di, Si);
    Tensor ln1 = view(ws.dln1, Di, Si), mod1 = view(ws.dmod1, Di, Si);
    ops::layer_norm(x_in, onev, zerov, eps, ln1, 0);
    ops::adaln_scale(ln1, modv, 0, tp, mod1, 0);

    Tensor q = view(ws.dq, Di, Si), k = view(ws.dk, Di, Si), v = view(ws.dv, Di, Si);
    ops::linear(mod1, Wq, q, 0);
    ops::linear(mod1, Wk, k, 0);
    ops::linear(mod1, Wv, v, 0);

    Tensor qh = view(ws.dq, 128, Hi * Si), kh = view(ws.dk, 128, Hi * Si);
    Tensor qhr = view(ws.dqr, 128, Hi * Si), khr = view(ws.dkr, 128, Hi * Si);
    ops::rmsnorm(qh, nq, eps, /*unit_offset=*/false, qhr, 0);
    ops::rmsnorm(kh, nk, eps, /*unit_offset=*/false, khr, 0);
    Tensor q3 = view(ws.dqr, 128, Hi, Si), k3 = view(ws.dkr, 128, Hi, Si), v3 = view(ws.dv, 128, Hi, Si);
    ops::rope_c64(rotv, q3, 0);
    ops::rope_c64(rotv, k3, 0);

    Tensor attn3 = view(ws.dattn, 128, Hi, Si);
    ops::attn_dit_joint(q3, k3, v3, tp, attn3, 0);
    Tensor attn2 = view(ws.dattn, Di, Si), attn_out = view(ws.dattn_out, Di, Si);
    ops::linear(attn2, Wout, attn_out, 0);

    Tensor x1 = view(ws.dx1, Di, Si);
    ops::adaln_gate_add(x_in, attn_out, modv, Di, tp, x1, 0);
    Tensor ln2 = view(ws.dln2, Di, Si), mod2 = view(ws.dmod2, Di, Si);
    ops::layer_norm(x1, onev, zerov, eps, ln2, 0);
    ops::adaln_scale(ln2, modv, 2 * Di, tp, mod2, 0);

    Tensor proj = view(ws.dproj, Mi, Si), gate = view(ws.dgate, Mi, Si), hidden = view(ws.dhidden, Mi, Si);
    ops::linear(mod2, Wproj, proj, 0);
    ops::linear(mod2, Wgate, gate, 0);
    ops::silu_mul(gate, proj, hidden, 0);
    Tensor mlp_out = view(ws.dmlp_out, Di, Si);
    ops::linear(hidden, Wmlpout, mlp_out, 0);
    Tensor out = view(dout, Di, Si);
    ops::adaln_gate_add(x1, mlp_out, modv, 3 * Di, tp, out, 0);
}

} // namespace infernix::models::qwen_image
