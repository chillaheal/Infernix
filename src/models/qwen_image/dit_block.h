#pragma once

// The Qwen-Image-2.1 DiT transformer block: one shared forward path used by the image job's
// 40-step denoise loop. Kept out of the host-only infernix_qwen_image config lib because it calls
// the CUDA op layer. The block's math is the qualified production sequence; the image tests and the
// in-process image service call this same function rather than each carrying a copy.

#include "core/weight.h"

namespace infernix::models::qwen_image {

// Per-block DiT activation scratch: one contiguous set of BF16 buffers, allocated by the caller
// once and reused for every block and every denoise step. Sized by (Di, Si, Hi, Mi).
struct DiTWorkspace {
    void *dln1, *dmod1, *dq, *dk, *dv, *dqr, *dkr, *dattn, *dattn_out, *dx1, *dln2, *dmod2,
         *dproj, *dgate, *dhidden, *dmlp_out;
};

// One DiT block forward through the production ops (BF16 linears, BF16 everything else). din/dout
// are the running joint hidden (viewed [Di, Si] = Si tokens x Di features). d_nq/d_nk are the head
// RMSNorm weights [128,1,1]; dmod is the [4*Di,2] adaLN table (text row selected where s<tp); drot
// is the interleaved cis [128,1,Si] rope table; d_ones/d_zeros back the unit/zero layer_norm views.
void run_block(void* din, void* dout,
               const infernix::Weight& Wq, const infernix::Weight& Wk, const infernix::Weight& Wv,
               const infernix::Weight& Wout, const infernix::Weight& Wproj,
               const infernix::Weight& Wgate, const infernix::Weight& Wmlpout,
               void* d_nq, void* d_nk,
               void* dmod, void* drot, void* d_ones, void* d_zeros, DiTWorkspace& ws,
               int Si, int Di, int Hi, int Mi, int tp, float eps);

} // namespace infernix::models::qwen_image
