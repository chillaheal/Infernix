#pragma once

// The Qwen-Image-2.1 DiT transformer block: one shared forward path used by the image job's
// 40-step denoise loop. Kept out of the host-only infernix_qwen_image config lib because it calls
// the CUDA op layer. The block's math is the qualified production sequence; the image tests and the
// in-process image service call this same function rather than each carrying a copy.

#include "core/weight.h"

#include <cuda_runtime.h>

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
// `stream` is the stream the block's kernels enqueue on. It defaults to the legacy default stream
// (0) so the oracle tests, which qualify this forward in isolation, are unchanged; the image job
// passes its explicit image stream so the block forward is stream-controlled and the weight copies
// can overlap it (bundle B fas 2).
//
// `d_target_mask` and `d_image_ids` are the edit route's two per-token layout arrays (device U8 and
// I32, [Si,1,1,1], both from build_joint_layout). When they are null the block is exactly today's:
// one prefix boundary tp picks the modulation row and the attention mask. When set, the modulation
// row and the mask come from the arrays -- the reference image rows read the t = 0 row like the text
// does, and each image block is internally bidirectional. They are passed together or not at all;
// the image job builds both from one JointLayout, and the ops reject either one whose length is not
// Si. They come after `stream` so every existing caller is untouched.
void run_block(void* din, void* dout,
               const infernix::Weight& Wq, const infernix::Weight& Wk, const infernix::Weight& Wv,
               const infernix::Weight& Wout, const infernix::Weight& Wproj,
               const infernix::Weight& Wgate, const infernix::Weight& Wmlpout,
               void* d_nq, void* d_nk,
               void* dmod, void* drot, void* d_ones, void* d_zeros, DiTWorkspace& ws,
               int Si, int Di, int Hi, int Mi, int tp, float eps, cudaStream_t stream = 0,
               void* d_target_mask = nullptr, void* d_image_ids = nullptr);

} // namespace infernix::models::qwen_image
