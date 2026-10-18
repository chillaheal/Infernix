infernix_add_op_test(infernix_image_load_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_load.cpp"
  LIBRARIES infernix_artifact)

# CPU-only host staging bound (coexistence task #13, bounded streaming loader): streams the entire real
# artifact object-by-object through one grow-only reused buffer via Reader::read_into, so peak host
# staging == the largest single weight object, not the sum; re-derives FP8 RowScale planes on the
# reused buffer for every FP8 tensor. Skips (77) without INFERNIX_TEST_ARTIFACT / when absent.
infernix_add_op_test(infernix_image_stream_host_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_stream_host.cpp"
  LIBRARIES infernix_artifact)
set_tests_properties(infernix_image_stream_host_test PROPERTIES SKIP_RETURN_CODE 77)

# CPU-only coexistence-feasibility bound (task #13 step-4 admission): the image lane's per-step device
# peak (streamed block weights + BF16 activation workspace + shared) as a function of token count,
# gated to fit inside ONE increment-2 yield (8192 MiB) at every supported resolution; asserts streaming
# (one block resident) is what makes a single yield sufficient. Pure published-dimension arithmetic;
# needs no oracle or artifact, always runs.
infernix_add_op_test(infernix_image_budget_host_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_budget_host.cpp")

# CPU numeric oracle: naive float32 Qwen-Image-2.1 DiT block re-derived against the upstream
# diffusers block-0 reference. Needs the oracle .npy dump; skips (77) when it is absent.
infernix_add_op_test(infernix_image_block_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_block.cpp")

# CPU numeric oracle: naive float32 full DiT forward (top level + block) against sample.npy.
infernix_add_op_test(infernix_image_forward_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_forward.cpp")

# CPU numeric oracle: flow-match scheduler schedule + Euler step against the upstream dump.
infernix_add_op_test(infernix_image_scheduler_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_scheduler.cpp")

# CPU numeric oracle: naive float32 single-frame VAE DECODER (AutoencoderKLQwenImage21) re-derived against
# the authoritative diffusers float32 forward. Single frame (T=1, first_chunk, no feat_cache) reduces to
# pure 2D: post_quant_conv(1x1) -> conv_in(3x3) -> mid(resnet->spatial-attn->resnet) -> 5 up_blocks
# (3 resnets + nearest-2x upsample + 3x3 + DupUp3D avg_shortcut) -> RMS_norm -> silu -> conv_out -> clamp.
# A-priori gate rel_fro < 1e-3 vs tensors_vae/vae_decode_frame0.npy; never narrowed to pass.
# Needs the VAE oracle dump (tensors_vae/ + tensors_vae_weights/); skips (77) when it is absent.
infernix_add_op_test(infernix_image_vae_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_vae.cpp")

# First GPU oracle: img_in dense projection on device vs the CPU oracle img_in.npy. Self-contained CUDA;
# skips (77) without INFERNIX_IMAGE_ORACLE and returns 2 (runnable, not counted as compile failure) with no device.
infernix_add_op_test(infernix_image_gemm_cuda_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_gemm_cuda.cu")

# GPU numeric oracle: full DiT block on device (naive CUDA, double accumulators) per-stage vs the shared
# CPU oracle primitives + end-to-end vs block_out.npy. Skips (77) without INFERNIX_IMAGE_ORACLE.
infernix_add_op_test(infernix_image_block_cuda_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_block_cuda.cu")

# GPU production-route qualification: the same DiT block composed entirely from repo ops
# (ops::linear, layer_norm, adaln_scale, rmsnorm, rope_c64, attn_dit_joint, adaln_gate_add, silu_mul)
# per-stage vs qioracle::block_forward on BF16-represented operands. Links infernix_ops. Skips (77)
# without INFERNIX_IMAGE_ORACLE; returns 2 (runnable) with no CUDA device.
infernix_add_op_test(infernix_image_block_prod_cuda_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_block_prod_cuda.cu"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_image_block_prod_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU production-route qualification, FP8 twin: the same DiT block but the seven linear projections are
# row-scaled FP8 E4M3 (W8A16) instead of BF16. Non-GEMM ops stay BF16. Oracle = qioracle::block_forward on
# the true BF16 weights, so the gap is pure FP8 quantization + kernel rounding. Fixed a-priori tol 4e-2.
# Links infernix_ops. Skips (77) without INFERNIX_IMAGE_ORACLE; returns 2 (runnable) with no CUDA device.
infernix_add_op_test(infernix_image_block_fp8_prod_cuda_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_block_fp8_prod_cuda.cu"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_image_block_fp8_prod_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU production-route qualification, FULL-DEPTH FP8: the complete 32-block DiT chain, each block's
# seven linear projections row-scaled FP8 E4M3 (W8A16), streamed block-by-block (load real weights ->
# quantize -> run -> free) so host RAM stays ~one block. The depth composition gate: a per-block LOCAL
# error (FP8 on the FP32 reference input at every depth) and the accumulated FP8 trajectory vs the
# FP32 trajectory. Oracle = the streamed authoritative diffusers full-depth dump
# (tensors_full_depth/). Fixed a-priori tol 4e-2 applied to the worst LOCAL (every block) and the
# deepest ACCUMULATED error; never narrowed to pass. Links infernix_ops. Skips (77) without the oracle.
infernix_add_op_test(infernix_image_depth_fp8_prod_cuda_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_depth_fp8_prod_cuda.cu"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_image_depth_fp8_prod_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU production-route qualification, DiT TOP LEVEL: the parts outside the 32 transformer blocks,
# composed from production ops (ops::linear general BF16 A16 route, rmsnorm unit_offset=ZeroCenter,
# gelu(Tanh), silu_mul, layer_norm affine=False, adaln_scale). The transformer blocks are NOT re-run:
# the norm_out/proj_out tail is fed the oracle block-0 output (block_out.npy) and the blocks are
# independently qualified (block + full-depth tests). Per-stage vs the BF16-represented qo:: host oracle
# (gated) and the diffusers FP32 .npy (reported). Links infernix_ops. Skips (77) without
# INFERNIX_IMAGE_ORACLE; returns 2 (runnable) with no CUDA device.
infernix_add_op_test(infernix_image_toplevel_prod_cuda_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_dit_toplevel_prod_cuda.cu"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_image_toplevel_prod_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU production-route qualification, Qwen3-VL TEXT TOWER: the full 36-layer language model
# (pre-final-RMSNorm output) composed entirely from repo ops (ops::rmsnorm unit_offset=FALSE,
# ops::linear general BF16 A16 route, per-head ops::rmsnorm, ops::rope 1-D text GQA 32/8 theta=5e6,
# test-local K/V 8->32 replication for GQA, ops::attn_dit_joint text_prefix=S (fully causal),
# ops::linear + ops::residual_add, ops::silu_mul). A host FP32 36-layer reference (bf16 weights,
# f32 activation math) localizes the divergent layer; the production gate is the model's own bf16
# pre_norm_hidden.npy (tensors_text_tower/). Fixed a-priori tol: gate device-vs-pre_norm < 2e-2,
# device-vs-host < 1e-1 (clears the bf16-ACTIVATION floor ~5.9e-2: device runs bf16 activations like
# the model, host runs f32; a real kernel defect exceeds that floor by an order of magnitude,
# cf. 1.56e+00 before the GQA K/V layout fix); never narrowed to pass. Links infernix_ops. Skips (77) without the
# weight dump (tensors_text_tower_weights/); returns 2 (runnable) with no CUDA device.
infernix_add_op_test(infernix_image_texttower_prod_cuda_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_texttower_prod_cuda.cu"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_image_texttower_prod_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU production-route qualification, VAE DECODER: the full Qwen-Image-2.1 single-frame VAE decoder
# (AutoencoderKLQwenImage21, T=1 first-chunk) composed from production ops (ops::conv2d, ops::silu,
# ops::add, ops::upsample_nearest_2x, ops::dupup3d, ops::spatial_attn, plus reused ops::linear +
# ops::add_bias for every 1x1 conv and ops::rmsnorm per-position), BF16 activations and weights.
# Per-stage vs a host FP32 (double-accumulation) re-derivation of the same graph on the same
# BF16-represented operands (localizes a divergent op in one offline run), and the final decoded image
# vs tensors_vae/vae_decode_frame0.npy (diffusers FP32). A-priori tol 5e-2 for every per-stage row and the
# final (a ~125-op deep BF16 chain, one notch above the 4e-2 deep-DiT precedent; a real op defect is
# O(0.1..1)); never narrowed to pass. Links infernix_ops. Skips (77) without INFERNIX_IMAGE_ORACLE;
# returns 2 (runnable) with no CUDA device.
infernix_add_op_test(infernix_image_vae_prod_cuda_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_vae_prod_cuda.cu"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_image_vae_prod_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU end-to-end ASSEMBLY of the Qwen-Image-2.1 image pipeline: the qualified text-tower output
# (prompt_embeds) + the qualified DiT (top-level ops + 32 production blocks) + the qualified VAE
# decoder, composed with the FlowMatchEulerDiscrete 40-step sampler, all BF16, producing the FIRST
# image on device and checking it against a fresh full-pipeline diffusers reference (tensors_pipeline/,
# dumped by image-spike/oracle/dump_pipeline.py running QwenImage21Pipeline.__call__ line-for-line).
# Every op is an already-qualified production op (top-level #24/#29, 32 blocks #29, attention #22,
# RoPE #21, AdaLN #23, VAE #31); this test is the ASSEMBLY gate. 256x256 -> latent 16x16 -> 256 image
# tokens + 11 text = 267 joint tokens. PRIMARY gate: final latent vs latents_final.npy rel_fro < 0.5
# (independent of the VAE). SECONDARY gate: final image vs image.npy rel_fro < 0.3. Both a-priori,
# never narrowed to pass. Links infernix_ops. Skips (77) without INFERNIX_IMAGE_ORACLE; returns 2
# (runnable) with no CUDA device.
infernix_add_op_test(infernix_image_pipeline_prod_cuda_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_pipeline_prod_cuda.cu"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_image_pipeline_prod_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

# CPU-oracle qualification of the BF16 linear's small-token (skinny GEMV) path (tokens <= 8), the
# route the Qwen-Image time-embedding linears use. Independent of any model/oracle data.
infernix_add_op_test(infernix_image_skinny_linear_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_skinny_linear.cu"
  LIBRARIES infernix_ops)
