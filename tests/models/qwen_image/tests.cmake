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

# C4 staging primitive (coexistence task #13): stream the entire real artifact object-by-object through
# the HostContextArena -- a fixed-capacity, NON-growing pinned pool sized to the largest single object --
# so peak host staging == the largest object and exhaustion is refused (allocate -> nullopt), not grown.
# Re-derives FP8 RowScale planes on the arena's pinned span for every FP8 tensor. Host + CUDA-context only
# (cudaMallocHost): no device VRAM, so it runs beside the live server (no blackout). Skips (77) without
# INFERNIX_TEST_ARTIFACT / when absent / with no usable CUDA device.
infernix_add_op_test(infernix_image_stream_arena_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_stream_arena.cpp"
  LIBRARIES infernix_artifact)
set_tests_properties(infernix_image_stream_arena_test PROPERTIES SKIP_RETURN_CODE 77)

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

# Production scheduler qualification: the infernix_qwen_image FlowMatchEulerDiscrete schedule
# (calculate_shift + build_schedule) and the geometry's token count against the real 512x512 40-step
# pipeline oracle (tensors_pipeline/ sigmas.npy + timesteps.npy, dumped by image-spike/oracle/
# dump_pipeline.py). Host-only, weights-free. A-priori tol 2e-5 on the f32 schedule, never narrowed.
# Skips (77) without INFERNIX_IMAGE_ORACLE_PIPELINE.
infernix_add_op_test(infernix_image_scheduler_prod_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_scheduler_prod.cpp"
  LIBRARIES infernix_qwen_image)

# Resolution-grid invariants of the production schedule (infernix_qwen_image): mu monotone in the
# latent token count, sigmas[0]=1 / sigmas[steps]=0 / sigmas[steps-1]=shift_terminal, strictly
# decreasing within [0,1], timesteps==sigmas*num_train_timesteps. Host-only, weights-free, no oracle,
# always runs -- guards the variable-resolution gap the single 512 pipeline oracle cannot cover.
infernix_add_op_test(infernix_image_schedule_grid_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_schedule_grid.cpp"
  LIBRARIES infernix_qwen_image)

# Seeded standard-normal init-latent noise (infernix_qwen_image::fill_normal): the image job's "seed"
# knob. Host-only, no oracle file -- the independent reference is the N(0,1) distribution, so the
# numeric gate is the sample moments (mean~0, std~1) over a large double-accumulated draw + the seed
# contract (same seed bit-identical, different seed differs) + the odd-count tail. A-priori SE-based
# tolerances, never narrowed to pass.
infernix_add_op_test(infernix_image_noise_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_noise.cpp"
  LIBRARIES infernix_qwen_image)

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
  LIBRARIES infernix_ops infernix_qwen_image)
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

# GPU qualification of the ops::spatial_attn ONLINE-SOFTMAX kernel vs a naive FP64 oracle (no model
# data): production C=1152 beyond the old O(N) shared-memory limit (N=16640) and on the full
# 2048x2048 latent grid (N=65536, sampled-row FP64 oracle + plane-wide finiteness), exhaustive small
# tile-tail/strided-C cases, the exact-48-KiB boundary C=6016, and the new cap guards (N<=65536,
# C<=6016). A-priori tol 2e-2 on every bound, never narrowed. Links infernix_ops. Returns 2
# (runnable) with no CUDA device.
infernix_add_op_test(infernix_image_vae_spatial_attn_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vae_spatial_attn.cu"
  LIBRARIES infernix_ops)

# GPU qualification of the two VAE-ENCODER ops (task #75 ref-image route) vs naive FP64 oracles:
# ops::conv2d_stride2 (Resample downsample: right/bottom-only ZeroPad, stride 2; exhaustive small
# planes + sampled production-width 384 case) and ops::avg_down (AvgDown3D image shortcut:
# channel-group mean with divisor = group_size, temporal front zero pad; group=1 identity,
# factor_t 1/2 exhaustive cases). Guards: stride-2 H<2/bad out extent, avg_down factor_t=3/odd H/
# indivisible channel pair. A-priori tol 2e-2, never narrowed. Links infernix_ops. Returns 2 with
# no CUDA device.
infernix_add_op_test(infernix_image_vae_encode_ops_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vae_encode_ops.cu"
  LIBRARIES infernix_ops)

# GPU production-route qualification of the #75 reference-image ENCODER: encode_reference_image()
# (full VAE encoder chain, BF16, via the artifact Reader + bind_image_vae) vs the independent
# diffusers FP32 oracle dumps (image-spike/oracle/dump_ref_encode.py) at the pipeline reference
# geometry 1248x832 -> 4056 tokens and the small 160x96 -> 60-token geometry. A-priori tol 5e-2
# on the packed [tokens,64] latents for both legs (VAE-decoder prod precedent; deep BF16 chain vs
# FP32 is O(1e-3..1e-2), an op/layout defect is O(0.1..10)); never narrowed. Links
# infernix_qwen_image_runtime. Skips (77) without INFERNIX_TEST_ARTIFACT /
# INFERNIX_REF_ENCODE_ORACLE; returns 2 with no CUDA device.
infernix_add_op_test(infernix_image_vae_encode_prod_cuda_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_vae_encode_prod_cuda.cu"
  LIBRARIES infernix_ops infernix_qwen_image_runtime)
set_tests_properties(infernix_image_vae_encode_prod_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

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
  LIBRARIES infernix_ops infernix_qwen_image infernix_qwen_image_runtime)
set_tests_properties(infernix_image_pipeline_prod_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

# CPU-oracle qualification of the BF16 linear's small-token (skinny GEMV) path (tokens <= 8), the
# route the Qwen-Image time-embedding linears use. Independent of any model/oracle data.
infernix_add_op_test(infernix_image_skinny_linear_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_skinny_linear.cu"
  LIBRARIES infernix_ops)

# Host-side text-tower tokenization gate (variable-input #37): the engine Qwen tokenizer, loaded from
# the Qwen3-VL processor's own tokenizer.json / tokenizer_config.json, must reproduce
# tensors_text_tower/input_ids.npy token-for-token for the oracle t2i template (prompt read from the
# manifest). Proves the host tokenize slice the tower input depends on; the tower forward itself is
# qualified on-device (infernix_image_texttower_prod_cuda_test). Host-only, no GPU. Links the
# qwen3_5 frontend (infernix_engine). Skips (77) without INFERNIX_QWENVL_PROCESSOR / INFERNIX_IMAGE_ORACLE.
infernix_add_op_test(infernix_image_tower_tokenize_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_tower_tokenize.cpp"
  LIBRARIES infernix_engine infernix_core infernix::json)
set_tests_properties(infernix_image_tower_tokenize_test PROPERTIES SKIP_RETURN_CODE 77)

# Step 8 (8C): the PRODUCTION prompt renderer (serve/image_text_tower.h) vs the same oracle: its
# compiled-Jinja chat_template.jinja render (add_generation_prompt included) and encode must equal
# the manifest template and input_ids.npy exactly, with the computed drop at drop_idx. Host-only.
# Skips (77) without the same environment.
infernix_add_op_test(infernix_image_tower_prompt_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_tower_prompt.cpp"
  LIBRARIES infernix_serve infernix_engine infernix_core infernix::json)
set_tests_properties(infernix_image_tower_prompt_test PROPERTIES SKIP_RETURN_CODE 77)

# Step 8 (8D): the PRODUCTION raw-prompt tower route (serve provider: compiled chat template +
# genuine tokenizer + FP8-v2 streamed weights + one-time FP8 embed dequant) on device; its
# pre-final-RMSNorm rows gated vs pre_norm_hidden[drop_idx,S) at the qualified tower test's a-priori
# 2e-2, with the #17 ledger gated at 0 before and after. Skips (77) without the oracle/processor/
# artifact environment; returns 2 with no CUDA device.
infernix_add_op_test(infernix_image_tower_provider_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_tower_provider_real.cu"
  LIBRARIES infernix_serve infernix_engine infernix_qwen_image infernix_ops infernix_core infernix::json)
set_tests_properties(infernix_image_tower_provider_real_test PROPERTIES SKIP_RETURN_CODE 77)

# Production post-tower text conditioning (infernix_qwen_image::build_prompt_embeds): gather each item's
# VALID tower rows, drop the leading drop_idx of them (mask-aware), right-pad to the longest kept length,
# emit encoder_attention_mask -- the reference pipeline's post-encode step (dump_text_tower.py), the pure
# data-movement between the (on-device-qualified) tower and the DiT. Synthetic gate (always runs): a
# multi-item batch with left- AND right-padding + distinct drop lengths, hand-built expected (a real
# reference, exercising the variable-N case the single oracle cannot). Oracle gate (INFERNIX_IMAGE_ORACLE):
# the real ASCII pre_norm_hidden.npy -> must reproduce prompt_embeds.npy + encoder_attention_mask.npy
# bitwise (pure selection/padding of the same f32 rows). Host-only, no GPU.
infernix_add_op_test(infernix_image_text_embeds_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_text_embeds.cpp"
  LIBRARIES infernix_qwen_image infernix::json)

# Production front-half embed_tokens gather (infernix_qwen_image::build_inputs_embeds): inputs_embeds[t] =
# widen(embed_tokens[ids[t]]) -- the pure BF16 row lookup that turns a prompt's token ids into the text
# tower's INPUT (paired bookend to build_prompt_embeds after the tower; the two are the data movement
# around the on-device-qualified forward -- gap #1). Synthetic gate (always runs): hand-built BF16 table of
# BF16-exact small ints + repeats + order -> row lookup/order/widening bitwise, out-of-range id refused
# (std::invalid_argument). Oracle gate (INFERNIX_IMAGE_ORACLE + INFERNIX_IMAGE_TEXT_ENCODER_ST): gather the
# real model.language_model.embed_tokens.weight (BF16, read by header+slice from the Qwen3-VL text-encoder
# safetensors, never the whole 5 GB file) rows for input_ids.npy -> reproduce tensors_text_tower_weights/
# inputs_embeds.npy bitwise (tol 0; mirrors verify_embed_gather.py in C++); SKIPS when the external
# artifact is absent (a prerequisite, not repo state). Host-only, no GPU.
infernix_add_op_test(infernix_image_embed_gather_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_embed_gather.cpp"
  LIBRARIES infernix_qwen_image infernix::json)

# Production variable-resolution joint-sequence rope (infernix_qwen_image::build_joint_rope): the host
# builder that replaces the fixed-geometry rotary .npy the interlude reads -- text_prefix + H*W joint
# sequence, 3-axis (frame/height/width) rotary [S,64,2] interleaved, reproducing QwenImage21Rope.forward
# for the text-only case. Grid invariants (shape S*128, unit modulus, text h/w-axis identity, shared
# image frame, two closed-form spots, determinism) across non-square/tiny/tp=0 geometries always run;
# two INDEPENDENT diffusers-dumped rotary anchors (block_rotary.npy @8,8,8 rel<1e-2; rot_inter.npy
# @11,16,16 rel<5e-2, tols copied from the pipeline self-check) run when INFERNIX_IMAGE_ORACLE is set.
# Host-only, no GPU. Guards the variable-resolution gap the single fixed-geometry dump cannot cover.
infernix_add_op_test(infernix_image_rope_prod_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_rope_prod.cpp"
  LIBRARIES infernix_qwen_image)

# Production joint-sequence img_mask (infernix_qwen_image::build_img_mask): the DiT joint mask is the text
# encoder's image_pad_mask_base row-concatenated with target_slots = (Hl*Wl)//4 packed-image ones
# (dump_pipeline.py:124). The base widens with the N condition images' image_pad slots, so one rule covers
# text-only and N-condition images -- replacing the fixed img_mask.npy the interlude reads. Synthetic gate
# (always runs): multi-item batch with condition 1-slots in the base + target_slots=0 edge, hand-built
# expected (base verbatim + ones). Oracle gate (INFERNIX_IMAGE_ORACLE): real image_pad_mask_base.npy +
# meta tokens -> reproduces img_mask.npy bitwise + matches meta img_mask_len. Host-only, no GPU.
infernix_add_op_test(infernix_image_img_mask_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_img_mask.cpp"
  LIBRARIES infernix_qwen_image infernix::json)

# Host loader gate (task #13 step 2): bind_dit/bind_text_encoder resolve EVERY logical DiT/text
# parameter the converter recipe names against the REAL artifact's Directory bindings, at config-
# derived shapes and the recipe's exact representations (FP8 E4M3 RowScale for the quantized roles,
# direct BF16 otherwise), and finish to a plan with zero device/pinned bytes (all streamed). Dims
# are asserted against the qualified architecture (32x32x128, MLP 3x, 36-layer GQA 32/8 tower);
# malformed names/shapes/formats are negative-tested to throw at open time. Skips (77) without
# INFERNIX_TEST_ARTIFACT / when absent. No GPU, no server.
infernix_add_op_test(infernix_image_dit_weights_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_dit_weights.cpp"
  LIBRARIES infernix_qwen_image)
set_tests_properties(infernix_image_dit_weights_test PROPERTIES SKIP_RETURN_CODE 77)

# Production pipeline-entry gate (task #13 step 3): bind_image_vae resolves all 128 direct-BF16
# VAE-decoder roles the qualified decode chain names; ONE Binder of DiT + VAE finishes to a
# streamed-only plan (zero device/pinned); the decoder's conv-shortcut structure and the deliberate
# time_conv non-binding are asserted against the route; the largest bound object is the
# 1152x1152x3x3 conv plane; and the run_image_job entry LINKS from infernix_qwen_image_runtime
# (never called -- no device work). A .cu so the entry header's cuda_runtime.h include is assured.
# Skips (77) without INFERNIX_TEST_ARTIFACT / when absent. No GPU, no server.
infernix_add_op_test(infernix_image_pipeline_gate_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_pipeline_gate.cu"
  LIBRARIES infernix_qwen_image_runtime)
set_tests_properties(infernix_image_pipeline_gate_test PROPERTIES SKIP_RETURN_CODE 77)

# Variable-resolution job-assembly INTEGRATION gate (infernix_qwen_image): the individually-oracle-verified
# modules (ImageGeometry/ImageJob scheduler, build_joint_rope, build_img_mask, fill_normal) must describe
# ONE consistent job for any (h,w). Across square / non-square / snapped-to-32 / tall / tiny geometries and
# varying text_len + steps, asserts the cross-module invariants a single fixed-resolution oracle cannot:
# rope rows == img_mask len == text_len + tokens/4; tokens == 4*packed; schedule length tracks steps; init
# latent count == packed*64 and finite. Host-only, no oracle, no GPU -- always runs.
infernix_add_op_test(infernix_image_job_assembly_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_job_assembly.cpp"
  LIBRARIES infernix_qwen_image)
