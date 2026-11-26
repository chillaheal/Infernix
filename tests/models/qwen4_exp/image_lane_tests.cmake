# The image lane's coexistence suites on the qwen4_exp Program.
#
# These are the LLM<->image suites: the one-shot yield that hands physical VRAM back, the first real
# image op and the whole DiT forward running on the yielded region, the production route through
# Engine::run_image_interlude, the interlude failure-state machine and the admission-queue budget across
# a closed submit door. Their sources exist only in this fork (test_image_yield_real.cpp,
# test_image_interlude_*_real.{cpp,cu}, test_image_queue_deadline_real.cu -- none of them at upstream,
# which has no image code to coexist with), so on a port they are a pure re-add.
# tests/models/qwen4_exp/tests.cmake reaches this file with one include line, and ${CMAKE_CURRENT_LIST_DIR}
# still means tests/models/qwen4_exp because this file lives beside it.
#
# Every one of them needs INFERNIX_QWEN4_ARTIFACT (the yield source) and, where it says so, the image
# artifact and the oracle; they skip with 77 without them, stay RUN_SERIAL because they take the card,
# and are labelled gpu;real so a host-only selection never offers them.

# LLM<->image coexistence: the Program's one-shot yield returns physical VRAM (cuMemGetInfo) and its
# release restores the pools, so a later generation is undisturbed. Skips without INFERNIX_QWEN4_ARTIFACT.
infernix_add_test(infernix_qwen4_exp_image_yield_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_yield_real.cpp"
  LIBRARIES infernix_engine infernix_model_runtime)
set_tests_properties(infernix_qwen4_exp_image_yield_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# LLM<->image coexistence (C4): the first real image op runs on the LLM Program's YIELDED VRAM,
# in-process. The DiT's real row-scaled-FP8 [4096,4096] projection is streamed from the image artifact
# through the bounded pinned HostContextArena, H2D'd into buffers allocated in the freed region, run
# with ops::linear on the drained device while the LLM is Quiesced/Leased, verified against a CPU
# dequant reference; the LLM then resumes undisturbed (ids == baseline). Needs both artifacts:
# INFERNIX_QWEN4_ARTIFACT (yield source) and INFERNIX_TEST_ARTIFACT (image artifact). Skips (77) without either.
infernix_add_test(infernix_qwen4_exp_image_interlude_op_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_interlude_op_real.cpp"
  LIBRARIES infernix_engine infernix_model_runtime infernix_ops infernix_artifact)
set_tests_properties(infernix_qwen4_exp_image_interlude_op_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# LLM<->image coexistence (C4, full step): the whole one-step DiT forward -- front-end + all 32
# production blocks (streamed one block at a time so the resident image working set peaks at ~one
# block ~0.9 GiB) + norm_out/proj_out -- runs in-process on the LLM Program's YIELDED 8 GiB while the
# LLM sits Quiesced/Leased, verified against noise_pred_step0.npy (step-0 velocity) and the LLM then
# resumes undisturbed (ids == baseline). Needs INFERNIX_QWEN4_ARTIFACT (yield source) and
# INFERNIX_IMAGE_ORACLE (weight dirs) + [INFERNIX_IMAGE_ORACLE_PIPELINE]. Skips (77) without them.
infernix_add_test(infernix_qwen4_exp_image_interlude_dit_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_interlude_dit_real.cpp"
  LIBRARIES infernix_engine infernix_model_runtime infernix_ops infernix_qwen_image_runtime)
set_tests_properties(infernix_qwen4_exp_image_interlude_dit_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# LLM<->image coexistence (C4, full pipeline): the whole image job -- 40-step FlowMatchEuler over
# all 32 production blocks (weights cached once in host RAM and H2D-streamed per-block per-step so
# the resident VRAM peaks at ~one block) + the VAE decode of the final latent -- runs in-process on
# the LLM Program's YIELDED 8 GiB while the LLM sits Quiesced/Leased, gated vs noise_pred_step0
# (step 0), latents_final.npy (rel_fro < 0.5) and image.npy (rel_fro < 0.3); the LLM then resumes
# undisturbed (ids == baseline). A .cu (not .cpp) because the VAE decode needs the channel<->token
# transpose kernel (no production transpose op). Needs INFERNIX_QWEN4_ARTIFACT (yield source) and
# INFERNIX_IMAGE_ORACLE (weight dirs) + [INFERNIX_IMAGE_ORACLE_PIPELINE]. Skips (77) without them.
infernix_add_test(infernix_qwen4_exp_image_interlude_full_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_interlude_dit_real.cu"
  LIBRARIES infernix_engine infernix_model_runtime infernix_ops infernix_qwen_image_runtime)
set_tests_properties(infernix_qwen4_exp_image_interlude_full_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# Step-4 certification: the PRODUCTION route -- Engine::run_image_interlude (public entry) driving
# run_image_job on weights streamed from the REAL image artifact (bind_dit + bind_image_vae), with
# the production ImageJob schedule. Gates: FINAL_LATENT < 0.5, FINAL_IMAGE < 0.3 vs the oracle,
# LLM undisturbed post-job, device-memory ledger at the quiescent points. Needs INFERNIX_QWEN4_ARTIFACT,
# INFERNIX_TEST_ARTIFACT and INFERNIX_IMAGE_ORACLE; skips (77) without them.
infernix_add_test(infernix_qwen4_exp_image_interlude_prod_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_interlude_prod_real.cu"
  LIBRARIES infernix_engine infernix_model_runtime infernix_ops infernix_qwen_image_runtime infernix_artifact)
set_tests_properties(infernix_qwen4_exp_image_interlude_prod_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# C5b+C5c certification: the interlude failure-state machine and the submit-door coverage on the real
# Engine with tiny jobs: success route, drain-deadline unwind, job-throw rethrow, single-flight, the
# door gate across a decoding LLM, and the terminal restoration-fault Degraded (runs last; leaves the
# Engine Degraded by design). Needs INFERNIX_QWEN4_ARTIFACT; skips (77) without it.
infernix_add_test(infernix_qwen4_exp_image_interlude_failures_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_interlude_failures_real.cu"
  LIBRARIES infernix_engine infernix_model_runtime)
set_tests_properties(infernix_qwen4_exp_image_interlude_failures_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# Admission-queue budget across an interlude: a request that arrives while the submit door is closed
# keeps its whole queue budget (the hold is credited back to it), and the budget still expires when the
# door is open and the lane is busy. Needs INFERNIX_QWEN4_ARTIFACT; skips (77) without it.
infernix_add_test(infernix_qwen4_exp_image_queue_deadline_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_queue_deadline_real.cu"
  LIBRARIES infernix_engine infernix_model_runtime)
set_tests_properties(infernix_qwen4_exp_image_queue_deadline_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)
