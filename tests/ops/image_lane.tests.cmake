# The image lane's own op suites.
#
# The four suites here qualify the DiT's kernels and the one ceiling the image routes share with them;
# their sources exist only in this fork (tests/ops/test_rope_c64.cu, test_attn_dit_joint.cu,
# test_dit_adaln.cu, test_image_joint_limits.cpp -- none of them at upstream), so on a port they are a
# pure re-add. tests/ops/tests.cmake reaches this file with one include line, the same idiom upstream
# already uses for linear/tests.cmake, and ${CMAKE_CURRENT_LIST_DIR} still means tests/ops because this
# file lives beside it. infernix_add_op_test / infernix_add_test / set_tests_properties are the parent
# directory's macros and are visible here.

# DiT complex 3-axis RoPE (use_real=false): synthetic BF16 q/k against a FP64 complex oracle over a
# real positional cis table. Skips (77) with no CUDA device.
infernix_add_op_test(infernix_rope_c64_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_rope_c64.cu"
  LIBRARIES infernix_ops)

# DiT joint block-causal attention against a FP64 masked-softmax oracle over random BF16 q/k/v.
# Skips (77) with no CUDA device.
infernix_add_op_test(infernix_attn_dit_joint_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_attn_dit_joint.cu"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_attn_dit_joint_test PROPERTIES SKIP_RETURN_CODE 77)

# DiT AdaLN-Zero modulation (adaln_scale, adaln_gate_add) against a FP64 oracle over random BF16
# activations and a [M,2] modulation table. Skips (77) with no CUDA device.
infernix_add_op_test(infernix_dit_adaln_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dit_adaln.cu"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_dit_adaln_test PROPERTIES SKIP_RETURN_CODE 77)

# One joint-sequence ceiling, shared by the image ops, /v1/capabilities and the chat tool
# schema. Host-only: the ops' token check runs before any device work is enqueued, so the
# refusals need no CUDA device and the suite must never skip.
infernix_add_test(infernix_image_joint_limits_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image_joint_limits.cpp"
  LIBRARIES infernix_ops infernix_serve)
