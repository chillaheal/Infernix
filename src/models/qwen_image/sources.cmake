# Qwen-Image-2.1: the in-process image job's host-side config and scheduler math
# (docs/maintainer/qwen-image-2-1-variables.md). Host-only; the CUDA image execution links it.
add_library(infernix_qwen_image STATIC
  "${CMAKE_CURRENT_LIST_DIR}/scheduler.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/noise.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/text_embeds.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/embed_gather.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/rope.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/img_mask.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/image_weights.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/artifact_constants.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/joint_layout.cpp")
infernix_internal_includes(infernix_qwen_image)
# The production weight loader binds logical parameters from the .infernix Directory.
target_link_libraries(infernix_qwen_image PUBLIC infernix_artifact)
# The artifact readers return std::expected, which this compiler only provides above C++20; this is a
# host-only library, so the CUDA runtime target stays at the project's C++20. The records in
# models/qwen_image/artifact_constants.h are plain data and stay visible to the C++20 lane.
target_compile_features(infernix_qwen_image PRIVATE cxx_std_23)

# The shared DiT block forward and the production image-job CUDA execution, calling the CUDA op
# layer. Separate from the host-only config lib so the image execution and its tests link it without
# dragging the ops dependency into host tests.
add_library(infernix_qwen_image_runtime STATIC
  "${CMAKE_CURRENT_LIST_DIR}/dit_block.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/image_pipeline.cu"
  "${CMAKE_CURRENT_LIST_DIR}/text_tower.cu"
  "${CMAKE_CURRENT_LIST_DIR}/vision_tower.cu")
infernix_internal_includes(infernix_qwen_image_runtime)
# The pipeline entry consumes the artifact reader (streamed weights) and the host rope module.
target_link_libraries(infernix_qwen_image_runtime PUBLIC infernix_ops infernix_core infernix_qwen_image)
