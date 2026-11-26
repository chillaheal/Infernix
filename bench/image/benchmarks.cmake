# Bundle B fas 0: the production image route's stage baseline -- the interlude's paused window, the whole
# request, the device job inside it, and the DiT block uploads' share of the loop.
add_executable(infernix_image_stage_bench
  "${CMAKE_CURRENT_LIST_DIR}/image_stage_bench.cpp")
infernix_internal_includes(infernix_image_stage_bench)
target_include_directories(infernix_image_stage_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party/spdlog/include)
target_link_libraries(infernix_image_stage_bench PRIVATE
  infernix_serve infernix_product_logging infernix_qwen_image CUDA::cudart)
