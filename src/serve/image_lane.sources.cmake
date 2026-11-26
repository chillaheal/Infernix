# The image lane's own build entries for the infernix_serve library.
#
# Everything the lane adds to the serve library is here, and src/serve/CMakeLists.txt reaches it with
# one include line: a file the lane does not own then carries no lane entry at all, so an upstream
# change to that list never conflicts with the lane and the lane never has to be re-added by hand after
# a merge. The include sits after add_library(infernix_serve ...) and after infernix_internal_includes,
# which is where target_sources and target_link_libraries belong for it; the same idiom upstream uses
# for its own split lists (see tests/ops/tests.cmake's include of linear/tests.cmake).
#
# The twelve sources below are the whole lane: the artifact-backed job store, the two image routes,
# the reference-picture decode, the server-executed chat tool, the service that yields the LLM's VRAM
# for a draw, the prompt enhancer's sidecar handle, the text tower, and the four seam files that keep
# the lane's text out of the protocol, options, capabilities and startup files
# (docs/maintainer/image-integration-points.md names each seam and what stays upstream).
#
# Relative paths resolve against this file's directory, which is src/serve, the same directory
# add_library(infernix_serve ...) was written in.

target_sources(infernix_serve PRIVATE
  image_job_store.cpp
  openai_images.cpp
  reference_image.cpp
  openai_images_http.cpp
  chat_image_tool.cpp
  image_generation_service.cpp
  prompt_enhancer.cpp
  image_text_tower.cpp
  image_serve_options.cpp
  image_capabilities.cpp
  image_startup.cpp
  image_chat_hook.cpp
)

# The three libraries only the lane needs. Upstream's own PRIVATE list keeps everything the plain
# server also uses (media acquire/decode, grammar, text, cuda runtime), so nothing outside the image
# code depends on these: infernix_qwen_image and its runtime are the DiT and VAE forwards the job
# executor binds, and infernix_media_encode is the PNG writer a finished picture is stored with.
target_link_libraries(infernix_serve PRIVATE
  infernix_qwen_image infernix_qwen_image_runtime infernix_media_encode)
