# The image lane's own test targets.
#
# These are the suites whose source files exist only in this fork (tests/test_image_*.cpp,
# tests/test_chat_image_tool.cpp, tests/test_openai_images_schema.cpp, tests/test_media_encode.cpp --
# none of them at upstream), so they are a pure re-add on a port and no upstream test file has to carry
# them. tests/cmake/ProductTests.cmake reaches this file with one include line, the same idiom upstream
# uses for its own split lists, and ${CMAKE_CURRENT_LIST_DIR} still means tests/cmake because this file
# lives beside it.
#
# The protocol suites that DO test image behaviour from the client side (the /v1/capabilities block, the
# injected tool, the image_tool_attempts field) are not here: they are cases added inside upstream's own
# test files, and docs/maintainer/image-integration-points.md carries them as anchors so a replay
# re-applies them and notices when upstream has moved one away.

# PNG encoder round trip through the engine's own decoder (opaque pixels byte-exact, multi-block
# stored DEFLATE, negative sizes).
infernix_add_test(infernix_media_encode_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_media_encode.cpp"
  LIBRARIES infernix_media_encode infernix_media_decode)

infernix_add_test(infernix_image_job_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_image_job_store.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_openai_images_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_images_schema.cpp"
  LIBRARIES infernix_serve)

# Synchronous image route service: failure mapping -> ApiException surface and store terminals.
# Every service in it is built with the artifact's own decoder geometry (the request's topology member is
# a reference with no default), which the suite reads through the production reader over the committed copy
# of the checkpoint's vae/config.json -- so it needs the source dir and the reader, which returns
# std::expected (host-only, compiled per target at C++23; the CUDA lane stays at C++20).
infernix_add_test(infernix_image_generation_service_test
  NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_image_generation_service.cpp"
  LIBRARIES infernix_serve infernix_qwen_image infernix_artifact infernix::json)
target_compile_features(infernix_image_generation_service_test PRIVATE cxx_std_23)

# The prompt enhancer's answer contract and the refusal a client gets when no sidecar is running: the
# rules the shipped edit system prompt puts on its own answer, and the codes that keep a bad rewrite
# from reaching the pipeline as if it were the client's instruction.
infernix_add_test(infernix_prompt_enhancer_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_prompt_enhancer.cpp"
  LIBRARIES infernix_serve)

# The image job's computed device-memory need: the number the serving layer refuses on before the LLM
# yields anything, qualified here against the certified route's measured peak. The ledger's decoder half
# has no constants of its own: the request carries the artifact's VaeTopology, which this suite reads
# through the PRODUCTION reader over the committed copy of the checkpoint's own vae/config.json
# (tests/models/qwen_image/vae_topology_fixture.h), so the target needs the source dir and the reader,
# which returns std::expected (host-only, compiled per target at C++23; the CUDA lane stays at C++20).
# infernix_serve is here because the ViT half of the ledger is a CLAIM ABOUT THE ROUTE: the suite proves
# the billed worst reference plane by running the real src/serve snap and the real
# prepare_reference_planes on encoded PNGs, so a change to the route's geometry moves this suite instead
# of agreeing with the mistake (tests/test_image_device_need.cpp). Host code only -- no CUDA enters here.
infernix_add_test(infernix_image_device_need_test
  NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_image_device_need.cpp"
  LIBRARIES infernix_serve infernix_qwen_image infernix_artifact infernix::json)
target_compile_features(infernix_image_device_need_test PRIVATE cxx_std_23)

# Server-executed chat image tool: injection gating, round feedback shape, caps and error surface,
# and the round loop's end contract of exactly one answer that carries every URL.
infernix_add_test(infernix_chat_image_tool_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_chat_image_tool.cpp"
  LIBRARIES infernix_serve)
