infernix_add_test(infernix_media_decode_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_media_decode.cpp"
  LIBRARIES infernix_media_decode)

# PNG encoder round trip through the engine's own decoder (opaque pixels byte-exact, multi-block
# stored DEFLATE, negative sizes).
infernix_add_test(infernix_media_encode_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_media_encode.cpp"
  LIBRARIES infernix_media_encode infernix_media_decode)

infernix_add_test(infernix_prompt_input_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_prompt_input.cpp"
  LIBRARIES infernix_product_prompt_input)

infernix_add_test(infernix_pretty_logging_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_pretty_logging.cpp"
  LIBRARIES infernix_product_logging infernix_media_decode)

# Stable per-statistic console colouring (product/log_colour): family classification of the
# operational line prefixes and clause-aware colouring of the pretty stats format.
infernix_add_test(infernix_log_colour_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_log_colour.cpp")

infernix_add_test(infernix_perplexity_evaluation_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_perplexity_evaluation.cpp"
          ${PROJECT_SOURCE_DIR}/apps/perplexity/evaluation.cpp
  LIBRARIES infernix_core)

target_include_directories(infernix_perplexity_evaluation_test PRIVATE
  ${PROJECT_SOURCE_DIR}/apps/perplexity)

infernix_add_test(infernix_llama_kld_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_llama_kld.cpp"
          ${PROJECT_SOURCE_DIR}/apps/certify/llama_kld.cpp
  LIBRARIES infernix_core infernix::json)

target_include_directories(infernix_llama_kld_test PRIVATE
  ${PROJECT_SOURCE_DIR}/apps/certify)

infernix_add_test(infernix_cli_options_test NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_cli_options.cpp" ${PROJECT_SOURCE_DIR}/apps/cli/options.cpp
  LIBRARIES infernix_runtime_support infernix_product_config infernix_product_logging)

target_include_directories(infernix_cli_options_test PRIVATE ${PROJECT_SOURCE_DIR}/apps/cli)

infernix_add_test(infernix_openai_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_schema.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_openai_responses_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_responses.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_openai_responses_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_responses_store.cpp"
  LIBRARIES infernix_serve)

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

infernix_add_test(infernix_decide_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_decide_schema.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_anthropic_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_anthropic_schema.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_tool_search_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_tool_search.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_serve_options_test NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_serve_options.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_request_log_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_request_log.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_metrics_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_metrics.cpp"
  LIBRARIES infernix_serve)
infernix_add_test(infernix_metrics_contract_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_metrics_contract.cpp"
  LIBRARIES infernix_serve)
infernix_add_test(infernix_prefix_miss_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_prefix_miss.cpp"
  LIBRARIES infernix_serve)
infernix_add_test(infernix_reasoning_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_reasoning_store.cpp"
  LIBRARIES infernix_serve)
infernix_add_test(infernix_console_stats_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_console_stats.cpp"
  LIBRARIES infernix_serve infernix_product_logging)

infernix_add_test(infernix_stop_control_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_stop_control.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_http_error_handler_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_error_handler.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_http_routes_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_routes.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_http_transport_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_transport.cpp"
  LIBRARIES infernix_serve)
