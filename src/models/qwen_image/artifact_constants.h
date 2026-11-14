#pragma once

// The Qwen-Image-2.1 image lane's model constants are read FROM the `.infernix` artifact and refused
// when the artifact cannot express the scheme the code implements. Nothing here may fall back to a
// compiled value: an artifact that says nothing about the denoise schedule must not be drawn with the
// schedule of another family (docs/maintainer/qwen-image-2-1-variables.md).
//
// Where each parameter's authority lies, so a reader can look it up instead of guessing:
//
//   | parameter                       | authority                              |
//   |---------------------------------|----------------------------------------|
//   | scheduler values and mode flags | the artifact                           |
//   | CUDA launch configuration       | the runtime implementation             |
//
// The scheduler document is the converter's verbatim copy of the checkpoint's own
// scheduler/scheduler_config.json (tools/convert/qwen_image.py), packed as the artifact's "scheduler"
// component: the reader adds no field, defaults no field, and coerces no type. A JSON integer is only
// ever an integer (256 ok, 256.0 refused), a boolean only ever a boolean (true ok, 1 refused), a
// string only ever a string, and a float must be finite.
//
// Host-only code: the readers are called while the artifact is bound, before any quiesce, yield, CUDA
// allocation or partially initialised model state.

#include <cstdint>
#include <string>

namespace infernix::models::qwen_image {

// The six values the resolution-dependent FlowMatchEulerDiscrete schedule is built from, exactly as
// the artifact declares them (an integer stays a JSON integer, so it is a distinct C++ type from the
// shifts and never silently accepts 256.0). Zero here means NEVER READ: the only producer is
// read_scheduler_config, and the image execution refuses an unread set before it allocates anything.
struct SchedulerConstants {
    std::int64_t base_image_seq_len = 0;
    double       base_shift         = 0.0;
    std::int64_t max_image_seq_len  = 0;
    double       max_shift          = 0.0;
    std::int64_t num_train_timesteps = 0;
    double       shift_terminal     = 0.0;
};

} // namespace infernix::models::qwen_image

// The readers return std::expected, which the project compiles per target at C++23 (the CUDA lane is
// pinned to C++20, where the standard library does not provide std::expected). The records above are
// plain data and stay visible to every translation unit; the reader declarations appear exactly where
// std::expected exists, which is where they are called -- the host binding path. They are never called
// from a device translation unit.
#if !defined(__CUDACC__)
#    if defined(_MSVC_LANG)
#        if defined(_HAS_CXX23) && _HAS_CXX23
#            define INFERNIX_IMAGE_READS_ARTIFACT_CONSTANTS 1
#        endif
#    elif __cplusplus >= 202302L
#        define INFERNIX_IMAGE_READS_ARTIFACT_CONSTANTS 1
#    endif
#endif

#ifdef INFERNIX_IMAGE_READS_ARTIFACT_CONSTANTS
#    include <nlohmann/json.hpp>

#    include <expected>

namespace infernix::models::qwen_image {

// Read and verify the artifact's scheduler document. Refusals name the key and, for a deviating value,
// the value itself, so the operator who passed --image-artifact sees WHY the artifact was refused; the
// caller prefixes which artifact it concerns. Every key the supported path reads must be present with
// the JSON type it promises, and every mode flag is verified, not just the six numbers:
// use_dynamic_shifting == true, time_shift_type == "exponential", use_karras_sigmas == false,
// use_beta_sigmas == false, use_exponential_sigmas == false, invert_sigmas == false, shift == 1.0,
// stochastic_sampling == false. A key the supported path has not verified is an unhandled scheme and is
// refused, because the schedule it would produce is not this schedule.
[[nodiscard]] std::expected<SchedulerConstants, std::string> read_scheduler_config(const nlohmann::json& config);

} // namespace infernix::models::qwen_image
#endif // INFERNIX_IMAGE_READS_ARTIFACT_CONSTANTS
