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
//   | DiT rope axes                   | the artifact                           |
//   | DiT rope theta                  | the pipeline implementation (code)     |
//   | CUDA launch configuration       | the runtime implementation             |
//
// The rope row is deliberately the odd one out: the shipped checkpoint declares no rope theta in
// transformer/config.json at all (verified against the live artifact's own "dit" component), yet the
// supported Diffusers Qwen-Image pipeline rotates with 10000, so naming it an artifact field would be
// an invention. rope.h carries it as kQwenImageRopeTheta, a named implementation default that changes
// only when the supported pipeline implementation is deliberately changed. An artifact that declares a
// "rope_theta" of its own is therefore not a source of the value but a claim about the scheme its weights
// were trained under: read_dit_rope_axes verifies such a claim against kQwenImageRopeTheta and refuses the
// artifact when it disagrees, instead of silently rotating those weights at 10000.
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

// The three axes the DiT rotates the joint sequence on (frame / height / width), exactly as the
// artifact's "dit" component declares them: positive even channel counts summing to attention_head_dim,
// so the emitted table is attention_head_dim / 2 interleaved (cos, sin) pairs wide. Nothing here is
// pinned at the shipped {16, 56, 56} -- the emitter's row width comes from these numbers. Zero here
// means NEVER READ: the only producer is read_dit_rope_axes, and the image execution refuses an unread
// set before it allocates anything. Theta is NOT a member, because the artifact does not declare one.
struct DitRopeAxes {
    std::uint32_t axes[3] = {0, 0, 0};
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

// Read and verify the DiT rope axes from the artifact's "dit" component config -- the checkpoint's own
// transformer/config.json, which the converter packed verbatim. "attention_head_dim" must be a JSON
// integer and "axes_dims_rope" a JSON array of exactly three positive EVEN JSON integers whose sum is
// that head width; anything else is a refusal naming the key, the offending value, and both numbers.
// Unlike the scheduler document, this one carries the whole transformer architecture, so keys the rope
// rotation does not read are the recipe's parameter contract's business and are not judged here. ONE
// exception, and it is a fail-closed check rather than a source of values: if the artifact DOES declare a
// "rope_theta", it must be a JSON float equal to this build's implementation default (rope.h,
// kQwenImageRopeTheta) or the artifact is refused naming the key, the declared value and the authority it
// was compared against. Absent is the shipped case and stays absent -- theta is never read FROM the
// artifact (the emitter always rotates with the code's constant), but an artifact that declares another
// base describes weights trained under a different rotary scheme, and drawing them at the code's base in
// silence is the divergence this lane refuses everywhere else.
[[nodiscard]] std::expected<DitRopeAxes, std::string> read_dit_rope_axes(const nlohmann::json& config);

} // namespace infernix::models::qwen_image
#endif // INFERNIX_IMAGE_READS_ARTIFACT_CONSTANTS
