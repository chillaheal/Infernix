// The strict artifact readers behind artifact_constants.h (see that header for the authority table).
// The Qwen-Image-2.1 image lane refuses an artifact instead of guessing: an absent key, a key with the
// wrong JSON type, a mode flag this path does not implement, or a key it has never verified all fail
// the binding, before any CUDA allocation and before any partially initialised model state.

#include "models/qwen_image/artifact_constants.h"

#ifdef INFERNIX_IMAGE_READS_ARTIFACT_CONSTANTS

// kQwenImageRopeTheta: the pipeline implementation default that an artifact-declared "rope_theta" has to
// agree with. The emitter in rope.cpp is the only thing that ROTATES, and it always rotates with that
// constant -- reading it here is a check on what the artifact CLAIMS, never a value the code adopts.
#include "models/qwen_image/rope.h"

// calculate_shift: the ONE authority for what the artifact's declared shift window means. The reader
// below has to ask the same question the schedule itself asks -- what mu does the declared window
// produce at a token count the lane can actually build -- and mirroring that arithmetic here is how a
// reader and a scheduler learn to disagree.
#include "models/qwen_image/scheduler.h"

// ops::kImageJointMaxTokens: the product's one joint-sequence ceiling, which is the largest token count
// the lane can ever hand the scheduler (a job's canvas tokens are a part of the joint sequence).
#include "infernix/ops/image_joint_limits.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::models::qwen_image {
namespace {

using Json = nlohmann::json;

std::string quoted_key(std::string_view text) { return "\"" + std::string(text) + "\""; }

// Every refusal prints the value as the artifact declared it, and nlohmann's dump() THROWS on a
// non-finite number: an artifact may declare inf or NaN in any numeric position, and a refusal that
// aborts the host process instead of naming the key is the failure mode this file exists to avoid.
std::string dump_safe(const Json& value) {
    if (value.is_number_float() && !std::isfinite(value.get<double>())) { return "a non-finite JSON number"; }
    return value.dump();
}

std::string prefix() { return "the scheduler config "; }

// Every refusal names the key it concerns; a deviating value is printed as the artifact declared it,
// so the operator running --image-artifact reads the reason instead of guessing it.
std::string absent_key(std::string_view key) {
    return prefix() + "has no " + quoted_key(key) +
           ", which the supported denoise schedule reads from the artifact";
}

std::string wrong_type(std::string_view key, const Json& value, std::string_view wanted) {
    return prefix() + quoted_key(key) + " is " + dump_safe(value) + " but must be " + std::string(wanted) +
           "; the reader does not coerce a JSON type";
}

std::string wrong_value(std::string_view key, const Json& value, std::string_view wanted) {
    return prefix() + quoted_key(key) + " is " + dump_safe(value) + " but the supported "
           "FlowMatchEulerDiscrete path requires " + std::string(wanted);
}

std::string out_of_supported_range(std::string_view key, const Json& value, std::string_view wanted) {
    return prefix() + quoted_key(key) + " is " + dump_safe(value) + " but the supported schedule requires " +
           std::string(wanted);
}

std::expected<std::int64_t, std::string> required_integer(const Json& config, std::string_view key) {
    if (!config.contains(key)) { return std::unexpected(absent_key(key)); }
    const Json& value = config.at(key);
    // is_number_integer, not is_number: 256 is the promise, 256.0 is a different one (and a float the
    // reader quietly truncated would let an artifact that never declared an integer bind).
    if (!value.is_number_integer()) { return std::unexpected(wrong_type(key, value, "a JSON integer")); }
    return value.get<std::int64_t>();
}

std::expected<double, std::string> required_float(const Json& config, std::string_view key) {
    if (!config.contains(key)) { return std::unexpected(absent_key(key)); }
    const Json& value = config.at(key);
    // is_number_float, not is_number: the document declares a float, and a non-finite one is refused
    // because it would silently make mu (and with it every sigma) NaN or inf.
    if (!value.is_number_float()) { return std::unexpected(wrong_type(key, value, "a JSON float")); }
    const double number = value.get<double>();
    if (!std::isfinite(number)) { return std::unexpected(out_of_supported_range(key, value, "a finite value")); }
    return number;
}

std::expected<void, std::string> required_boolean(const Json& config, std::string_view key,
                                                  const bool expected) {
    if (!config.contains(key)) { return std::unexpected(absent_key(key)); }
    const Json& value = config.at(key);
    // is_boolean, not a truthiness test: 1 and 0 are numbers, and a mode flag the artifact spells as a
    // number is not the flag this path verified.
    if (!value.is_boolean()) { return std::unexpected(wrong_type(key, value, "a JSON boolean")); }
    if (value.get<bool>() != expected) {
        return std::unexpected(wrong_value(key, value, expected ? "true" : "false"));
    }
    return {};
}

std::expected<void, std::string> required_text(const Json& config, std::string_view key,
                                               std::string_view expected) {
    if (!config.contains(key)) { return std::unexpected(absent_key(key)); }
    const Json& value = config.at(key);
    if (!value.is_string()) { return std::unexpected(wrong_type(key, value, "a JSON string")); }
    if (value.get<std::string>() != expected) {
        return std::unexpected(wrong_value(key, value, quoted_key(expected)));
    }
    return {};
}

// The keys the supported path has verified, and nothing else. "_diffusers_version" travels with the
// checkpoint's document as provenance and is never judged (a version bump is not a scheme change);
// any OTHER key would let the artifact declare a mode the schedule here does not implement.
bool known_key(std::string_view key) {
    constexpr std::string_view known[] = {
        "_class_name",           "_diffusers_version",   "base_image_seq_len",
        "base_shift",            "max_image_seq_len",    "max_shift",
        "num_train_timesteps",   "shift_terminal",       "shift",
        "use_dynamic_shifting",  "time_shift_type",      "use_karras_sigmas",
        "use_beta_sigmas",       "use_exponential_sigmas", "invert_sigmas",
        "stochastic_sampling",
    };
    for (const std::string_view candidate : known) {
        if (candidate == key) { return true; }
    }
    return false;
}

// --- the "dit" component document -------------------------------------------------------------
//
// The dit config is the checkpoint's own transformer/config.json and carries the whole transformer
// architecture, so these helpers differ from the scheduler ones in one deliberate way: no key allowlist
// (the recipe's parameter contract owns the rest) and a prefix naming the document the operator can
// match against --image-artifact. The strictness that does apply is the same: no coercion, no defaults.
constexpr std::string_view kDitDocument    = "the dit config ";
constexpr std::string_view kRopeAxesKey    = "axes_dims_rope";
constexpr std::string_view kRopeHeadDimKey = "attention_head_dim";
constexpr std::string_view kRopeThetaKey   = "rope_theta";

std::string dit_absent_key(std::string_view key) {
    return std::string(kDitDocument) + "has no " + quoted_key(key) +
           ", which the DiT rope rotation reads from the artifact";
}

std::string dit_wrong_type(std::string_view key, const Json& value, std::string_view wanted) {
    return std::string(kDitDocument) + quoted_key(key) + " is " + dump_safe(value) + " but must be " +
           std::string(wanted) + "; the reader does not coerce a JSON type";
}

std::expected<std::int64_t, std::string> dit_required_integer(const Json& config, std::string_view key) {
    if (!config.contains(key)) { return std::unexpected(dit_absent_key(key)); }
    const Json& value = config.at(key);
    if (!value.is_number_integer()) { return std::unexpected(dit_wrong_type(key, value, "a JSON integer")); }
    return value.get<std::int64_t>();
}

// A refusal about the axes list always names the key (the operator greps for it), prints the document's
// own declaration, and says which axis is the problem.
std::string dit_axes_refusal(const Json& axes, std::string_view what) {
    return std::string(kDitDocument) + quoted_key(kRopeAxesKey) + " is " + dump_safe(axes) + " but the DiT "
           "rope rotation requires " + std::string(what);
}

std::string theta_authority() {
    return "the supported Diffusers Qwen-Image pipeline's implementation default " +
           std::to_string(static_cast<double>(kQwenImageRopeTheta)) +
           " (src/models/qwen_image/rope.h, kQwenImageRopeTheta)";
}

// Theta is this build's pipeline implementation default and the artifact is never ASKED to declare one --
// the shipped transformer/config.json carries no "rope_theta" key at all (verified against the live
// artifact's own "dit" component), and no key was invented so the artifact would look like the source of a
// value it never declared. But an artifact that DOES declare one is stating which rotary scheme its weights
// were trained under, and the emitter would rotate them at the code's base regardless: that is a silent
// numerical divergence between the artifact's declaration and the code's behaviour, the same class of hole
// this task exists to close (the scheduler side refuses a foreign "_class_name" or an unverified mode flag
// for exactly this reason, and this lane's text tower requires its rope_theta from the artifact and refuses
// without it, image_weights.cpp). So a declared theta is VERIFIED, never ADOPTED: absent is the normal
// case, present must be a JSON float equal to the code's value, and anything else refuses by key and value
// and names whose value it is compared against.
std::expected<void, std::string> dit_theta_agrees(const Json& config) {
    if (!config.contains(kRopeThetaKey)) { return {}; }
    const Json& value = config.at(kRopeThetaKey);
    if (!value.is_number_float()) {
        return std::unexpected(
            dit_wrong_type(kRopeThetaKey, value, "a JSON float equal to " + theta_authority()));
    }
    const double declared = value.get<double>();
    // Before any dump(): nlohmann's dump() THROWS on a non-finite number, so a NaN or inf declaration must
    // not reach it. Non-finite declares no rotary scheme either way.
    if (!std::isfinite(declared)) {
        return std::unexpected(std::string(kDitDocument) + quoted_key(kRopeThetaKey) +
                               " is not a finite number, so it declares no rotary scheme this build could "
                               "compare with " + theta_authority());
    }
    if (declared != static_cast<double>(kQwenImageRopeTheta)) {
        return std::unexpected(std::string(kDitDocument) + quoted_key(kRopeThetaKey) + " is " +
                               value.dump() + " but this build rotates the DiT with " + theta_authority() +
                               ": those weights were trained under a different rotary scheme, so the "
                               "artifact is refused rather than drawn at a base it does not declare. "
                               "Changing theta is a deliberate change of the supported pipeline "
                               "implementation, not an artifact field.");
    }
    return {};
}

// --- the "vae" component document -------------------------------------------------------------------
//
// Same rule as the dit document above: no key allowlist (the recipe's parameter contract owns the keys
// this route does not read), and the same strictness about the keys it does -- no coercion across JSON
// types, no default for an absent key, and every refusal names the key and prints the value exactly as
// the artifact declared it. What is different here is that some keys are VERIFIED rather than ADOPTED:
// the graph in image_pipeline.cu runs one decoder chain, so an artifact that declares a different
// resnet count, a different number of doublings, a non-residual block, attention outside the mid block
// or another temporal scheme describes weights this build cannot decode, and the alternative to
// refusing is drawing them through a chain they were not trained under -- the silent divergence this
// lane refuses everywhere else (dit_theta_agrees above, the scheduler's mode flags).

constexpr std::string_view kVaeDocument = "the vae config ";

// The chain as THIS build runs it, read off image_pipeline.cu: five up blocks whose last one does not
// upsample (four 2x doublings), three resnets per up block (dup_up's resnets.0/.1/.2 at :329-332), and
// the live-buffer inventory of each stage (VaeStage::planes). An artifact's claim about any of these is
// compared against them. The block and resnet counts are the header's, because the binding that resolves
// the operand shapes and the graph that runs them have to read the same numbers this reader verifies
// against; the per-stage plane counts are this reader's alone -- nothing but the topology it returns
// consumes them.
constexpr std::size_t   kVaeUpBlocks           = kQwenImageVaeUpBlocks;
constexpr std::uint32_t kVaeResnetsPerUpBlock  = kQwenImageVaeResnetsPerUpBlock;
constexpr std::uint32_t kVaeMidStagePlanes  = 5;
constexpr std::uint32_t kVaeUpStagePlanes   = 3;
// The encoder blocks that carry a temporal declaration: every up block except the last one (the last
// neither downsamples nor upsamples), each paired with the decoder's up block kQwenImageVaeUpBlocks - 2 -
// index. The claim itself is compared against the graph's own scheme -- kQwenImageVaeUpBlockTemporal in
// the header, which is what image_pipeline.cu's dup_up() passes to dupup3d -- so there is one table of
// what this build runs, not one per file.
constexpr std::size_t kVaeEncoderTemporalBlocks = kQwenImageVaeUpBlocks - 1;

// No plausible convolution width, and far below an int64 overflow of any product made from it: an
// absurd declaration is refused by name instead of wrapping into a channel count the graph would
// allocate and then fail on.
constexpr std::int64_t kVaeLargestPlausibleValue = 1LL << 20;

std::string vae_absent_key(std::string_view key) {
    return std::string(kVaeDocument) + "has no " + quoted_key(key) +
           ", which the decoder topology this build runs is read from";
}

std::string vae_wrong_type(std::string_view key, const Json& value, std::string_view wanted) {
    return std::string(kVaeDocument) + quoted_key(key) + " is " + dump_safe(value) + " but must be " +
           std::string(wanted) + "; the reader does not coerce a JSON type";
}

// A refusal about a claim the graph cannot honour: the key, the declared value, and what the chain
// running these weights actually requires.
std::string vae_wrong_claim(std::string_view key, const Json& value, std::string_view required) {
    return std::string(kVaeDocument) + quoted_key(key) + " is " + dump_safe(value) + " but the decoder "
           "chain this build runs requires " + std::string(required) + ": this artifact would be decoded "
           "through a chain its weights were not trained under, so it is refused rather than drawn.";
}

std::expected<std::int64_t, std::string> vae_required_integer(const Json& config, std::string_view key) {
    if (!config.contains(key)) { return std::unexpected(vae_absent_key(key)); }
    const Json& value = config.at(key);
    // is_number_integer, not is_number: 144 is the promise, 144.0 is a different one.
    if (!value.is_number_integer()) { return std::unexpected(vae_wrong_type(key, value, "a JSON integer")); }
    const std::int64_t declared = value.get<std::int64_t>();
    if (declared >= kVaeLargestPlausibleValue || declared <= -kVaeLargestPlausibleValue) {
        // The two ends are EXCLUSIVE, because that is what the test above refuses: a declaration exactly
        // equal to either bound is refused, so the message has to say "more than / less than" rather than
        // "between ... and", or the value it prints satisfies the bound it is being refused for.
        return std::unexpected(std::string(kVaeDocument) + quoted_key(key) + " is " + dump_safe(value) +
                               " but must be more than " + std::to_string(-kVaeLargestPlausibleValue) +
                               " and less than " + std::to_string(kVaeLargestPlausibleValue) +
                               " (neither end included: a declaration equal to either is refused too)");
    }
    return declared;
}

std::expected<void, std::string> vae_required_boolean(const Json& config, std::string_view key,
                                                      const bool required) {
    if (!config.contains(key)) { return std::unexpected(vae_absent_key(key)); }
    const Json& value = config.at(key);
    // is_boolean, not a truthiness test: 1 and 0 are numbers, and a scheme flag the artifact spells as a
    // number is not the flag this route verified.
    if (!value.is_boolean()) { return std::unexpected(vae_wrong_type(key, value, "a JSON boolean")); }
    if (value.get<bool>() != required) {
        return std::unexpected(vae_wrong_claim(key, value, required ? "true" : "false"));
    }
    return {};
}

// One array of JSON integers, each a positive channel multiplier: "dim_mult". An entry written 8.0 or
// "8" is a different promise than the width the convolutions are built from, and the reader coerces
// nothing.
std::expected<std::vector<std::int64_t>, std::string> vae_positive_integer_array(const Json& config,
                                                                                std::string_view key,
                                                                                const std::size_t expected_size) {
    if (!config.contains(key)) { return std::unexpected(vae_absent_key(key)); }
    const Json& value = config.at(key);
    if (!value.is_array() || value.size() != expected_size) {
        return std::unexpected(std::string(kVaeDocument) + quoted_key(key) + " is " + dump_safe(value) +
                               " but must be a JSON array of exactly " + std::to_string(expected_size) +
                               " entries, which is the number of " +
                               (expected_size == kVaeUpBlocks ? "levels the decoder's up chain runs"
                                                              : "entries this route reads") +
                               "; the reader does not coerce a JSON type");
    }
    std::vector<std::int64_t> out;
    out.reserve(expected_size);
    for (std::size_t i = 0; i < expected_size; ++i) {
        const Json& entry = value[i];
        if (!entry.is_number_integer()) {
            return std::unexpected(
                std::string(kVaeDocument) + quoted_key(key) + " holds " + dump_safe(entry) + " at index " +
                std::to_string(i) + " but every entry must be a JSON integer; the reader does not coerce a "
                "JSON type");
        }
        const std::int64_t declared = entry.get<std::int64_t>();
        if (declared <= 0 || declared >= kVaeLargestPlausibleValue) {
            return std::unexpected(std::string(kVaeDocument) + quoted_key(key) + " holds " +
                                   dump_safe(entry) + " at index " + std::to_string(i) +
                                   " but every entry must be a positive channel multiplier below " +
                                   std::to_string(kVaeLargestPlausibleValue));
        }
        out.push_back(declared);
    }
    return out;
}

std::expected<void, std::string> vae_finite_number_array_of(const Json& config, std::string_view key,
                                                           const std::size_t expected_size) {
    if (!config.contains(key)) { return std::unexpected(vae_absent_key(key)); }
    const Json& value = config.at(key);
    if (!value.is_array()) { return std::unexpected(vae_wrong_type(key, value, "a JSON array")); }
    if (value.size() != expected_size) {
        return std::unexpected(std::string(kVaeDocument) + quoted_key(key) + " has " +
                               std::to_string(value.size()) + " entries but the latent width " +
                               quoted_key("z_dim") + " is " + std::to_string(expected_size) +
                               ": the per-channel statistics and the latent plane are one fact, and an "
                               "artifact that disagrees with itself is refused");
    }
    for (const Json& entry : value) {
        // is_number, matching the pair's existing reader (read_vae_denorm, which is what consumes these
        // values): the route takes a number per channel, and a non-number or a non-finite one normalizes
        // nothing -- it silently makes every latent NaN.
        if (!entry.is_number()) {
            return std::unexpected(std::string(kVaeDocument) + quoted_key(key) + " holds " +
                                   dump_safe(entry) + " but every entry must be a JSON number");
        }
        if (!std::isfinite(entry.get<double>())) {
            return std::unexpected(std::string(kVaeDocument) + quoted_key(key) +
                                   " holds a non-finite number, which would denormalize every latent to NaN");
        }
    }
    return {};
}

std::expected<void, std::string> vae_empty_array(const Json& config, std::string_view key,
                                                std::string_view required) {
    if (!config.contains(key)) { return std::unexpected(vae_absent_key(key)); }
    const Json& value = config.at(key);
    if (!value.is_array()) { return std::unexpected(vae_wrong_type(key, value, "a JSON array")); }
    if (!value.empty()) { return std::unexpected(vae_wrong_claim(key, value, required)); }
    return {};
}

// The graph's dup_up() takes its dupup3d shortcut factor per up block, and the pure-2D image route
// implements exactly one scheme. The checkpoint declares the temporal behaviour on the ENCODER's blocks,
// so the decoder's factors are that list mirrored -- compared entry by entry against what the graph
// passes, because an artifact that declares another scheme would have its shortcut duplicated wrong.
std::expected<void, std::string> vae_temporal_scheme(const Json& config) {
    constexpr std::string_view key = "temperal_downsample"; // the checkpoint's own spelling
    if (!config.contains(key)) { return std::unexpected(vae_absent_key(key)); }
    const Json& value = config.at(key);
    if (!value.is_array() || value.size() != kVaeEncoderTemporalBlocks) {
        return std::unexpected(
            std::string(kVaeDocument) + quoted_key(key) + " is " + dump_safe(value) + " but must be a JSON "
            "array of exactly " + std::to_string(kVaeEncoderTemporalBlocks) +
            " booleans, one per encoder block that downsamples");
    }
    for (std::size_t i = 0; i < kVaeEncoderTemporalBlocks; ++i) {
        const Json& entry = value[i];
        if (!entry.is_boolean()) {
            return std::unexpected(std::string(kVaeDocument) + quoted_key(key) + " holds " + dump_safe(entry) +
                                   " at index " + std::to_string(i) + " but every entry must be a JSON boolean");
        }
        // The encoder block at index i is the decoder's up block (kVaeUpBlocks - 2 - i): the decoder walks
        // the chain from the deepest level outward, and the graph's dup_up() passes this build's own
        // scheme (kQwenImageVaeUpBlockTemporal) as the dupup3d shortcut factor there.
        const std::size_t up_block = kVaeUpBlocks - 2 - i;
        const bool required        = kQwenImageVaeUpBlockTemporal[up_block];
        if (entry.get<bool>() != required) {
            return std::unexpected(vae_wrong_claim(
                key, entry, std::string(required ? "true" : "false") + " at index " + std::to_string(i) +
                                ", which is the decoder's up block " + std::to_string(up_block) +
                                " whose dupup3d shortcut factor this build implements as " +
                                (required ? "2" : "1")));
        }
    }
    return {};
}

} // namespace

std::expected<SchedulerConstants, std::string> read_scheduler_config(const nlohmann::json& config) {
    if (!config.is_object()) {
        return std::unexpected("the artifact declares no scheduler config object to read");
    }
    for (const auto& [key, value] : config.items()) {
        (void)value;
        if (!known_key(key)) {
            return std::unexpected(
                prefix() + quoted_key(key) +
                " is a key the supported FlowMatchEulerDiscrete path has never verified, so the "
                "schedule this build produces is not the schedule the artifact declares");
        }
    }

    // The sampler class is part of what the artifact declares: this build implements the Euler flow
    // match scheduler and nothing else, so a document naming another one is a different scheme.
    if (auto verdict = required_text(config, "_class_name", "FlowMatchEulerDiscreteScheduler");
        !verdict.has_value()) {
        return std::unexpected(verdict.error());
    }
    if (config.contains("_diffusers_version") && !config.at("_diffusers_version").is_string()) {
        return std::unexpected(wrong_type("_diffusers_version", config.at("_diffusers_version"),
                                          "a JSON string"));
    }

    // Every mode flag, not only the six numbers: each of these selects a different sigma construction
    // in the upstream scheduler, and this path implements exactly one of them.
    for (const auto& [key, expected] : {std::pair<std::string_view, bool>{
                 "use_dynamic_shifting", true},
             {"use_karras_sigmas", false}, {"use_beta_sigmas", false}, {"use_exponential_sigmas", false},
             {"invert_sigmas", false}, {"stochastic_sampling", false}}) {
        if (auto verdict = required_boolean(config, key, expected); !verdict.has_value()) {
            return std::unexpected(verdict.error());
        }
    }
    if (auto verdict = required_text(config, "time_shift_type", "exponential"); !verdict.has_value()) {
        return std::unexpected(verdict.error());
    }

    SchedulerConstants out;
    if (auto value = required_float(config, "shift"); !value.has_value() || *value != 1.0) {
        if (!value.has_value()) { return std::unexpected(value.error()); }
        return std::unexpected(
            wrong_value("shift", config.at("shift"), "1.0 (this path applies no static shift)"));
    }
    if (auto value = required_integer(config, "base_image_seq_len"); !value.has_value()) {
        return std::unexpected(value.error());
    } else {
        out.base_image_seq_len = *value;
    }
    if (auto value = required_integer(config, "max_image_seq_len"); !value.has_value()) {
        return std::unexpected(value.error());
    } else {
        out.max_image_seq_len = *value;
    }
    if (auto value = required_integer(config, "num_train_timesteps"); !value.has_value()) {
        return std::unexpected(value.error());
    } else {
        out.num_train_timesteps = *value;
    }
    if (auto value = required_float(config, "base_shift"); !value.has_value()) {
        return std::unexpected(value.error());
    } else {
        out.base_shift = *value;
    }
    if (auto value = required_float(config, "max_shift"); !value.has_value()) {
        return std::unexpected(value.error());
    } else {
        out.max_shift = *value;
    }
    if (auto value = required_float(config, "shift_terminal"); !value.has_value()) {
        return std::unexpected(value.error());
    } else {
        out.shift_terminal = *value;
    }

    // The three values the schedule divides by: a zero or a reversed pair would turn mu into inf/NaN
    // and every sigma with it, so they are refused by name rather than computed with.
    if (out.base_image_seq_len <= 0) {
        return std::unexpected(
            out_of_supported_range("base_image_seq_len", config.at("base_image_seq_len"), "a positive token count"));
    }
    if (out.max_image_seq_len <= out.base_image_seq_len) {
        return std::unexpected(
            out_of_supported_range("max_image_seq_len", config.at("max_image_seq_len"),
                         "a token count above base_image_seq_len, which is " +
                             std::to_string(out.base_image_seq_len)));
    }
    if (out.num_train_timesteps <= 0) {
        return std::unexpected(
            out_of_supported_range("num_train_timesteps", config.at("num_train_timesteps"), "a positive count"));
    }
    if (!(out.shift_terminal >= 0.0 && out.shift_terminal < 1.0)) {
        return std::unexpected(out_of_supported_range("shift_terminal", config.at("shift_terminal"),
                                            "a value in [0, 1): the stretch divides by 1 - shift_terminal"));
    }

    // The two shifts and the timestep count are bounded, not merely checked for finiteness (which
    // required_float above already did), because nothing downstream looks at what the schedule emits.
    // calculate_shift extrapolates LINEARLY in the latent token count
    // (mu = seq_len * m + b, m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)), and
    // build_schedule then runs std::exp(mu) on it and multiplies the sigmas by num_train_timesteps. An
    // artifact whose declared window is too narrow to carry the shift it also declares -- the mistyped
    // "base_image_seq_len": 256, "max_image_seq_len": 257 a hand-edited metadata file produces, or a
    // "base_shift"/"max_shift" beyond log(DBL_MAX) ~= 709.78 -- makes mu enormous: exp() returns inf, the
    // float32 cast of it feeds emu / (emu + inv), and EVERY sigma and timestep comes out NaN. The job is
    // then DRAWN as garbage rather than refused, with no operator-visible reason -- and before this branch
    // those two numbers were compiled constants, so the hole is the artifact's, not the arithmetic's. The
    // file already refuses every other unrepresentable declaration this way (kVaeLargestPlausibleValue
    // bounds every VAE integer, and the divisor checks above are "refused by name rather than computed
    // with"); the same rule and the same wording apply here, by name, before a single allocation.
    //
    // Why these two numbers: exp() overflows double to inf above log(DBL_MAX) = 709.78, so mu must stay
    // under it for exp(mu) to be finite at all; at the other end the float32 cast of exp(mu) flushes to
    // 0 below about -87.3, and the first input sigma is exactly 1.0, so emu / (emu + (1/sigma - 1)) would
    // divide 0 by 0 and produce NaN there too. Both constants sit inside those bounds with room, and they
    // are far above any shift any checkpoint declares (the shipped document declares 0.5 and 0.9), so an
    // honest document never meets them -- a refusal here is a mis-authored metadata file, and the message
    // is the reason the operator can read.
    constexpr double kLargestFiniteMu = 700.0;
    constexpr double kSmallestFiniteMu = -80.0;
    for (const std::string_view key : {"base_shift", "max_shift"}) {
        const double declared = key == "base_shift" ? out.base_shift : out.max_shift;
        // Written as the negation of the inclusive test so a NaN that slipped past required_float refuses
        // too, rather than comparing its way into the schedule.
        if (!(declared >= kSmallestFiniteMu && declared <= kLargestFiniteMu)) {
            return std::unexpected(
                out_of_supported_range(key, config.at(key),
                                       "a shift at least " + std::to_string(kSmallestFiniteMu) +
                                           " and at most " + std::to_string(kLargestFiniteMu) +
                                           ", which is the range std::exp stays finite in and the float32 "
                                           "cast of it never flushes to zero: outside it every sigma and "
                                           "timestep of the schedule would be NaN and the picture would be "
                                           "drawn rather than refused"));
        }
    }
    // The window, on the same terms: mu is LINEAR in the token count, so holding it inside that range at
    // both ends of the span the lane can hand the scheduler holds it there for every count in between.
    // ops::kImageJointMaxTokens is the product's one joint-sequence ceiling (include/infernix/ops/
    // image_joint_limits.h) and a job's canvas tokens are a part of the joint sequence, so 20480 is the
    // largest count the schedule is ever asked for. The LOW end is deliberately below what the lane ever
    // asks for: the smallest canvas the serving layer admits is kVramCanvasFloorPx (256 px a side,
    // src/serve/image_generation_service.cpp:220), which at the 8x latent downsample and the 2x2 patchify
    // is 256 latent tokens, so the true floor of the probe is 256 and 1 is one probe the route never
    // needs. It stays because mu is LINEAR in the token count: holding it inside the range at 1 and at
    // 20480 brackets every count in [1, 20480], which strictly CONTAINS the counts the lane can build, and
    // an extra point on a monotone segment can only ever refuse more -- never admit a schedule the 256
    // probe would have caught. Over-refusal here means one mis-authored metadata document that no honest
    // document meets (the shipped window is 256/8192 with shifts 0.5/0.9, whose mu(1) is 0.4871), and
    // under-refusal means a drawn picture of NaNs; the conservatism is the side to err on. A
    // document refused here is one whose declared window cannot keep a real job's schedule finite, which
    // is exactly the case no later stage checks.
    for (const std::size_t tokens :
         {std::size_t{1}, static_cast<std::size_t>(::infernix::ops::kImageJointMaxTokens)}) {
        const double mu = calculate_shift(out, tokens);
        if (!(mu >= kSmallestFiniteMu && mu <= kLargestFiniteMu)) {
            return std::unexpected(
                prefix() + quoted_key("base_image_seq_len") + " = " +
                std::to_string(out.base_image_seq_len) + " with " + quoted_key("max_image_seq_len") +
                " = " + std::to_string(out.max_image_seq_len) + " and " + quoted_key("base_shift") +
                " = " + std::to_string(out.base_shift) + " with " + quoted_key("max_shift") + " = " +
                std::to_string(out.max_shift) + " put the schedule's mu at " + std::to_string(mu) +
                " for a " + std::to_string(tokens) + "-token sequence, outside the range [" +
                std::to_string(kSmallestFiniteMu) + ", " + std::to_string(kLargestFiniteMu) +
                "] the schedule can compute: exp(mu) would overflow to inf (or flush to 0 at the other "
                "end), every sigma and timestep would come out NaN, and the picture would be drawn as "
                "garbage instead of refused. A window this narrow cannot cover the token counts this "
                "lane builds -- the two declared token counts and the two declared shifts are one line, "
                "and all four have to agree on it.");
        }
    }
    // num_train_timesteps has an upper bound for the same reason: it is a JSON integer, and build_schedule
    // computes timesteps as sigmas * num_train_timesteps in float32, so an enormous declaration reaches
    // +-inf and the schedule is garbage the same way. The bound is this file's own idiom -- an absurd
    // declaration is refused by name, not carried into arithmetic to see what happens (the shipped
    // document declares 1000).
    constexpr std::int64_t kLargestPlausibleTrainTimesteps = 1LL << 20;
    if (out.num_train_timesteps > kLargestPlausibleTrainTimesteps) {
        return std::unexpected(
            out_of_supported_range("num_train_timesteps", config.at("num_train_timesteps"),
                                   "a positive count at most " +
                                       std::to_string(kLargestPlausibleTrainTimesteps) +
                                       ": the schedule multiplies its sigmas by it in float32, so a larger "
                                       "declaration would overflow the timesteps to inf and the job would "
                                       "be drawn rather than refused"));
    }
    return out;
}

std::expected<DitRopeAxes, std::string> read_dit_rope_axes(const nlohmann::json& config) {
    if (!config.is_object()) {
        return std::unexpected(std::string(kDitDocument) +
                               "is not a JSON object, so the DiT rope axes cannot be read from the artifact");
    }
    // The rotary scheme first, the arithmetic after -- the same order the scheduler reader uses, where a
    // foreign "_class_name" or an unverified mode flag refuses before any number is read. A declared theta
    // this build does not implement is a different scheme, so nothing about the axes can rescue it.
    if (auto verdict = dit_theta_agrees(config); !verdict.has_value()) {
        return std::unexpected(verdict.error());
    }
    std::int64_t head_dim = 0;
    if (auto value = dit_required_integer(config, kRopeHeadDimKey); !value.has_value()) {
        return std::unexpected(value.error());
    } else {
        head_dim = *value;
    }
    if (!config.contains(kRopeAxesKey)) { return std::unexpected(dit_absent_key(kRopeAxesKey)); }
    const Json& axes = config.at(kRopeAxesKey);
    if (!axes.is_array() || axes.size() != 3) {
        return std::unexpected(dit_axes_refusal(
            axes, "a JSON array of exactly the three axes the DiT rotates on (frame, height, width)"));
    }

    // The three values are read as the JSON integers they promise to be before any arithmetic: an axis
    // written 16.0 or "16" is a different promise than the channel count the rotation divides by, and the
    // reader coerces nothing.
    std::int64_t value[3] = {0, 0, 0};
    std::int64_t sum      = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        const Json& axis = axes[i];
        if (!axis.is_number_integer()) {
            return std::unexpected(
                dit_axes_refusal(axes, "three JSON integers, but axis " + std::to_string(i) + " is " +
                                       dump_safe(axis) + " and the reader does not coerce a JSON type"));
        }
        value[i] = axis.get<std::int64_t>();
        // An absurd channel count could overflow the sum below and turn the refusal into a nonsense
        // number, so it is refused outright -- by name and value. No DiT head is remotely this wide, and
        // three values under this bound cannot overflow an int64 between them.
        constexpr std::int64_t kLargestPlausibleHead = 1LL << 32;
        if (value[i] >= kLargestPlausibleHead || value[i] <= -kLargestPlausibleHead) {
            return std::unexpected(dit_axes_refusal(
                axes, "each axis below " + std::to_string(kLargestPlausibleHead) + " channels, but axis " +
                      std::to_string(i) + " is " + std::to_string(value[i])));
        }
        sum += value[i];
    }

    // The three axes together rotate the whole head: that is the arithmetic fact the rope table rests on
    // (its width is sum / 2 == attention_head_dim / 2 interleaved (cos, sin) pairs, which is what
    // ops::rope_c64 uploads), so a document that adds up to anything else is refused by both keys and
    // both numbers. A head that is not a positive even channel count cannot be rotated either.
    if (head_dim <= 0 || (head_dim % 2) != 0) {
        return std::unexpected(std::string(kDitDocument) + quoted_key(kRopeHeadDimKey) + " is " +
                               std::to_string(head_dim) + " but the DiT head must be a positive even count "
                               "of channels for " + quoted_key(kRopeAxesKey) + " = " + axes.dump() +
                               " to rotate it");
    }
    if (sum != head_dim) {
        return std::unexpected(std::string(kDitDocument) + quoted_key(kRopeAxesKey) + " = " + axes.dump() +
                               " sums to " + std::to_string(sum) + " but " + quoted_key(kRopeHeadDimKey) +
                               " is " + std::to_string(head_dim) + ": the three axes have to add up to the "
                               "head they rotate, so the rope table is attention_head_dim / 2 interleaved "
                               "(cos, sin) pairs wide");
    }

    // Each axis is a count of head channels it rotates, so it must be positive and even: the emitter
    // writes one interleaved (cos, sin) pair per two channels, and a zero, a negative or an odd axis
    // would silently truncate the row. Checked after the sum so a document that cannot describe this
    // architecture is refused for the arithmetic reason first, naming both keys.
    DitRopeAxes out;
    for (std::size_t i = 0; i < 3; ++i) {
        if (value[i] <= 0 || (value[i] % 2) != 0) {
            return std::unexpected(dit_axes_refusal(axes,
                "positive even channel counts, but axis " + std::to_string(i) + " is " + std::to_string(value[i])));
        }
        out.axes[i] = static_cast<std::uint32_t>(value[i]);
    }
    return out;
}

std::expected<VaeTopology, std::string> read_vae_topology(const nlohmann::json& config) {
    if (!config.is_object()) {
        return std::unexpected(std::string(kVaeDocument) +
                               "is not a JSON object, so the decoder topology cannot be read from the artifact");
    }

    // The scheme claims this build implements unconditionally come first, before a single width is
    // adopted -- the same order the scheduler reader uses, where a foreign "_class_name" or an
    // unverified mode flag refuses before any sigma is read. An artifact that declares a chain the CUDA
    // graph does not run describes weights this build cannot decode, and the alternative to refusing is
    // drawing them through a chain they were not trained under.
    if (auto verdict = vae_required_boolean(config, "is_residual", true); !verdict.has_value()) {
        return std::unexpected(verdict.error());
    }
    if (auto verdict = vae_empty_array(
            config, "attn_scales",
            "the empty array: this build places its one attention in the decoder's mid block and runs no "
            "per-block attention, so a listed scale describes attention that never runs");
        !verdict.has_value()) {
        return std::unexpected(verdict.error());
    }
    if (auto verdict = vae_temporal_scheme(config); !verdict.has_value()) {
        return std::unexpected(verdict.error());
    }
    if (auto value = vae_required_integer(config, "num_res_blocks"); !value.has_value()) {
        return std::unexpected(value.error());
    } else if (*value != static_cast<std::int64_t>(kVaeResnetsPerUpBlock) - 1) {
        // Three resnets per up block is what dup_up() runs, and three is num_res_blocks + 1 in the
        // diffusers chain's convention -- a declaration of any other count is a chain with a resnet
        // missing or an extra one, whose bound convolutions this build would hand the wrong weights.
        return std::unexpected(vae_wrong_claim(
            "num_res_blocks", config.at("num_res_blocks"),
            std::to_string(static_cast<std::int64_t>(kVaeResnetsPerUpBlock) - 1) +
                ", because image_pipeline.cu's dup_up() runs " + std::to_string(kVaeResnetsPerUpBlock) +
                " resnets per up block (num_res_blocks + 1)"));
    }

    // The widths, read as the JSON integers they promise to be: the reader coerces nothing, so a 144.0
    // or a "144" refuses naming the key and printing the declaration as it stands.
    auto levels = vae_positive_integer_array(config, "dim_mult", kVaeUpBlocks);
    if (!levels.has_value()) { return std::unexpected(levels.error()); }
    // The last level is the shallowest one, the width norm_out normalizes at and conv_out convolves:
    // that is decoder_base_dim, so the multiplier beside it has to say 1 or the artifact disagrees with
    // itself about its own output width.
    if ((*levels)[0] != 1) {
        return std::unexpected(std::string(kVaeDocument) + quoted_key("dim_mult") + " holds " +
                               dump_safe(config.at("dim_mult")[0]) + " at index 0 but the shallowest level "
                               "is the width " + quoted_key("decoder_base_dim") + " names and the final "
                               "convolution runs at, so its multiplier must be 1");
    }

    std::int64_t base_dim = 0;
    if (auto value = vae_required_integer(config, "decoder_base_dim"); !value.has_value()) {
        return std::unexpected(value.error());
    } else if (*value <= 0) {
        return std::unexpected(
            vae_wrong_claim("decoder_base_dim", config.at("decoder_base_dim"), "a positive channel count"));
    } else {
        base_dim = *value;
    }
    std::int64_t latent_channels = 0;
    if (auto value = vae_required_integer(config, "z_dim"); !value.has_value()) {
        return std::unexpected(value.error());
    } else if (*value <= 0) {
        return std::unexpected(vae_wrong_claim("z_dim", config.at("z_dim"),
                                               "a positive latent channel count"));
    } else {
        latent_channels = *value;
    }
    std::int64_t out_channels = 0;
    if (auto value = vae_required_integer(config, "out_channels"); !value.has_value()) {
        return std::unexpected(value.error());
    } else if (*value <= 0) {
        return std::unexpected(
            vae_wrong_claim("out_channels", config.at("out_channels"), "a positive channel count"));
    } else {
        out_channels = *value;
    }
    std::int64_t scale_factor_spatial = 0;
    if (auto value = vae_required_integer(config, "scale_factor_spatial"); !value.has_value()) {
        return std::unexpected(value.error());
    } else if (*value <= 0) {
        return std::unexpected(vae_wrong_claim("scale_factor_spatial", config.at("scale_factor_spatial"),
                                               "a positive number of pixels per latent row"));
    } else {
        scale_factor_spatial = *value;
    }

    // The per-channel statistics ARE the latent width: the route divides each latent plane by its own
    // std, so an artifact whose statistics are any other length than its own latent declaration is
    // refused naming both numbers rather than reading past the end or normalizing nothing.
    if (auto verdict = vae_finite_number_array_of(config, "latents_mean",
                                                 static_cast<std::size_t>(latent_channels));
        !verdict.has_value()) {
        return std::unexpected(verdict.error());
    }
    if (auto verdict = vae_finite_number_array_of(config, "latents_std",
                                                 static_cast<std::size_t>(latent_channels));
        !verdict.has_value()) {
        return std::unexpected(verdict.error());
    }

    // The decoder walks the chain from the deepest level outward, so the level widths are the declared
    // multipliers reversed and scaled by the decoder's own base width: the shipped family's
    // [1, 2, 4, 8, 8] x 144 becomes 1152, 1152, 576, 288, 144 -- the channels every bound convolution
    // carries and the numbers the VRAM ledger charges for.
    //
    // Each factor is bounded on its own by kVaeLargestPlausibleValue, so the int64 product cannot wrap --
    // but that bound says nothing about the PRODUCT, and these widths are stored and priced as
    // std::uint32_t. A hand-authored document with decoder_base_dim 524288 and a dim_mult entry of 8192 --
    // each one under its own bound, so each one passes it -- multiplies to exactly 2^32, which narrows to
    // 0. A zero-channel stage is then bound at zero-width weights and charged ZERO bytes of vae_peak by the
    // ledger, admitting a job whose real peak is gigabytes: precisely the admission blindness this plan
    // exists to close. So every product is refused here, naming both keys, both factors and the product,
    // BEFORE a single narrowing cast happens. The ceiling is the widest channel count the field the ledger
    // multiplies can hold exactly; nothing is widened, because the fields are the contract.
    constexpr std::int64_t kLargestStorableChannels =
        static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max());
    std::vector<std::int64_t> widths(kVaeUpBlocks, 0);
    for (std::size_t b = 0; b < kVaeUpBlocks; ++b) {
        const std::size_t level   = kVaeUpBlocks - 1 - b;
        const std::int64_t channels = base_dim * (*levels)[level];
        if (channels > kLargestStorableChannels) {
            return std::unexpected(std::string(kVaeDocument) + quoted_key("decoder_base_dim") + " = " +
                                   std::to_string(base_dim) + " times " + quoted_key("dim_mult") + "[" +
                                   std::to_string(level) + "] = " + std::to_string((*levels)[level]) + " is " +
                                   std::to_string(channels) + " channels for the decoder's up stage " +
                                   std::to_string(b) + ", more than the " +
                                   std::to_string(kLargestStorableChannels) + " a stage's channel count is "
                                   "stored and priced in: each declaration is plausible on its own, but the "
                                   "chain's width is their PRODUCT, and a stage that did not fit would be "
                                   "bound and admitted at its truncated width");
        }
        widths[b] = channels;
    }

    // The doublings the chain performs are not adopted from the artifact but verified against it: every
    // up block except the last one upsamples, and the pixels per latent row the route divides the
    // request geometry by has to be what those stages actually produce. A declaration of another factor
    // would make the decode emit an image of a size nothing admitted.
    std::uint64_t produced = 1ULL;
    for (std::size_t b = 0; b + 1 < kVaeUpBlocks; ++b) { produced *= 2ULL; }
    if (produced != static_cast<std::uint64_t>(scale_factor_spatial)) {
        return std::unexpected(
            vae_wrong_claim("scale_factor_spatial", config.at("scale_factor_spatial"),
                            std::to_string(produced) + ", which is the only factor the " +
                                std::to_string(kVaeUpBlocks) + " up blocks " + quoted_key("dim_mult") +
                                " declares can produce: " + std::to_string(kVaeUpBlocks - 1) +
                                " of them upsample by 2 and the last one does not"));
    }

    // The casts below are honest because of where their values come from, not by luck: latent_channels,
    // spatial_downscale and out_channels each arrived through vae_required_integer, which refuses anything
    // at or above kVaeLargestPlausibleValue (2^20) -- a single declaration, never a product -- and every
    // width in `widths` was bounded to kLargestStorableChannels just above, by the same argument the
    // product needs and a single bounded value does not.
    VaeTopology out;
    out.latent_channels   = static_cast<std::uint32_t>(latent_channels);
    out.spatial_downscale = static_cast<std::uint32_t>(scale_factor_spatial);
    out.mid_stage = VaeStage{static_cast<std::uint32_t>(widths.front()),
                             static_cast<std::uint32_t>(widths.front()), kVaeMidStagePlanes, false};
    for (std::size_t b = 0; b < kVaeUpBlocks; ++b) {
        const std::int64_t channels_in = (b == 0) ? widths.front() : widths[b - 1];
        out.up_stages.push_back(VaeStage{static_cast<std::uint32_t>(channels_in),
                                         static_cast<std::uint32_t>(widths[b]), kVaeUpStagePlanes,
                                         /*doubles_side=*/b + 1 < kVaeUpBlocks});
    }
    // The final convolution's width is the shallowest level plus the artifact's declared output channels:
    // the same narrowing class as the stage widths, so it is bounded the same way -- and by subtraction,
    // so the bound's own arithmetic cannot overflow. conv_out is the term the ledger charges on the FULL
    // pixel canvas, so a truncated one under-charges a whole image, not one stage.
    if (widths.back() > kLargestStorableChannels - out_channels) {
        // The predicate is on the SUM, so the message prints the sum, the ceiling it is compared against,
        // and the most out_channels could have been under this same shallowest level. Naming only the
        // ceiling left a reader to work out whether the refused value actually exceeded it (and this line
        // once attributed the level width to "decoder_base_dim", which is one of its two factors, not the
        // result): every number a refusal quotes has to be checkable against the refusal.
        return std::unexpected(std::string(kVaeDocument) + quoted_key("out_channels") + " = " +
                               std::to_string(out_channels) + " on top of the shallowest level width " +
                               std::to_string(widths.back()) + " (the " + quoted_key("decoder_base_dim") +
                               " = " + std::to_string(base_dim) + " times the shallowest " +
                               quoted_key("dim_mult") + " entry) is " +
                               std::to_string(widths.back() + out_channels) + " channels for the final "
                               "convolution, more than the " + std::to_string(kLargestStorableChannels) +
                               " it is stored and priced in -- at most " +
                               std::to_string(kLargestStorableChannels - widths.back()) +
                               " " + quoted_key("out_channels") + " fits beside that level -- and that "
                               "stage is charged on the full pixel canvas");
    }
    out.final_channels  = static_cast<std::uint32_t>(widths.back() + out_channels);
    out.denorm_elements = static_cast<std::uint32_t>(latent_channels);
    // weight_resident_bytes is deliberately left zero: it is a calibration of THIS build's allocator and
    // ops, not a declaration of the checkpoint's. The family's binding stamps it, and check_vae_topology
    // refuses an unstamped topology -- an admission decision taken from a zero allowance would admit
    // anything.
    return out;
}

} // namespace infernix::models::qwen_image

#endif // INFERNIX_IMAGE_READS_ARTIFACT_CONSTANTS
