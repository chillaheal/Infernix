// The strict artifact readers behind artifact_constants.h (see that header for the authority table).
// The Qwen-Image-2.1 image lane refuses an artifact instead of guessing: an absent key, a key with the
// wrong JSON type, a mode flag this path does not implement, or a key it has never verified all fail
// the binding, before any CUDA allocation and before any partially initialised model state.

#include "models/qwen_image/artifact_constants.h"

#ifdef INFERNIX_IMAGE_READS_ARTIFACT_CONSTANTS

#include <cmath>
#include <string>
#include <string_view>

namespace infernix::models::qwen_image {
namespace {

using Json = nlohmann::json;

std::string quoted_key(std::string_view text) { return "\"" + std::string(text) + "\""; }

std::string prefix() { return "the scheduler config "; }

// Every refusal names the key it concerns; a deviating value is printed as the artifact declared it,
// so the operator running --image-artifact reads the reason instead of guessing it.
std::string absent_key(std::string_view key) {
    return prefix() + "has no " + quoted_key(key) +
           ", which the supported denoise schedule reads from the artifact";
}

std::string wrong_type(std::string_view key, const Json& value, std::string_view wanted) {
    return prefix() + quoted_key(key) + " is " + value.dump() + " but must be " + std::string(wanted) +
           "; the reader does not coerce a JSON type";
}

std::string wrong_value(std::string_view key, const Json& value, std::string_view wanted) {
    return prefix() + quoted_key(key) + " is " + value.dump() + " but the supported "
           "FlowMatchEulerDiscrete path requires " + std::string(wanted);
}

std::string out_of_supported_range(std::string_view key, const Json& value, std::string_view wanted) {
    return prefix() + quoted_key(key) + " is " + value.dump() + " but the supported schedule requires " +
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
    return out;
}

} // namespace infernix::models::qwen_image

#endif // INFERNIX_IMAGE_READS_ARTIFACT_CONSTANTS
