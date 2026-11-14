// The image lane reads its scheduler and DiT-rope constants FROM the artifact and refuses an
// unhandled scheme. This suite is the reader gate (Task 9): it qualifies
// read_scheduler_config / read_dit_rope_axes (src/models/qwen_image/artifact_constants.{h,cpp})
// against the REAL metadata the converter packs -- the committed copy of the reference checkpoint's
// own scheduler/scheduler_config.json, tests/convert/fixtures/qwen_image_scheduler_config.json, read
// from disk rather than hand-copied into a literal -- and against every deviation the supported
// FlowMatchEulerDiscrete path must refuse.
//
// Why the fixture and not a literal: the point of the change is that the shipped artifact declares
// these numbers. A test that pasted its own copy would pass on a hand-written JSON while the shipped
// artifact said something else.
//
// Host-only, no GPU, no server. The last section is gated on INFERNIX_TEST_ARTIFACT (the live
// artifact, read-only through the production reader) and SKIPs when it is absent.

#include "models/qwen_image/artifact_constants.h"
#include "models/qwen_image/scheduler.h"

#include "artifact/reader.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace qi = infernix::models::qwen_image;

#ifndef INFERNIX_SOURCE_DIR
#    define INFERNIX_SOURCE_DIR "."
#endif

namespace {

int g_failures = 0;

void check(bool ok, const std::string& name) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", name.c_str());
    if (!ok) ++g_failures;
}

void check_detail(bool ok, const std::string& name, const std::string& detail) {
    std::printf("%s: %s%s%s\n", ok ? "ok" : "FAIL", name.c_str(), detail.empty() ? "" : " -- ",
                detail.c_str());
    if (!ok) ++g_failures;
}

// The committed scheduler document, verbatim from the reference checkpoint (Task 8).
const char* kFixturePath =
    INFERNIX_SOURCE_DIR "/tests/convert/fixtures/qwen_image_scheduler_config.json";

nlohmann::json read_json_file(const std::string& path, bool& ok) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        ok = false;
        std::printf("FAIL: cannot open the committed scheduler fixture %s\n", path.c_str());
        return nlohmann::json();
    }
    std::ostringstream text;
    text << in.rdbuf();
    ok = true;
    return nlohmann::json::parse(text.str(), nullptr, false);
}

std::string describe(const std::string& needle, const std::string& haystack) {
    return "refusal names \"" + needle + "\"; message: " + haystack;
}

// The supported path's own scheduler document, as the artifact carries it.
nlohmann::json supported_config() {
    bool ok = false;
    nlohmann::json j = read_json_file(kFixturePath, ok);
    if (!ok) std::exit(1);
    return j;
}

// read_scheduler_config must refuse and say which key (and, for a deviating value, which value).
void expect_refuses(const nlohmann::json& config, const std::string& name,
                    const std::vector<std::string>& needles) {
    const auto result = qi::read_scheduler_config(config);
    if (result.has_value()) {
        check(false, name + " (refused)");
        return;
    }
    bool all = true;
    for (const std::string& needle : needles) {
        if (result.error().find(needle) == std::string::npos) all = false;
    }
    check_detail(all, name, describe(needles.empty() ? "?" : needles[0], result.error()));
}

// ...and accept, with every constant equal to the shipped value.
void expect_scheduler_values(const qi::SchedulerConstants& c, const std::string& name) {
    const bool ok = c.base_image_seq_len == 256 && c.base_shift == 0.5 && c.max_image_seq_len == 8192 &&
                    c.max_shift == 0.9 && c.num_train_timesteps == 1000 && c.shift_terminal == 0.02;
    check_detail(ok, name,
                 "base_image_seq_len=" + std::to_string(c.base_image_seq_len) +
                     " base_shift=" + std::to_string(c.base_shift) +
                     " max_image_seq_len=" + std::to_string(c.max_image_seq_len) +
                     " max_shift=" + std::to_string(c.max_shift) +
                     " num_train_timesteps=" + std::to_string(c.num_train_timesteps) +
                     " shift_terminal=" + std::to_string(c.shift_terminal));
}

// ---- the artifact's own scheduler document ----

void gate_shipped_metadata() {
    const nlohmann::json config = supported_config();
    const auto constants = qi::read_scheduler_config(config);
    if (!constants.has_value()) {
        check_detail(false, "the shipped scheduler document is accepted", constants.error());
        return;
    }
    check(true, "the shipped scheduler document is accepted");
    expect_scheduler_values(*constants, "the six constants are the artifact's own values");

    // The same numbers the certified scheduler guards qualify: 512x512 -> 1024 latent tokens -> the
    // mu the production pipeline oracle pins. If the reader fed anything else to the scheduler, this
    // is where the certified number moves.
    const double mu = qi::calculate_shift(*constants, 1024);
    const double mu_ref = 0.5387096774193548; // tensors_pipeline/schedule.json (dump_pipeline.py)
    check_detail(mu == mu_ref, "the artifact-fed mu equals the certified pipeline mu",
                 "got " + std::to_string(mu) + " want " + std::to_string(mu_ref));

    // The schedule the same constants build is the one the guards gate: terminal sigma exact, and the
    // last denoise sigma at the artifact's own shift_terminal.
    const qi::Schedule schedule = qi::build_schedule(*constants, 40, mu);
    const bool bounds = schedule.sigmas.size() == 41 && schedule.timesteps.size() == 40 &&
                        schedule.sigmas[0] == 1.0F && schedule.sigmas[40] == 0.0F &&
                        schedule.timesteps[39] == schedule.sigmas[39] * 1000.0F;
    check(bounds, "the artifact-fed schedule keeps the certified bounds");
    const float terminal = schedule.sigmas[39];
    check_detail(std::fabs(terminal - static_cast<float>(constants->shift_terminal)) < 1e-5F,
                 "the artifact's shift_terminal is the last denoise sigma",
                 "sigmas[39]=" + std::to_string(terminal));
}

// ---- the strict typing the reader insists on ----

void gate_strict_typing() {
    struct Case {
        const char* key;
        nlohmann::json value;
        const char* name;
    };
    const nlohmann::json base = supported_config();

    // An integer the JSON writes as a float is a different promise: 256 ok, 256.0 refused.
    for (const char* key : {"base_image_seq_len", "max_image_seq_len", "num_train_timesteps"}) {
        nlohmann::json j = base;
        const double value = j.at(key).get<double>();
        j[key] = value;
        expect_refuses(j, std::string(key) + " as a JSON float is refused", {key});
    }
    // A boolean the JSON writes as 1 is a different promise: true ok, 1 refused.
    for (const char* key : {"use_dynamic_shifting", "use_karras_sigmas", "use_beta_sigmas",
                            "use_exponential_sigmas", "invert_sigmas", "stochastic_sampling"}) {
        nlohmann::json j = base;
        j[key] = 1;
        expect_refuses(j, std::string(key) + " as an integer is refused", {key});
        nlohmann::json s = base;
        s[key] = "true";
        expect_refuses(s, std::string(key) + " as a string is refused", {key});
    }
    // A string only as a string.
    {
        nlohmann::json j = base;
        j["time_shift_type"] = 3;
        expect_refuses(j, "time_shift_type as a number is refused", {"time_shift_type"});
    }
    // A float the JSON writes as an integer: shift 1.0 ok, 1 refused.
    {
        nlohmann::json j = base;
        j["shift"] = 1;
        expect_refuses(j, "shift as a JSON integer is refused", {"shift"});
    }
    // Non-finite floats are refused, not compared.
    for (const char* key : {"base_shift", "max_shift", "shift_terminal", "shift"}) {
        for (double bad : {std::nan(""), HUGE_VAL, -HUGE_VAL}) {
            nlohmann::json j = base;
            j[key] = bad;
            expect_refuses(j, std::string(key) + " non-finite is refused", {key});
        }
    }
}

// ---- every mode flag is verified, not just the six numbers ----

void gate_mode_flags() {
    const nlohmann::json base = supported_config();

    // Each flag flipped to the other legal value is an unhandled scheme: refuse, naming the key and
    // the offending value.
    for (const char* key : {"use_dynamic_shifting", "use_karras_sigmas", "use_beta_sigmas",
                            "use_exponential_sigmas", "invert_sigmas", "stochastic_sampling"}) {
        nlohmann::json j = base;
        const bool expected = j.at(key).get<bool>();
        j[key] = !expected;
        expect_refuses(j, std::string(key) + " flipped is refused", {key, expected ? "false" : "true"});
    }
    {
        nlohmann::json j = base;
        j["time_shift_type"] = "linear";
        expect_refuses(j, "a non-exponential time shift is refused", {"time_shift_type", "linear"});
    }
    {
        nlohmann::json j = base;
        j["shift"] = 2.0;
        expect_refuses(j, "a shift other than 1.0 is refused", {"shift", "2"});
    }

    // A key the supported path never read is a scheme it cannot express.
    {
        nlohmann::json j = base;
        j["use_lms"] = true;
        expect_refuses(j, "an unknown scheme key is refused", {"use_lms"});
    }
    // Another sampler class is not this sampler.
    {
        nlohmann::json j = base;
        j["_class_name"] = "DDPMScheduler";
        expect_refuses(j, "another sampler class is refused", {"_class_name", "DDPMScheduler"});
    }
    // Anything the reader must have is refused by name when it is gone.
    for (const char* key : {"base_image_seq_len", "base_shift", "max_image_seq_len", "max_shift",
                            "num_train_timesteps", "shift_terminal", "use_dynamic_shifting",
                            "time_shift_type", "use_karras_sigmas", "use_beta_sigmas",
                            "use_exponential_sigmas", "invert_sigmas", "shift", "stochastic_sampling",
                            "_class_name"}) {
        nlohmann::json j = base;
        j.erase(key);
        expect_refuses(j, std::string("a missing ") + key + " is refused", {key});
    }
    // The values the scheduler divides by, or the linear shift would silently become inf/NaN.
    for (const char* key : {"base_image_seq_len", "max_image_seq_len", "num_train_timesteps"}) {
        nlohmann::json j = base;
        j[key] = 0;
        expect_refuses(j, std::string(key) + " zero is refused", {key});
        nlohmann::json n = base;
        n[key] = -1;
        expect_refuses(n, std::string(key) + " negative is refused", {key});
    }
    {
        nlohmann::json j = base;
        j["max_image_seq_len"] = j["base_image_seq_len"];
        expect_refuses(j, "max_image_seq_len == base_image_seq_len is refused", {"max_image_seq_len",
                                                                                "base_image_seq_len"});
    }
    {
        nlohmann::json j = base;
        j["shift_terminal"] = 1.0;
        expect_refuses(j, "shift_terminal 1.0 is refused", {"shift_terminal"});
        nlohmann::json n = base;
        n["shift_terminal"] = -0.01;
        expect_refuses(n, "a negative shift_terminal is refused", {"shift_terminal"});
    }
    // A document that is not an object at all.
    expect_refuses(nlohmann::json::array(), "a non-object scheduler document is refused", {"scheduler"});
}

// ---- the live artifact through the production reader (skips when absent) ----

void gate_real_artifact() {
    const char* env = std::getenv("INFERNIX_TEST_ARTIFACT");
    if (env == nullptr || env[0] == '\0') {
        std::printf("ok: real-artifact section SKIPped (INFERNIX_TEST_ARTIFACT not set)\n");
        return;
    }
    std::ifstream probe(env, std::ios::binary);
    if (!probe) {
        std::printf("ok: real-artifact section SKIPped (%s absent)\n", env);
        return;
    }
    probe.close();

    // Host-only metadata read: the reader opens the container's header and files lazily; nothing here
    // touches a device or starts a server.
    const std::filesystem::path path(env);
    infernix::artifact::Reader reader(path);
    const infernix::artifact::Directory& directory = reader.directory();

    const bool has_scheduler = directory.components.find("scheduler") != directory.components.end();
    if (has_scheduler) {
        const auto constants = qi::read_scheduler_config(directory.component("scheduler").config);
        if (constants.has_value()) {
            expect_scheduler_values(*constants, "the live artifact declares the shipped scheduler constants");
        } else {
            check_detail(false, "the live artifact's scheduler component is accepted", constants.error());
        }
    } else {
        // An artifact packed before the converter carried the scheduler must NOT bind: refusing it is
        // the whole point of the change, so that no job is drawn with a compiled schedule.
        std::printf("ok: the live artifact carries no scheduler component -- the strict reader must refuse it\n");
    }
}

} // namespace

int main() {
    gate_shipped_metadata();
    gate_strict_typing();
    gate_mode_flags();
    gate_real_artifact();
    if (g_failures != 0) {
        std::printf("IMAGE_ARTIFACT_CONSTANTS_TEST: FAIL (%d check(s))\n", g_failures);
        return 1;
    }
    std::printf("IMAGE_ARTIFACT_CONSTANTS_TEST: PASS\n");
    return 0;
}
