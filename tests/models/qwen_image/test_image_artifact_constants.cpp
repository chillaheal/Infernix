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
#include "models/qwen_image/rope.h"
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

// ---- the DiT rope axes the artifact declares ----

// A dit document that declares the shipped architecture: the axes the checkpoint's
// transformer/config.json carries (verified against the live artifact's own dit component below, and
// against two independent diffusers-dumped rotary tables in infernix_image_rope_prod_test) and the head
// width those axes have to add up to. The deviations below are hand-written against THIS document
// because the strictness is what is under test; the shipped values themselves are gated twice, once
// here through the production reader and once by the rope oracle's bitwise comparison.
nlohmann::json supported_dit_config() {
    return nlohmann::json{{"_class_name", "QwenImage2Transformer2DModel"},
                          {"attention_head_dim", 128},
                          {"axes_dims_rope", {16, 56, 56}}};
}

void expect_rope_axes(const nlohmann::json& config, const std::string& name,
                      const std::uint32_t a, const std::uint32_t b, const std::uint32_t c) {
    const auto axes = qi::read_dit_rope_axes(config);
    if (!axes.has_value()) {
        check_detail(false, name, axes.error());
        return;
    }
    const bool ok = axes->axes[0] == a && axes->axes[1] == b && axes->axes[2] == c;
    check_detail(ok, name, "axes=" + std::to_string(axes->axes[0]) + "," +
                               std::to_string(axes->axes[1]) + "," + std::to_string(axes->axes[2]));
}

void expect_rope_refuses(const nlohmann::json& config, const std::string& name,
                         const std::vector<std::string>& needles) {
    const auto result = qi::read_dit_rope_axes(config);
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

void gate_rope_axes() {
    expect_rope_axes(supported_dit_config(), "the artifact's rope axes are accepted", 16, 56, 56);

    // An axis the JSON writes as a float or a string is a different promise: 16 ok, 16.0 refused.
    for (const char* bad : {"16.0", "\"16\""}) {
        nlohmann::json j = supported_dit_config();
        j["axes_dims_rope"] = nlohmann::json::parse(
            std::string("[") + bad + ", 56, 56]");
        expect_rope_refuses(j, std::string("a rope axis written as ") + bad + " is refused",
                            {"axes_dims_rope"});
    }
    // Not an array at all, or not exactly the three axes the DiT rotates on.
    {
        nlohmann::json j = supported_dit_config();
        j["axes_dims_rope"] = 128;
        expect_rope_refuses(j, "rope axes that are not an array are refused", {"axes_dims_rope"});
    }
    for (const char* document : {"[16, 56]", "[16, 56, 56, 8]"}) {
        nlohmann::json j = supported_dit_config();
        j["axes_dims_rope"] = nlohmann::json::parse(document);
        expect_rope_refuses(j, std::string("a rope axis list of ") + std::string(document) +
                                   " is refused",
                            {"axes_dims_rope", "three"});
    }
    {
        nlohmann::json j = supported_dit_config();
        j.erase("axes_dims_rope");
        expect_rope_refuses(j, "a missing axes_dims_rope is refused", {"axes_dims_rope"});
        nlohmann::json h = supported_dit_config();
        h.erase("attention_head_dim");
        expect_rope_refuses(h, "a missing attention_head_dim is refused", {"attention_head_dim"});
        nlohmann::json f = supported_dit_config();
        f["attention_head_dim"] = 128.0;
        expect_rope_refuses(f, "attention_head_dim as a JSON float is refused", {"attention_head_dim"});
    }
    // A zero, a negative or an odd axis cannot emit whole (cos, sin) pairs.
    for (const char* document : {"[0, 56, 72]", "[-16, 56, 144]", "[17, 55, 56]"}) {
        nlohmann::json j = supported_dit_config();
        j["axes_dims_rope"] = nlohmann::json::parse(document);
        expect_rope_refuses(j, std::string("the rope axes ") + std::string(document) + " are refused",
                            {"axes_dims_rope"});
    }
    // The axes have to rotate the whole head: their sum is attention_head_dim, and the emitted table is
    // sum/2 interleaved (cos, sin) pairs -- attention_head_dim/2 -- wide. A document that adds up to
    // anything else is refused by name and by both numbers.
    {
        nlohmann::json j = supported_dit_config();
        j["axes_dims_rope"] = {16, 56, 55}; // 127, and odd
        expect_rope_refuses(j, "rope axes that do not add up to the head are refused",
                            {"axes_dims_rope", "attention_head_dim"});
        nlohmann::json h = supported_dit_config();
        h["axes_dims_rope"] = {16, 56, 40}; // 112, all even: still not the head
        expect_rope_refuses(h, "rope axes narrower than the head are refused",
                            {"axes_dims_rope", "attention_head_dim"});
        nlohmann::json o = supported_dit_config();
        o["attention_head_dim"] = 126; // the axes sum to 128, an odd-narrowed head cannot be rotated
        expect_rope_refuses(o, "an attention_head_dim the axes cannot add up to is refused",
                            {"axes_dims_rope", "attention_head_dim"});
    }
    // A document that is not an object at all.
    expect_rope_refuses(nlohmann::json::array(), "a non-object dit document is refused", {"dit"});

    // The table's width comes from the ARTIFACT's axes: a different, legal architecture gets its own
    // width, so nothing in the emitter is pinned at the shipped 64 pairs.
    {
        const auto other = qi::read_dit_rope_axes(nlohmann::json{{"attention_head_dim", 24},
                                                                {"axes_dims_rope", {8, 8, 8}}});
        if (!other.has_value()) {
            check_detail(false, "a second, legal rope architecture is accepted", other.error());
        } else {
            const std::size_t row_floats = qi::rope_row_floats(*other);
            const std::vector<float> table = qi::build_joint_rope(*other, 3, 2, 2);
            check_detail(row_floats == 24 && table.size() == 7 * 24,
                         "a legal other architecture emits its own table width",
                         "row=" + std::to_string(row_floats) + " floats=" + std::to_string(table.size()));
        }
        const auto shipped = qi::read_dit_rope_axes(supported_dit_config());
        if (!shipped.has_value()) {
            check_detail(false, "the shipped architecture's axes are readable", shipped.error());
        } else {
            const std::vector<float> table = qi::build_joint_rope(*shipped, 3, 2, 2);
            check_detail(qi::rope_row_floats(*shipped) == 128 && table.size() == 7 * 128,
                         "the shipped architecture keeps the certified table width",
                         "row=" + std::to_string(qi::rope_row_floats(*shipped)) +
                             " floats=" + std::to_string(table.size()));
        }
    }
    // Theta is the supported pipeline implementation's default and the artifact is never ASKED to declare
    // one (the shipped dit config declares none, and no theta key was introduced for tidiness). But an
    // artifact that DOES declare one is stating which rotary scheme it was trained under, so a theta that
    // disagrees with this build's has to refuse rather than be silently rotated at the code's value: the
    // same fail-closed rule the scheduler side applies to a foreign "_class_name" or a "use_lms" flag, and
    // the contract this lane's text tower already keeps (its rope_theta is read from the artifact and
    // refused when absent, image_weights.cpp). An AGREING declaration is the one theta key that binds, and
    // it changes no byte -- the emitter still rotates at the code's theta, so the certified numbers cannot
    // move.
    {
        check(qi::kQwenImageRopeTheta == 10000.0F,
              "the rope theta is the documented pipeline implementation default");
        expect_rope_refuses(nlohmann::json{{"attention_head_dim", 128}, {"axes_dims_rope", {16, 56, 56}},
                                           {"rope_theta", 7000.0}},
                            "an artifact-declared rope_theta that disagrees with the pipeline is refused",
                            {"rope_theta", "7000.0", "kQwenImageRopeTheta"});
        expect_rope_refuses(nlohmann::json{{"attention_head_dim", 128}, {"axes_dims_rope", {16, 56, 56}},
                                           {"rope_theta", 10000}},
                            "a rope_theta written as a JSON integer is refused",
                            {"rope_theta", "JSON float"});
        expect_rope_refuses(nlohmann::json{{"attention_head_dim", 128}, {"axes_dims_rope", {16, 56, 56}},
                                           {"rope_theta", "10000.0"}},
                            "a rope_theta written as a string is refused",
                            {"rope_theta", "JSON float"});
        expect_rope_refuses(nlohmann::json{{"attention_head_dim", 128}, {"axes_dims_rope", {16, 56, 56}},
                                           {"rope_theta", nullptr}},
                            "a rope_theta that is not a number is refused", {"rope_theta"});
        const auto agreeing =
            qi::read_dit_rope_axes(nlohmann::json{{"attention_head_dim", 128},
                                                  {"axes_dims_rope", {16, 56, 56}},
                                                  {"rope_theta", 10000.0}});
        if (!agreeing.has_value()) {
            check_detail(false, "a rope_theta that agrees with the pipeline is accepted",
                         agreeing.error());
        } else {
            const std::vector<float> declared  = qi::build_joint_rope(*agreeing, 3, 2, 2);
            const auto plain                   = qi::read_dit_rope_axes(supported_dit_config());
            const std::vector<float> untouched = qi::build_joint_rope(*plain, 3, 2, 2);
            check_detail(declared == untouched && declared.size() == 7 * 128,
                         "an agreeing rope_theta declares the same scheme and changes no byte",
                         "floats=" + std::to_string(declared.size()));
        }
    }
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

    // The DiT rope axes are the checkpoint's own transformer/config.json keys, so the live artifact is
    // the only place they can be qualified: the shipped architecture must read as 16/56/56 over a
    // 128-wide head, which is what the certified rotary tables and ops::rope_c64's 64-pair contract
    // require.
    const bool has_dit = directory.components.find("dit") != directory.components.end();
    if (has_dit) {
        expect_rope_axes(directory.component("dit").config,
                         "the live artifact declares the shipped rope axes", 16, 56, 56);
    } else {
        check(false, "the live artifact declares a dit component");
    }
}

} // namespace

int main() {
    gate_shipped_metadata();
    gate_strict_typing();
    gate_mode_flags();
    gate_rope_axes();
    gate_real_artifact();
    if (g_failures != 0) {
        std::printf("IMAGE_ARTIFACT_CONSTANTS_TEST: FAIL (%d check(s))\n", g_failures);
        return 1;
    }
    std::printf("IMAGE_ARTIFACT_CONSTANTS_TEST: PASS\n");
    return 0;
}
