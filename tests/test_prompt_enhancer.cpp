// The prompt enhancer's contract, host-side: what an answer from Qwen/Qwen-Image-2.1-PE-I2I must look
// like before the route trusts it, and what the client says when the sidecar is not there. No GPU, no
// worker process -- the rules under test are the shipped system prompt's own
// (image-edit/refs/q21_system_prompt_edit.txt:185-192) and the refusal the route owes a client that
// asked for a rewriter this deployment does not have.

#include "serve/openai_images.h"
#include "serve/prompt_enhancer.h"
#include "serve/request.h"
#include "serve/request_json.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = infernix::serve::RequestJson;
using namespace infernix::serve;

constexpr std::chrono::milliseconds kTinyTimeout{50};

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

template <typename Function>
ApiError api_error(Function&& function) {
    try {
        function();
    } catch (const ApiException& exception) { return exception.error(); }
    return ApiError{.status = 0, .message = "no exception"};
}

// One well-formed edit answer: the three shipped fields, the canvas following the first reference.
Json good_answer() {
    return Json{{"rewritten_prompt", "Replace the sky with a warm sunset. Keep everything else."},
                {"wh_ratio", ""},
                {"ratio_follow", "<image1>"},
                {"thinking", "the picture is a landscape under a grey sky"},
                {"parse_ok", true}};
}

// The code every rejection must carry: a bad answer is the server's failure, not the client's.
bool is_bad_answer(const ApiError& error) {
    return error.status == 500 && error.code == "image_prompt_enhancer_bad_answer" &&
           error.param == "prompt_enhance";
}

// The error one answer earns, or status 0 when it passes. The answer itself is consumed on the way
// through: a contract check that threw away what it validated could not tell a passing answer from a
// call that never ran.
ApiError answer_error(const Json& reply, const std::size_t references) {
    return api_error([&] {
        const EnhancerAnswer answer = validate_enhancer_answer(reply, references);
        return answer.rewritten_prompt.size();
    });
}

bool accepted(const Json& reply, const std::size_t references) {
    return answer_error(reply, references).status == 0;
}

int test_accepts_a_well_formed_edit_answer() {
    int failures = 0;
    const auto answer = validate_enhancer_answer(good_answer(), 2);
    failures += check(answer.rewritten_prompt ==
                          "Replace the sky with a warm sunset. Keep everything else.",
                      "the instruction is what the model said");
    failures += check(answer.wh_ratio.empty() && answer.ratio_follow == "<image1>",
                      "ratio_follow carries the canvas");
    failures += check(answer.thinking == "the picture is a landscape under a grey sky",
                      "the thinking block travels with the answer");
    failures += check(answer.new_tokens == 0, "an answer that names no token count says so with 0");

    // Whitespace padding is the same answer.
    Json padded = good_answer();
    padded["rewritten_prompt"] = "  Smooth the water.  ";
    padded["ratio_follow"]     = " <image2> ";
    const auto trimmed         = validate_enhancer_answer(padded, 2);
    failures += check(trimmed.rewritten_prompt == "Smooth the water.", "padding is not content");
    failures += check(trimmed.ratio_follow == "<image2>", "the tag keeps its meaning");
    return failures;
}

int test_rejects_a_malformed_answer() {
    int failures = 0;
    const auto refused = [](const Json& reply, const std::size_t references,
                            const std::string& label) {
        return check(is_bad_answer(answer_error(reply, references)), label);
    };
    const auto mutate = [](const char* field, Json value) {
        Json reply = good_answer();
        reply[field] = std::move(value);
        return reply;
    };

    Json unparsed     = good_answer();
    unparsed["parse_ok"] = false;
    failures += refused(unparsed, 2, "an answer that did not parse is not a rewrite");
    failures += refused(mutate("rewritten_prompt", ""), 2, "an empty instruction cannot condition a DiT");
    failures += refused(mutate("rewritten_prompt", Json("   ")), 2, "whitespace is not an instruction");
    failures += refused(mutate("rewritten_prompt", "one sentence.\nanother sentence."), 2,
                        "the contract is one paragraph");
    failures += refused(mutate("rewritten_prompt", "Redraw it at 1024x1024 pixels."), 2,
                        "a resolution inside the instruction belongs in wh_ratio");
    failures += refused(mutate("rewritten_prompt", "Make it 16:9 wider."), 2,
                        "an aspect inside the instruction belongs in wh_ratio");
    failures += refused(mutate("rewritten_prompt", "Keep the aspect ratio of the scene."), 2,
                        "the words for a ratio are as good as the number");
    failures += refused(mutate("rewritten_prompt", std::string(8001, 'a')), 2,
                        "an instruction past the bound is not honoured");
    failures += refused(mutate("wh_ratio", Json("16:9")), 2,
                        "wh_ratio and ratio_follow are mutually exclusive");
    failures += refused(mutate("wh_ratio", Json("16/9")), 2, "wh_ratio must be the W:H form");
    failures += refused(mutate("ratio_follow", Json("image1")), 2,
                        "a tag needs the <imageN> shape the engine renders");
    failures += refused(mutate("ratio_follow", Json("<image3>")), 2,
                        "a canvas must be a picture the request actually carries");
    failures += refused(mutate("rewritten_prompt", Json(7)), 2, "the instruction is a string");

    Json missing = good_answer();
    missing.erase("wh_ratio");
    failures += refused(missing, 2, "a missing field is a missing field");
    Json not_an_object = Json("text");
    failures += refused(not_an_object, 2, "the answer is an object");
    return failures;
}

int test_the_canvas_answer_is_checked_against_what_the_request_carries() {
    int failures = 0;
    // An edit whose answer names neither channel is still servable: the route follows the request's own
    // reference (openai_images.h resolve_canvas_geometry), which is the same canvas ratio_follow would
    // have named -- so neither-empty is accepted for an edit. For text-to-image nothing else decides the
    // shape, and an answer with neither channel would have named no canvas at all: there it is refused.
    Json neither = good_answer();
    neither["ratio_follow"] = "";
    failures += check(accepted(neither, 1),
                      "an edit may leave the canvas to the reference it carried");
    failures += check(!accepted(neither, 0),
                      "a text-to-image answer with no canvas named nothing");

    // The tag is bounded by the reference COUNT, not by any guess about which picture was meant.
    Json follow = good_answer();
    follow["ratio_follow"] = "<image4>";
    failures += check(accepted(follow, 4),
                      "<image4> is the fourth reference when there are four");
    failures += check(!accepted(follow, 3),
                      "and is refused when there are three");
    return failures;
}

int test_says_so_when_there_is_no_enhancer() {
    int failures = 0;
    const PromptEnhancer none_of_it("", kTinyTimeout);
    failures += check(!none_of_it.configured(), "an empty URL is not a configuration");
    failures += check(!none_of_it.healthy(), "nothing answers an empty URL");
    const auto refused = api_error([&] {
        const EnhancerAnswer answer =
            none_of_it.rewrite("make it an owl", {"aGVsbG8="}, 42ULL, 1);
        return answer.rewritten_prompt.size();
    });
    failures += check(refused.status == 503 && refused.code == "image_prompt_enhancer_unavailable" &&
                          refused.param == "prompt_enhance",
                      "an unconfigured enhancer is the documented 503, not a silent pass-through");
    failures += check(refused.message.find("PE-I2I") != std::string::npos,
                      "the refusal names the model the client asked for");
    failures += check(refused.message.find("capabilities") != std::string::npos,
                      "and points at the document that tells it what to write instead");
    return failures;
}

int test_a_configured_but_silent_sidecar_is_still_the_documented_503() {
    int failures = 0;
    // Port 1 answers nothing on this machine: the state is "configured, not running", which a client
    // must be able to tell from "not configured" only by the message, not by a different contract.
    const PromptEnhancer silent("127.0.0.1:1", kTinyTimeout);
    failures += check(silent.configured(), "a URL is a configuration");
    failures += check(!silent.healthy(), "and silence is not health");
    const auto refused = api_error([&] { silent.load(); });
    failures += check(refused.status == 503 && refused.code == "image_prompt_enhancer_unavailable",
                      "load on a silent sidecar is the 503");
    failures += check(refused.message.find("pe_worker.py") != std::string::npos,
                      "the message names the thing to start");
    return failures;
}

int test_a_malformed_enhancer_url_is_refused_not_guessed() {
    int failures = 0;
    for (const char* url : {"127.0.0.1", "127.0.0.1:8899/extra", "127.0.0.1:notaport",
                            "127.0.0.1:0"}) {
        const PromptEnhancer misconfigured(url, kTinyTimeout);
        const auto error = api_error([&] { misconfigured.load(); });
        failures += check(error.status == 503 &&
                              error.code == "image_prompt_enhancer_unavailable",
                          std::string("a URL that is not host:port is refused: ") + url);
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_accepts_a_well_formed_edit_answer();
    failures += test_rejects_a_malformed_answer();
    failures += test_the_canvas_answer_is_checked_against_what_the_request_carries();
    failures += test_says_so_when_there_is_no_enhancer();
    failures += test_a_configured_but_silent_sidecar_is_still_the_documented_503();
    failures += test_a_malformed_enhancer_url_is_refused_not_guessed();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
