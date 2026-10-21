// Server-executed chat image tool host contract: the hidden tool is injected only where the client
// left tool choice free, a round is fed back as the same assistant+tool turns a client round-trip
// would send, the per-request picture cap and bad arguments become tool errors instead of pictures,
// a failing pipeline still propagates, and the round loop ends in one answer that carries every URL.
// CHAT_IMAGE_TOOL_TEST: PASS
#include "serve/chat_image_tool.h"

#include <infernix/types.h>

#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace infernix::serve;

int failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "ok: " : "FAIL: ") << what << '\n';
    if (!ok) { ++failures; }
}

constexpr std::string_view kUrlBase = "http://127.0.0.1:8888/v1/images/";
constexpr std::string_view kImageModel = "infernix-image";

[[nodiscard]] GenerationOutcome prose_outcome(std::string text) {
    GenerationOutcome outcome;
    outcome.text          = std::move(text);
    outcome.finish_reason = infernix::FinishReason::StopToken;
    return outcome;
}

[[nodiscard]] GenerationOutcome image_call_outcome(std::vector<std::string> arguments) {
    GenerationOutcome outcome;
    outcome.text          = "Drawing that now.";
    outcome.finish_reason = infernix::FinishReason::StopToken;
    for (std::string& arguments_json : arguments) {
        outcome.tool_calls.push_back(infernix::GeneratedToolCall{
            .name           = std::string(kChatImageToolName),
            .arguments_json = std::move(arguments_json),
        });
    }
    return outcome;
}

// Stands in for ImageGenerationService::generate: one URL per accepted request, and the same
// ApiException a real pipeline failure raises.
struct FakeGenerator {
    std::vector<OpenAIImagesRequest> requests;
    std::vector<std::string> urls;
    int fail_status   = 0;
    int calls         = 0;

    RequestJson operator()(const OpenAIImagesRequest& request) {
        ++calls;
        requests.push_back(request);
        if (fail_status != 0) {
            ApiError error;
            error.status  = fail_status;
            error.type    = "server_error";
            error.message = "the image pipeline is busy";
            error.code    = "image_pipeline_unavailable";
            throw ApiException(std::move(error));
        }
        const std::string id = "img-" + std::to_string(requests.size());
        const std::string url = std::string(kUrlBase) + id;
        urls.push_back(url);
        return RequestJson{{"created", 1},
                           {"data", RequestJson::array({RequestJson{{"url", url}}})}};
    }
};

// The round takes the generator as a type-erased callable, which would otherwise copy this observer
// and lose every record of what it was asked to draw.
[[nodiscard]] ChatImageGenerator observer_of(FakeGenerator& generator) { return std::ref(generator); }

[[nodiscard]] std::string tool_result_text(const GenerationRequest& request, std::size_t index) {
    const ChatTurn& turn = request.messages[index];
    return turn.content.empty() ? std::string() : turn.content.front().text;
}

void test_injection() {
    GenerationRequest free_choice;
    append_chat_image_tool(free_choice, true);
    check(free_choice.tools.size() == 1 && free_choice.tools[0].name == kChatImageToolName,
          "a free tool choice gains the hidden tool");

    const ToolDefinition definition = make_chat_image_tool_definition();
    check(RequestJson::parse(definition.input_schema_json).contains("required"),
          "the hidden tool carries a parseable input schema with a required prompt");

    GenerationRequest disabled;
    disabled.tool_choice.mode = ToolChoiceMode::None;
    append_chat_image_tool(disabled, true);
    check(disabled.tools.empty(), "tool_choice none keeps a request without tools");

    GenerationRequest pinned;
    pinned.tool_choice.allowed_names = std::vector<std::string>{"lookup"};
    append_chat_image_tool(pinned, true);
    check(pinned.tools.empty(), "a pinned tool selection is not widened");

    GenerationRequest owns_the_name;
    owns_the_name.tools.push_back(ToolDefinition{.name = std::string(kChatImageToolName)});
    append_chat_image_tool(owns_the_name, true);
    check(owns_the_name.tools.size() == 1 && owns_the_name.tools[0].input_schema_json.empty(),
          "a client tool of this name is never taken over");

    GenerationRequest no_pipeline;
    append_chat_image_tool(no_pipeline, false);
    check(no_pipeline.tools.empty(), "without an image pipeline nothing is injected");
}

void test_round_detection() {
    check(!outcome_is_chat_image_tool_round(prose_outcome("hi")), "prose is not an image round");
    check(outcome_is_chat_image_tool_round(image_call_outcome({R"({"prompt":"a"})"})),
          "a lone image call is an internal round");

    GenerationOutcome mixed = image_call_outcome({R"({"prompt":"a"})"});
    mixed.tool_calls.push_back(
        infernix::GeneratedToolCall{.name = "lookup", .arguments_json = "{}"});
    check(!outcome_is_chat_image_tool_round(mixed),
          "a round that mixes in a client tool is handed to the client");
}

void test_single_round() {
    GenerationRequest request;
    request.messages.push_back(ChatTurn{.role = infernix::ChatRole::User});
    FakeGenerator generator;

    const ChatImageToolRound round = run_chat_image_tool_round(
        request, image_call_outcome({R"({"prompt":"a red fox","size":"1024x1024","steps":7})"}),
        kImageModel, kUrlBase, 0, 0, observer_of(generator));

    check(round.executed == 1 && round.rejected == 0 && round.urls.size() == 1 &&
              round.ids.size() == 1 && round.ids[0] == "img-1",
          "one call produced one picture and its artifact id");
    check(generator.requests.size() == 1 && generator.requests[0].prompt == "a red fox" &&
              generator.requests[0].width == 1024 && generator.requests[0].steps == 7 &&
              generator.requests[0].model == kImageModel &&
              generator.requests[0].response_format == ImageResponseFormat::Url,
          "the model's arguments reach the images contract, and delivery fields are the server's");
    check(request.messages.size() == 3, "the round is appended as assistant + tool turns");
    const ChatTurn& assistant = request.messages[1];
    const ChatTurn& tool      = request.messages[2];
    check(assistant.role == infernix::ChatRole::Assistant && assistant.tool_calls.size() == 1 &&
              assistant.tool_calls[0].id == "infernix_img_0" &&
              tool.role == infernix::ChatRole::Tool && tool.tool_call_id == assistant.tool_calls[0].id &&
              tool.tool_result_name == kChatImageToolName && !tool.tool_result_is_error,
          "the two turns correlate by call id as a client round-trip would");
    check(tool_result_text(request, 2).find(round.urls[0]) != std::string::npos,
          "the tool result carries the picture URL the answer must quote");
}

void test_cap_and_bad_arguments() {
    GenerationRequest request;
    FakeGenerator generator;
    const ChatImageToolRound over_cap =
        run_chat_image_tool_round(request, image_call_outcome({R"({"prompt":"a"})"}), kImageModel,
                                  kUrlBase, 0, kMaximumChatImageGenerations,
                                  observer_of(generator));
    check(over_cap.executed == 0 && over_cap.rejected == 1 && generator.calls == 0,
          "a request at its picture cap draws nothing more");
    check(tool_result_text(request, 1).find("image_generation_limit_reached") != std::string::npos,
          "the refused call gets an error result the model can read");

    GenerationRequest malformed;
    const ChatImageToolRound bad_json = run_chat_image_tool_round(
        malformed, image_call_outcome({"this is not json"}), kImageModel, kUrlBase, 0, 0, observer_of(generator));
    check(bad_json.executed == 0 && bad_json.rejected == 0 &&
              tool_result_text(malformed, 1).find("image_arguments_invalid") != std::string::npos,
          "unparseable arguments become a tool error, not a failure of the request");

    GenerationRequest bad_size;
    const ChatImageToolRound rejected_size = run_chat_image_tool_round(
        bad_size, image_call_outcome({R"({"prompt":"a","size":"123"})"}), kImageModel, kUrlBase, 0,
        0, observer_of(generator));
    check(rejected_size.executed == 0 && generator.calls == 0,
          "an argument set the images contract refuses never reaches the pipeline");
    check(tool_result_text(bad_size, 1).find("error") != std::string::npos,
          "and comes back as an error result the model can correct");
}

void test_pipeline_failure_propagates() {
    GenerationRequest request;
    FakeGenerator generator;
    generator.fail_status = 503;
    bool propagated       = false;
    bool settled          = true;
    try {
        const ChatImageToolRound round = run_chat_image_tool_round(
            request, image_call_outcome({R"({"prompt":"a"})"}), kImageModel, kUrlBase, 0, 0,
            observer_of(generator));
        (void)round;
    } catch (const ApiException& exception) {
        settled    = false;
        propagated = exception.error().status == 503;
    }
    check(!settled && propagated,
          "a pipeline failure surfaces as the error the images route would return");
    check(request.messages.empty(),
          "and the failed round is not fed back as a tool result the model must answer");
}

void test_url_suffix() {
    const std::vector<std::string> urls{"http://h/v1/images/a", "http://h/v1/images/b"};
    check(missing_image_url_suffix("see http://h/v1/images/a and http://h/v1/images/b", urls)
              .empty(),
          "an answer that quoted every URL gains nothing");
    const std::string suffix = missing_image_url_suffix("I drew two pictures", urls);
    check(suffix.find(urls[0]) != std::string::npos && suffix.find(urls[1]) != std::string::npos,
          "URLs the answer dropped are appended");
}

void test_round_loop() {
    GenerationRequest request;
    FakeGenerator generator;

    GenerationOutcome first    = image_call_outcome({R"({"prompt":"one"})"});
    first.reasoning            = "they want a fox";
    first.prompt_tokens        = 100;
    first.completion_tokens    = 5;
    GenerationOutcome second   = image_call_outcome({R"({"prompt":"two"})"});
    second.reasoning           = "and now a whale";
    second.prompt_tokens       = 140;
    second.completion_tokens   = 6;
    GenerationOutcome final    = prose_outcome("here is your second picture");
    final.prompt_tokens        = 180;
    final.completion_tokens    = 7;

    std::vector<GenerationOutcome> pending{std::move(second), std::move(final)};
    const ChatImageRounds loop = run_chat_image_rounds(
        request, std::move(first), kImageModel, kUrlBase, 0, observer_of(generator), [&] {
            GenerationOutcome next = std::move(pending.front());
            pending.erase(pending.begin());
            return next;
        });

    check(loop.rounds == 2 && loop.generations == 2 && loop.urls.size() == 2 &&
              generator.calls == 2,
          "both picture rounds ran before the prose answer ended the loop");
    check(loop.answer.text == "Drawing that now.Drawing that now.here is your second picture",
          "the answer is every round's text in the order it was spoken");
    check(loop.answer.reasoning == "they want a fox",
          "only the first round's thinking is reported, since it is already behind content");
    check(loop.answer.prompt_tokens == 420 && loop.answer.completion_tokens == 18,
          "token usage is summed over every round the request really ran");
    check(loop.answer.tool_calls.empty(), "the client answer carries no call to a hidden tool");
    check(request.messages.size() == 4, "both rounds were appended to the conversation");

    const std::string suffix = missing_image_url_suffix(loop.answer.text, loop.urls);
    check(suffix.find(loop.urls[0]) != std::string::npos &&
              suffix.find(loop.urls[1]) != std::string::npos,
          "the route's suffix step guarantees both URLs in the reply");
}

void test_round_bound_and_cancellation() {
    GenerationRequest request;
    FakeGenerator generator;
    const ChatImageRounds bounded = run_chat_image_rounds(
        request, image_call_outcome({R"({"prompt":"again"})"}), kImageModel, kUrlBase, 0, observer_of(generator),
        [] { return image_call_outcome({R"({"prompt":"again"})"}); });

    check(bounded.rounds == kMaximumChatImageRounds,
          "one request stops asking for pictures at the round bound");
    check(bounded.generations == kMaximumChatImageGenerations,
          "and at the picture cap well before that, which it reports as refusals");
    check(bounded.answer.tool_calls.empty() &&
              bounded.answer.finish_reason == infernix::FinishReason::StopToken,
          "the answer at the bound is an ordinary reply, not a call to a hidden tool");

    GenerationRequest cancelled_request;
    FakeGenerator idle;
    GenerationOutcome cancelled = image_call_outcome({R"({"prompt":"a"})"});
    cancelled.finish_reason     = infernix::FinishReason::Cancelled;
    const ChatImageRounds stopped =
        run_chat_image_rounds(cancelled_request, std::move(cancelled), kImageModel, kUrlBase, 0,
                              observer_of(idle), [] { return prose_outcome("unreachable"); });
    check(stopped.rounds == 0 && idle.calls == 0 && !stopped.answer.tool_calls.empty(),
          "a cancelled generation draws nothing and is reported as it is");
}

} // namespace

int main() {
    test_injection();
    test_round_detection();
    test_single_round();
    test_cap_and_bad_arguments();
    test_pipeline_failure_propagates();
    test_url_suffix();
    test_round_loop();
    test_round_bound_and_cancellation();

    std::cout << (failures == 0 ? "CHAT_IMAGE_TOOL_TEST: PASS" : "CHAT_IMAGE_TOOL_TEST: FAIL")
              << " (" << (failures == 0 ? "all" : std::to_string(failures) + " failed") << ")\n";
    return failures == 0 ? 0 : 1;
}
