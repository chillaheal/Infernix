// Server-executed chat image tool host contract: the hidden tool is injected only where the client
// left tool choice free, a round is fed back as the same assistant+tool turns a client round-trip
// would send, the per-request picture cap and bad arguments become tool errors instead of pictures,
// and so does a failing pipeline -- the model's request was for an answer, not for the HTTP status
// of a draw. The round loop ends in one answer that carries every URL.
// CHAT_IMAGE_TOOL_TEST: PASS
#include "serve/chat_image_tool.h"
#include "product/media_acquire/source.h"
#include "serve/image_generation_service.h"

#include <infernix/types.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
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
    const char* fail_code = "image_pipeline_unavailable";
    int calls         = 0;

    RequestJson operator()(const OpenAIImagesRequest& request) {
        ++calls;
        requests.push_back(request);
        if (fail_status != 0) {
            ApiError error;
            error.status  = fail_status;
            error.type    = "server_error";
            error.message = "the image pipeline is busy";
            error.code    = fail_code;
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

void test_pipeline_failure_becomes_a_tool_result() {
    // The three ways a draw can fail after the model asked for it: this size does not fit now,
    // another picture is already on the device, the job fell over. None is the model's mistake and
    // none may take the user's answer with it.
    for (const char* code : {"image_vram_insufficient", "image_interlude_in_flight",
                             "image_generation_failed"}) {
        GenerationRequest request;
        FakeGenerator generator;
        generator.fail_status = 503;
        generator.fail_code   = code;
        const ChatImageToolRound round = run_chat_image_tool_round(
            request, image_call_outcome({R"({"prompt":"a"})"}), kImageModel, kUrlBase, 0, 0,
            observer_of(generator));
        const std::string what{code};

        check(round.executed == 0 && round.urls.empty() && round.ids.empty(),
              what + " leaves no URL the answer could promise");
        check(request.messages.size() == 2 && request.messages[0].tool_calls.size() == 1,
              what + " settles as one assistant call and one tool result, as a client round-trip "
                    "would");
        const std::string result = tool_result_text(request, 1);
        check(result.find(code) != std::string::npos &&
                  result.find("the image pipeline is busy") != std::string::npos,
              what + " reaches the model through the tool result with the pipeline's own reason "
                    "(the size that would fit travels in it), not as a failed request");
        check(request.messages[1].tool_result_is_error,
              what + " is marked as an error, so the model may retry smaller or answer in words");
    }
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

// The chat `image` argument: three forms are resolved into the route's own base64 reference, and
// every other shape -- including the base64 a model cannot type -- is refused with the accepted forms
// in the text, because the model is the one that has to retry.
void test_image_argument_forms() {
    using MediaSource = infernix::product::media_acquire::Source;
    using MediaKind   = infernix::product::media_acquire::SourceKind;

    const ToolDefinition definition = make_chat_image_tool_definition();
    const RequestJson schema        = RequestJson::parse(definition.input_schema_json);
    const std::string described     = schema.at("properties").at("image").dump();
    check(schema.at("properties").at("image").at("type") == "string",
          "the hidden tool advertises an optional string image argument");
    check(described.find("img-") != std::string::npos &&
              described.find("attachment:N") != std::string::npos &&
              described.find("base64") != std::string::npos &&
              described.find("another machine") != std::string::npos,
          "its description names exactly the forms that resolve, and the two that do not");

    // A request carrying one inline attachment, as the protocols parse one.
    const std::string data_uri = "data:image/png;base64,iVBORw0KGgo=";
    auto with_attachment       = [&](std::size_t pictures) {
        GenerationRequest request;
        ChatTurn user;
        user.role = infernix::ChatRole::User;
        user.content.push_back(ContentPart{.kind = ContentKind::Text, .text = "look at this"});
        for (std::size_t index = 0; index < pictures; ++index) {
            user.content.push_back(ContentPart{
                .kind   = ContentKind::Image,
                .source = MediaSource{.kind = MediaKind::Data, .value = data_uri},
            });
        }
        request.messages.push_back(std::move(user));
        return request;
    };

    {
        // The file a picture this server drew leaves on disk, named by the id it came back with.
        std::error_code ignored;
        std::filesystem::create_directories("images", ignored);
        const std::vector<std::uint8_t> png{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
        {
            std::ofstream file("images/img-100-1.png", std::ios::binary | std::ios::trunc);
            file.write(reinterpret_cast<const char*>(png.data()),
                       static_cast<std::streamsize>(png.size()));
        }
        const std::string expected = base64_encode_bytes(png);

        for (const std::string name : {std::string("img-100-1"), std::string(kUrlBase) + "img-100-1",
                                       std::string("/v1/images/img-100-1")}) {
            GenerationRequest request;
            FakeGenerator generator;
            const ChatImageToolRound round = run_chat_image_tool_round(
                request, image_call_outcome({R"({"prompt":"again","image":")" + name + R"("})"}),
                kImageModel, kUrlBase, 0, 0, observer_of(generator));
            check(round.executed == 1 && generator.requests.size() == 1 &&
                      generator.requests[0].image == expected,
                  "the picture named \"" + (name.size() > 40 ? name.substr(0, 40) : name) +
                      "\" is read back off disk as the route's reference");
        }

        // Every refusal: no draw, an error result, and the accepted forms in its text.
        struct Refusal {
            std::string argument;
            std::string why;
        };
        const std::vector<Refusal> refusals{
            {"img-7-7", "a drawn picture that is not on disk"},
            {"http://elsewhere.example/v1/images/img-100-1", "a picture on another machine"},
            {"iVBORw0KGgoAAAANSUhEUg", "raw base64 the model cannot have typed"},
            {"data:image/png;base64,iVBORw0KGgo=", "a data URI in the argument"},
            {"attachment:1", "an attachment index past what the request carries"},
            {"attachment:x", "a non-numeric attachment index"},
        };
        for (const Refusal& refusal : refusals) {
            GenerationRequest request = with_attachment(1);
            FakeGenerator generator;
            const ChatImageToolRound round = run_chat_image_tool_round(
                request, image_call_outcome({R"({"prompt":"again","image":")" + refusal.argument +
                                                       R"("})"}),
                kImageModel, kUrlBase, 0, 0, observer_of(generator));
            const std::string result = tool_result_text(request, request.messages.size() - 1);
            check(round.executed == 0 && generator.calls == 0,
                  "the refused form (" + refusal.why + ") never reaches the pipeline");
            check(result.find("image_arguments_invalid") != std::string::npos &&
                      result.find("attachment:N") != std::string::npos &&
                      result.find(refusal.argument) != std::string::npos,
                  "and comes back naming the argument, the reason and the forms that would work");
            check(request.messages.back().tool_result_is_error,
                  "the refusal is marked as an error result for the model to act on");
        }

        // attachment:N is the user's own picture, in the shape the route already takes.
        for (const std::string argument : {"attachment:0", "attachment:1"}) {
            GenerationRequest request = with_attachment(1);
            FakeGenerator generator;
            const ChatImageToolRound round = run_chat_image_tool_round(
                request, image_call_outcome({R"({"prompt":"again","image":")" + argument + R"("})"}),
                kImageModel, kUrlBase, 0, 0, observer_of(generator));
            const bool accepted = argument == "attachment:0";
            check(generator.calls == (accepted ? 1 : 0) && round.executed == (accepted ? 1 : 0) &&
                      round.rejected == 0,
                  std::string(accepted ? "attachment:0 resolves to " : "the index in ") + argument +
                      (accepted ? " the request's own attachment" : " that this request lacks"));
            if (accepted) {
                // The route's own parser keeps the base64 of a data: URI and drops the prefix, so
                // "unchanged" here means the same payload bytes, not the same string.
                check(generator.requests[0].image == data_uri.substr(data_uri.find(',') + 1),
                      "an inline attachment travels to the route as its own payload, not decoded "
                      "and re-typed");
            }
        }

        // Bytes, the shape a server-internal caller can carry.
        {
            GenerationRequest request;
            ChatTurn user;
            user.role = infernix::ChatRole::User;
            const std::vector<std::uint8_t> bytes{1, 2, 3, 4, 5};
            user.content.push_back(ContentPart{
                .kind = ContentKind::Image,
                .source = MediaSource{.kind = MediaKind::Bytes, .bytes = bytes},
            });
            request.messages.push_back(std::move(user));
            FakeGenerator generator;
            const ChatImageToolRound round = run_chat_image_tool_round(
                request, image_call_outcome({R"({"prompt":"again","image":"attachment:0"})"}),
                kImageModel, kUrlBase, 0, 0, observer_of(generator));
            check(generator.calls == 1 && round.executed == 1 &&
                      generator.requests[0].image == base64_encode_bytes(bytes),
                  "an attachment carried as bytes is encoded into the route's reference");
        }

        // Over the reference limit: refused here rather than decoded and failed deeper down.
        {
            GenerationRequest request;
            ChatTurn user;
            user.role = infernix::ChatRole::User;
            std::vector<std::uint8_t> huge(kMaximumImageInputBytes + 1, 0x20);
            user.content.push_back(ContentPart{
                .kind   = ContentKind::Image,
                .source = MediaSource{.kind = MediaKind::Bytes, .bytes = std::move(huge)},
            });
            request.messages.push_back(std::move(user));
            FakeGenerator generator;
            const ChatImageToolRound round = run_chat_image_tool_round(
                request, image_call_outcome({R"({"prompt":"again","image":"attachment:0"})"}),
                kImageModel, kUrlBase, 0, 0, observer_of(generator));
            check(generator.calls == 0 && round.executed == 0 &&
                      request.messages.back().tool_result_is_error,
                  "an attachment larger than a reference may be is refused before any work");
        }

        std::filesystem::remove("images/img-100-1.png", ignored);
        std::filesystem::remove("images", ignored);
    }
}

} // namespace

int main() {
    test_injection();
    test_image_argument_forms();
    test_round_detection();
    test_single_round();
    test_cap_and_bad_arguments();
    test_pipeline_failure_becomes_a_tool_result();
    test_url_suffix();
    test_round_loop();
    test_round_bound_and_cancellation();

    std::cout << (failures == 0 ? "CHAT_IMAGE_TOOL_TEST: PASS" : "CHAT_IMAGE_TOOL_TEST: FAIL")
              << " (" << (failures == 0 ? "all" : std::to_string(failures) + " failed") << ")\n";
    return failures == 0 ? 0 : 1;
}
