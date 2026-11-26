// Server-executed chat image tool host contract: the hidden tool is injected only where the client
// left tool choice free, a round is fed back as the same assistant+tool turns a client round-trip
// would send, the per-request picture cap and bad arguments become tool errors instead of pictures,
// and so does a failing pipeline -- the model's request was for an answer, not for the HTTP status
// of a draw. The round loop ends in one answer that carries every URL. And the renderer half, which
// the handlers cannot check for themselves: the call this process runs for itself never reaches the
// client on any of the three protocols, while the internal witness keeps it.
// CHAT_IMAGE_TOOL_TEST: PASS
#include "serve/chat_image_tool.h"
#include "product/media_acquire/source.h"
#include "serve/image_generation_service.h"

// The three protocol renderers the hidden-call rule is asserted against, and the lane's own doors
// they all read it through (serve/image_chat_hook.h).
#include "serve/anthropic_messages.h"
#include "serve/image_chat_hook.h"
#include "serve/openai_chat.h"
#include "serve/openai_responses.h"

#include <infernix/types.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
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

// A request carrying `pictures` inline attachments, in the shape the protocols parse one. Hoisted out
// of the argument-forms test because the attempt report needs a request with a picture to name, too.
[[nodiscard]] GenerationRequest request_with_attachments(std::size_t pictures,
                                                         const std::string& data_uri) {
    using MediaSource = infernix::product::media_acquire::Source;
    using MediaKind   = infernix::product::media_acquire::SourceKind;

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
}

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
    // The schema is the only place the model reads its limits, so the joint-sequence ceiling belongs in
    // its text: a model that learns the ceiling only from a 400 spends a round per guess to find it,
    // which is the five wasted rounds the refusal was measured costing.
    check(definition.input_schema_json.find("20480") != std::string::npos &&
              definition.input_schema_json.find("(W/16)x(H/16)") != std::string::npos,
          "the schema states the ceiling the canvas and the words share and how a canvas is counted");

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
    const auto with_attachment = [&](std::size_t pictures) {
        return request_with_attachments(pictures, data_uri);
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
                      generator.requests[0].images == std::vector<std::string>{expected},
                  "the picture named \"" + (name.size() > 40 ? name.substr(0, 40) : name) +
                      "\" is read back off disk as the route's reference");
        }

        // The list form. The route takes up to kMaximumImageReferences pictures and the chat argument
        // has the same two shapes, each element resolved by its own rule -- which is what lets an
        // instruction address <image1> and <image2> instead of one picture flattened in by hand. Before
        // this, a list was left exactly as the model typed it and every element died as an undecodable
        // reference, so the capability the route has was invisible to the chat path.
        {
            GenerationRequest request = with_attachment(1);
            FakeGenerator generator;
            const ChatImageToolRound round = run_chat_image_tool_round(
                request,
                image_call_outcome(
                    {R"({"prompt":"keep the pose","image":["attachment:0","img-100-1"]})"}),
                kImageModel, kUrlBase, 0, 0, observer_of(generator));
            check(round.executed == 1 && generator.requests.size() == 1 &&
                      generator.requests[0].images.size() == 2 &&
                      generator.requests[0].images[0] ==
                          data_uri.substr(data_uri.find(',') + 1) &&
                      generator.requests[0].images[1] == expected,
                  "a list of two forms reaches the route as two references, the attachment's own "
                  "payload first and the drawn picture's bytes second");
        }
        {
            // The same list, typed as a STRING. The schema declares `image` as a string, and a model
            // that obeys that type while still passing the list embeds it as text -- which is exactly
            // what /v1/chat/completions sent on 2026-11-02 for a two-reference instruction. It must
            // resolve like the array it holds, not come back as an unusable job id.
            GenerationRequest request = with_attachment(1);
            FakeGenerator generator;
            const ChatImageToolRound round = run_chat_image_tool_round(
                request,
                image_call_outcome({
                    R"({"prompt":"keep the pose","image":"[\"attachment:0\",\"img-100-1\"]"})"}),
                kImageModel, kUrlBase, 0, 0, observer_of(generator));
            check(round.executed == 1 && generator.requests.size() == 1 &&
                      generator.requests[0].images.size() == 2 &&
                      generator.requests[0].images[0] ==
                          data_uri.substr(data_uri.find(',') + 1) &&
                      generator.requests[0].images[1] == expected,
                  "a list typed as one string is still the list form: both entries resolve and the "
                  "route gets two references");
        }
        {
            // A bracket that is not a JSON array is not the list form either: it stays the string the
            // model typed and comes back refused with the accepted forms, so nothing new is accepted.
            GenerationRequest request = with_attachment(1);
            FakeGenerator generator;
            const ChatImageToolRound round = run_chat_image_tool_round(
                request, image_call_outcome({R"({"prompt":"a","image":"[not json]"})"}),
                kImageModel, kUrlBase, 0, 0, observer_of(generator));
            check(round.executed == 0 && generator.calls == 0,
                  "a string that only looks like a list draws nothing");
            check(tool_result_text(request, request.messages.size() - 1)
                      .find("attachment:N") != std::string::npos,
                  "and it is refused with the forms the model may type instead");
        }
        {
            // Past the cap the refusal is the ROUTE's, not one this layer invents: the elements are
            // resolved first and the count is still refused, so the model gets the code and the wording
            // the capability document promises for that mistake.
            GenerationRequest request = with_attachment(1);
            FakeGenerator generator;
            const std::string five =
                R"({"prompt":"a","image":["img-100-1","img-100-1","img-100-1","img-100-1","img-100-1"]})";
            const ChatImageToolRound round = run_chat_image_tool_round(
                request, image_call_outcome({five}), kImageModel, kUrlBase, 0, 0,
                observer_of(generator));
            check(round.executed == 0 && generator.calls == 0,
                  "five references in one list draw nothing");
            check(tool_result_text(request, request.messages.size() - 1)
                      .find("reference images") != std::string::npos,
                  "and the refusal that reaches the model is the route's count error in its own words");
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
                check(generator.requests[0].images ==
                          std::vector<std::string>{data_uri.substr(data_uri.find(',') + 1)},
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
                      generator.requests[0].images ==
                          std::vector<std::string>{base64_encode_bytes(bytes)},
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

// The client's own witness of every picture this request asked for. The rounds were invisible by
// construction: the tool result went back to the model, which had seen the refusal and told the user
// something else (2026-11-02, a face swap refused by the model itself, then reported as a VRAM
// shortage of a figure nobody had read from anywhere). These are the facts the response now carries
// beside the answer -- the ask as the call named it, and which half of it the server answered.
void test_attempts_are_reported() {
    {
        // A picture that came: both halves, ask and result, and no error half.
        GenerationRequest request =
            request_with_attachments(1, "data:image/png;base64,iVBORw0KGgo=");
        FakeGenerator generator;
        const ChatImageToolRound round = run_chat_image_tool_round(
            request,
            image_call_outcome({R"({"prompt":"a","size":"768x768","steps":20,"seed":7,)"
                               R"("image":"attachment:0","prompt_enhance":"auto"})"}),
            kImageModel, kUrlBase, 0, 0, observer_of(generator));
        const RequestJson attempts = chat_image_attempts_json(round.attempts);
        check(round.attempts.size() == 1 && attempts.size() == 1,
              "one tool call is one attempt");
        const RequestJson& item = attempts.at(0);
        check(item.at("outcome") == "succeeded" && item.at("tool_call_id") == "infernix_img_0",
              "a drawn picture is reported as succeeded, tied to the call that asked for it");
        const RequestJson& asked = item.at("request");
        check(asked.at("size") == "768x768" && asked.at("steps") == 20 &&
                  asked.at("image_count") == 1 && asked.at("seed") == 7 &&
                  asked.at("prompt_enhance") == "auto",
              "the attempt reports the size, steps, reference count, seed and rewrite the call asked "
              "for -- read before the reference was replaced by base64");
        const RequestJson& result = item.at("result");
        check(result.at("url") == round.urls.front() && result.at("id") == round.ids.front(),
              "and the artifact the server actually handed back");
        check(!item.contains("error"), "a picture that came carries no error half");
    }
    {
        // A refusal keeps the ask: the client needs the size that was refused to know what to retry.
        GenerationRequest request;
        FakeGenerator generator;
        generator.fail_status = 503;
        generator.fail_code   = "image_vram_insufficient";
        const ChatImageToolRound round = run_chat_image_tool_round(
            request, image_call_outcome({R"({"prompt":"a","size":"2048x2048"})"}), kImageModel,
            kUrlBase, 0, 0, observer_of(generator));
        const RequestJson item = chat_image_attempts_json(round.attempts).at(0);
        check(item.at("outcome") == "refused" &&
                  item.at("error").at("code") == "image_vram_insufficient" &&
                  item.at("error").at("http_status") == 503,
              "a route refusal is reported as refused with the route's own code and status");
        check(item.at("request").at("size") == "2048x2048" && !item.contains("result"),
              "and the size that was refused travels with it, without a result half");
    }
    {
        // A 5xx is not an argument: the same record, a different outcome, so a client retries the
        // request instead of rewriting it.
        GenerationRequest request;
        FakeGenerator generator;
        generator.fail_status = 500;
        generator.fail_code   = "image_generation_failed";
        const ChatImageToolRound round = run_chat_image_tool_round(
            request, image_call_outcome({R"({"prompt":"a"})"}), kImageModel, kUrlBase, 0, 0,
            observer_of(generator));
        const RequestJson item = chat_image_attempts_json(round.attempts).at(0);
        check(item.at("outcome") == "failed" &&
                  item.at("error").at("code") == "image_generation_failed",
              "a draw that broke inside the server is reported as failed, not refused");
    }
    {
        // The same word, on the operator's side. Both witnesses answer one question about one draw, and
        // they used to answer it differently: the image_job line mapped any 5xx to "failed", so a 503
        // image_vram_insufficient -- a refusal the client was correctly told is a refusal -- read
        // afterwards as a broken server to whoever opened requests.jsonl. One function now, over every
        // status an image route answers with.
        check(std::string(image_outcome_word(500)) == "failed" &&
                  std::string(image_outcome_word(400)) == "refused" &&
                  std::string(image_outcome_word(409)) == "refused" &&
                  std::string(image_outcome_word(429)) == "refused" &&
                  std::string(image_outcome_word(503)) == "refused",
              "a 500 is the only failed answer; the admission gates' 400, 409, 429 and 503 are refusals "
              "in requests.jsonl too, so the operator's line and the client's attempt cannot disagree "
              "about the same draw");
    }
    {
        // The per-request cap and unparseable arguments are refusals too, with the status a client
        // would have seen on the route. The cap refuses before the arguments are read, so it has no
        // ask to report -- and says so by leaving the field out rather than inventing one.
        GenerationRequest capped;
        FakeGenerator generator;
        const ChatImageToolRound over_cap = run_chat_image_tool_round(
            capped, image_call_outcome({R"({"prompt":"a","size":"512x512"})"}), kImageModel, kUrlBase,
            0, kMaximumChatImageGenerations, observer_of(generator));
        const RequestJson limited = chat_image_attempts_json(over_cap.attempts).at(0);
        check(limited.at("outcome") == "refused" &&
                  limited.at("error").at("code") == "image_generation_limit_reached" &&
                  limited.at("error").at("http_status") == 429 && !limited.contains("request"),
              "the picture cap is reported as a refusal with its status, and with no ask it never read");

        GenerationRequest malformed;
        const ChatImageToolRound bad_json = run_chat_image_tool_round(
            malformed, image_call_outcome({"this is not json"}), kImageModel, kUrlBase, 0, 0,
            observer_of(generator));
        const RequestJson bad = chat_image_attempts_json(bad_json.attempts).at(0);
        check(bad.at("outcome") == "refused" &&
                  bad.at("error").at("code") == "image_arguments_invalid" &&
                  bad.at("error").at("http_status") == 400,
              "arguments that never were a request are reported as the 400 the route would have said");
    }

    {
        // Through the loop: the round number is the loop's, so a client can tell three tries from one,
        // and the whole array rides to the response on the answer itself.
        GenerationRequest request;
        FakeGenerator generator;
        GenerationOutcome first  = image_call_outcome({R"({"prompt":"one"})"});
        GenerationOutcome second = image_call_outcome({R"({"prompt":"two"})"});
        std::vector<GenerationOutcome> pending{std::move(second),
                                               prose_outcome("both drawn")};
        const ChatImageRounds loop = run_chat_image_rounds(
            request, std::move(first), kImageModel, kUrlBase, 0, observer_of(generator), [&] {
                GenerationOutcome next = std::move(pending.front());
                pending.erase(pending.begin());
                return next;
            });
        const RequestJson attempts = chat_image_attempts_for_response(loop.answer);
        check(attempts.is_array() && attempts.size() == 2 && loop.attempts.size() == 2,
              "both rounds' attempts reach the answer as one array");
        check(attempts.at(0).at("round") == 1 && attempts.at(1).at("round") == 2,
              "each is numbered by the round it ran in, not by its own call");
        check(loop.answer.image_tool_attempts_json.has_value(),
              "the answer carries the serialized form the response writers emit");
    }
    {
        // The byte-identity rule: a request that drew nothing has no field at all -- not an empty
        // array, which a client would read as "it tried and produced nothing".
        GenerationRequest request;
        FakeGenerator generator;
        const ChatImageRounds loop = run_chat_image_rounds(
            request, prose_outcome("just an answer"), kImageModel, kUrlBase, 0,
            observer_of(generator), [] { return prose_outcome("unreachable"); });
        check(!loop.answer.image_tool_attempts_json.has_value() &&
                  loop.attempts.empty() && chat_image_attempts_for_response(loop.answer).is_null(),
              "a plain chat answer asks for no picture and gains no field");
    }
}

// ---------------------------------------------------------------------------
// The renderer half of the hidden-call rule.
//
// A call this process runs for itself is decided by `chat_image_tool_hidden` in each of the three
// protocol renderers -- not by the handlers, which have no view of what they wrote to the socket. A
// replay onto upstream tip can lose the call in one renderer, in one half of a renderer, or in one of
// the two carriers a stream ships the witness on, and no compiler reports any of it: the stream builds,
// answers, and either leaks the server's own call to the client or loses the picture's fate with it. So
// these cases drive the renderers the way serve/openai_chat_http.cpp, serve/anthropic_messages_http.cpp
// and serve/openai_responses_http.cpp do -- the deltas of the picture round included -- and read only the
// bytes a client is handed.
// ---------------------------------------------------------------------------

// What one tool-call delta published, whatever shape this tree's adapter answers with. This fork's
// adapters refuse an unusable delta with an error instead of throwing, which upstream tip has no notion
// of, and a suppressed call costs the empty chunk: so only the bytes that reached the wire travel on,
// and a delta this suite fed badly is said so rather than read as a silence.
template <class Rendered>
[[nodiscard]] std::vector<std::string> published(Rendered&& rendered) {
    using Value = std::decay_t<Rendered>;
    std::vector<std::string> chunks;
    if constexpr (std::is_same_v<Value, std::string>) {
        if (!rendered.empty()) { chunks.push_back(std::move(rendered)); }
    } else if constexpr (std::is_same_v<Value, std::vector<std::string>>) {
        for (std::string& line : rendered) {
            if (!line.empty()) { chunks.push_back(std::move(line)); }
        }
    } else {
        if constexpr (requires { rendered.ok(); }) {
            check(rendered.ok(), "a tool-call delta of the shape the handlers feed was refused");
        }
        if constexpr (requires { rendered.chunk; }) {
            if (!rendered.chunk.empty()) { chunks.push_back(std::move(rendered.chunk)); }
        } else {
            for (std::string& line : rendered.events) {
                if (!line.empty()) { chunks.push_back(std::move(line)); }
            }
        }
    }
    return chunks;
}

template <class Value>
void append_all(std::vector<Value>& wire, std::vector<Value> next) {
    wire.insert(wire.end(), std::make_move_iterator(next.begin()), std::make_move_iterator(next.end()));
}

// The bytes of one tool call as the constrained route emits them, which is the form the renderer's own
// positional check matches by: the first delta names it, the next carry the argument bytes, the last
// completes it, and the whole of it concatenates to the arguments the round's result holds.
template <class Stream>
void stream_call(Stream& stream, std::vector<std::string>& wire, std::uint32_t engine_index,
                 std::string_view name, std::string_view arguments) {
    const std::size_t half = arguments.size() / 2U;
    append_all(wire, published(stream.tool_call_delta(
                         {.index     = engine_index,
                          .name      = std::string(name),
                          .arguments = std::string(arguments.substr(0, half))})));
    append_all(wire, published(stream.tool_call_delta(
                         {.index     = engine_index,
                          .arguments = std::string(arguments.substr(half))})));
    append_all(wire, published(stream.tool_call_delta(
                         {.index = engine_index, .complete = true})));
}

// One SSE frame's document. The two OpenAI protocols ship `data: <json>` on its own; the Messages
// route and the Responses route prefix the same line with their own `event:` line, and a Chat stream
// closes with the terminator rather than a document.
[[nodiscard]] std::string_view data_of(std::string_view frame) {
    std::size_t start = 0;
    if (frame.starts_with("data: ")) {
        start = 6;
    } else if (frame.starts_with("event: ")) {
        const std::size_t line = frame.find("\ndata: ");
        if (line == std::string_view::npos) { return {}; }
        start = line + 1 + 6;
    } else {
        return {};
    }
    std::string_view data = frame.substr(start);
    if (data.ends_with("\n\n")) { data.remove_suffix(2); }
    return data == "[DONE]" ? std::string_view{} : data;
}

[[nodiscard]] bool is_frame(const std::string& wire) { return !data_of(wire).empty(); }

[[nodiscard]] RequestJson frame_of(const std::string& wire) {
    return RequestJson::parse(std::string(data_of(wire)));
}

// The lane's own field, wherever a protocol's document hides it: the Chat chunk carries it beside
// usage, the Anthropic terminal event beside stop_reason, the Responses terminal event inside its
// response. It is the one part of a response allowed to speak of the draw, so it comes out before the
// wire is searched for the call, and a client reads it once per response.
void collect_field(RequestJson& document, std::string_view field, std::vector<RequestJson>& found) {
    const std::string key(field);
    if (document.is_object()) {
        if (document.contains(key)) {
            found.push_back(document.at(key));
            document.erase(key);
        }
        for (auto& entry : document) { collect_field(entry, field, found); }
    } else if (document.is_array()) {
        for (auto& element : document) { collect_field(element, field, found); }
    }
}

[[nodiscard]] std::vector<RequestJson> witnesses_in(RequestJson document, std::string_view field) {
    std::vector<RequestJson> found;
    collect_field(document, field, found);
    return found;
}

void strip_field(RequestJson& document, std::string_view field) {
    std::vector<RequestJson> kept;
    collect_field(document, field, kept);
}

[[nodiscard]] std::vector<RequestJson> witnesses_of(const std::vector<std::string>& frames,
                                                    std::string_view field) {
    std::vector<RequestJson> found;
    for (const std::string& frame : frames) {
        if (!is_frame(frame)) { continue; }
        append_all(found, witnesses_in(frame_of(frame), field));
    }
    return found;
}

// Everything a client was handed for this response, with the witness taken out and the protocol's
// framing stripped, so a hit is a name or an argument byte that rode some field it never belonged on.
[[nodiscard]] std::string client_text(const std::vector<std::string>& frames, std::string_view field) {
    std::string text;
    for (const std::string& frame : frames) {
        if (!is_frame(frame)) { continue; }
        RequestJson document = frame_of(frame);
        document.erase("usage"); // the client's own prompt, which is not the server's call
        strip_field(document, field);
        text += document.dump();
    }
    return text;
}

// Reading what a renderer wrote: a field the protocol did not put there costs a failed check, never
// the suite, because what is under test is the shape of the whole response rather than one key.
[[nodiscard]] RequestJson at_or_null(const RequestJson& document, const char* key) {
    return document.is_object() && document.contains(key) ? document.at(key) : RequestJson();
}

[[nodiscard]] std::string text_at(const RequestJson& document, const char* key) {
    const RequestJson value = at_or_null(document, key);
    return value.is_string() ? value.get<std::string>() : std::string();
}

[[nodiscard]] int int_at(const RequestJson& document, const char* key) {
    const RequestJson value = at_or_null(document, key);
    return value.is_number_integer() ? value.get<int>() : -1;
}

// The delta of one streamed Chat chunk, whose answer a client accumulates. A usage-only chunk carries
// no choice at all, which is why this is read rather than indexed.
[[nodiscard]] RequestJson delta_of(const RequestJson& document) {
    const RequestJson choices = at_or_null(document, "choices");
    if (!choices.is_array() || choices.empty() || !choices.at(0).is_object()) { return RequestJson(); }
    return at_or_null(choices.at(0), "delta");
}

// A renderer's refusal of a call sequence it cannot reconcile is a std::logic_error, which a handler
// turns into one terminal error event: on the wire that is a committed stream that never got its answer.
// So the suite asks for the throw and for its words, which name the rule that was not met -- and takes
// any other escape from the renderer as a failed check too, rather than letting it end the suite.
[[nodiscard]] std::string throw_from(const std::function<void()>& action) {
    try {
        action();
    } catch (const std::exception& error) {
        return error.what();
    }
    return {};
}

// One hidden call the lane really ran, so the renderer's cases read the tool's own writing: the picture
// drawn, the round numbered, the attempt tied to the call id that asked for it, and the prose answer the
// round was replaced by.
struct PictureRound {
    GenerationRequest request;
    FakeGenerator generator;
    ChatImageRounds rounds;
};

[[nodiscard]] PictureRound drew_one_picture(std::string_view arguments, std::string_view prose) {
    PictureRound state;
    state.request.messages.push_back(ChatTurn{.role = infernix::ChatRole::User});
    state.rounds = run_chat_image_rounds(
        state.request, image_call_outcome({std::string(arguments)}), kImageModel, kUrlBase, 0,
        observer_of(state.generator), [&] { return prose_outcome(std::string(prose)); });
    return state;
}

// The name every protocol puts the draw's fate under, and the whole record one drawn picture leaves
// there: the ask as the call named it, the artifact the server handed back, the round and the call id
// that tie them. The renderer is what has to carry it while carrying no call for it.
constexpr std::string_view kAttemptField = "image_tool_attempts";

void check_attempt_keeps_the_draw(const RequestJson& witness, const std::string& url) {
    if (!witness.is_array() || witness.size() != 1) {
        check(false, "the draw's witness is not the one record this response owes the client");
        return;
    }
    const RequestJson item = witness.at(0);
    check(text_at(item, "outcome") == "succeeded" && int_at(item, "round") == 1 &&
              text_at(item, "tool_call_id") == "infernix_img_0",
          "the internal witness lost the round and the call the server ran for itself");
    const RequestJson asked = at_or_null(item, "request");
    check(text_at(asked, "size") == "512x512" && int_at(asked, "steps") == 4,
          "the internal witness lost the ask the call named");
    check(text_at(at_or_null(item, "result"), "url") == url, "the internal witness lost the picture it drew");
    check(!item.contains("error"), "a picture that came is reported to the client as an error");
}

// The Chat renderer, streaming, on the answer the picture rounds left behind: no frame of the response
// carries the server's own call in any field, the renderer's own bookkeeping does not keep it either,
// and the one chunk that closes the stream is the one that carries what really happened.
void test_chat_renderer_hides_the_server_run_call() {
    const std::string arguments = R"({"prompt":"a red fox","size":"512x512","steps":4})";
    PictureRound picture        = drew_one_picture(arguments, " and here it is");
    check(picture.rounds.rounds == 1 && picture.rounds.generations == 1 &&
              picture.rounds.attempts.size() == 1 && picture.rounds.answer.tool_calls.empty(),
          "the renderer case is not standing on the round it means");

    const OpenAIChatResponseIdentity identity = make_openai_chat_response_identity("infernix");
    OpenAIChatStream stream(identity, /*include_usage =*/true);
    std::vector<std::string> wire;
    append_all(wire, published(stream.start()));
    append_all(wire, published(stream.content_delta("Drawing that now.")));
    // The handler feeds every delta of the live round to the encoder, the server's own call included:
    // keeping it off the wire is the renderer's job and nobody else's.
    stream_call(stream, wire, 0, kChatImageToolName, arguments);

    std::vector<std::string> terminal;
    const std::string refused =
        throw_from([&] { terminal = stream.finish(picture.rounds.answer); });
    // The round replaced its own answer, so the answer this stream closes on names no call at all.
    // Upstream's positional check has to be settled by the lane's hidden-call rule rather than by a
    // throw, which would end a committed stream with an error event and no answer.
    check(refused.empty(),
          "the chat renderer refused the replaced answer of a picture round: " + refused);
    append_all(wire, terminal);
    check(!wire.empty() && wire.back() == "data: [DONE]\n\n",
          "the chat stream ended without closing as an answered one");

    check(client_text(wire, kAttemptField).find(std::string(kChatImageToolName)) == std::string::npos,
          "a chunk reached the client naming the tool this process runs for itself");
    check(client_text(wire, kAttemptField).find("512x512") == std::string::npos,
          "a chunk reached the client carrying the hidden call's arguments");
    bool published_a_call = false;
    for (const std::string& frame : wire) {
        if (!is_frame(frame)) { continue; }
        published_a_call = published_a_call || delta_of(frame_of(frame)).contains("tool_calls");
    }
    check(!published_a_call, "the terminal answer of a picture round published a tool call");
    check(stream.tool_call_ids().empty(),
          "the chat renderer's own call bookkeeping kept the call the client never saw");

    const std::vector<RequestJson> witnesses = witnesses_of(wire, kAttemptField);
    check(witnesses.size() == 1,
          "the chat stream did not carry the picture's fate to the client exactly once");
    if (witnesses.size() == 1) { check_attempt_keeps_the_draw(witnesses[0], picture.rounds.urls[0]); }

    // The same answer on the non-streaming route, which is the other reader of the lane's two doors.
    const RequestJson body =
        RequestJson::parse(make_chat_completion_response(identity, picture.rounds.answer));
    const std::vector<RequestJson> aggregate = witnesses_in(body, kAttemptField);
    const RequestJson choice = at_or_null(body, "choices").is_array() && !at_or_null(body, "choices").empty()
                                   ? at_or_null(body, "choices").at(0)
                                   : RequestJson();
    check(aggregate.size() == 1 && !at_or_null(choice, "message").contains("tool_calls") &&
              text_at(choice, "finish_reason") == "stop",
          "the aggregate chat answer either lost the draw's fate or answered with a call for it");
    if (aggregate.size() == 1) { check_attempt_keeps_the_draw(aggregate[0], picture.rounds.urls[0]); }
}

// The other half of the same rule, and the half a renderer can lose while the first still holds: the
// hidden call earns no published index, so the calls the client does execute are numbered as if it had
// never been in the turn. A round that mixes a client tool in is the handler's own round, and it carries
// both calls, which is the shape the renderer's positional check is written for.
void test_chat_renderer_keeps_client_call_numbering_dense() {
    const std::string lookup_first  = R"({"q":"one"})";
    const std::string lookup_second = R"({"q":"two"})";
    const std::string hidden        = R"({"prompt":"a whale","size":"512x512","steps":4})";

    GenerationOutcome mixed = image_call_outcome({hidden});
    mixed.text              = "";
    mixed.tool_calls.insert(
        mixed.tool_calls.begin(),
        infernix::GeneratedToolCall{.name = "lookup", .arguments_json = lookup_first});
    mixed.tool_calls.push_back(
        infernix::GeneratedToolCall{.name = "lookup", .arguments_json = lookup_second});

    const OpenAIChatResponseIdentity identity = make_openai_chat_response_identity("infernix");
    OpenAIChatStream stream(identity, /*include_usage =*/false);
    std::vector<std::string> wire;
    append_all(wire, published(stream.start()));
    stream_call(stream, wire, 0, "lookup", lookup_first);
    // The Engine numbers a turn's calls from zero and counts the server's own call among them: a
    // renderer that published the Engine's number, or that let the hidden call take a published one,
    // leaves this round's client calls at 1 and 2 on a protocol whose clients index them from 0.
    stream_call(stream, wire, 1, kChatImageToolName, hidden);
    stream_call(stream, wire, 2, "lookup", lookup_second);

    std::vector<std::string> terminal;
    const std::string refused = throw_from([&] { terminal = stream.finish(mixed); });
    check(refused.empty(), "the chat renderer refused a round it was handed: " + refused);
    append_all(wire, terminal);

    std::vector<std::pair<int, std::string>> opened;
    std::vector<std::string> published_ids;
    std::string arguments_of_client_calls;
    for (const std::string& frame : wire) {
        if (!is_frame(frame)) { continue; }
        const RequestJson delta = delta_of(frame_of(frame));
        if (!delta.contains("tool_calls")) { continue; }
        for (const RequestJson& call : delta.at("tool_calls")) {
            const RequestJson function = at_or_null(call, "function");
            arguments_of_client_calls += text_at(function, "arguments");
            const std::string id = text_at(call, "id");
            if (id.empty()) { continue; } // a continuation chunk: it reopens nothing
            opened.emplace_back(int_at(call, "index"), text_at(function, "name"));
            check(id.starts_with("call_"),
                  "a published call carried an id the client cannot answer with");
            published_ids.push_back(id);
        }
    }
    const std::vector<std::pair<int, std::string>> expected{
        {0, "lookup"}, {1, "lookup"}};
    check(opened == expected,
          "the calls a client has to execute were not numbered 0,1 with the server's own call between "
          "them");
    // Each call's chunks concatenate to the arguments its tool was declared with, which is what a
    // client executes them from, and what the renderer matched the result against.
    check(arguments_of_client_calls == lookup_first + lookup_second,
          "a client call lost or gained argument bytes on the way to the client");
    // Two calls opened, so two identifiers earned: the hidden call took no chunk and therefore none
    // of its own, and the client's two are the ones the renderer reports back to its handler.
    const std::vector<std::string> bookkeeping = stream.tool_call_ids();
    check(published_ids.size() == 2 && published_ids[0] != published_ids[1] &&
              std::find(bookkeeping.begin(), bookkeeping.end(), published_ids[0]) !=
                  bookkeeping.end() &&
              std::find(bookkeeping.begin(), bookkeeping.end(), published_ids[1]) !=
                  bookkeeping.end(),
          "the calls the renderer reports to its handler are not the ones it published");
    check(client_text(wire, kAttemptField).find(std::string(kChatImageToolName)) == std::string::npos &&
              client_text(wire, kAttemptField).find("a whale") == std::string::npos,
          "the server's own call named itself, or carried its ask, on a wire the client reads");
}

// The Messages route, which is the one protocol whose body and terminal event are the request's own
// ordered document: it reaches the lane through the `ordered_json` half of `add_chat_image_attempts`,
// and a call the server runs for itself earns no content block at all -- not even the block that would
// have closed the text block beside it.
void test_anthropic_renderer_hides_the_server_run_call() {
    const std::string arguments = R"({"prompt":"a red fox","size":"512x512","steps":4})";
    PictureRound picture        = drew_one_picture(arguments, " and here it is");

    const AnthropicResponseIdentity identity = make_anthropic_response_identity("req_hidden", "claude-local");
    AnthropicMessagesStream stream(identity, /*input_tokens =*/10);
    std::vector<std::string> wire{stream.start()};
    append_all(wire, published(stream.content_delta("Drawing that now.")));
    stream_call(stream, wire, 0, kChatImageToolName, arguments);

    std::vector<std::string> terminal;
    const std::string refused =
        throw_from([&] { terminal = stream.finish(picture.rounds.answer); });
    check(refused.empty(),
          "the Messages renderer refused the replaced answer of a picture round: " + refused);
    append_all(wire, terminal);

    // The text blocks the prose streamed as are the response's whole content: a call the server ran
    // for itself earns neither a `tool_use` block of its own nor the block that would close the text
    // beside it.
    int tool_blocks = 0;
    for (const std::string& frame : wire) {
        if (!is_frame(frame)) { continue; }
        const RequestJson event = frame_of(frame);
        if (text_at(event, "type") != "content_block_start") { continue; }
        const std::string kind = text_at(at_or_null(event, "content_block"), "type");
        if (kind == "tool_use" || kind == "server_tool_use") { ++tool_blocks; }
    }
    check(client_text(wire, kAttemptField).find(std::string(kChatImageToolName)) == std::string::npos,
          "a Messages content block named the tool this process runs for itself");
    check(client_text(wire, kAttemptField).find("512x512") == std::string::npos,
          "a Messages block carried the hidden call's input");
    check(tool_blocks == 0, "the server's own picture call earned the client a content block");

    const std::vector<RequestJson> witnesses = witnesses_of(wire, kAttemptField);
    check(witnesses.size() == 1,
          "the Messages stream did not carry the picture's fate to the client exactly once");
    if (witnesses.size() == 1) { check_attempt_keeps_the_draw(witnesses[0], picture.rounds.urls[0]); }

    const RequestJson body =
        RequestJson::parse(make_anthropic_messages_response(identity, picture.rounds.answer));
    const std::vector<RequestJson> aggregate = witnesses_in(body, kAttemptField);
    int aggregate_blocks = 0;
    if (at_or_null(body, "content").is_array()) {
        for (const RequestJson& block : body.at("content")) {
            const std::string kind = text_at(block, "type");
            if (kind == "tool_use" || kind == "server_tool_use") { ++aggregate_blocks; }
        }
    }
    check(aggregate.size() == 1 && aggregate_blocks == 0 && text_at(body, "stop_reason") == "end_turn",
          "the aggregate Messages answer either lost the draw's fate or answered with a block for it");
    if (aggregate.size() == 1) { check_attempt_keeps_the_draw(aggregate[0], picture.rounds.urls[0]); }

    // A round that mixes a client tool in is handed over as the segment it generated, and that is where
    // the hidden call can still do damage: it must earn no block and no position, so the client's call
    // keeps the index the text block left it and its input still arrives whole.
    const std::string lookup_arguments = R"({"q":"one"})";
    AnthropicMessagesStream mixed_stream(identity, /*input_tokens =*/10);
    std::vector<std::string> mixed_wire{mixed_stream.start()};
    append_all(mixed_wire, published(mixed_stream.content_delta("One moment.")));
    stream_call(mixed_stream, mixed_wire, 0, kChatImageToolName, arguments);
    stream_call(mixed_stream, mixed_wire, 1, "lookup", lookup_arguments);

    // The calls in the order the round streamed them, because that is the order the renderer's own
    // positional check matches by.
    GenerationOutcome mixed = image_call_outcome({arguments});
    mixed.text              = "One moment.";
    mixed.tool_calls.push_back(
        infernix::GeneratedToolCall{.name = "lookup", .arguments_json = lookup_arguments});
    std::vector<std::string> mixed_terminal;
    const std::string mixed_refused =
        throw_from([&] { mixed_terminal = mixed_stream.finish(mixed); });
    check(mixed_refused.empty(), "the Messages renderer refused a round it was handed: " + mixed_refused);
    append_all(mixed_wire, mixed_terminal);

    int mixed_tool_blocks                 = 0;
    int published_block_index             = -1;
    std::string delivered_input;
    for (const std::string& frame : mixed_wire) {
        if (!is_frame(frame)) { continue; }
        const RequestJson event = frame_of(frame);
        const std::string type  = text_at(event, "type");
        if (type == "content_block_start") {
            const RequestJson block = at_or_null(event, "content_block");
            if (text_at(block, "type") != "tool_use") { continue; }
            ++mixed_tool_blocks;
            published_block_index = int_at(event, "index");
            check(text_at(block, "name") == "lookup" && text_at(block, "id").starts_with("toolu_"),
                  "the block a client has to execute was not named and identified as its own");
        } else if (type == "content_block_delta" &&
                   text_at(at_or_null(event, "delta"), "type") == "input_json_delta") {
            delivered_input += text_at(at_or_null(event, "delta"), "partial_json");
        }
    }
    check(mixed_tool_blocks == 1 && published_block_index == 1 && delivered_input == lookup_arguments,
          "the server's own call took a content block or a block index from the client's call");
    check(client_text(mixed_wire, kAttemptField).find("a red fox") == std::string::npos &&
              client_text(mixed_wire, kAttemptField).find(std::string(kChatImageToolName)) ==
                  std::string::npos,
          "the hidden call's ask or name reached the client beside the call that did");
}

// The Responses route, where the same rule is a whole Item that must not exist: a hidden call is
// recorded, its round's result counts it, and no `response.output_item.added`, no argument delta and no
// terminal Item ever names it -- while the Items around it never notice it was there.
void test_responses_renderer_hides_the_server_run_call() {
    const RequestJson body =
        RequestJson{{"model", "m"},
                    {"input", "draw a fox"},
                    {"tools", RequestJson::array({RequestJson{
                                  {"type", "function"},
                                  {"name", "lookup"},
                                  {"parameters", RequestJson{{"type", "object"},
                                                             {"properties",
                                                              RequestJson{{"q", RequestJson{
                                                                  {"type", "string"}}}}}}}}})}};
    const OpenAIResponsesCreateRequest request =
        parse_openai_responses_create_request(body, RequestLimits{});
    const std::string hidden_arguments = R"({"prompt":"a whale","size":"512x512","steps":4})";
    const std::string lookup_arguments = R"({"q":"one"})";

    // One hidden call the lane ran, for the answer the renderer closes on.
    PictureRound picture = drew_one_picture(hidden_arguments, " and here it is");

    OpenAIResponsesEventStream hidden_only("resp_hidden", 1, request, {});
    std::vector<std::string> wire = hidden_only.start();
    append_all(wire, published(hidden_only.content_delta("Drawing that now.")));
    stream_call(hidden_only, wire, 0, kChatImageToolName, hidden_arguments);
    OpenAIResponsesStreamFinish finish;
    const std::string refused = throw_from([&] { finish = hidden_only.finish(picture.rounds.answer); });
    check(refused.empty(),
          "the Responses renderer refused the replaced answer of a picture round: " + refused);
    append_all(wire, finish.events_before_terminal);
    if (refused.empty()) { append_all(wire, {hidden_only.terminal(finish.response)}); }

    check(client_text(wire, kAttemptField).find(std::string(kChatImageToolName)) == std::string::npos &&
              client_text(wire, kAttemptField).find("a whale") == std::string::npos,
          "a Responses event named the server's own call, or carried its ask");
    for (const std::string& frame : wire) {
        if (!is_frame(frame)) { continue; }
        const RequestJson event = frame_of(frame);
        if (text_at(event, "type") != "response.output_item.added") { continue; }
        check(text_at(at_or_null(event, "item"), "name") != std::string(kChatImageToolName),
              "the server's own call became an Item the client has to answer");
    }
    const std::vector<RequestJson> witnesses = witnesses_of(wire, kAttemptField);
    check(witnesses.size() == 1,
          "the Responses stream did not carry the picture's fate to the client exactly once");
    if (witnesses.size() == 1) { check_attempt_keeps_the_draw(witnesses[0], picture.rounds.urls[0]); }

    // A round that mixes a client call in is handed over as it is, and this is the half of the rule that
    // a hidden call in the middle of it can break: the client's call keeps the output position it would
    // have had, and its arguments still arrive.
    OpenAIResponsesEventStream mixed("resp_mixed", 1, request, {});
    std::vector<std::string> mixed_wire = mixed.start();
    append_all(mixed_wire, published(mixed.content_delta("One moment.")));
    stream_call(mixed, mixed_wire, 0, kChatImageToolName, hidden_arguments);
    stream_call(mixed, mixed_wire, 1, "lookup", lookup_arguments);

    GenerationOutcome round = image_call_outcome({hidden_arguments});
    round.text              = "One moment.";
    round.tool_calls.push_back(
        infernix::GeneratedToolCall{.name = "lookup", .arguments_json = lookup_arguments});
    OpenAIResponsesStreamFinish mixed_finish;
    const std::string mixed_refused = throw_from([&] { mixed_finish = mixed.finish(round); });
    check(mixed_refused.empty(), "the Responses renderer refused a round it was handed: " + mixed_refused);
    append_all(mixed_wire, mixed_finish.events_before_terminal);
    if (mixed_refused.empty()) { append_all(mixed_wire, {mixed.terminal(mixed_finish.response)}); }

    std::vector<int> call_positions;
    std::string delivered_arguments;
    for (const std::string& frame : mixed_wire) {
        if (!is_frame(frame)) { continue; }
        const RequestJson event = frame_of(frame);
        const std::string type  = text_at(event, "type");
        if (type == "response.output_item.added" &&
            text_at(at_or_null(event, "item"), "type") == "function_call") {
            call_positions.push_back(int_at(event, "output_index"));
            check(text_at(at_or_null(event, "item"), "name") == "lookup",
                  "a Responses Item named a tool the client never declared");
        } else if (type == "response.function_call_arguments.delta") {
            delivered_arguments += text_at(event, "delta");
        }
    }
    check(call_positions == std::vector<int>{1} && delivered_arguments == lookup_arguments,
          "the client's call lost its output position, or its arguments, to the server's own call "
          "beside it");

    const BuiltOpenAIResponse aggregate =
        make_openai_response_object("resp_aggregate", 1, request, OpenAIResponsesRuntimeValues{},
                                    picture.rounds.answer);
    // The Responses route builds its body as a plain document, not the request's ordered one, so it is
    // read back as the bytes a client parses rather than as this suite's own document type.
    const RequestJson built_body       = RequestJson::parse(aggregate.body.dump());
    const std::vector<RequestJson> aggregate_witnesses = witnesses_in(built_body, kAttemptField);
    const RequestJson output = at_or_null(built_body, "output");
    check(aggregate_witnesses.size() == 1 && output.is_array() && output.size() == 1 &&
              text_at(output.at(0), "type") == "message",
          "the aggregate Response either lost the draw's fate or published an Item for the server's own "
          "call");
    if (aggregate_witnesses.size() == 1) {
        check_attempt_keeps_the_draw(aggregate_witnesses[0], picture.rounds.urls[0]);
    }
}

} // namespace

// The rewriter's switch on the chat path. The tool's arguments are parsed by the images route's own
// parser (run_chat_image_tool_round), so prompt_enhance is not a field this layer has to learn about --
// what has to be proven is that it SURVIVES into the request the pipeline is handed, in both spellings
// the parser takes, that a call which did not ask still gets no rewrite, and that a value the contract
// refuses becomes a correctable tool error rather than a silent no-rewrite. The schema itself is held
// to the capability document by the http routes test, not here.
void test_prompt_enhance_passes_through() {
    struct Case {
        const char* arguments;
        PromptEnhanceMode expected;
        const char* what;
    };
    const Case cases[] = {
        {R"({"prompt":"a","prompt_enhance":"auto"})", PromptEnhanceMode::Auto,
         "auto reaches the images request as auto"},
        {R"({"prompt":"a","prompt_enhance":true})", PromptEnhanceMode::On,
         "the boolean true reaches it as on"},
        {R"({"prompt":"a","prompt_enhance":false})", PromptEnhanceMode::Off,
         "the boolean false reaches it as off"},
        {R"({"prompt":"a"})", PromptEnhanceMode::Off,
         "a call that never asked is left with no rewrite"},
    };
    for (const Case& item : cases) {
        GenerationRequest request;
        FakeGenerator generator;
        const ChatImageToolRound round = run_chat_image_tool_round(
            request, image_call_outcome({item.arguments}), kImageModel, kUrlBase, 0, 0,
            observer_of(generator));
        check(round.executed == 1 && generator.requests.size() == 1 &&
                  generator.requests[0].prompt_enhance == item.expected,
              item.what);
    }

    GenerationRequest request;
    FakeGenerator generator;
    const ChatImageToolRound refused = run_chat_image_tool_round(
        request, image_call_outcome({R"({"prompt":"a","prompt_enhance":"yes"})"}), kImageModel,
        kUrlBase, 0, 0, observer_of(generator));
    check(refused.executed == 0 && generator.calls == 0 &&
              tool_result_text(request, 1).find("prompt_enhance") != std::string::npos,
          "a prompt_enhance that is neither a boolean nor \"auto\" never reaches the pipeline and "
          "comes back naming the field the model has to fix");

    // The promise the schema text makes to a client that never read a document: asking for a rewrite on
    // a build with no rewriter is the documented refusal INSIDE the tool result, which the model can
    // work around by answering in words -- not a chat request that dies with a 503 the caller cannot
    // act on. This is the difference between offering the switch on the chat path and advertising a
    // route that takes the user's question down with it.
    GenerationRequest asked;
    FakeGenerator no_rewriter;
    no_rewriter.fail_status = 503;
    no_rewriter.fail_code   = "image_prompt_enhancer_unavailable";
    const ChatImageToolRound refused_hard = run_chat_image_tool_round(
        asked, image_call_outcome({R"({"prompt":"a","prompt_enhance":true})"}), kImageModel, kUrlBase,
        0, 0, observer_of(no_rewriter));
    const std::string said = tool_result_text(asked, 1);
    check(refused_hard.executed == 0 && refused_hard.urls.empty() &&
              asked.messages.size() == 2 && asked.messages[1].tool_result_is_error &&
              said.find("image_prompt_enhancer_unavailable") != std::string::npos,
          "a true on a build without a rewriter settles as that documented refusal in the tool result, "
          "so the picture is not drawn and the answer still reaches the user");
}

int main() {
    test_injection();
    test_image_argument_forms();
    test_round_detection();
    test_single_round();
    test_prompt_enhance_passes_through();
    test_cap_and_bad_arguments();
    test_pipeline_failure_becomes_a_tool_result();
    test_attempts_are_reported();
    test_url_suffix();
    test_round_loop();
    test_round_bound_and_cancellation();
    test_chat_renderer_hides_the_server_run_call();
    test_chat_renderer_keeps_client_call_numbering_dense();
    test_anthropic_renderer_hides_the_server_run_call();
    test_responses_renderer_hides_the_server_run_call();

    std::cout << (failures == 0 ? "CHAT_IMAGE_TOOL_TEST: PASS" : "CHAT_IMAGE_TOOL_TEST: FAIL")
              << " (" << (failures == 0 ? "all" : std::to_string(failures) + " failed") << ")\n";
    return failures == 0 ? 0 : 1;
}
