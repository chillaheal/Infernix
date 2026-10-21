#include "serve/chat_image_tool.h"

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::serve {
namespace {

// The tool schema the model sees. Everything but `prompt` is optional and validated by the images
// route's own parser, so a wrong `size` or `steps` comes back as a tool error the model can correct.
constexpr std::string_view kChatImageToolSchema = R"JSON({
  "type": "object",
  "properties": {
    "prompt": {"type": "string", "description": "What to draw: subject, style, lighting, composition."},
    "size": {"type": "string", "description": "WIDTHxHEIGHT, each a multiple of 32 in [32,2048]. Default 768x768."},
    "steps": {"type": "integer", "description": "Denoise steps, 2 to 100. Default 40."},
    "negative_prompt": {"type": "string", "description": "What must not appear. Only used when true_cfg_scale is above 1."},
    "true_cfg_scale": {"type": "number", "description": "Guidance 1 to 20; 1 (default) runs one forward per step."},
    "seed": {"type": "integer", "description": "Repeatable draw."}
  },
  "required": ["prompt"]
})JSON";

[[nodiscard]] std::string json_error_result(std::string_view code, std::string_view message) {
    return RequestJson{{"error", code}, {"message", message}}.dump();
}

} // namespace

ToolDefinition make_chat_image_tool_definition() {
    return ToolDefinition{
        .name        = std::string(kChatImageToolName),
        .description = "Generate a new picture from a text prompt using this server's image model. "
                       "Call it when the user asks for an image, illustration, photo, poster or "
                       "diagram, and call it again for a second, different picture. The result is "
                       "the URL of the finished PNG; put that URL in your reply.",
        .input_schema_json = std::string(kChatImageToolSchema),
        // An internal definition has no client request path to name in an error.
        .schema_param = "infernix.image_tool.input_schema",
    };
}

void append_chat_image_tool(GenerationRequest& request, bool image_pipeline_configured) {
    if (!image_pipeline_configured) { return; }
    // Only a free tool choice may gain a tool: a disabled, pinned or forced selection is the client's
    // own decision about which tools exist, and a client that named this tool owns that name.
    if (request.tool_choice.mode != ToolChoiceMode::Auto || request.tool_choice.allowed_names) {
        return;
    }
    for (const ToolDefinition& tool : request.tools) {
        if (tool.name == kChatImageToolName) { return; }
    }
    request.tools.push_back(make_chat_image_tool_definition());
}

bool outcome_is_chat_image_tool_round(const GenerationOutcome& outcome) noexcept {
    if (outcome.tool_calls.empty()) { return false; }
    for (const infernix::GeneratedToolCall& call : outcome.tool_calls) {
        if (call.name != kChatImageToolName) { return false; }
    }
    return true;
}

ChatImageToolRound run_chat_image_tool_round(GenerationRequest& request,
                                            const GenerationOutcome& outcome,
                                            std::string_view image_model_id,
                                            std::string_view artifact_url_base, std::int64_t now,
                                            std::size_t already_generated,
                                            const ChatImageGenerator& generate) {
    ChatImageToolRound round;
    (void)now; // the generator stamps `created`; the round itself needs no clock

    ChatTurn assistant;
    assistant.role = ChatRole::Assistant;
    if (!outcome.text.empty()) {
        assistant.content.push_back(
            ContentPart{.kind = ContentKind::Text, .text = outcome.text});
    }

    std::vector<ChatTurn> results;
    results.reserve(outcome.tool_calls.size());
    for (std::size_t index = 0; index < outcome.tool_calls.size(); ++index) {
        const infernix::GeneratedToolCall& call = outcome.tool_calls[index];
        // Server-internal: these ids correlate the two turns inside this request and never reach a
        // client, so they are unique here rather than in any protocol's id space.
        const std::string call_id = "infernix_img_" + std::to_string(already_generated + index);
        assistant.tool_calls.push_back(ToolCall{
            .id             = call_id,
            .name           = std::string(kChatImageToolName),
            .arguments_json = call.arguments_json,
        });

        std::string result;
        bool is_error = false;
        if (already_generated + round.executed >= kMaximumChatImageGenerations) {
            is_error = true;
            ++round.rejected;
            result = json_error_result(
                "image_generation_limit_reached",
                "one chat request may generate at most " + std::to_string(kMaximumChatImageGenerations) +
                    " pictures");
        } else {
            try {
                RequestJson body = RequestJson::parse(call.arguments_json);
                if (!body.is_object()) { throw std::runtime_error("arguments are not an object"); }
                // Whatever the model asked for is validated by the images contract itself; only the
                // delivery fields are the server's, because the answer carries a URL.
                body["model"]             = image_model_id;
                body["n"]                 = 1;
                body["response_format"]   = "url";
                const OpenAIImagesRequest image_request = parse_openai_images_request(body);
                const RequestJson response = generate(image_request);
                const std::string url      = response.at("data").at(0).at("url").get<std::string>();
                const std::string id = url.size() > artifact_url_base.size() &&
                                              url.rfind(artifact_url_base, 0) == 0
                                          ? url.substr(artifact_url_base.size())
                                          : url.substr(url.find_last_of('/') + 1);
                round.ids.push_back(id);
                round.urls.push_back(url);
                ++round.executed;
                result = RequestJson{{"id", id}, {"url", url}}.dump();
            } catch (const ApiException& exception) {
                // Bad arguments are the model's to fix; a failing pipeline is not its to swallow.
                if (exception.error().status != 400) { throw; }
                is_error = true;
                result   = json_error_result(exception.error().code.empty() ? "image_arguments_invalid"
                                                                            : exception.error().code,
                                             exception.error().message);
            } catch (const std::exception& exception) {
                is_error = true;
                result   = json_error_result("image_arguments_invalid", exception.what());
            }
        }

        ChatTurn tool;
        tool.role                        = ChatRole::Tool;
        tool.tool_call_id                = call_id;
        tool.tool_result_name            = std::string(kChatImageToolName);
        tool.tool_result_is_error        = is_error;
        tool.content.push_back(ContentPart{.kind = ContentKind::Text, .text = std::move(result)});
        results.push_back(std::move(tool));
    }

    request.messages.push_back(std::move(assistant));
    for (ChatTurn& tool : results) { request.messages.push_back(std::move(tool)); }
    return round;
}

std::string missing_image_url_suffix(std::string_view answer_text,
                                     const std::vector<std::string>& urls) {
    std::string suffix;
    for (const std::string& url : urls) {
        if (answer_text.find(url) != std::string_view::npos) { continue; }
        suffix += "\n\n![generated image](" + url + ")";
    }
    return suffix;
}

ChatImageRounds run_chat_image_rounds(GenerationRequest& request, GenerationOutcome answer,
                                      std::string_view image_model_id,
                                      std::string_view artifact_url_base, std::int64_t now,
                                      const ChatImageGenerator& make_picture,
                                      const std::function<GenerationOutcome()>& run_next_answer) {
    ChatImageRounds loop;
    loop.answer = std::move(answer);

    int prompt_tokens     = 0;
    int completion_tokens = 0;
    int reasoning_tokens  = 0;
    // Everything the rounds before the answer said is part of the one answer the client gets, in the
    // order it was produced; a streaming client may already have received the first round's text, and
    // the terminal output must still contain everything already streamed. Thinking is not: a round
    // after the first cannot put reasoning behind content already on the wire in any of the three
    // protocols, so only the first round's thinking is reported.
    std::string spoken_text   = loop.answer.text;
    std::string first_reasoning = loop.answer.reasoning;

    while (loop.answer.finish_reason != infernix::FinishReason::Cancelled &&
           outcome_is_chat_image_tool_round(loop.answer) && loop.rounds < kMaximumChatImageRounds) {
        ++loop.rounds;
        // An internal round only asked for pictures: its token counts are work this request really
        // did on top of the answer, because each round re-ran its own prompt.
        prompt_tokens += loop.answer.prompt_tokens;
        completion_tokens += loop.answer.completion_tokens;
        reasoning_tokens += loop.answer.reasoning_tokens;

        const ChatImageToolRound round =
            run_chat_image_tool_round(request, loop.answer, image_model_id, artifact_url_base, now,
                                      loop.generations, make_picture);
        loop.generations += round.executed;
        loop.urls.insert(loop.urls.end(), round.urls.begin(), round.urls.end());

        loop.answer = run_next_answer();
        spoken_text += loop.answer.text;
    }

    loop.answer.text     = std::move(spoken_text);
    loop.answer.reasoning = std::move(first_reasoning);

    // The loop can only end on a call the client cannot execute if the round budget ran out. The
    // collected URLs still make the reply usable, so it is reported as an ordinary answer.
    if (loop.answer.finish_reason != infernix::FinishReason::Cancelled &&
        outcome_is_chat_image_tool_round(loop.answer)) {
        loop.answer.tool_calls.clear();
        loop.answer.finish_reason = infernix::FinishReason::StopToken;
    }
    loop.answer.prompt_tokens += prompt_tokens;
    loop.answer.completion_tokens += completion_tokens;
    loop.answer.reasoning_tokens += reasoning_tokens;
    return loop;
}

} // namespace infernix::serve
