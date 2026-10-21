#pragma once

// #83: the server-executed image tool. When this process serves an image pipeline, every chat
// request also receives one hidden tool that NO client executes: the model calls it, this process
// runs the picture through ImageGenerationService in the same HTTP request, and the finished
// picture's URL is fed back so the answer can carry it. The client never sees the call, the tool
// result or the extra generation round -- it sees one assistant answer with a URL in it.
//
// The round is built from the same pieces a client round-trip already sends (an assistant
// tool_calls turn plus one role=tool turn), so the Engine path is the certified one, and the
// argument sets go through the images route's own parser, so the images contract has one owner.

#include "serve/generation_service.h"
#include "serve/openai_images.h"
#include "serve/request_json.h"

#include <infernix/types.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::serve {

// The hidden tool's name. A client that declares a tool with this name keeps its own: the server
// never takes a client's tool over.
inline constexpr std::string_view kChatImageToolName = "generate_image";

// One picture costs a full image interlude (the LLM yields ~8 GiB and is restored afterwards), so a
// request bounds how many it may trigger, including several calls in one round. Calls past the
// limit get an error tool result instead of another picture.
inline constexpr std::size_t kMaximumChatImageGenerations = 4;

// The single definition every protocol injects, so the rendered prompt is identical everywhere.
[[nodiscard]] ToolDefinition make_chat_image_tool_definition();

// Adds the hidden tool when `image_pipeline_configured` and the client left tool choice free: a
// request with `tool_choice:"none"`, a forced single tool, a required call, or its own tool of this
// name keeps exactly the tool list it sent.
void append_chat_image_tool(GenerationRequest& request, bool image_pipeline_configured);

// True when the model answered with ONLY internal image calls. A round that mixes in a client tool
// is handed to the client untouched, so no client call is ever swallowed here.
[[nodiscard]] bool outcome_is_chat_image_tool_round(const GenerationOutcome& outcome) noexcept;

// Runs one image job for one validated request. Production binds this to
// ImageGenerationService::generate; the host test injects a fake.
using ChatImageGenerator = std::function<RequestJson(const OpenAIImagesRequest&)>;

struct ChatImageToolRound {
    std::vector<std::string> ids;   // stored job ids, the artifact route's ids
    std::vector<std::string> urls;  // absolute GET /v1/images/<id> URLs, parallel to `ids`
    std::size_t executed  = 0;      // pictures actually produced in this round
    std::size_t rejected  = 0;      // calls refused as past kMaximumChatImageGenerations
};

// Executes every internal call of `outcome` and appends the round to `request`: one assistant
// tool_calls turn carrying the round's text, then one tool result per call (its URL as the result,
// or an error result for a call the model argued badly or pushed past the limit).
//
// A bad argument set is the model's mistake and becomes an error tool result, so it can retry. A
// pipeline failure (429/500/503) is not, and propagates as the ApiException the images route would
// have returned. `already_generated` is how many pictures this request produced earlier.
// [[nodiscard]] so a caller cannot run the pictures and forget to feed the results back.
[[nodiscard]] ChatImageToolRound run_chat_image_tool_round(
    GenerationRequest& request, const GenerationOutcome& outcome, std::string_view image_model_id,
    std::string_view artifact_url_base, std::int64_t now, std::size_t already_generated,
    const ChatImageGenerator& generate);

// The answer must carry the picture. A model that described the image instead of quoting its URL
// would otherwise drop the only usable reference, so any URL still missing is appended. Empty when
// the answer already contains every URL.
[[nodiscard]] std::string missing_image_url_suffix(std::string_view answer_text,
                                                   const std::vector<std::string>& urls);

// Bound on internal rounds per request. kMaximumChatImageGenerations bounds the pictures, but a call
// that failed on bad arguments costs a round without costing a picture, and a model can repeat that;
// the round bound is what keeps one HTTP request from looping on the Engine forever.
inline constexpr std::size_t kMaximumChatImageRounds = 8;

// Everything one chat request produced: the answer the client receives and its pictures.
struct ChatImageRounds {
    // The answer the client receives: the last round's outcome, with token usage summed over every
    // round the request ran.
    GenerationOutcome answer;
    std::vector<std::string> urls;  // every picture the request produced, in generation order
    std::size_t generations = 0;    // pictures produced
    std::size_t rounds      = 0;    // internal rounds executed
};

// Runs the internal rounds to their end. `answer` is the outcome of the caller's first generation;
// while it contains only internal image calls, the round is executed, appended to `request`, and
// `run_next_answer` produces the next answer for the request as it now stands (the caller re-prepares
// and re-runs, so it also owns the consumer mode, observers and transport). Stops when the model
// answers in prose, when the Engine reports the generation cancelled, or at kMaximumChatImageRounds.
//
// The answer's text is every round's text in order (a streaming client may already hold the first
// round's, and the terminal output must contain everything already sent). Its reasoning is only the
// first round's: none of the three protocols can place thinking behind content that is already on the
// wire, so thinking from a round after the first is not reported. A streaming caller runs every round
// after the first with a sink that carries nothing but cancellation, which also keeps one generation's
// start, prompt-progress and per-token timing state from being opened twice; the SSE heartbeat covers
// the silence while a picture is drawn.
//
// Ending on the round bound cannot hand the client a call to a tool it never saw, so that last
// outcome is turned into a plain answer: its hidden calls are dropped and the finish reason is
// `stop`. Token usage is summed over every round because every round really ran.
[[nodiscard]] ChatImageRounds run_chat_image_rounds(GenerationRequest& request,
                                                   GenerationOutcome answer,
                                                   std::string_view image_model_id,
                                                   std::string_view artifact_url_base,
                                                   std::int64_t now,
                                                   const ChatImageGenerator& make_picture,
                                                   const std::function<GenerationOutcome()>& run_next_answer);

} // namespace infernix::serve
