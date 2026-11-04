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

// The wire word for what a request asked the rewriter to do. Public because the same word has to be
// written in two places that must not drift -- the attempt a client reads and the line the operator
// reads -- and an operator comparing them is entitled to assume a difference is a bug.
[[nodiscard]] const char* prompt_enhance_word(PromptEnhanceMode mode) noexcept;

// The word for what became of a picture, for BOTH witnesses: the attempt a client reads in
// image_tool_attempts and the image_job line the operator reads in requests.jsonl. One definition
// because the two answer one question about one draw, and a `status >= 500` rule at the log side used
// to call a 503 VRAM refusal "failed" while the client was told "refused" for the very same attempt --
// the operator saw a broken server and the client saw an ask to change. "failed" is a 500 only: the
// draw fell over after nobody objected to anything, so only a retry helps. Every answer the admission
// gates give is a refusal (400 joint sequence too long, 409 another draw in flight, 429 queue
// pressure, 503 no room for this canvas): the caller can change the ask, and capabilities documents
// them that way.
[[nodiscard]] const char* image_outcome_word(int http_status) noexcept;

// What one picture attempt the server made on the client's behalf, in the form the CLIENT is
// allowed to read. The model already sees every refusal (it is the tool's own error result), and the
// header above is right that the client never sees the round -- but that invisibility was total, and
// it cost the user a wrong answer: a third-party client whose draw was refused had nothing to read,
// so it told its user "VRAM is the bottleneck, unload the chat model, 7.9 GB needed", a number it had
// taken from /v1/models weight size. An attempt the client can see costs that story nothing; an
// attempt it cannot see lets the model invent one. These records are the server's own codes and
// messages, never the reference bytes and never the client's prompt, and they go ONLY here -- they are
// not fed back to the model, so nothing new enters a model context through this door.
struct ChatImageAttempt {
    std::size_t round = 0;         // the 1-based internal round that made this attempt
    // What became of it, on the same vocabulary the operator's log line uses: "succeeded", "refused"
    // (the route or this tool said no, and says why in `code`), or "failed" (the picture could not be
    // made even though nothing objected to it).
    std::string outcome;
    std::string code;              // the images route's own code; empty when it succeeded
    std::string message;           // its message, or the per-request limit's
    int http_status = 0;           // the status that refusal would have carried on the route itself;
                                   // 0 when the attempt never reached one. The chat response stays 200
                                   // either way -- this is the picture's status, not the answer's.
    std::string tool_call_id;      // this attempt's internal call id, shared with the log record
    std::string id;                // stored job id, when a picture came out
    std::string url;               // its artifact URL, when a picture came out
    // What the model asked for, as the route's parser settled it. A refused attempt is unreadable
    // without the parameters it was refused for: "image_size_invalid" means nothing to a client that
    // cannot see it asked for 500x500.
    bool canvas_set    = false;    // false when the arguments never parsed into a request
    std::string size;              // "WIDTHxHEIGHT"
    std::uint32_t steps = 0;
    bool has_seed      = false;
    std::uint64_t seed   = 0;
    std::size_t image_count = 0;   // references the attempt carried
    std::string prompt_enhance;    // "off", "true" or "auto", as asked
    // What the rewriter really did with that ask, on the route's own vocabulary ("applied",
    // "skipped_no_device", ...). It is the difference between a picture drawn from the client's words
    // and one drawn from a rewrite, and a client cannot tell them apart from the pixels.
    std::string enhancer_outcome;
    // The canvas the picture came out at when it is NOT the one `size` asked for: the card could not
    // hold the asked canvas, so the server drew the largest that fits. Empty in the ordinary case where
    // they agree. Without it a model announces "here is your 1024x1024" over a 704x704 picture.
    std::string drawn_size;
};

// The same records in the wire form every protocol puts under `image_tool_attempts`. An empty vector
// yields an empty array, and the callers drop the field entirely then -- a request that never touched
// the image tool must keep the response byte-for-byte what it was before this field existed.
[[nodiscard]] RequestJson chat_image_attempts_json(const std::vector<ChatImageAttempt>& attempts);

// The array for one protocol's response body, taken from the outcome the rounds left behind, or null
// when this answer drew no picture. One owner for all three protocols: a field a client reads has to
// read the same way from /v1/chat/completions as from /v1/messages, and three parsers of one string is
// three ways for that to stop being true. The string is this process's own writing (the line above),
// never socket input, so a parse that fails is a bug -- answered by no field, not a thrown response in
// the middle of a stream a client is already reading.
[[nodiscard]] RequestJson chat_image_attempts_for_response(const GenerationOutcome& outcome);

struct ChatImageToolRound {
    std::vector<std::string> ids;   // stored job ids, the artifact route's ids
    std::vector<std::string> urls;  // absolute GET /v1/images/<id> URLs, parallel to `ids`
    std::size_t executed  = 0;      // pictures actually produced in this round
    std::size_t rejected  = 0;      // calls refused as past kMaximumChatImageGenerations
    std::vector<ChatImageAttempt> attempts;  // one per call, ok and refused alike
};

// Executes every internal call of `outcome` and appends the round to `request`: one assistant
// tool_calls turn carrying the round's text, then one tool result per call (its URL as the result,
// or an error result for a call the model argued badly or pushed past the limit).
//
// A bad argument set is the model's mistake and becomes an error tool result, so it can retry. A
// pipeline failure (429 no room for this size, 429 another draw in flight, 500 a failed job) is not
// the model's mistake, but it is still this tool's answer: it comes back as an error result with the
// code and reason the images route would have returned, so the model can retry smaller, wait, or
// answer in words. The user's request was for an answer, not for the HTTP status of a draw.
// `already_generated` is how many pictures this request produced earlier.
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
    // Every attempt of every round, in order, so a protocol can answer "what happened to my picture"
    // without the model being asked to remember it.
    std::vector<ChatImageAttempt> attempts;
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
