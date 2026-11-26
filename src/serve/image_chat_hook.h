#pragma once

// The attach point of the image lane for the three protocol handlers and the three response
// renderers: the server-executed `generate_image` tool, the rounds that run it, the SSE note a
// drawing job leaves on a committed stream, and the `image_tool_attempts` witness a client reads.
// Everything the lane says about those -- the tool it injects, the wording of the progress note, the
// field's name, the null rule that keeps a plain answer's bytes unchanged, the hidden-call rule --
// lives in this file's implementation (serve/image_chat_hook.cpp), which is a thin facade over
// serve/chat_image_tool.{h,cpp}, the tool's real owner. The handlers keep what only they can own:
// their service_, their request, their re-prepare/re-run callbacks and their own route label, and
// they reach the lane through the typed calls below; there is no registry and no name-driven
// dispatch.

#include "serve/chat_image_tool.h"
#include "serve/http_transport.h"
#include "serve/openai_images.h"
#include "serve/request_json.h"
#include "serve/serve_options.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::serve {

class HttpServer;
// The job's live progress state (serve/image_job_progress.h), named here only so the door below can
// hand it over: this header does not read its members.
struct ImageJobProgress;

// One chat request's share of the image lane: the hidden tool, the picture generator bound to this
// process's image artifact, the URLs and the attempts a client reads. Created per request, handed to
// the streaming provider by shared_ptr because a provider runs after its handler has returned.
// Null when this process has no image pipeline -- which is exactly when no tool is injected, no round
// is ever wanted and no attempts field is ever written, so a serve without --image-artifact behaves
// byte-for-byte as it does today.
class ChatImageHook;

// The two doors only a member of HttpServer can open, handed over by the handler that owns this
// request: the attached pipeline itself, and the one door a picture goes through (the images route's
// own witness, so a draw the chat tool ran is recorded exactly like one the route ran). The handler
// keeps nothing of what they are used for: the source, the protocol and the URL a draw is recorded
// with are read here, and so is the rule for what a drawing job says on a committed stream.
struct ChatImageDoors {
    // HttpServer keeps its image service behind a private member, so the handler hands the pointer
    // over without naming its type: null is the lane's own answer that this process draws no
    // picture, which is exactly the `images_ != nullptr` the handlers read before this existed. The
    // job's real percent is read from it here, where the rule lives.
    std::function<void*()> pipeline;
    // The handler's own `HttpServer::run_image_job`, with the three trailing arguments left open
    // because they belong to the lane: the url base it reads from this request, the witness source
    // and protocol it names.
    std::function<RequestJson(const OpenAIImagesRequest&, std::string_view artifact_url_base,
                              std::string_view source, std::string_view protocol,
                              std::string correlation)>
        run_job;
};

// A request's hook, or nullptr when this process draws no picture -- the exact condition the handler
// tested by reading its own `images_` before this existed. `route_label` is the handler's own existing
// word ("openai_chat", "openai_responses", "anthropic_messages"), which is what the picture's witness
// line is filed under; `req_id` is the handler's already-allocated request id, so the draw and the
// chat turn it belongs to share one correlation. `options` is the handler's own ServeOptions, which
// HttpServer keeps private: the URL the picture's address is built from -- and the image model's id,
// read from the server itself -- are read here rather than in the handler, because a streaming
// provider runs after the handler has returned.
[[nodiscard]] std::shared_ptr<ChatImageHook>
make_chat_image_hook(HttpServer& server, const httplib::Request& req, std::string_view route_label,
                     std::uint64_t req_id, const ServeOptions& options, const ChatImageDoors& doors);

// Adds the hidden tool to a request that left tool choice free. nullptr-tolerant, so a handler calls
// it unconditionally: a null hook means no pipeline and therefore no tool, which is what the handler
// passed as `image_pipeline_configured` before this existed. Not noexcept, because the tool it appends
// is a push_back into the request's tool list and a throw there is the handler's to answer with its
// own 500, exactly as it was when the handler called the tool directly.
void inject_chat_image_tool(const std::shared_ptr<ChatImageHook>& hook,
                            GenerationRequest& request);

// True when the rounds below owe this request work: a pipeline in this process and an answer made of
// nothing but the server's own calls. The two halves are the lane's rule; the handler keeps the
// branch.
[[nodiscard]] bool chat_image_round_needed(const std::shared_ptr<ChatImageHook>& hook,
                                           const GenerationOutcome& outcome) noexcept;

// True when this process could still answer a picture request, which is what tells a handler whether
// the conversation has to stay whole for the rounds that would re-render it. The handler decides what
// that costs it; the rule for when it applies is here.
[[nodiscard]] bool chat_image_replay_needed(const std::shared_ptr<ChatImageHook>& hook) noexcept;

// Runs the internal rounds to their end, stamping the hour the same clock the handlers stamped them
// with and drawing through the hook. `run_next_answer` stays the handler's: it owns service_, the
// request and the consumer mode, exactly as it did when it was an inline lambda.
[[nodiscard]] ChatImageRounds run_chat_image_rounds(
    const ChatImageHook& hook, GenerationRequest& request, GenerationOutcome answer,
    const std::function<GenerationOutcome()>& run_next_answer);

// The answer must carry the picture: any URL the answer still does not quote is appended here. The
// hook is the only door this rule is reachable through; the rule itself is the answer's text and the
// urls it drew.
[[nodiscard]] std::string chat_image_missing_url_suffix(const ChatImageHook& hook,
                                                         std::string_view answer_text,
                                                         const std::vector<std::string>& urls);

// The streaming "one draw is at NN%" note: the percent rule and the prose belong to the lane, so a
// committed stream asks the hook for the comment instead of reading the job. Empty means nothing to
// say this tick, which is what keeps the heartbeat's own silence byte-for-byte what it was.
[[nodiscard]] std::string chat_image_progress_note(const ChatImageHook& hook);

// The job this request's hidden tool round is drawing, for the ticker's stall watchdog. It is the same
// live state chat_image_progress_note reads, handed over rather than read here, because the thread that
// has to notice a wedged draw is the ticker's and not the handler's -- the handler is blocked inside
// the draw. Null when this process has no image pipeline or no job yet; the only reader is
// serve/http_transport.h's HeartbeatTicker, and it measures nothing until the job's own DiT loop opens
// the window (serve/image_job_progress.h's `armed`, set by src/models/qwen_image/image_pipeline.cu).
[[nodiscard]] std::shared_ptr<ImageJobProgress> chat_image_watch_progress(const ChatImageHook& hook);

// The renderer side, for all three protocols. One owner so the field cannot drift.
[[nodiscard]] RequestJson chat_image_attempts_json(const GenerationOutcome& outcome);

// Adds `image_tool_attempts` to a response body, a chunk payload or a terminal SSE event, and emits
// nothing at all when this request drew no picture -- which is the whole reason a plain chat answer
// is still the exact bytes it was before the field existed. A caller that has two possible carriers
// asks only one of them, because the field is read once per response.
//
// The two shapes are the renderers' own document types, spelled out here because both aliases are
// file-local to theirs: two of them build plain `nlohmann::json` bodies, and the Anthropic response
// carries its parts in the request's ordered document. Both read the attempts through the sorted
// document first, which is what makes the keys of one written attempt come out in the same order on
// all three protocols rather than in the order the tool happened to write them.
void add_chat_image_attempts(nlohmann::json& payload, const GenerationOutcome& outcome);
void add_chat_image_attempts(nlohmann::ordered_json& payload, const GenerationOutcome& outcome);

// Was `delta.name == kChatImageToolName` in three upstream renderers: the call the server runs itself
// is the one the client never sees.
[[nodiscard]] bool chat_image_tool_hidden(std::string_view tool_name) noexcept;

} // namespace infernix::serve
