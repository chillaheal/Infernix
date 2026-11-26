#include "serve/image_chat_hook.h"

#include "serve/chat_image_tool.h"
#include "serve/http_server.h"
#include "serve/image_generation_service.h"
#include "serve/image_job_progress.h"
#include "serve/openai_common.h"
#include "serve/serve_options.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace infernix::serve {
namespace {

// The document type the three renderers add the field to (`using Json = nlohmann::json;` in each of
// them), spelled out here because that alias is file-local to theirs.
using Json = nlohmann::json;

// Where the chat tool's pictures are filed in the operator's request log. The images route files its
// own draws under "images_route"; every picture a chat request asked for is filed under this source
// on all three protocols, with the handler's own route label beside it -- the pair is what lets one
// log say which door a draw came in by. It was written in the three handlers; it is one constant
// here so the three of them cannot drift.
constexpr std::string_view kChatImageToolWitnessSource = "chat_tool";

} // namespace

// The lane's one handle on one chat request. Held by shared_ptr because a streaming provider runs
// after its handler has returned, and read by nobody but the calls below: this is the only door from
// a protocol file into the image lane, and the only place outside serve/chat_image_tool.cpp that says
// anything about the hidden tool.
class ChatImageHook {
public:
    ChatImageHook(HttpServer& server, const ServeOptions& options, const httplib::Request& req,
                  std::string_view route_label, std::uint64_t req_id, const ChatImageDoors& doors)
        : image_model_(server.image_model_id()), url_base_(artifact_url_base(req, options)),
          route_label_(route_label), req_id_(req_id), doors_(doors) {}

    // The tool's pictures go through the same witness the images route uses -- one door, one record --
    // and they carry the chat request that asked for them, so one log line names both halves of what
    // the client experiences as one call. Built after `req_id` for exactly that reason.
    [[nodiscard]] RequestJson draw(const OpenAIImagesRequest& image_request) const {
        return doors_.run_job(image_request, url_base_, kChatImageToolWitnessSource, route_label_,
                              std::to_string(req_id_));
    }

    // The picture draws with the Engine loop stopped: this ticker is the
    // stream's only traffic while it runs, and it carries the job's real
    // percent as an SSE comment.
    [[nodiscard]] std::string progress_note() const {
        auto* const images = static_cast<ImageGenerationService*>(doors_.pipeline());
        if (images == nullptr) { return {}; }
        const auto progress = images->last_job_progress();
        if (progress == nullptr) { return {}; }
        const int percent = progress->percent.load(std::memory_order_relaxed);
        if (percent < 0) { return {}; }
        return ": image " + std::to_string(percent) + "%\n\n";
    }

    // The job this round is drawing, for the ticker's stall watchdog: the same read as the note above,
    // handed to the thread that has to measure it instead of being read here. The handler that owns this
    // hook is blocked inside the draw, so this is the only way its thread can reach the state.
    [[nodiscard]] std::shared_ptr<ImageJobProgress> watch_progress() const {
        auto* const images = static_cast<ImageGenerationService*>(doors_.pipeline());
        if (images == nullptr) { return {}; }
        return images->last_job_progress();
    }

    [[nodiscard]] const std::string& image_model_id() const noexcept { return image_model_; }
    // Named for what it holds, not after the free `artifact_url_base(req, options)` this hook was built
    // with, which the constructor still calls below.
    [[nodiscard]] const std::string& url_base() const noexcept { return url_base_; }

private:
    // The image resource's public id, read from the server this request belongs to: the hidden tool is
    // offered under it, exactly as the handlers read it before it was a member here.
    std::string image_model_;
    std::string url_base_;
    // The handler's own word for the protocol the picture was asked on, kept as it arrived.
    std::string route_label_;
    std::uint64_t req_id_ = 0;
    ChatImageDoors doors_;
};

std::shared_ptr<ChatImageHook>
make_chat_image_hook(HttpServer& server, const httplib::Request& req, std::string_view route_label,
                     std::uint64_t req_id, const ServeOptions& options, const ChatImageDoors& doors) {
    // With an image pipeline in this process the request also carries the server-executed image
    // tool, so the picture rounds below are the ones a client would otherwise have to run. The URL
    // base and the model id are read here because a streaming provider runs after this handler has
    // returned, and the generator is bound to this request's Host so the URL resolves for that client.
    //
    // Whether this process attached a pipeline at all is the handler's read -- HttpServer keeps the
    // service private, so the handler hands the pointer over and the lane only ever asks whether it is
    // there. A process that attached none gets no hook, which is the exact condition the handlers
    // tested by hand before this existed: a serve without --image-artifact injects no tool, wants no
    // round and writes no attempts field, byte for byte the answers it gives today.
    if (!doors.pipeline || !doors.run_job || doors.pipeline() == nullptr) { return nullptr; }
    return std::make_shared<ChatImageHook>(server, options, req, route_label, req_id, doors);
}

void inject_chat_image_tool(const std::shared_ptr<ChatImageHook>& hook,
                            GenerationRequest& request) {
    // The handler asks unconditionally; a null hook is the lane's own answer that this process has no
    // image pipeline, which is what `image_pipeline_configured` said when the handler computed it.
    append_chat_image_tool(request, hook != nullptr);
}

bool chat_image_round_needed(const std::shared_ptr<ChatImageHook>& hook,
                             const GenerationOutcome& outcome) noexcept {
    // The two halves of the handler's old `images_configured && outcome_is_chat_image_tool_round(...)`,
    // in that order: no pipeline means no round whatever the answer says.
    return hook != nullptr && outcome_is_chat_image_tool_round(outcome);
}

bool chat_image_replay_needed(const std::shared_ptr<ChatImageHook>& hook) noexcept {
    return hook != nullptr;
}

ChatImageRounds run_chat_image_rounds(const ChatImageHook& hook, GenerationRequest& request,
                                     GenerationOutcome answer,
                                     const std::function<GenerationOutcome()>& run_next_answer) {
    // Every round of one request is one HTTP request: the same observation wiring, and the same
    // request log, so an image round is visible as part of this request rather than as a new one.
    // The handler keeps its own observation and log identity; the rounds are what they exist for.
    //
    // The model answered with nothing but the server's own calls, so the lane runs them: the hour is
    // stamped by the clock the handlers always stamped it with, the model id and the URL base come
    // from the hook, and every picture goes through the handler's one draw door.
    //
    // A round replaces the answer, so a handler re-derives its own call bookkeeping from the answer
    // that comes back. That bookkeeping is the handler's; what makes it necessary is the lane's hidden
    // call, so the reason is stated here. The Anthropic handler's rule in its own words, in both the
    // aggregate and the streamed form:
    //
    // The segment's calls follow its generation: the hidden image call of the round
    // before is no longer part of this one, and any call the answer now carries is a
    // client call.
    //
    // The segment's calls follow its generation: the hidden image call of the
    // round before is no longer part of this one.
    return run_chat_image_rounds(request, std::move(answer), hook.image_model_id(),
                                 hook.url_base(), unix_time_now(),
                                 [&hook](const OpenAIImagesRequest& image_request) {
                                     return hook.draw(image_request);
                                 },
                                 run_next_answer);
}

std::string chat_image_missing_url_suffix(const ChatImageHook& hook, std::string_view answer_text,
                                          const std::vector<std::string>& urls) {
    (void)hook; // the door this rule is reachable through; the rule itself is text and urls
    return missing_image_url_suffix(answer_text, urls);
}

std::string chat_image_progress_note(const ChatImageHook& hook) { return hook.progress_note(); }

std::shared_ptr<ImageJobProgress> chat_image_watch_progress(const ChatImageHook& hook) {
    return hook.watch_progress();
}

// What the server-executed image tool tried, in the field every protocol carries under the same name.
// The string is this process's own writing (serve/chat_image_tool.cpp), never socket input, so a parse
// that fails is a bug -- and the answer to a bug here is no field, not a failed response. Absent when
// this request drew no picture, which is what keeps a plain answer's bytes unchanged.
RequestJson chat_image_attempts_json(const GenerationOutcome& outcome) {
    return chat_image_attempts_for_response(outcome);
}

void add_chat_image_attempts(nlohmann::json& payload, const GenerationOutcome& outcome) {
    // The attempt array, or null when this request drew no picture -- and null means the field is not
    // emitted at all, which is the whole reason a plain chat answer is the same bytes it was before the
    // field existed. The reading itself belongs to the tool that writes the string: see
    // chat_image_attempts_for_response.
    //
    // Additive, and absent whenever this request drew no picture: the picture's fate is the server's
    // own record, and the client is not left to take the model's word for it.
    //
    // The hidden tool's attempts belong on the terminal chunk only, beside usage and timings: they
    // describe work that is finished, and a client accumulating deltas should read them once.
    //
    // A sibling of usage, not a content block: the picture's fate is a fact about this request, and a
    // client must be able to read it without believing anything the model said about it.
    //
    // Inside the response object, which is where the Responses protocol puts everything about the
    // response: the streaming terminal event ships this same body, so one write covers both shapes.
    //
    // Beside usage and stop_reason, in the one event that closes the message: content blocks are
    // already on the wire and indexed, and this is not content -- it is what the server did.
    if (Json attempts = chat_image_attempts_json(outcome); !attempts.is_null()) {
        payload["image_tool_attempts"] = std::move(attempts);
    }
}

// The Anthropic response and its terminal event are built with the request's own ordered document type,
// so that renderer reaches the same rule through this shape. The attempts are still read into the
// sorted document above first and copied over as it, which is what keeps one written attempt's keys in
// the same order on all three protocols.
void add_chat_image_attempts(nlohmann::ordered_json& payload, const GenerationOutcome& outcome) {
    if (Json attempts = chat_image_attempts_json(outcome); !attempts.is_null()) {
        payload["image_tool_attempts"] = std::move(attempts);
    }
}

bool chat_image_tool_hidden(std::string_view tool_name) noexcept {
    return tool_name == kChatImageToolName;
}

} // namespace infernix::serve
