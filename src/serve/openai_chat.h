#pragma once

// OpenAI Chat Completions wire adapter. Parsing produces one protocol envelope plus an executable
// GenerationRequest; response builders consume protocol-neutral GenerationOutcome values.

#include "serve/request.h"
#include "serve/request_json.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace infernix::serve {

struct GenerationOutcome;

struct OpenAIChatRequest {
    std::string model;
    GenerationRequest generation;
    bool stream                 = false;
    bool include_usage          = false;
    bool output_tokens_explicit = false;
    // llama.cpp-compatible response observations. Final timings remain unconditional;
    // timings_per_token controls only cumulative timing snapshots on streamed output chunks.
    bool timings_per_token = false;
    bool return_progress   = false;
};

OpenAIChatRequest parse_chat_completion_request(const RequestJson& body,
                                                const RequestLimits& limits);

// Appends a Chat Completions message content value (a string or content parts) to `turn`, whose
// role decides which media parts are accepted; `index` names the message in errors.
void parse_openai_message_content(const RequestJson& content, ChatTurn& turn, std::size_t index);

struct OpenAIChatResponseIdentity {
    std::string id;
    std::string model;
    std::int64_t created = 0;
};

OpenAIChatResponseIdentity make_openai_chat_response_identity(std::string model);
// `call_ids`, when given, receives the ids of the response's tool calls in order.
std::string make_chat_completion_response(const OpenAIChatResponseIdentity& identity,
                                          const GenerationOutcome& outcome,
                                          std::vector<std::string>* call_ids = nullptr);

// Why one tool-call delta could not be rendered. Those deltas come from the model's constrained
// decoding, so an unusable one is model output arriving on a trust boundary: it fails the request,
// and it is never an exception -- an exception that leaves the SSE content provider lands on a
// worker thread with no handler, and the server dies with the request.
enum class OpenAIChatProtocolError : std::uint8_t {
    None,
    NotStarted,
    AlreadyFinished,
    MissingToolName,
    DeltaAfterComplete,
    ToolChangedMidCall,
};

const char* describe_openai_chat_protocol_error(OpenAIChatProtocolError error) noexcept;

// What one tool-call delta produced: the chunk it earned, or the reason it was refused.
struct OpenAIChatToolCallChunk {
    std::string chunk;
    OpenAIChatProtocolError error = OpenAIChatProtocolError::None;
    [[nodiscard]] bool ok() const noexcept { return error == OpenAIChatProtocolError::None; }
};

class OpenAIChatStream {
public:
    OpenAIChatStream(OpenAIChatResponseIdentity identity, bool include_usage,
                     bool timings_per_token = false, bool return_progress = false,
                     bool usage_chunk_choice = false);

    std::string start();
    void note_start(const infernix::GenerationStart& start);
    std::string initial_prompt_progress();
    std::string prompt_progress(const infernix::PromptProgress& progress);
    void note_timing(const infernix::GenerationTimingObservation& timing);
    std::string reasoning_delta(const std::string& text);
    std::string content_delta(const std::string& text);
    // A constrained call's step: its first opens the call (id, type, name), every one extends its
    // arguments. Total: a delta this adapter cannot use is refused with `error`, never thrown.
    OpenAIChatToolCallChunk tool_call_delta(const infernix::ToolCallDelta& delta);
    std::vector<std::string> finish(const GenerationOutcome& outcome);
    // After finish: the ids of the result's tool calls, in order.
    [[nodiscard]] const std::vector<std::string>& tool_call_ids() const noexcept { return call_ids_; }

private:
    nlohmann::json live_timings_json() const;

    struct StreamedCall {
        // The Engine's number for this call. It is what a delta is matched by: the Engine numbers a
        // turn's calls from zero and restarts that numbering every continuation round, so it is
        // neither this list's position nor the number the client reads.
        std::uint32_t engine_index = 0;
        // The number this message gives the call, counting only the calls it publishes. A call the
        // server runs for itself stays in this list -- the terminal outcome counts it, and the
        // result's calls are matched against it by position -- but it earns no index and no chunk.
        std::uint32_t published_index = 0;
        bool hidden                   = false;
        std::string id;
        std::string name;
        std::string arguments;
        bool complete = false;
    };

    OpenAIChatResponseIdentity identity_;
    std::string reasoning_;
    std::string content_;
    std::vector<StreamedCall> calls_;
    std::vector<std::string> call_ids_;
    std::optional<infernix::GenerationTimingObservation> live_timing_;
    std::uint32_t prompt_tokens_            = 0;
    std::uint32_t cached_tokens_            = 0;
    std::uint32_t last_progress_tokens_     = 0;
    std::uint64_t last_progress_elapsed_ns_ = 0;
    // The next `index` this message publishes. Hidden calls never take one, so the numbers a client
    // reads stay 0, 1, 2 ... whatever the Engine numbered between them.
    std::uint32_t next_published_index_     = 0;
    bool include_usage_                     = false;
    bool timings_per_token_                 = false;
    bool return_progress_                   = false;
    bool usage_chunk_choice_                = false;
    bool started_                           = false;
    bool admitted_                          = false;
    bool progress_started_                  = false;
    bool content_started_                   = false;
    bool finished_                          = false;
};

} // namespace infernix::serve
