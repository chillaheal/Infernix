#include "serve/prompt_enhancer.h"

#include "serve/openai_images.h"
#include "serve/request_json.h"

#include <httplib.h>

#include <cctype>
#include <exception>
#include <utility>

namespace infernix::serve {
namespace {

constexpr std::string_view kBadAnswerCode     = "image_prompt_enhancer_bad_answer";
constexpr std::string_view kUnavailableCode   = "image_prompt_enhancer_unavailable";
constexpr std::string_view kFailedCode        = "image_prompt_enhancer_failed";
constexpr std::string_view kFieldParam        = "prompt_enhance";
constexpr std::size_t kMaxRewrittenPrompt     = 8000;
constexpr std::chrono::milliseconds kHealthBound{2000};

[[noreturn]] void bad_answer(const std::string& reason) {
    throw ApiException{ApiError{500, "server_error",
                                "the prompt enhancer answered outside its contract: " + reason,
                                std::string(kFieldParam), std::string(kBadAnswerCode)}};
}

[[noreturn]] void enhancer_unavailable(const std::string& reason) {
    throw ApiException{ApiError{503, "server_error", reason, std::string(kFieldParam),
                                std::string(kUnavailableCode)}};
}

bool is_space(char character) noexcept {
    return character == ' ' || character == '\t' || character == '\v' || character == '\f';
}

// True when the text spells a size or an aspect: a digit run, a separator, another digit run
// ("1024x1024", "16:9", "1024 x 1024"), or one of the words that only appear when somebody is
// talking about the canvas instead of the picture. The shipped system prompt forbids both
// (refs/q21_system_prompt_edit.txt:188: no resolution or ratio in rewritten_prompt -- size travels
// only in wh_ratio / ratio_follow), and an instruction that carries its own size is the one case
// where the two channels can disagree with the canvas this route already decided on.
bool names_size_or_ratio(const std::string& text) {
    static constexpr std::string_view words[] = {"aspect ratio", "resolution", "pixels",
                                                 "portrait orientation", "landscape orientation"};
    std::string lowered;
    lowered.reserve(text.size());
    for (const char character : text) {
        lowered.push_back(
            static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
    }
    for (const auto word : words) {
        if (lowered.find(word) != std::string::npos) { return true; }
    }
    std::size_t index = 0;
    while (index < text.size()) {
        if (!std::isdigit(static_cast<unsigned char>(text[index]))) {
            ++index;
            continue;
        }
        std::size_t run = index;
        while (run < text.size() && std::isdigit(static_cast<unsigned char>(text[run]))) { ++run; }
        std::size_t probe = run;
        while (probe < text.size() && is_space(text[probe])) { ++probe; }
        if (probe < text.size() && (text[probe] == 'x' || text[probe] == 'X' ||
                                    text[probe] == ':' || text[probe] == '*')) {
            ++probe;
            while (probe < text.size() && is_space(text[probe])) { ++probe; }
            if (probe < text.size() && std::isdigit(static_cast<unsigned char>(text[probe]))) {
                return true;
            }
        }
        index = run;
    }
    return false;
}

[[nodiscard]] std::string string_field(const RequestJson& record, const char* name,
                                       const std::string& field) {
    if (!record.contains(field)) { bad_answer(std::string("the answer has no ") + name); }
    const auto& value = record.at(field);
    if (!value.is_string()) { bad_answer(std::string("the answer's ") + name + " is not a string"); }
    return value.get<std::string>();
}

} // namespace

EnhancerAnswer validate_enhancer_answer(const RequestJson& reply,
                                        const std::size_t reference_count) {
    if (!reply.is_object()) { bad_answer("the answer is not a JSON object"); }
    if (reply.contains("parse_ok")) {
        const auto& flag = reply.at("parse_ok");
        if (!flag.is_boolean()) { bad_answer("parse_ok is not a boolean"); }
        if (!flag.get<bool>()) {
            bad_answer("its answer did not parse as the expected JSON object");
        }
    }

    EnhancerAnswer answer;
    answer.rewritten_prompt = string_field(reply, "rewritten_prompt", "rewritten_prompt");
    answer.wh_ratio         = string_field(reply, "wh_ratio", "wh_ratio");
    answer.ratio_follow     = string_field(reply, "ratio_follow", "ratio_follow");
    if (reply.contains("thinking")) {
        const auto& thinking = reply.at("thinking");
        if (!thinking.is_string()) { bad_answer("thinking is not a string"); }
        answer.thinking = thinking.get<std::string>();
    }
    if (reply.contains("new_tokens")) {
        const auto& tokens = reply.at("new_tokens");
        if (!tokens.is_number_integer()) { bad_answer("new_tokens is not an integer"); }
        answer.new_tokens = tokens.get<std::int64_t>();
    }

    // Trim: a model that pads its answer with whitespace said the same thing.
    const auto trim = [](std::string& text) {
        const auto first = text.find_first_not_of(" \t\r\n\v\f");
        if (first == std::string::npos) {
            text.clear();
            return;
        }
        const auto last = text.find_last_not_of(" \t\r\n\v\f");
        text = text.substr(first, last - first + 1);
    };
    trim(answer.rewritten_prompt);
    trim(answer.wh_ratio);
    trim(answer.ratio_follow);

    if (answer.rewritten_prompt.empty()) { bad_answer("rewritten_prompt is empty"); }
    if (answer.rewritten_prompt.size() > kMaxRewrittenPrompt) {
        bad_answer("rewritten_prompt is longer than " + std::to_string(kMaxRewrittenPrompt) +
                   " characters");
    }
    // One paragraph: the contract is a single block of prose (refs/q21_system_prompt_edit.txt:185-192).
    if (answer.rewritten_prompt.find('\n') != std::string::npos ||
        answer.rewritten_prompt.find('\r') != std::string::npos) {
        bad_answer("rewritten_prompt spans more than one paragraph");
    }
    if (names_size_or_ratio(answer.rewritten_prompt)) {
        bad_answer("rewritten_prompt carries a resolution or an aspect ratio, which belong only in "
                   "wh_ratio or ratio_follow");
    }

    const bool has_ratio  = !answer.wh_ratio.empty();
    const bool has_follow = !answer.ratio_follow.empty();
    if (has_ratio && has_follow) {
        bad_answer("both wh_ratio and ratio_follow were answered; exactly one may carry a value");
    }
    if (!has_ratio && !has_follow && reference_count == 0) {
        // Text-to-image has no picture to follow, so an answer with neither names no canvas at all.
        bad_answer("neither wh_ratio nor ratio_follow was answered");
    }
    if (has_ratio) {
        try {
            (void)parse_image_aspect_ratio(answer.wh_ratio);
        } catch (const ApiException& exception) {
            bad_answer("wh_ratio is not a usable \"W:H\" aspect (" + exception.error().message + ")");
        }
    }
    if (has_follow) {
        std::size_t index = 0;
        if (!parse_image_reference_tag(answer.ratio_follow, reference_count, index)) {
            bad_answer("ratio_follow does not name one of the " + std::to_string(reference_count) +
                       " references as <imageN>");
        }
    }
    return answer;
}

PromptEnhancer::PromptEnhancer(std::string base_url, const std::chrono::milliseconds timeout)
    : base_url_(std::move(base_url)), timeout_(timeout) {}

namespace {
// Split "http://host:port" (or a bare "host:port") into the two parts httplib::Client wants. A URL
// with a path, a query or a second colon is a misconfigured flag, not something to guess about.
struct Endpoint {
    std::string host;
    int port = 0;
};

[[nodiscard]] Endpoint split_endpoint(const std::string& base_url) {
    std::string value = base_url;
    if (const auto scheme = value.find("://"); scheme != std::string::npos) {
        value = value.substr(scheme + 3);
    }
    if (value.find('/') != std::string::npos) {
        enhancer_unavailable("the configured prompt enhancer URL must be a host:port with no path");
    }
    const auto colon = value.rfind(':');
    if (colon == std::string::npos) {
        enhancer_unavailable("the configured prompt enhancer URL must name a port");
    }
    Endpoint endpoint;
    endpoint.host = value.substr(0, colon);
    const std::string port = value.substr(colon + 1);
    // At most five digits before converting: a longer run is out of any port's range, and stoll would
    // say so with an exception type the route has no mapping for.
    if (endpoint.host.empty() || port.empty() || port.size() > 5 ||
        port.find_first_not_of("0123456789") != std::string::npos) {
        enhancer_unavailable("the configured prompt enhancer URL is not host:port");
    }
    const long long number = std::stoll(port);
    if (number <= 0 || number > 65535) {
        enhancer_unavailable("the configured prompt enhancer URL names a port outside 1-65535");
    }
    endpoint.port = static_cast<int>(number);
    return endpoint;
}

// The one place a sidecar's silence becomes an ApiError. httplib hands back an empty Result for a
// refused or timed-out connection, which is "not started"/"wrong port" to an operator and reads that
// way here; a reply with a body is the worker's own words.
[[nodiscard]] RequestJson call(httplib::Client& client, const Endpoint& endpoint,
                               const std::string& path, const RequestJson& body,
                               const std::string& what) {
    const auto result = client.Post(path, body.is_null() ? "{}" : body.dump(), "application/json");
    if (!result) {
        enhancer_unavailable(what + " is not answering at " + endpoint.host + ":" +
                             std::to_string(endpoint.port) + " (" +
                             httplib::to_string(result.error()) + "); start "
                             "tools/prompt_enhancer/pe_worker.py or send the request without "
                             "prompt_enhance");
    }
    if (result->status < 200 || result->status >= 300) {
        std::string detail;
        try {
            const auto parsed = RequestJson::parse(result->body);
            if (parsed.is_object() && parsed.contains("error") && parsed.at("error").is_string()) {
                detail = ": " + parsed.at("error").get<std::string>();
            }
        } catch (const std::exception&) { /* the body is not the worker's JSON; keep the status */ }
        throw ApiException{ApiError{500, "server_error",
                                    what + " answered HTTP " + std::to_string(result->status) +
                                        detail,
                                    std::string(kFieldParam), std::string(kFailedCode)}};
    }
    try {
        return RequestJson::parse(result->body);
    } catch (const std::exception&) {
        throw ApiException{ApiError{500, "server_error", what + " answered something that is not JSON",
                                    std::string(kFieldParam), std::string(kFailedCode)}};
    }
}
} // namespace

bool PromptEnhancer::healthy() const {
    if (!configured()) { return false; }
    try {
        const Endpoint endpoint = split_endpoint(base_url_);
        httplib::Client client(endpoint.host, endpoint.port);
        const auto bound = timeout_ < kHealthBound ? timeout_ : kHealthBound;
        client.set_connection_timeout(bound);
        client.set_read_timeout(bound);
        const auto result = client.Get("/health");
        if (!result || result->status != 200) { return false; }
        const auto reply = RequestJson::parse(result->body);
        return reply.is_object() && reply.contains("ok") && reply.at("ok").is_boolean() &&
               reply.at("ok").get<bool>();
    } catch (const std::exception&) {
        return false;
    }
}

RequestJson PromptEnhancer::post(const char* path, const RequestJson& body) const {
    if (!configured()) {
        enhancer_unavailable("this build has no prompt enhancer configured; the family's is " +
                             std::string(kImagePromptEnhancerModel) +
                             ", and in its absence the client writes the instruction itself (see GET "
                             "/v1/capabilities image_edit.instruction)");
    }
    const Endpoint endpoint = split_endpoint(base_url_);
    httplib::Client client(endpoint.host, endpoint.port);
    client.set_connection_timeout(timeout_);
    client.set_read_timeout(timeout_);
    return call(client, endpoint, path, body,
                "the prompt enhancer at " + base_url_);
}

void PromptEnhancer::load() const {
    (void)post("/load", RequestJson::object());
}

void PromptEnhancer::unload() const {
    (void)post("/unload", RequestJson::object());
}

EnhancerAnswer PromptEnhancer::rewrite(const std::string& prompt,
                                       const std::vector<std::string>& images_base64,
                                       const std::uint64_t seed,
                                       const std::size_t reference_count) const {
    RequestJson body = RequestJson::object();
    body["prompt"]   = prompt;
    body["images"]   = images_base64;
    body["seed"]     = seed;
    const auto reply = post("/rewrite", body);
    return validate_enhancer_answer(reply, reference_count);
}

} // namespace infernix::serve
