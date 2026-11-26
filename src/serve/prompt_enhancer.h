#pragma once

// The family's prompt enhancer, as the image route uses it: a SEPARATE shipped checkpoint
// (Qwen/Qwen-Image-2.1-PE-I2I, a fine-tuned Qwen3.5-VL 9B) that reads the user's instruction and the
// source pictures and answers {rewritten_prompt, wh_ratio, ratio_follow}
// (image-edit/refs/q21_prompt_rewrite_README.md:1-12, refs/q21_system_prompt_edit.txt:175-183). It is
// not a template that can be put on an existing tower -- the image artifact's own vision-language
// tower is a different checkpoint, never trained against the enhancer's system prompt, so running it
// would return the picture's embeddings rather than a better instruction.
//
// The checkpoint is served by a sidecar process (tools/prompt_enhancer/pe_worker.py) rather than
// loaded into this one: its BF16 weights are about 20 GB and this card also holds the resident LLM and
// the DiT, which is why the route drives it through /load and /unload inside the image-job interlude
// instead of keeping it resident. A separate process is an architectural boundary (a crash in the
// enhancer cannot take the server down); the VRAM boundary is the lease, not the process.

#include "serve/request.h"
#include "serve/request_json.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace infernix::serve {

// One enhancer answer after validation. The ratio fields are empty when the model left them empty;
// exactly one of them is ever non-empty for the edit task, and that is what the validator enforces.
struct EnhancerAnswer {
    std::string rewritten_prompt;
    std::string wh_ratio;
    std::string ratio_follow;
    std::string thinking;
    std::int64_t new_tokens = 0;
};

// The rules the shipped edit system prompt puts on its own answer
// (refs/q21_system_prompt_edit.txt:185-192, refs/q21_prompt_rewrite_README.md:107-121): exactly the
// three fields; wh_ratio XOR ratio_follow, one of them ""; a single paragraph with no newline; no
// resolution or ratio inside rewritten_prompt. A shape the two existing parsers already own is checked
// by them -- parse_image_aspect_ratio for "W:H", parse_image_reference_tag for "<imageN>" naming a
// picture the request actually carries.
//
// An answer that breaks a rule is the SERVER's failure, not the client's: the client asked for
// enhancement, the component this deployment added produced the bad text. It is a 500
// image_prompt_enhancer_bad_answer, never a 400 that blames a parameter the client never sent.
[[nodiscard]] EnhancerAnswer validate_enhancer_answer(const RequestJson& reply,
                                                     std::size_t reference_count);

// Loopback HTTP client for the sidecar. Every failure is an ApiException carrying an
// image_prompt_enhancer_* code, so the route's answer names which part is missing.
class PromptEnhancer {
public:
    PromptEnhancer(std::string base_url, std::chrono::milliseconds timeout);
    virtual ~PromptEnhancer() = default;

    // True when a URL was configured. It says nothing about whether the sidecar answers -- that is
    // healthy(), and a caller must not claim enhancement is available before asking.
    [[nodiscard]] bool configured() const noexcept { return !base_url_.empty(); }
    [[nodiscard]] const std::string& base_url() const noexcept { return base_url_; }

    // GET /health. False when the sidecar does not answer at all (not started, wrong port); the
    // caller turns that into 503 image_prompt_enhancer_unavailable.
    //
    // These four are virtual for one reason: the service's decision about an enhancer (refuse, skip,
    // run) is host-verifiable logic, and a host gate must be able to stand a rewriter that answers
    // without a sidecar listening on a port. Nothing else derives from it.
    [[nodiscard]] virtual bool healthy() const;

    // Bring the weights onto the card and put them back. Both are idempotent on the worker side, so
    // the lease can call them per batch without caring what the last batch did. Const like the rest:
    // the client owns no state but its address and timeout -- the weights are the sidecar's business.
    virtual void load() const;
    virtual void unload() const;

    // One edit rewrite. `images_base64` is the request's own references in request order -- the order
    // is the <image1>, <image2>, ... the rewritten prompt addresses them by, so it is never reordered
    // here. `reference_count` is how many there are, which is what bounds a ratio_follow tag.
    [[nodiscard]] virtual EnhancerAnswer rewrite(const std::string& prompt,
                                         const std::vector<std::string>& images_base64,
                                         std::uint64_t seed, std::size_t reference_count) const;

private:
    [[nodiscard]] RequestJson post(const char* path, const RequestJson& body) const;

    std::string base_url_;
    std::chrono::milliseconds timeout_;
};

} // namespace infernix::serve
