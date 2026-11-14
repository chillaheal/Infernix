// One joint-sequence ceiling, observable from the three places that must never disagree: the image
// ops' own ceiling check (it must refuse above the shared number and admit up to it), /v1/capabilities,
// and the chat tool schema the model reads. Public behaviour only -- no mirror of the validation code.
// IMAGE_JOINT_LIMITS_TEST: PASS
#include "infernix/ops/image_joint_limits.h"
#include "infernix/ops/attn_dit_joint.h"
#include "infernix/ops/rope_c64.h"
#include "infernix/ops/dit_adaln.h"
#include "serve/chat_image_tool.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "ok: " : "FAIL: ") << what << '\n' << std::flush;
    if (!ok) { ++failures; }
}

constexpr std::int32_t kCeiling = infernix::ops::kImageJointMaxTokens;

// An entry point must refuse with exactly the message the shared constant builds. Comparing whole
// text (not just the digits) is what makes a second, hand-written limit visible as a failure.
template <typename Fn>
bool refuses_with(const Fn& fn, const std::string& op) {
    try {
        fn();
    } catch (const std::invalid_argument& error) {
        return std::string(error.what()) == infernix::ops::image_joint_range_message(op);
    } catch (...) {
        return false;
    }
    return false;
}

// The other half of the boundary. An entry handed a token count the shared ceiling admits must get PAST
// the joint-range gate: with the null data pointer these operands keep, it then stops one step later on
// a different validation with its own text. Requiring exactly that text is what makes the check able to
// fail -- it proves the entry was really reached and that the shared gate, not a private bound, is what
// let S through. An entry that quietly kept (or reintroduced) its own lower ceiling answers here with the
// shared refusal text, which is the failure this suite exists to prevent: serve admits the job, the op
// refuses, the client gets a 500 instead of a 400. Throwing nothing would mean a launch, so that counts
// as a failure too -- this suite must never reach a device.
template <typename Fn>
bool accepts_past_the_ceiling_check(const Fn& fn, const std::string& op,
                                    const std::string& expected_next_refusal) {
    try {
        fn();
    } catch (const std::invalid_argument& error) {
        const std::string text = error.what();
        return text == expected_next_refusal &&
               text != infernix::ops::image_joint_range_message(op);
    } catch (...) {
        return false;
    }
    return false;
}

// The ops read the sequence length from the tensor's public shape, and Tensor's own constructor
// rejects a zero dimension -- so an operand is built with positive dims and S is written onto its
// token axis afterwards. The data pointer stays null on purpose: each entry validates the token
// count before it looks at an operand's bytes, strides or contiguity, so the refusal happens before
// a pointer is read and before any launch, and this needs no GPU work.
infernix::Tensor bf16_tokens(std::array<std::int32_t, 4> shape, std::int32_t tokens,
                             std::int32_t token_axis) {
    // Every caller passes a positive placeholder on the token axis, which the line below replaces.
    infernix::Tensor tensor{nullptr, infernix::DType::BF16, {shape[0], shape[1], shape[2], shape[3]}};
    tensor.ne[token_axis] = tokens;
    return tensor;
}

// One case per public entry point, each handed the token axis as a parameter so the boundary values
// below drive every entry through the same two refusals.
void entry_attn(std::int32_t tokens) {
    const auto operand = [tokens] { return bf16_tokens({128, 32, 1, 1}, tokens, 2); };
    infernix::Tensor q = operand();
    infernix::Tensor k = operand();
    infernix::Tensor v = operand();
    infernix::Tensor out = operand();
    infernix::ops::attn_dit_joint(q, k, v, 0, out, nullptr);
}

void entry_attn_block_causal(std::int32_t tokens) {
    const auto operand = [tokens] { return bf16_tokens({128, 32, 1, 1}, tokens, 2); };
    infernix::Tensor q = operand();
    infernix::Tensor k = operand();
    infernix::Tensor v = operand();
    infernix::Tensor out = operand();
    const infernix::Tensor image_ids{};
    infernix::ops::attn_dit_joint_block_causal(q, k, v, image_ids, out, nullptr);
}

void entry_rope_pair(std::int32_t tokens) {
    const infernix::Tensor rot{};
    infernix::Tensor q = bf16_tokens({128, 32, 1, 1}, tokens, 2);
    infernix::Tensor k = bf16_tokens({128, 32, 1, 1}, tokens, 2);
    infernix::ops::rope_c64(rot, q, k, nullptr);
}

void entry_rope_single(std::int32_t tokens) {
    const infernix::Tensor rot{};
    infernix::Tensor x = bf16_tokens({128, 32, 1, 1}, tokens, 2);
    infernix::ops::rope_c64(rot, x, nullptr);
}

void entry_adaln_scale(std::int32_t tokens) {
    infernix::Tensor x = bf16_tokens({64, 1, 1, 1}, tokens, 1);
    infernix::Tensor out = bf16_tokens({64, 1, 1, 1}, tokens, 1);
    const infernix::Tensor mod{};
    infernix::ops::adaln_scale(x, mod, 0, 0, out, nullptr);
}

void entry_adaln_gate_add(std::int32_t tokens) {
    infernix::Tensor residual = bf16_tokens({64, 1, 1, 1}, tokens, 1);
    infernix::Tensor branch = bf16_tokens({64, 1, 1, 1}, tokens, 1);
    infernix::Tensor out = bf16_tokens({64, 1, 1, 1}, tokens, 1);
    const infernix::Tensor mod{};
    infernix::ops::adaln_gate_add(residual, branch, mod, 0, 0, out, nullptr);
}

void entry_adaln_scale_masked(std::int32_t tokens) {
    infernix::Tensor x = bf16_tokens({64, 1, 1, 1}, tokens, 1);
    infernix::Tensor out = bf16_tokens({64, 1, 1, 1}, tokens, 1);
    const infernix::Tensor mod{};
    const infernix::Tensor target_mask{};
    infernix::ops::adaln_scale_masked(x, mod, 0, target_mask, out, nullptr);
}

void entry_adaln_gate_add_masked(std::int32_t tokens) {
    infernix::Tensor residual = bf16_tokens({64, 1, 1, 1}, tokens, 1);
    infernix::Tensor branch = bf16_tokens({64, 1, 1, 1}, tokens, 1);
    infernix::Tensor out = bf16_tokens({64, 1, 1, 1}, tokens, 1);
    const infernix::Tensor mod{};
    const infernix::Tensor target_mask{};
    infernix::ops::adaln_gate_add_masked(residual, branch, mod, 0, target_mask, out, nullptr);
}

struct boundary_case {
    const char* entry;           // the public entry point, named in the check line
    const char* prefix;          // the op prefix every refusal of this domain carries
    void (*run)(std::int32_t tokens);
    const char* next_refusal;    // what the SAME entry must refuse with once the joint gate lets S
                                 // through: the first validation after it, which the null data pointer
                                 // alone makes refuse
};

const boundary_case kCases[] = {
    {"attn_dit_joint", "attn_dit_joint", &entry_attn, "attn_dit_joint: invalid q"},
    {"attn_dit_joint_block_causal", "attn_dit_joint", &entry_attn_block_causal,
     "attn_dit_joint: invalid image_ids [S,1,1,1] I32"},
    {"rope_c64(rot, q, k)", "rope_c64", &entry_rope_pair, "rope_c64: invalid rot"},
    {"rope_c64(rot, x)", "rope_c64", &entry_rope_single, "rope_c64: invalid rot"},
    {"adaln_scale", "dit_adaln", &entry_adaln_scale, "dit_adaln: invalid activation x"},
    {"adaln_gate_add", "dit_adaln", &entry_adaln_gate_add, "dit_adaln: invalid activation residual"},
    {"adaln_scale_masked", "dit_adaln", &entry_adaln_scale_masked,
     "dit_adaln: invalid activation x"},
    {"adaln_gate_add_masked", "dit_adaln", &entry_adaln_gate_add_masked,
     "dit_adaln: invalid activation residual"},
};

} // namespace

int main() {
    check(kCeiling == 20480, "the shared ceiling is the 20480 the image lane was built for");
    check(infernix::ops::image_joint_tokens_ok(1) &&
              infernix::ops::image_joint_tokens_ok(kCeiling - 1) &&
              infernix::ops::image_joint_tokens_ok(kCeiling),
          "a sequence up to and including the ceiling is accepted");
    check(!infernix::ops::image_joint_tokens_ok(0) &&
              !infernix::ops::image_joint_tokens_ok(-1) &&
              !infernix::ops::image_joint_tokens_ok(kCeiling + 1),
          "an empty or over-long sequence is rejected");

    // Both sides of the boundary for every public entry, through one table: refusing at ceiling+1 and
    // at 0 is only half the invariant -- an entry that kept its own private bound would refuse nowhere
    // near the shared number and still satisfy those two checks while the serve admitted the job. So
    // each entry must also let an admitted token count past the joint check, at the ceiling itself and
    // one under it, landing on the validation that follows.
    for (const boundary_case& c : kCases) {
        const std::string name = std::string(c.entry) + " [" + c.prefix + "]";
        check(refuses_with([&] { c.run(kCeiling + 1); }, c.prefix),
              name + " refuses one over the ceiling naming the shared number");
        check(refuses_with([&] { c.run(0); }, c.prefix),
              name + " refuses an empty sequence naming the same number");
        check(accepts_past_the_ceiling_check([&] { c.run(kCeiling); }, c.prefix, c.next_refusal),
              name + " admits a sequence at the ceiling, refusing only past the joint check");
        check(accepts_past_the_ceiling_check([&] { c.run(kCeiling - 1); }, c.prefix, c.next_refusal),
              name + " admits a sequence one under the ceiling, refusing only past the joint check");
    }

    const std::string ceiling = std::to_string(kCeiling);
    check(infernix::serve::make_chat_image_tool_definition().input_schema_json.find(ceiling) !=
              std::string::npos,
          "the tool schema the model reads names the shared ceiling");
    const std::string capabilities = infernix::serve::make_capabilities_json(
        infernix::serve::ServeOptions{}, "chat-model", "image-model");
    check(capabilities.find("\"ceiling_tokens\":" + ceiling) != std::string::npos,
          "/v1/capabilities publishes the same ceiling");

    std::cout << (failures == 0 ? "IMAGE_JOINT_LIMITS_TEST: PASS" : "IMAGE_JOINT_LIMITS_TEST: FAIL")
              << '\n';
    return failures == 0 ? 0 : 1;
}
