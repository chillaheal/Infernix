#pragma once
// The one joint-sequence ceiling for the image lane. Attention, rope_c64, dit_adaln,
// canvas admission, the chat tool schema and /v1/capabilities all read THIS symbol, so
// a job can never be admitted past a limit an op will refuse. Raising it is a capacity
// decision (scratch, the VRAM ledger, attention cost), never an edit here and there.
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace infernix::ops {

inline constexpr std::int32_t kImageJointMaxTokens = 20480;

[[nodiscard]] constexpr bool image_joint_tokens_ok(std::int32_t tokens) noexcept {
    return tokens >= 1 && tokens <= kImageJointMaxTokens;
}

// The refusal text, built from the constant so it cannot drift from the number it names.
[[nodiscard]] inline std::string image_joint_range_message(std::string_view op) {
    return std::string(op) + ": S must be 1.." + std::to_string(kImageJointMaxTokens);
}

} // namespace infernix::ops
