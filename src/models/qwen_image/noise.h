#pragma once

// Seeded standard-normal noise for the Qwen-Image-2.1 initial latents (the image job's "seed" knob,
// docs/maintainer/qwen-image-2-1-variables.md). Host-only, no device: the in-process image runtime
// fills its init-latent buffer from this before the VAE scaling. The sequence is a stateful
// SplitMix64 (the same 64-bit constants as qwen4_exp ngram_hash.cpp) mapped through Box-Muller to
// N(0,1); a fixed seed gives a reproducible latent for a given element count.

#include <cstdint>
#include <span>

namespace infernix::models::qwen_image {

// Fill `out` (any length, odd or even) with reproducible standard-normal samples seeded by `seed`.
// Independent of resolution: the caller supplies the latent element count (channels * Hl * Wl) and
// reshapes; this only owns the seeded N(0,1) stream.
void fill_normal(std::uint64_t seed, std::span<float> out) noexcept;

} // namespace infernix::models::qwen_image
