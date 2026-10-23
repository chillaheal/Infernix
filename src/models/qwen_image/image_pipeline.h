#pragma once

// The production image job's CUDA execution: the FlowMatchEuler denoise loop through the streamed
// .infernix DiT weights plus the VAE decode, as ONE reusable entry that runs on the VRAM the LLM's
// interlude yielded. Merge authority: tests/models/qwen4_exp/test_image_interlude_dit_real.cu (the
// device-qualified route: txt_in computed once, one block resident per step from a host cache,
// RAII everywhere, every image allocation freed before the lease restores the LLM's pools). No
// oracle files and no debug gates live here: the weights come from the loader's streamed
// references, the job inputs from the caller, and the numeric re-qualification of this exact
// composition against the diffusers reference happens in the combined certification run.
//
// The caller supplies `stream` (the interlude's dedicated image stream, created blocking) and all
// ops run on it. Host/device transfers are synchronous cudaMemcpy, which the legacy default stream
// orders against every blocking stream, so they stay correctly sequenced with work the shared
// block forward submits on the legacy stream.

#include "models/qwen_image/image_weights.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace infernix::artifact { class Reader; }

namespace infernix::models::qwen_image {

// Everything about one job that is neither a weight nor a model constant. The arrays are BF16 bit
// payloads / float32 the caller produced with the host modules (build_schedule, fill_normal,
// ImageGeometry); height/width are the snapped job dimensions in pixels.
struct ImageJobInputs {
    std::vector<std::uint16_t> prompt_embeds; // [text_tokens, hidden] token-major BF16 bits
    std::vector<std::uint16_t> init_latent;   // [latents.tokens, image_in] token-major BF16 bits
    std::vector<float> timesteps;             // [steps]
    std::vector<float> sigmas;                // [steps + 1]
    std::vector<float> latents_mean;          // [image_in] VAE latent denormalization
    std::vector<float> latents_std;           // [image_in]
    std::uint32_t height_px = 0;              // snapped job height (multiple of 16)
    std::uint32_t width_px = 0;               // snapped job width (multiple of 16)
    // #75 reference/edit images (may be empty = plain text-to-image). ref_latents is the
    // concatenation of the encoded reference blocks (encode_reference_image output, already
    // normalized) in order, ref_grids their (Hl_i, Wl_i) latent grids; the joint sequence becomes
    // [text, ref tokens (constant across steps), target tokens] exactly like the reference
    // pipeline's latent_model_input = cat([input_images_latents, latents]).
    std::vector<std::uint16_t> ref_latents;               // [ref_tokens, image_in] token-major BF16 bits
    std::vector<std::pair<std::uint32_t, std::uint32_t>> ref_grids; // per reference (height_lat, width_lat)
    // True CFG: with true_cfg_scale != 1 the step runs the DiT a second time over the negative prompt
    // (its own token count, rope table and joint sequence) and the velocity becomes the BF16-rounded
    // neg + true_cfg_scale * (cond - neg), as the reference pipeline's true_cfg branch does. The
    // default (1.0 with empty negatives) is the single-forward route, byte for byte.
    std::vector<std::uint16_t> negative_prompt_embeds;    // [neg_tokens, hidden] token-major BF16 bits
    float true_cfg_scale = 1.0F;
};

// The job's output. channels is the decoded image ([4, height, width] channel-major float, the
// VAE's own semantic output range, clamped to [-1,1]); latent is the final denoised latent
// ([tokens, image_in] token-major BF16 bits) for reporting or re-runs.
struct GeneratedImage {
    std::uint32_t height = 0, width = 0;
    std::vector<float> channels;
    std::vector<std::uint16_t> latent;
};

// #75 reference/edit image: one source image encoded to normalized reference latents with the SAME
// VAE weights the job decodes with. pixels is the caller-prepared RGBA plane in the VAE's own input
// range: channel-major [4, height_px, width_px] float in [-1,1] where the alpha channel is carried
// RAW as the fourth input channel (the reference pipeline's VaeImageProcessor path never composites
// it; the white composite belongs to the Qwen3-VL text side only); height/width are multiples of 16
// (the VAE's total spatial factor). The output is the
// DiT-ready reference block: [height_lat*width_lat, image_in] token-major BF16 bits with
// height_lat = height_px/16, already mode-sampled and channel-normalized with latents_mean/std --
// exactly the tensor prepended to the joint sequence (ref latents stay constant across the steps).
struct ReferenceEncodeInputs {
    std::vector<float> pixels;       // [4, height_px, width_px] channel-major float in [-1,1]
    std::vector<float> latents_mean; // [image_in]
    std::vector<float> latents_std;  // [image_in]
    std::uint32_t height_px = 0, width_px = 0;
};

struct EncodedReference {
    std::uint32_t height_lat = 0, width_lat = 0;
    std::vector<std::uint16_t> latents; // [height_lat*width_lat, image_in] token-major BF16 bits
};

// Runs the VAE encode on `stream`; every device allocation is owned here and freed before the
// return, success or throw alike. Bad inputs or a CUDA failure throw std::runtime_error.
[[nodiscard]] EncodedReference encode_reference_image(artifact::Reader& reader, const VaeWeights& vae,
                                                      const ReferenceEncodeInputs& in, cudaStream_t stream);

// #17 release gate: this module's device-memory ledger. Buf is its ONLY device allocator, so
// image_device_bytes_live() is the exact count of image-owned device bytes alive right now -- it
// MUST be 0 between jobs, on the success and the throw path alike. image_device_bytes_peak() is the
// high-water mark since the last reset (reset_image_device_peak() re-baselines it to the live count).
[[nodiscard]] std::uint64_t image_device_bytes_live() noexcept;
[[nodiscard]] std::uint64_t image_device_bytes_peak() noexcept;
void reset_image_device_peak() noexcept;

// Runs one complete image job on `stream` inside the yielded VRAM. Every device allocation is
// owned here and freed before the return, on success and on failure alike (the caller's interlude
// restore then returns the LLM's pools). Bad inputs or a CUDA failure throw std::runtime_error.
// The 32 blocks' weights are read from the artifact once per job into a host cache (peak ~ one
// block's roles x layers) and streamed H2D per block per step, so the 40-step loop never re-reads
// them from disk and the device working set peaks at ~one block.
// What the serving caller passes to observe a running job (the models layer must not know serve
// types): one call per completed unit -- each denoise step, then the VAE decode -- and a line sink
// for the job's stage telemetry. `step` returns false when the caller's watchdog has flagged the
// job; the loop then throws so the interlude unwinds and restores the LLM. Both default to
// nothing: the certified route is bit-identical when they are absent.
class ImageJobProgressView {
public:
    virtual ~ImageJobProgressView() = default;
    [[nodiscard]] virtual bool step(std::size_t units_done, std::size_t units_total) = 0;
};

// The Host bytes one job page-locks for its DiT block cache: every block's nine roles, held for the
// whole job. Summed from the stored geometry the cache itself allocates from, so a caller reserving
// room for the cache (serve's RAM-ledger reservation) cannot drift from what the job really takes.
[[nodiscard]] std::uint64_t host_block_cache_bytes(const artifact::Reader& reader,
                                                  const DiTWeights& dit);

[[nodiscard]] GeneratedImage run_image_job(artifact::Reader& reader, const DiTWeights& dit,
                                           const VaeWeights& vae, const ImageJobInputs& inputs,
                                           cudaStream_t stream,
                                           ImageJobProgressView* progress = nullptr,
                                           std::function<void(const std::string&)> log_line = {});

} // namespace infernix::models::qwen_image
