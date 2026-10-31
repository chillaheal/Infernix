// Qualification of image_device_need() against the pipeline's own device-memory ledger (#17): for a
// matrix of jobs, the module's high-water mark must never exceed the number the serving layer refused
// on. The zero-filled inputs make the arithmetic trivial and the allocation identical -- this test
// measures the inventory, not the picture. It also re-asserts the module's ledger contract: zero live
// image bytes before and after every job, on the throw path alike.
//
// Skips (77) without INFERNIX_TEST_ARTIFACT or when the artifact is absent; returns 2 with no CUDA
// device. It runs the real pipeline on the real GPU, so it must not overlap a live serve.

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "models/qwen_image/image_device_need.h"
#include "models/qwen_image/image_job.h"
#include "models/qwen_image/image_pipeline.h"
#include "models/qwen_image/image_weights.h"
#include "models/qwen_image/vision_tower.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::artifact;
using namespace infernix::models::qwen_image;

constexpr std::uint32_t kTextTokens = 8;
constexpr std::uint32_t kSteps      = 2; // the peak is per step: fewer steps measures the same high-water

struct Case {
    std::uint32_t side;
    std::uint32_t ref_side; // 0 = no reference image
    std::uint32_t neg_tokens; // 0 = single forward (no true CFG)
    const char*   what;
    // Run the ViT encode inside the SAME peak window as the DiT loop and reserve for it. The interlude
    // holds both at once -- the encode that produces the conditioning runs on the yielded VRAM -- so
    // the edit route's ledger row has to predict the ViT's live set and this row has to measure it.
    bool          vit = false;
};

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::vector<std::uint16_t> zero_bits(std::size_t elements) { return std::vector<std::uint16_t>(elements, 0U); }

} // namespace

int main() {
    try {
        const char* env = std::getenv("INFERNIX_TEST_ARTIFACT");
        if (env == nullptr || std::string_view(env).empty()) {
            std::cout << "INFERNIX_TEST_ARTIFACT unset; skipping the need/ledger qualification\n";
            return 77;
        }
        const std::filesystem::path path(env);
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            std::cout << "artifact absent; skipping the need/ledger qualification: " << env << "\n";
            return 77;
        }
        if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
            std::cout << "no CUDA device; the qualification is runnable\n";
            return 2;
        }

        Reader        reader(path);
        Binder        binder(reader);
        DiTWeights    dit = bind_dit(binder);
        VaeWeights    vae = bind_image_vae(binder);
        const VaeDenorm denorm = read_vae_denorm(binder);
        // Bound for the edit row below: the same artifact, the same streamed-only plan. Its dims are
        // what the ledger's vision term is derived from, so the row reserves for the ViT the artifact
        // actually ships rather than for a number written into the header.
        VisionWeights vision = bind_image_vision(binder);
        (void)std::move(binder).finish();

        const auto lane_of = [](const VisionDims& v) {
            VisionLaneDims out;
            out.hidden       = v.hidden;
            out.intermediate = v.intermediate;
            out.head_dim     = v.head_dim;
            out.patch        = v.patch;
            out.temporal     = v.temporal;
            out.merge        = v.merge;
            out.out_hidden   = v.out_hidden;
            return out;
        };

        const std::vector<Case> cases = {
            {512, 0, 0, "the certified route"},
            {768, 0, 0, "the chat default"},
            {768, 128, 0, "the review's case: same target plus a reference"},
            {768, 0, 24, "true CFG (a second, differently sized pass)"},
            {1024, 0, 0, "one megapixel"},
            // M7's own row: a reference on the 1024^2 area cap (4096 latent tokens == 4 * 1024 vision
            // tokens) whose pad rows are the whole prompt, with the ViT encode measured in the same
            // window. This is the edit route the serving layer admits on this number.
            {768, 1024, 0, "the edit route: a 1024^2 reference, the ViT lane and the joint sequence", true},
        };

        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess) { throw std::runtime_error("no image stream"); }

        int failures = 0;
        require(image_device_bytes_live() == 0, "the ledger holds bytes before any job");
        for (const auto& c : cases) {
            const ImageGeometry geom = ImageGeometry::derive(c.side, c.side);
            ImageJobInputs      inputs;
            inputs.prompt_embeds = zero_bits(static_cast<std::size_t>(kTextTokens) * dit.dims.hidden);
            inputs.latents_mean  = denorm.mean;
            inputs.latents_std   = denorm.std;
            inputs.height_px     = geom.height;
            inputs.width_px      = geom.width;
            inputs.init_latent   = zero_bits(static_cast<std::size_t>(geom.tokens) * dit.dims.image_in);
            const Schedule sched = ImageJob{geom.height, geom.width, kSteps}.schedule(SchedulerConfig{});
            inputs.timesteps = sched.timesteps;
            inputs.sigmas    = sched.sigmas;
            if (c.ref_side != 0) {
                const ImageGeometry ref = ImageGeometry::derive(c.ref_side, c.ref_side);
                inputs.ref_grids.emplace_back(ref.hl, ref.wl);
                inputs.ref_latents = zero_bits(static_cast<std::size_t>(ref.tokens) * dit.dims.image_in);
                // A reference makes this the edit route, whose joint sequence is built from the prompt's
                // image-pad mask: one set bit per 2x2 group of reference latents, in the kept prompt rows
                // (the job appends the target's own slots itself). The mask carries exactly one byte per
                // kept row (run_image_job throws otherwise), so the prompt grows by the pad rows rather
                // than the mask shrinking to them: kTextTokens text rows, then one set bit per 2x2 group.
                const std::size_t pads = static_cast<std::size_t>(ref.tokens) / 4ULL;
                inputs.prompt_embeds = zero_bits((static_cast<std::size_t>(kTextTokens) + pads) *
                                                 dit.dims.hidden);
                inputs.prompt_image_pad_mask.assign(static_cast<std::size_t>(kTextTokens), 0U);
                inputs.prompt_image_pad_mask.insert(inputs.prompt_image_pad_mask.end(), pads, 1U);
            }
            if (c.neg_tokens != 0) {
                inputs.negative_prompt_embeds = zero_bits(static_cast<std::size_t>(c.neg_tokens) *
                                                         dit.dims.hidden);
                inputs.true_cfg_scale         = 3.5F;
            }

            ImageDeviceNeedRequest req;
            req.geometry        = geom;
            req.dims            = dit.dims;
            // The prompt's kept rows, read back from what the job is actually handed: a reference grew
            // the prompt by its pad rows above, and the ledger must size the same sequence the job builds.
            req.text_tokens     = static_cast<std::uint32_t>(inputs.prompt_embeds.size() /
                                                             static_cast<std::size_t>(dit.dims.hidden));
            req.negative_tokens = c.neg_tokens;
            req.ref_tokens      = static_cast<std::uint32_t>(inputs.ref_latents.size() / dit.dims.image_in);
            req.blocks_resident = kImageResidentBlocks; // same symbol the serving layer and the
                             // pipeline's CopyLane assert; the allowance was calibrated at this value
            if (c.vit) {
                // The reference's vision tokens are the prompt here: 4096 patches merge to 1024 rows,
                // all of them <|image_pad|>, so the joint sequence is 0 text + 4096 reference + target.
                // Declaring them is the point -- the same 1024 rows must not also be counted as text.
                const std::size_t merged = 64ULL * 64ULL /
                                           (static_cast<std::size_t>(vision.dims.merge) * vision.dims.merge);
                inputs.prompt_embeds         = zero_bits(merged * dit.dims.hidden);
                inputs.prompt_image_pad_mask.assign(merged, 1U);
                req.text_tokens              = static_cast<std::uint32_t>(merged);
                req.vision_tokens            = static_cast<std::uint32_t>(merged);
                req.vision                   = lane_of(vision.dims);
            }
            const std::uint64_t predicted = image_device_need(req).total();

            reset_image_device_peak();
            if (c.vit) {
                // The encode runs on the yielded VRAM, inside the same window as the loop below, so the
                // high-water mark this row compares against the ledger includes the ViT's live set.
                const VisionItemGeometry grid{1U, 64U, 64U};
                const std::size_t patches = 64ULL * 64ULL;
                (void)run_vision_tower(reader, vision,
                                       zero_bits(patches * vision.dims.patch_width), grid);
            }
            (void)run_image_job(reader, dit, vae, inputs, stream);
            const std::uint64_t measured = image_device_bytes_peak();
            const std::uint64_t mib      = 1024ULL * 1024ULL;
            std::cout << c.what << ": " << c.side << "x" << c.side << " ref=" << c.ref_side
                      << " neg=" << c.neg_tokens << (c.vit ? " vit=yes" : "")
                      << " predicted " << predicted / mib
                      << " MiB (vision lane " << image_device_need(req).vision_peak / mib
                      << " MiB), measured peak " << measured / mib << " MiB\n";
            if (measured > predicted) {
                std::cout << "FAIL: the measured peak exceeds the predicted need by "
                          << (measured - predicted) / mib << " MiB\n";
                ++failures;
            }
            // Reservation floor. The serving layer keeps 512 MiB between the predicted need and the
            // device it yields (kInterludeMarginBytes), so a row that clears the real peak by less than
            // that means kImageWeightResidentBytes was cut below the footprint plus its reservation. This
            // is the pair to "predicted >= measured": it catches the next change that lowers the constant
            // too far, exactly the mistake the 3072 MiB tightening must not be re-lowered into.
            const std::uint64_t kFloor = 512ULL * mib;
            if (predicted >= measured && predicted < measured + kFloor) {
                std::cout << "FAIL: " << c.what << " clears the measured peak by only "
                          << (predicted - measured) / mib << " MiB, under the " << kFloor / mib
                          << " MiB reservation floor\n";
                ++failures;
            }
            if (image_device_bytes_live() != 0) {
                std::cout << "FAIL: the ledger holds " << image_device_bytes_live()
                          << " bytes after the job\n";
                ++failures;
            }
        }
        cudaStreamDestroy(stream);
        if (failures != 0) {
            std::cout << failures << " case(s) failed; lower kImageWeightResidentBytes or widen the "
                         "inventory in image_device_need.h\n";
            return 1;
        }
        std::cout << "the computed need covers every measured peak\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "image need qualification failed: " << error.what() << "\n";
        return 1;
    }
}
