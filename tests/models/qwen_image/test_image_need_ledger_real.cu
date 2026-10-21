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
        (void)std::move(binder).finish();

        const std::vector<Case> cases = {
            {512, 0, 0, "the certified route"},
            {768, 0, 0, "the chat default"},
            {768, 128, 0, "the review's case: same target plus a reference"},
            {768, 0, 24, "true CFG (a second, differently sized pass)"},
            {1024, 0, 0, "one megapixel"},
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
            }
            if (c.neg_tokens != 0) {
                inputs.negative_prompt_embeds = zero_bits(static_cast<std::size_t>(c.neg_tokens) *
                                                         dit.dims.hidden);
                inputs.true_cfg_scale         = 3.5F;
            }

            ImageDeviceNeedRequest req;
            req.geometry        = geom;
            req.dims            = dit.dims;
            req.text_tokens     = kTextTokens;
            req.negative_tokens = c.neg_tokens;
            req.ref_tokens      = static_cast<std::uint32_t>(inputs.ref_latents.size() / dit.dims.image_in);
            req.blocks_resident = 1;
            const std::uint64_t predicted = image_device_need(req).total();

            reset_image_device_peak();
            (void)run_image_job(reader, dit, vae, inputs, stream);
            const std::uint64_t measured = image_device_bytes_peak();
            const std::uint64_t mib      = 1024ULL * 1024ULL;
            std::cout << c.what << ": " << c.side << "x" << c.side << " ref=" << c.ref_side
                      << " neg=" << c.neg_tokens << " predicted " << predicted / mib
                      << " MiB, measured peak " << measured / mib << " MiB\n";
            if (measured > predicted) {
                std::cout << "FAIL: the measured peak exceeds the predicted need by "
                          << (measured - predicted) / mib << " MiB\n";
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
