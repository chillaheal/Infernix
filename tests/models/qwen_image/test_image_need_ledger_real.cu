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
    std::uint32_t side_h = 0; // 0 = square (== side); else the height, with `side` as the width
    std::uint32_t ref_side; // 0 = no reference image
    std::uint32_t neg_tokens; // 0 = single forward (no true CFG)
    const char*   what;
    // Run the ViT encode inside the SAME peak window as the DiT loop and reserve for it. The interlude
    // holds both at once -- the encode that produces the conditioning runs on the yielded VRAM -- so
    // the edit route's ledger row has to predict the ViT's live set and this row has to measure it.
    bool          vit = false;
    // Certified anchors, in MiB. A non-zero entry must reproduce exactly: the predicted need (the number
    // the serving layer refuses on) and the measured high-water (what the pipeline really allocated).
    //
    // Where BOTH are set, the row is one of the four the plan certifies -- the numbers at
    // docs/superpowers/plans/2026-11-14-image-fail-closed.md:688,781. Both sanctioned runs whose stdout is on
    // record reproduce them exactly (the fix-round-1 re-certification and the fix-round-2 restored run, in
    // build-windows\task7_logs\); the first certified run's table was never captured, as disclosed in the
    // task report, so the claim stops at the two runs that carry evidence.
    //
    // The first five rows pin the MEASURED side only. Their high-water is the 2026-10-23 certification
    // recorded in .agent/STATE.md:861-862 (1845/2746/2760/2751/4007) and it still reproduces exactly on
    // this build, so the footprint is frozen there. Their predicted side was recorded by the same older
    // run (4570/5921/5935/5925/7812) and the current inventory computes 432-1728 MiB less for those five
    // canvases; that number was never in the plan's certified table, so pinning it would freeze a stale
    // prediction rather than a certified one.
    //
    // 1280/1536/1792 pin the MEASURED side too, and their source is different in kind: no earlier
    // certification names them (.agent/STATE.md has no rows for those canvases, and the plan's table does
    // not), so 5628/7609/9951 MiB are the peaks THIS suite printed on its own sanctioned runs against the
    // shipped artifact, first recorded here. That makes them a freeze rather than a reproduction -- they
    // cannot catch a footprint that was already wrong when they were read, only one that drifts from here
    // -- which is why the predicted side of every row the plan does not certify is left to the two bounds
    // below (measured must not outgrow the inventory, and the inventory must clear it by the serving
    // layer's own 512 MiB reservation). Every row pins its measured high-water, so a zero now marks only an
    // unpinned PREDICTED side.
    std::uint64_t certified_predicted_mib = 0;
    std::uint64_t certified_measured_mib  = 0;
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
        // The denormalization pair is read against the topology the binding just verified, so its length
        // is checked against the artifact's own declared z_dim rather than adopted at whatever length the
        // blob happens to carry.
        const VaeDenorm denorm = read_vae_denorm(binder, vae.topology);
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
            // The measured high-water of these five rows is the 2026-10-23 certification in
            // .agent/STATE.md:861-862, copied verbatim. The predicted side stays unpinned there for the
            // reason above; the floor gate (measured + 512 MiB) still bounds it from below.
            {512, 0, 0, 0, "the certified route", false, 0, 1845},
            {768, 0, 0, 0, "the chat default", false, 0, 2746},
            {768, 0, 128, 0, "the review's case: same target plus a reference", false, 0, 2760},
            {768, 0, 0, 24, "true CFG (a second, differently sized pass)", false, 0, 2751},
            {1024, 0, 0, 0, "one megapixel", false, 0, 4007},
            // M7's own row: a reference on the 1024^2 area cap (4096 latent tokens == 4 * 1024 vision
            // tokens) whose pad rows are the whole prompt, with the ViT encode measured in the same
            // window. This is the edit route the serving layer admits on this number.
            // The anchors are the certified row for row: prediction 6010 MiB against a measured
            // 3634 MiB, the numbers the pre-scheduler artifact was certified with.
            {768, 0, 1024, 0, "the edit route: a 1024^2 reference, the ViT lane and the joint sequence", true,
             6010, 3634},
            // The canvas-scaled sweep: every certified square the serving layer can be asked for, so
            // the prediction-vs-measured row exists for the exact canvases the walk and the net decide
            // on. Pure geometry (no reference, no CFG, no vision): the four canvas-scaled terms are all
            // that change across these rows, which is what makes the comparison attributable to them.
            {1024, 1152, 0, 0, "the trained edit canvas 1024x1152", false, 6409, 4367},
            {1280, 0, 0, 0, "certified 1280", false, 0, 5628},
            {1536, 0, 0, 0, "certified 1536", false, 0, 7609},
            {1792, 0, 0, 0, "certified 1792", false, 0, 9951},
            {2016, 0, 0, 0, "certified 2016", false, 13548, 12295},
            {2048, 0, 0, 0, "certified 2048", false, 13869, 12653},
        };

        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess) { throw std::runtime_error("no image stream"); }

        int failures = 0;
        require(image_device_bytes_live() == 0, "the ledger holds bytes before any job");
        for (const auto& c : cases) {
            const ImageGeometry geom =
                ImageGeometry::derive(c.side_h != 0 ? c.side_h : c.side, c.side);
            ImageJobInputs      inputs;
            inputs.prompt_embeds = zero_bits(static_cast<std::size_t>(kTextTokens) * dit.dims.hidden);
            inputs.latents_mean  = denorm.mean;
            inputs.latents_std   = denorm.std;
            inputs.height_px     = geom.height;
            inputs.width_px      = geom.width;
            inputs.init_latent   = zero_bits(static_cast<std::size_t>(geom.tokens) * dit.dims.image_in);
            // The artifact's own constants, bound with the weights above: the need this row predicts is
            // the need of a job drawn on the schedule the artifact declares.
            const Schedule sched = ImageJob{geom.height, geom.width, kSteps}.schedule(dit.scheduler);
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

            // The topology is the one THIS artifact's binding produced and verified against every stored
            // decoder weight -- the same object the job decodes with two statements below -- so the
            // prediction this row compares against the measured peak is priced from the decoder that
            // actually ran, not from a table this file could have kept stale.
            ImageDeviceNeedRequest req{
                .geometry        = geom,
                .dims            = dit.dims,
                // The prompt's kept rows, read back from what the job is actually handed: a reference
                // grew the prompt by its pad rows above, and the ledger must size the same sequence the
                // job builds.
                .text_tokens     = static_cast<std::uint32_t>(inputs.prompt_embeds.size() /
                                                             static_cast<std::size_t>(dit.dims.hidden)),
                .negative_tokens = c.neg_tokens,
                .ref_tokens      = static_cast<std::uint32_t>(inputs.ref_latents.size() / dit.dims.image_in),
                .blocks_resident = kImageResidentBlocks, // same symbol the serving layer and the
                             // pipeline's CopyLane assert; the calibration was measured at this value
                .vae             = vae.topology,
            };
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
            const ImageDeviceNeed need = image_device_need(req);
            const std::uint64_t   predicted = need.total();

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
            const std::uint32_t print_h = c.side_h != 0 ? c.side_h : c.side;
            std::cout << c.what << ": " << c.side << "x" << print_h << " ref=" << c.ref_side
                      << " neg=" << c.neg_tokens << (c.vit ? " vit=yes" : "")
                      << " predicted " << predicted / mib
                      << " MiB [dit " << need.dit_activations / mib
                      << " lat " << need.latent_and_velocity / mib
                      << " vae " << need.vae_peak / mib
                      << " rope " << need.rope_and_tables / mib
                      << " res " << need.resident_weights / mib
                      << " vision " << need.vision_peak / mib
                      << "], measured peak " << measured / mib << " MiB\n";
            // The certified equality, enforced here rather than eyeballed in a transcript. The two
            // bounds below cannot catch a shrinking prediction: "measured > predicted" only fails when
            // the footprint OUTGROWS the inventory, and the reservation floor only fails when the
            // slack drops under 512 MiB -- so a row could lose 511 MiB of prediction (an inventory term
            // dropped from image_device_need.h, the resident calibration re-lowered) and the suite stay
            // green. Nor is the measured side ever compared to a constant. These pins are what make the
            // row identical to the certification instead of merely compatible with it.
            if (c.certified_predicted_mib != 0 && predicted / mib != c.certified_predicted_mib) {
                std::cout << "FAIL: " << c.what << " predicts " << predicted / mib
                          << " MiB, the certified number is " << c.certified_predicted_mib << " MiB\n";
                ++failures;
            }
            if (c.certified_measured_mib != 0 && measured / mib != c.certified_measured_mib) {
                std::cout << "FAIL: " << c.what << " measured " << measured / mib
                          << " MiB, the certified high-water is " << c.certified_measured_mib << " MiB\n";
                ++failures;
            }
            if (measured > predicted) {
                std::cout << "FAIL: the measured peak exceeds the predicted need by "
                          << (measured - predicted) / mib << " MiB\n";
                ++failures;
            }
            // Reservation floor. The serving layer keeps 512 MiB between the predicted need and the
            // device it yields (kInterludeMarginBytes), so a row that clears the real peak by less than
            // that means the topology's resident calibration was cut below the footprint plus its
            // reservation. This is the pair to "predicted >= measured": it catches the next change that
            // lowers the constant too far, exactly the mistake the 3072 MiB tightening must not be
            // re-lowered into.
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
            std::cout << failures << " case(s) failed; lower the resident calibration "
                         "(kQwenImageWeightResidentBytes, which the binding stamps onto the topology) or "
                         "widen the inventory in image_device_need.h\n";
            return 1;
        }
        std::cout << "the computed need covers every measured peak\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "image need qualification failed: " << error.what() << "\n";
        return 1;
    }
}
