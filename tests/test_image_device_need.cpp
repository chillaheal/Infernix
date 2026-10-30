// The image job's computed device-memory need: the model layer's own account of what a job will hold
// at its highest point. The serving layer refuses on this number before the LLM yields any VRAM, so
// the cases that matter are the ones an area-scaled guess got wrong: a reference image and true CFG
// must raise the need, and the model must never predict below the certified route's measured peak.

#include "models/qwen_image/image_device_need.h"

#include <cstdio>

namespace qi = infernix::models::qwen_image;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

qi::DiTDims test_dims() {
    qi::DiTDims d;
    d.layers       = 32;
    d.heads        = 32;
    d.head_dim     = 128;
    d.hidden       = 4096;
    d.intermediate = 12288;
    d.image_in     = 64;
    d.image_out    = 64;
    d.context_in   = 4096;
    return d;
}

qi::VisionLaneDims test_vision() {
    // The shipped vision_config (image-edit/NOTES-M0.md, read off the artifact's component config):
    // depth 27, hidden 1152, out_hidden 4096 == the text tower's hidden, patch 16, temporal 2,
    // spatial_merge 2. The intermediate width is the ViT MLP's; the ledger only ever multiplies it by
    // the patch count, so any plausible width exercises the same code path here.
    qi::VisionLaneDims v;
    v.hidden       = 1152;
    v.intermediate = 4304;
    v.head_dim     = 72;
    v.patch        = 16;
    v.temporal     = 2;
    v.merge        = 2;
    v.out_hidden   = 4096;
    return v;
}

qi::ImageDeviceNeed need_struct(std::uint32_t side, std::uint32_t text, std::uint32_t neg,
                                std::uint32_t ref, std::uint32_t vision_tokens, bool with_vision) {
    qi::ImageDeviceNeedRequest req;
    req.geometry        = qi::ImageGeometry::derive(side, side);
    req.dims            = test_dims();
    req.text_tokens     = text;
    req.negative_tokens = neg;
    req.ref_tokens      = ref;
    req.vision_tokens   = vision_tokens;
    if (with_vision) { req.vision = test_vision(); }
    // The shipped route's resident-slot count is the header's single constant (bundle B fas 2).
    return qi::image_device_need(req);
}

std::uint64_t need_of(std::uint32_t side, std::uint32_t text, std::uint32_t neg, std::uint32_t ref) {
    return need_struct(side, text, neg, ref, 0, false).total();
}

// The certified text-to-image admission in BYTES -- this test's own report line prints the number on
// the tree where the edit route's fields were added (both zero for this route). The lock is the
// point: adding a vision lane must not move one byte of the route that is already certified. If this
// number ever changes, the text-to-image admission regressed; explain the diff, do not edit this.
constexpr std::uint64_t kGoldenT2I768Bytes = 6214898688ULL; // 5926 MiB, this test's own report line

} // namespace

int main() {
    // The review's failure case: the same target size with a reference must cost more than without.
    check(need_of(768, 200, 0, 0) < need_of(768, 200, 0, 4096), "a reference raises the need");
    // True CFG sizes every sequence buffer by the longer joint sequence.
    check(need_of(768, 200, 0, 0) < need_of(768, 200, 512, 0), "true CFG raises the need");
    // Whichever pass is longer drives s_max, so the need does not depend on which one is negative.
    check(need_of(512, 64, 4096, 0) == need_of(512, 4096, 64, 0), "s_max is the longer joint sequence");
    // Area dominates: 2048^2 is far past 768^2, and past the same size plus one reference.
    check(need_of(768, 200, 0, 0) < need_of(2048, 200, 0, 0), "area raises the need");
    check(need_of(1024, 200, 0, 4096) < need_of(2048, 200, 0, 0), "area beats one reference");
    // The 8dc9c63 calibration's floor for the shipped (two-resident-block, bundle B fas 2) route:
    // the allowance plus the 512 MiB interlude floor. The old 6183 MiB floor measured the serialized
    // one-block route's high-water peak and retired with that route; the live guarantee-0 ledger
    // cert (f77e16b) is the oracle for the current route's peak. This check keeps the shipped route
    // from ever predicting below the calibrated floor.
    check(need_of(512, 200, 0, 0) >= qi::kImageWeightResidentBytes + 512ULL * qi::kImageMiB,
          "covers the calibrated 512 floor");
    // The allowance is the conservative part; the sequence terms must still be the ones that grow.
    check(need_of(2048, 200, 0, 0) > 3ULL * 6183ULL * 1024ULL * 1024ULL,
          "the growing terms carry the large sizes");

    qi::ImageDeviceNeedRequest req;
    req.geometry        = qi::ImageGeometry::derive(768, 768);
    req.dims            = test_dims();
    req.text_tokens     = 200;
    req.blocks_resident = 1;
    const auto need     = qi::image_device_need(req);
    check(need.total() == need.dit_activations + need.latent_and_velocity + need.resident_weights +
                              need.vae_peak + need.rope_and_tables + need.vision_peak,
          "total is the sum of its terms");
    check(need.vae_peak > 0 && need.dit_activations > 0 && need.resident_weights > 0, "every term is set");

    // ---- the edit route's ledger (M7) ----
    // The certified text-to-image route must not move a byte when the edit fields are absent.
    check(kGoldenT2I768Bytes != 0ULL && need_of(768, 32, 0, 0) == kGoldenT2I768Bytes,
          "the certified 768x768 text-to-image admission is byte-identical");
    {
        // A 768^2 target with a reference on the 1024^2 area cap: ref_tokens 4096 == 4 * 1024 vision
        // tokens (reference_image.h vision_token_count), and the rendered prompt carries those 1024
        // rows inside its kept length -- so the joint sequence is 32 + 4096 + 2304, NOT 1056 + 4096.
        constexpr std::uint32_t kRef = 4096U, kVis = 1024U, kNonPad = 32U;
        const auto declared = need_struct(768, kNonPad + kVis, 0, kRef, kVis, true);
        const auto undecl = need_struct(768, kNonPad + kVis, 0, kRef, 0, true);
        const auto honest = need_struct(768, kNonPad, 0, kRef, 0, true);
        check(declared.dit_activations < undecl.dit_activations &&
                  declared.rope_and_tables < undecl.rope_and_tables,
              "declaring the pad rows takes them out of the joint sequence");
        check(declared.dit_activations == honest.dit_activations &&
                  declared.rope_and_tables == honest.rope_and_tables,
              "the image pad rows are counted once, in ref_tokens");
        // The tower's output really does carry the pad rows, so the text-side stages must still be
        // sized by the full kept length. Only the JOINT sequence deduplicates them.
        check(declared.latent_and_velocity == undecl.latent_and_velocity,
              "the text stages keep the full kept length");
        check(need_struct(768, kNonPad, 0, 0, 0, false).vision_peak == 0,
              "no vision lane, no vision bytes");
        check(declared.vision_peak > 100ULL * qi::kImageMiB &&
                  declared.vision_peak < 256ULL * qi::kImageMiB,
              "the ViT lane reserves its live set (172 MiB at the shipped config: eleven "
              "[hidden,patches] planes 104 MiB, d_up 33 MiB, d_patches 12 MiB, the merger rows 8 MiB, "
              "the three index planes 0.1 MiB, one block's roles 19 MiB -- vision_tower.cu:197-219 plus "
              "the streamed block, over the worst legal reference of 4096 patches)");
        // The vision term is purely additive: the same request with NO lane, with the two joint-shaped
        // terms shrunk by exactly what declaring the pad rows saved, plus the vision term, reproduces
        // total() byte for byte. Nothing else in the ledger moved.
        const auto nolid = need_struct(768, kNonPad + kVis, 0, kRef, 0, false);
        check(declared.total() ==
                  nolid.total() - (nolid.dit_activations - declared.dit_activations) -
                      (nolid.rope_and_tables - declared.rope_and_tables) + declared.vision_peak,
              "the vision term is summed like vae_peak, never maxed");
        // The row the plan asks for in MiB: the same 768^2 target as the certified route plus a
        // reference and the ViT lane. Expected ~1.0-1.2 GiB above the text-only need (the 4096 extra
        // latent tokens at ~0.2 MiB per joint row plus the ~130 MiB vision lane).
        const auto plain = need_struct(768, kNonPad, 0, 0, 0, false);
        const std::uint64_t delta = declared.total() > plain.total()
                                        ? declared.total() - plain.total()
                                        : 0ULL;
        check(delta > declared.vision_peak && delta > 512ULL * qi::kImageMiB,
              "a reference plus the ViT lane raises the need by its structural terms");
        std::printf("edit row 768x768 text32 ref4096: %llu MiB (vision lane %llu MiB, delta %llu MiB)\n",
                    static_cast<unsigned long long>(declared.total() / qi::kImageMiB),
                    static_cast<unsigned long long>(declared.vision_peak / qi::kImageMiB),
                    static_cast<unsigned long long>(delta / qi::kImageMiB));
    }

    // Printed before the verdict so a broken golden constant can be read off a failing run.
    std::printf("golden 768x768 text32 ref0: %llu bytes\n",
                static_cast<unsigned long long>(need_of(768, 32, 0, 0)));

    if (failures != 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    const auto report = [](const char* name, std::uint64_t side, std::uint32_t neg, std::uint32_t ref) {
        std::printf("%s %ux%u neg=%u ref=%u: %llu MiB\n", name, static_cast<unsigned>(side),
                    static_cast<unsigned>(side), static_cast<unsigned>(neg), static_cast<unsigned>(ref),
                    static_cast<unsigned long long>(need_of(side, 200, neg, ref) / (1024ULL * 1024ULL)));
    };
    report("need", 512, 0, 0);
    report("need", 768, 0, 0);
    report("need", 768, 0, 4096);
    report("need", 768, 512, 0);
    report("need", 1024, 0, 0);
    report("need", 2048, 0, 0);
    std::printf("image_device_need: ok\n");
    return 0;
}
