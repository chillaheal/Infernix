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

std::uint64_t need_of(std::uint32_t side, std::uint32_t text, std::uint32_t neg, std::uint32_t ref) {
    qi::ImageDeviceNeedRequest req;
    req.geometry        = qi::ImageGeometry::derive(side, side);
    req.dims            = test_dims();
    req.text_tokens     = text;
    req.negative_tokens = neg;
    req.ref_tokens      = ref;
    // The shipped route's resident-slot count is the header's single constant (bundle B fas 2).
    return qi::image_device_need(req).total();
}

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
                              need.vae_peak + need.rope_and_tables,
          "total is the sum of its terms");
    check(need.vae_peak > 0 && need.dit_activations > 0 && need.resident_weights > 0, "every term is set");

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
