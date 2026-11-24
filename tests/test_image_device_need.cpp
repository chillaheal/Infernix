// The image job's computed device-memory need: the model layer's own account of what a job will hold
// at its highest point. The serving layer refuses on this number before the LLM yields any VRAM, so
// the cases that matter are the ones an area-scaled guess got wrong: a reference image and true CFG
// must raise the need, and the model must never predict below the certified route's measured peak.
//
// The decoder half of the ledger has no constants of its own any more: the request carries the
// artifact's own VaeTopology, and this suite gets it the only way a guard may -- from the committed copy
// of the checkpoint's vae/config.json through the production reader, the same object the CUDA decode
// graph runs on. A hand-written table here would keep passing while the shipped artifact declared a
// different decoder, which is the failure this seam closes.

#include "models/qwen_image/image_device_need.h"
#include "models/qwen_image/vae_topology_fixture.h"

// The reference route the ViT charge has to match is src/serve's, so the bound is proven against the
// code that produces the planes rather than against a copy of its arithmetic: the exported snap, the
// exported prepare_reference_planes, and its base64 helper. See the block that uses them below for why
// a reimplementation would have kept the 4096 alive.
#include "media/encode/png_encode.h"
#include "serve/image_generation_service.h" // srv::base64_encode_bytes
#include "serve/openai_images.h"            // srv::snap_to_32
#include "serve/reference_image.h"          // srv::prepare_reference_planes

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qi = infernix::models::qwen_image;
namespace qt = infernix::tests::qwen_image;

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

// The decoder geometry every row here is priced from, read ONCE and shared by all of them the way one
// admission hands one topology to every step of its walk: through the production reader over the
// committed copy of the checkpoint's own vae/config.json, with the resident calibration stamped the way
// bind_image_vae stamps it (tests/models/qwen_image/vae_topology_fixture.h). It is a reference on the
// request, so nothing below can price a job against a copy that drifted from the artifact's.
const qi::VaeTopology& shipped_topology() {
    static const qi::VaeTopology topology = qt::qwen_image_2_1_topology();
    return topology;
}

qi::ImageDeviceNeed need_struct(std::uint32_t side, std::uint32_t text, std::uint32_t neg,
                                std::uint32_t ref, std::uint32_t vision_tokens, bool with_vision) {
    // Designated aggregate initialization, in declaration order: the request's last member is a
    // reference with no default, so a field-by-field construction that forgot it would not even be a
    // legal way to ask the question -- every row must say which artifact's decoder it is pricing.
    const qi::ImageDeviceNeedRequest req{
        .geometry        = qi::ImageGeometry::derive(side, side),
        .dims            = test_dims(),
        .text_tokens     = text,
        .negative_tokens = neg,
        .ref_tokens      = ref,
        // The shipped route's resident-slot count is the header's single constant (bundle B fas 2).
        .blocks_resident = qi::kImageResidentBlocks,
        .vision_tokens   = vision_tokens,
        .vision          = with_vision ? test_vision() : qi::VisionLaneDims{},
        .vae             = shipped_topology(),
    };
    return qi::image_device_need(req);
}

std::uint64_t need_of(std::uint32_t side, std::uint32_t text, std::uint32_t neg, std::uint32_t ref) {
    return need_struct(side, text, neg, ref, 0, false).total();
}

// The certified text-to-image admission in BYTES -- this test's own report line prints the number on
// the tree where the edit route's fields were added (both zero for this route). The lock is the
// point: adding a vision lane must not move one byte of the route that is already certified. If this
// number ever changes, the text-to-image admission regressed; explain the diff, do not edit this.
//
// The value below is the one THIS commit's tree prints, and it is not the number this line carried
// before Task 5 touched the ledger. It moved under 8986a6e3 ("serve: charge the VAE decoder what it
// actually holds"), which replaced the decoder's max(channels_in, channels_out) charge with the
// stage's true output width and thereby lowered the 768^2 row by exactly the 972 MiB the {576,288}
// stage used to double-count -- but that commit never re-certified this lock, so the committed
// constant still priced the over-charge. Verified, not inferred: with Task 5's changes stashed away
// (git stash), HEAD itself printed "golden 768x768 text32 ref0: 5195682816 bytes" and failed this
// same check, and with them restored this suite prints the same 5195682816 bytes. So the shared
// artifact-owned topology moved no byte of the certified admission; the lock was stale by one commit
// and is re-certified to the HEAD-verified figure here. Nothing about the route's guarantee changed:
// The old constant priced the over-estimate and this one the honest charge; the qualification test
// (tests/models/qwen_image/test_image_need_ledger_real.cu) is still the oracle for predicted >=
// measured. A reviewer who wants the old number back has to re-introduce the double charge, which is
// the over-estimate that commit deliberately removed.
constexpr std::uint64_t kGoldenT2I768Bytes = 5195682816ULL; // this test's own report line, in bytes

} // namespace

int main() {
    // The decoder geometry is the artifact's, and there is no fallback if it cannot be read: a suite that
    // could not open the committed document would otherwise be printing a need computed from a decoder
    // nobody declared. Fail loudly instead.
    try {
        (void)shipped_topology();
    } catch (const std::exception& error) {
        std::printf("FAIL: no VAE topology to price the ledger with: %s\n", error.what());
        return 1;
    }
    // The review's failure case: the same target size with a reference must cost more than without.
    check(need_of(768, 200, 0, 0) < need_of(768, 200, 0, 4096), "a reference raises the need");
    // True CFG sizes every sequence buffer by the longer joint sequence.
    check(need_of(768, 200, 0, 0) < need_of(768, 200, 512, 0), "true CFG raises the need");
    // Whichever pass is longer drives s_max, so the need does not depend on which one is negative.
    check(need_of(512, 64, 4096, 0) == need_of(512, 4096, 64, 0), "s_max is the longer joint sequence");
    // Area dominates: 2048^2 is far past 768^2, and past the same size plus one reference.
    check(need_of(768, 200, 0, 0) < need_of(2048, 200, 0, 0), "area raises the need");
    check(need_of(1024, 200, 0, 4096) < need_of(2048, 200, 0, 0), "area beats one reference");
    // The 8dc9c63 calibration's floor for the shipped (two-resident-block, bundle B fas 2) route: the
    // topology's resident calibration plus the 512 MiB interlude floor. The old 6183 MiB floor measured
    // the serialized one-block route's high-water peak and retired with that route; the live guarantee-0
    // ledger cert (f77e16b) is the oracle for the current route's peak. This check keeps the shipped
    // route from ever predicting below the calibrated floor -- and it reads the calibration off the
    // topology the ledger itself multiplies by, so a route priced against a decoder that was never
    // stamped cannot clear it either.
    check(need_of(512, 200, 0, 0) >=
              shipped_topology().weight_resident_bytes + 512ULL * qi::kImageMiB,
          "covers the calibrated 512 floor");
    // The calibration is the conservative part; the sequence terms must still be the ones that grow.
    // The floor is now stated against the topology's OWN calibration instead of a MiB figure from a
    // retired route: `3 * 6183 MiB` here dated 8dc9c63's serialized one-block route, and 8986a6e3's
    // honest decoder charge (this row's widest stage at 972 MiB instead of the 1944 MiB the
    // max(channels_in, channels_out) version billed) dropped the 2048^2 total under it without any
    // guarantee weakening -- which is why the check failed at HEAD. What it guards is unchanged: a big
    // canvas is carried by the growing terms, not by the allowance, so the prediction stays past three
    // times the calibration the request itself hands in.
    check(need_of(2048, 200, 0, 0) > 3ULL * shipped_topology().weight_resident_bytes,
          "the growing terms carry the large sizes");

    const qi::ImageDeviceNeedRequest req{
        .geometry        = qi::ImageGeometry::derive(768, 768),
        .dims            = test_dims(),
        .text_tokens     = 200,
        .blocks_resident = 1,
        .vae             = shipped_topology(),
    };
    const auto need = qi::image_device_need(req);
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
        // The bracket is the doubled charge's own arithmetic, read off vision_tower.cu's live set at the
        // worst plane the route can hand the tower (65536x32 = 8192 patches, twice the square reference
        // that used to be billed -- see the derivation beside the constant and the enumeration below):
        // eleven [hidden,patches] planes 198 MiB, d_up 67 MiB, d_patches 24 MiB, the merger rows 16 MiB,
        // the three index planes 0.3 MiB, one block's roles 19 MiB = 325 MiB, and no fudge around it.
        check(declared.vision_peak > 300ULL * qi::kImageMiB &&
                  declared.vision_peak < 400ULL * qi::kImageMiB,
              "the ViT lane reserves its live set over the route's worst legal reference plane");
        // The vision term is purely additive: the same request with NO lane, with the two joint-shaped
        // terms shrunk by exactly what declaring the pad rows saved, plus the vision term, reproduces
        // total() byte for byte. Nothing else in the ledger moved.
        const auto nolid = need_struct(768, kNonPad + kVis, 0, kRef, 0, false);
        check(declared.total() ==
                  nolid.total() - (nolid.dit_activations - declared.dit_activations) -
                      (nolid.rope_and_tables - declared.rope_and_tables) + declared.vision_peak,
              "the vision term is summed like vae_peak, never maxed");
        // The row the plan asks for in MiB: the same 768^2 target as the certified route plus a
        // reference and the ViT lane. Expected ~1.2-1.4 GiB above the text-only need (the 4096 extra
        // latent tokens at ~0.2 MiB per joint row plus the ~325 MiB the lane is billed at once the snap's
        // rounding is priced in -- this is the row whose predicted figure the controller re-certifies).
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

    // ---- the ViT charge is the route's worst plane, derived there and enforced here ----
    // Everything above prices the vision lane at kVisionWorstReferencePatches, and the serving layer
    // yields on the total it prices. That promise is only worth what the lane ENFORCES: run_vision_tower
    // sizes every live plane from the count its caller hands it (src/models/qwen_image/vision_tower.cu),
    // so a count above the number this header bills would hold live device bytes no admission decision
    // took -- the resident LLM starves on a stale figure, which is the blindness this branch exists to
    // close. The guard therefore sits beside the charge (require_patches_within_vision_ledger,
    // image_device_need.h) and is exercised HERE rather than in the CUDA lane: the boundary is
    // arithmetic, so the host suite can pin both ends of it without a device in the way -- and the
    // production route cannot be called from a host test at all, because the CUDA target is deliberately
    // kept out of host tests (src/models/qwen_image/sources.cmake).
    //
    // WHAT the number has to be is a question about the ROUTE, not about this header, so it is asked of
    // the route: the real serve::snap_to_32 (src/serve/openai_images.cpp:75-89) and the real
    // prepare_reference_planes (src/serve/reference_image.cpp) on encoded PNGs. The snap is the EXPORTED
    // function rather than a copy on purpose -- a change to the route's rounding then moves this
    // enumeration with it instead of agreeing with the mistake, which is exactly how the 4096 that used
    // to live beside the charge survived: the constant and the test were both told the same wrong story
    // (the 32-grid rounds each side independently and can round BOTH up, so a 4:3 reference is 1184x896 =
    // 4144 patches and the widest admitted reference is 65536x32 = 8192). The two route constants the
    // header restates -- the 1024*1024 target area and the 32-grid with its below-32 gate
    // (reference_image.cpp:21, :350-352, both file-static in src/serve) -- are restated here too, and the
    // part-D rows pin them against what the real route actually produces, so a route change that moved
    // them without repricing the ledger fails THIS suite rather than passing both sides.
    {
        namespace srv = infernix::serve;
        const auto refused = [](std::uint64_t patches) -> std::string {
            try {
                qi::require_patches_within_vision_ledger(patches);
                return {};
            } catch (const std::exception& error) {
                return error.what();
            }
        };
        constexpr std::uint64_t kCap   = qi::kVisionWorstReferencePatches;
        constexpr double        kArea  = 1024.0 * 1024.0; // reference_image.cpp:21 (file-static there)
        constexpr int           kGrid  = 32;              // serve::snap_to_32's grid
        const auto patches_of = [](std::uint32_t w, std::uint32_t h) {
            return (std::uint64_t(w) / 16ULL) * (std::uint64_t(h) / 16ULL);
        };
        // The route as src/serve writes it: each side snapped on its own, and the pair refused when
        // either snapped side fell below the grid (reference_image.cpp:346-352).
        const auto route_plane = [](double ratio) -> std::pair<int, int> {
            const int w = srv::snap_to_32(std::sqrt(kArea * ratio));
            const int h = srv::snap_to_32(std::sqrt(kArea / ratio));
            if (w < kGrid || h < kGrid) { return {0, 0}; }
            return {w, h};
        };

        // ---- A. the whole aspect space, through the route's own snap ----
        // Two candidate families. Every RATIONAL aspect a client can hand the decoder (coprime source
        // sizes, both orientations, so nothing about the enumeration leans on the header's arithmetic),
        // and every SNAPPED BOUNDARY, where an exact side lands on the half-grid tie that decides
        // round-up-from-under: exact == k*16 for either side, i.e. ratio = (k*16)^2 / AREA and its
        // reciprocal, each probed either side of the tie.
        std::uint64_t     worst   = 0;
        std::uint64_t     worst_px = 0;
        std::size_t       legal  = 0;
        const std::size_t considered = [&] {
            std::size_t seen = 0;
            const auto consider = [&](double ratio) {
                ++seen;
                const auto [w, h] = route_plane(ratio);
                if (w == 0 || h == 0) { return; }
                ++legal;
                const std::uint64_t patches = patches_of(unsigned(w), unsigned(h));
                // The direction that would have caught C1: no aspect the route ADMITS may hand the
                // tower more patches than the ledger billed for.
                check(patches <= kCap,
                      "no reference aspect the route admits produces more patches than the ViT is billed for");
                if (patches > worst) {
                    worst    = patches;
                    worst_px = std::uint64_t(w) * std::uint64_t(h);
                }
            };
            for (int num = 1; num <= 600; ++num) {
                for (int den = 1; den <= 600; ++den) {
                    if (std::gcd(num, den) != 1) { continue; }
                    consider(double(num) / double(den));
                    consider(double(den) / double(num));
                }
            }
            for (int k = 1; k <= 4200; ++k) {
                const double edge = double(k) * 16.0;
                for (const double base : {(edge * edge) / kArea, kArea / (edge * edge)}) {
                    for (const double eps : {-1e-9, 0.0, 1e-9}) { consider(base * (1.0 + eps)); }
                }
            }
            return seen;
        }();
        // The bound is neither under- nor over-priced: the maximum the route can actually produce IS the
        // charge (reachable, so not padded), and nothing above it was ever seen (so the guard cannot
        // refuse a legal reference -- the C1 regression, in one check).
        check(worst == kCap, "the billed charge is exactly the worst plane the reference route can produce");
        check(worst_px == qi::kVisionWorstReferencePlanePixels,
              "the worst plane is the widest admitted reference, snapped to one grid-wide short side");
        check(legal > 100000ULL, "the enumeration really swept the aspect space it claims to cover");
        std::printf("reference route: %zu aspects considered, %zu admitted, worst plane %llu px = %llu patches "
                    "(charged %llu)\n",
                    considered, legal, static_cast<unsigned long long>(worst_px),
                    static_cast<unsigned long long>(worst), static_cast<unsigned long long>(kCap));

        // ---- B. the shapes the product ships with are served, and priced ----
        // C1 was a legal 4:3 photograph edit turning into a 500, so the ordinary aspects are pinned as
        // data at the planes the snap really produces -- and each must clear the guard and be priced by
        // the ledger. These expectations are the route's own geometry, not derived from the constant.
        struct LegalReference {
            const char* what;
            double      source_ratio;
            int         width;
            int         height;
        };
        const std::uint64_t bill = need_struct(768, 32, 0, 4096, 1024, true).vision_peak;
        for (const LegalReference& row : {LegalReference{"square 1:1", 1.0, 1024, 1024},
                                          LegalReference{"4:3", 4.0 / 3.0, 1184, 896},
                                          LegalReference{"3:4", 3.0 / 4.0, 896, 1184},
                                          LegalReference{"16:9", 16.0 / 9.0, 1376, 768},
                                          LegalReference{"9:16", 9.0 / 16.0, 768, 1376},
                                          LegalReference{"2:1", 2.0, 1440, 736}}) {
            const auto [w, h] = route_plane(row.source_ratio);
            const bool shape  = (w == row.width && h == row.height);
            check(shape, "an ordinary reference aspect still snaps to the plane the route documents");
            if (!shape) { std::printf("  %s: got %dx%d, want %dx%d\n", row.what, w, h, row.width, row.height); }
            // Above the square target for every non-square shape: the snap's slack is real, and a charge
            // that ignored it (the old 4096) refused these.
            check(refused(patches_of(unsigned(row.width), unsigned(row.height))).empty(),
                  "every ordinary reference aspect is admitted by the guard that enforces the charge");
            // The bill is the route's, not the request's: a reference of any of these shapes, and any
            // prompt length, is priced identically -- which is why the probe can refuse before yielding.
            const std::uint64_t priced =
                need_struct(768, 32 + unsigned(patches_of(unsigned(row.width), unsigned(row.height)) / 4ULL),
                            0, 4096, 1024, true)
                    .vision_peak;
            check(priced == bill, "the vision lane is billed at the fixed worst plane, not the reference's size");
        }
        // The bill covers what that plane holds live (vision_tower.cu's eleven [hidden, patches] planes
        // alone) and is not a loose multiple of it: it prices the derived plane, nothing padded around it.
        const std::uint64_t eleven =
            11ULL * kCap * test_vision().hidden * qi::kImageElementBytes;
        check(bill >= eleven, "the vision bill at least covers the live planes of the worst legal reference");
        check(bill < 2ULL * eleven, "the vision bill is the worst plane's own live set, not a padded figure");

        // ---- C. the guard enforces exactly the billed number, readably ----
        check(refused(kCap).empty(), "the worst legal reference plane the ledger bills is admitted");
        const std::string one_over = refused(kCap + 1ULL);
        check(!one_over.empty(), "one patch above the billed worst case is refused");
        const auto names = [&](const std::string& text, const char* needle) {
            return !text.empty() && text.find(needle) != std::string::npos;
        };
        check(names(one_over, std::to_string(kCap + 1ULL).c_str()),
              "the refusal prints the caller's patch count it refused");
        check(names(one_over, std::to_string(kCap).c_str()),
              "the refusal prints the bound it enforces");
        // The refusal's story is the derivation, so whoever hits it can tell a pathological reference from
        // a ledger that stopped matching the snap: the target area, the grid, the patch width and the
        // route and file that own them, not the square this header once claimed.
        check(names(one_over, "reference_image.cpp") && names(one_over, "ledger"),
              "the refusal says which route's promise the cap is and what it prices");
        check(names(one_over, std::to_string(qi::kReferenceTargetAreaPx).c_str()) &&
                  names(one_over, std::to_string(qi::kReferenceSnapGridPx).c_str()) &&
                  names(one_over, std::to_string(qi::kVisionPatchSidePx).c_str()),
              "the refusal states how the bound is derived: the target area, the snap grid, the patch width");
        // The case the review found: the joint ceiling alone would admit 20480 merged rows, i.e. 81920
        // patches, ten times the billed plane -- ~3.0 GiB live against the 325 MiB this header bills.
        // Refused, and refused with the same readable reason rather than as an int-overflow accident.
        const std::string joint = refused(81920ULL);
        check(names(joint, "81920") && names(joint, std::to_string(kCap).c_str()),
              "a joint-ceiling-sized plane is refused by name, not by an overflow guard");
        // Nothing below the cap is refused, so the admitted route cannot be squeezed out of its own
        // geometry by the guard: every legal snapped reference size on the 32-grid passes.
        check(refused(0ULL).empty() && refused(1ULL).empty() && refused(4095ULL).empty(),
              "the guard refuses only what the ledger does not bill");

        // ---- D. the route end to end, on encoded references ----
        // The same prepare_reference_planes the HTTP route runs, so the decode, the aspect snap and the
        // refusal the client gets are the shipped ones: this is what ties the two route constants this
        // file and the header both restate to the code that uses them, and what proves the equality case
        // (the widest admitted aspect, landing exactly on the charge) is SERVED rather than refused.
        const auto reference = [](std::uint32_t w, std::uint32_t h) {
            const std::vector<std::uint8_t> rgba(std::size_t(w) * std::size_t(h) * 4U, 90U);
            const std::vector<std::uint8_t> png =
                infernix::media::encode::encode_png_rgba8(w, h, std::span<const std::uint8_t>(rgba));
            return srv::base64_encode_bytes(std::span<const std::uint8_t>(png));
        };
        const auto prepared_pane = [&](std::uint32_t w, std::uint32_t h, const char* what) {
            std::pair<std::uint32_t, std::uint32_t> plane{0, 0};
            try {
                const srv::ReferencePlanes planes = srv::prepare_reference_planes(reference(w, h));
                plane = {planes.vae.width_px, planes.vae.height_px};
            } catch (const std::exception& error) {
                ++failures;
                std::printf("FAIL: %s reference refused: %s\n", what, error.what());
                return plane;
            }
            check(refused(patches_of(plane.first, plane.second)).empty(),
                  "the plane the reference route produced is admitted by the vision guard");
            return plane;
        };
        {
            // A 4:3 photograph at the ratio's smallest pixels (the snap reads only the ratio), and the
            // certified square.
            const auto wide = prepared_pane(40, 30, "4:3");
            check(wide.first == 1184U && wide.second == 896U,
                  "the shipped route still serves a 4:3 reference at the plane the snap produces");
            const auto square = prepared_pane(64, 64, "square");
            check(square.first == 1024U && square.second == 1024U,
                  "the shipped route still serves a square reference, and the certified {1,64,64} grid");
            // The reachable boundary of the charge: the widest aspect the route's own below-32 gate
            // admits, landing EXACTLY on kVisionWorstReferencePatches. Equality must paint.
            const auto widest = prepared_pane(4095, 1, "the widest admitted reference");
            check(widest.first * widest.second == qi::kVisionWorstReferencePlanePixels,
                  "the widest legal reference is the worst plane the ledger bills, and it is served");
        }
        {
            // One pixel narrower on the short side's exact value, and the route's own gate refuses it --
            // the documented std::invalid_argument that reaches the client as a 400, before any resize.
            // Beyond THAT there is no plane to bill for, which is why the serve boundary's over-the-cap
            // arm has no reachable input: the only legal way to meet it is the equality above.
            bool refused_by_the_route = false;
            try {
                (void)srv::prepare_reference_planes(reference(4096, 1));
            } catch (const std::invalid_argument& error) {
                refused_by_the_route = std::string(error.what()).find("aspect ratio is too extreme") !=
                                       std::string::npos;
            } catch (const std::exception&) {
            }
            check(refused_by_the_route,
                  "a reference past the route's aspect gate is refused as a client error, not a 500");
        }
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
