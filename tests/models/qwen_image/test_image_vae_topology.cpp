// The image VRAM ledger and the CUDA decoder graph read ONE artifact-owned VAE topology (plan
// 2026-11-14-image-fail-closed, Task 5): the stage and channel data the ledger charges for is the same
// data the graph runs, built by the binding from the artifact's own "vae" component config and never
// from a compiled header literal.
//
// This suite proves two things, in the plan's own words: that the shipped family's certified numbers
// did not move when the literals became fields, and that ANOTHER family is expressible. A second
// topology with its own channels and its own decoder width must drive a different vae_peak; if the
// ledger still read its geometry from a header array, the foreign row below could not move and the seam
// would not be real.
//
// The shipped topology is produced by the PRODUCTION reader (read_vae_topology) over the committed
// artifact document, so neither a test-side literal nor an edited fixture can quietly move what the
// ledger certifies. The bound-weight shapes check_vae_topology compares against are written as the
// artifact stores them (the converter packs every vae/* tensor one-to-one at its source shape) rather
// than derived from the topology, so the acceptance is a comparison of two independent representations
// and not a tautology.
//
// Host-only: no GPU, no server, no device allocation. The last section opens the live .infernix container
// named by INFERNIX_TEST_ARTIFACT through the production reader and FAILS (exit 1) when that variable is
// unset or names an unreadable file. The registration helper grants the entry a SKIP_RETURN_CODE of 77
// (tests/cmake/InfernixTests.cmake:32-36), but that code counts only when the test exits 77 and this gate
// no longer does that for a missing artifact -- ctest suppresses stdout on success, and a suite that
// reported a clean pass while never opening the shipped container certified nothing.
// The fixture-only gates above still run first, so a host without the container gets the
// reader's and the ledger's verdicts -- it just cannot also claim the shipped artifact was verified.

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/schema.h"
#include "models/qwen_image/artifact_constants.h"
#include "models/qwen_image/image_device_need.h"
#include "models/qwen_image/image_weights.h"
#include "models/qwen_image/vae_topology_fixture.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace qi      = infernix::models::qwen_image;
namespace qt      = infernix::tests::qwen_image;
namespace artifact = infernix::artifact;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& name) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", name.c_str());
    if (!ok) ++g_failures;
}

void check_detail(bool ok, const std::string& name, const std::string& detail) {
    std::printf("%s: %s%s%s\n", ok ? "ok" : "FAIL", name.c_str(), detail.empty() ? "" : " -- ",
                detail.c_str());
    if (!ok) ++g_failures;
}

std::string shapes(const artifact::Shape& expected, const artifact::Shape& actual) {
    const auto join = [](const artifact::Shape& s) {
        std::string text = "[";
        for (std::size_t i = 0; i < s.size(); ++i) {
            if (i != 0) text += ",";
            text += std::to_string(s[i]);
        }
        return text + "]";
    };
    return "expected " + join(expected) + ", the bound weight is " + join(actual);
}

// ---- the certified route, copied from tests/test_image_device_need.cpp row for row ----
//
// The ledger's own suite pins this route (its kGoldenT2I768Bytes is this test's kCertifiedShippedTotal,
// the same number its report line prints). Nothing here is invented: the same DiT dimensions, the same
// snapped 768x768 canvas, the same 32 kept prompt rows, the same two resident DiT block slots, and the
// zeroed edit route. If any of it moves, the certified text-to-image admission regressed.
qi::DiTDims certified_dims() {
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

// The byte-for-byte pin is copied from the ledger suite's own lock (its kGoldenT2I768Bytes), which is
// where this route's certified admission lives. Two facts belong with that number, both verified rather
// than assumed. (1) The value is the one THIS tree prints, and it is not the value the ledger suite's
// file committed: 8986a6e3 ("serve: charge the VAE decoder what it actually holds") replaced the
// decoder's max(channels_in, channels_out) charge with the stage's true output width, lowering this row
// by exactly the 972 MiB the {576,288} stage double-counted, and never re-certified the lock -- so HEAD
// itself printed "golden 768x768 text32 ref0: 5195682816 bytes" and failed its own byte-identical
// check (verified by running the suite with this task's changes stashed away, and again with them
// restored: the same 5195682816 both times). (2) Therefore Task 5 moved no byte of the certified
// admission: the artifact-owned topology reproduces the header arrays it replaced, stage for stage,
// planes and channels and doublings alike, which expect_shipped() below checks field by field. The
// suite's own comment on that constant says what to do with a diff here: explain it, do not edit it --
// the explanation is 8986a6e3, and the figure is the one HEAD measures.
constexpr std::uint64_t kCertifiedShippedTotal = 5195682816ULL; // the ledger suite's own pin

qi::ImageDeviceNeedRequest request_with(const qi::VaeTopology& topology) {
    return qi::ImageDeviceNeedRequest{
        .geometry        = qi::ImageGeometry::derive(768, 768),
        .dims            = certified_dims(),
        .text_tokens     = 32,
        .negative_tokens = 0,
        .ref_tokens      = 0,
        .blocks_resident = qi::kImageResidentBlocks,
        .vision_tokens   = 0,
        .vision          = {},
        .vae             = topology,
    };
}

// ---- the shipped family's certified geometry, row by row ----

void expect_shipped(const qi::VaeTopology& t, const std::string& who) {
    check(t.latent_channels == 64, who + ": latent_channels is the artifact's z_dim (64)");
    check(t.spatial_downscale == 16, who + ": spatial_downscale is the artifact's scale_factor_spatial (16)");
    check(t.mid_stage.channels_in == 1152 && t.mid_stage.channels_out == 1152 && t.mid_stage.planes == 5 &&
              !t.mid_stage.doubles_side,
          who + ": the mid stage is the certified {1152, 1152, 5, false}");
    const qi::VaeStage certified_up[5] = {
        {1152, 1152, 3, true}, {1152, 1152, 3, true}, {1152, 576, 3, true},
        {576, 288, 3, true},   {288, 144, 3, false},
    };
    check(t.up_stages.size() == 5, who + ": the decoder chain has five up stages");
    for (std::size_t i = 0; i < t.up_stages.size() && i < 5; ++i) {
        const auto& s = t.up_stages[i];
        const auto& c = certified_up[i];
        check(s.channels_in == c.channels_in && s.channels_out == c.channels_out && s.planes == c.planes &&
                  s.doubles_side == c.doubles_side,
              who + ": up stage " + std::to_string(i) + " is the certified {" +
                  std::to_string(c.channels_in) + ", " + std::to_string(c.channels_out) + ", " +
                  std::to_string(c.planes) + ", " + (c.doubles_side ? "true" : "false") + "}");
    }
    check(t.final_channels == 144 + 4, who + ": the final conv is the certified 144-wide plus 4 channels");
    check(t.denorm_elements == 64, who + ": the denormalization guard reads 64 per-channel elements");
    check(t.weight_resident_bytes == 3072ULL * qi::kImageMiB,
          who + ": the calibration allowance is the certified 3072 MiB");
}

// ---- the artifact's stored decoder rows, as the converter packs them ----

struct Row {
    std::string name;
    artifact::Shape shape;
};

// Every "vae/..." operand of the DECODER chain the qualified route consumes, at the shape the production
// artifact stores it (tools/convert/qwen_image.py maps each source tensor one-to-one as a direct BF16
// parameter). These are the artifact's numbers, typed in as the artifact carries them, so that
// check_vae_topology comparing them against the artifact-DERIVED topology is a real cross-check.
std::vector<Row> shipped_decoder_rows() {
    std::vector<Row> rows;
    const auto add = [&rows](const std::string& name, artifact::Shape shape) {
        rows.push_back(Row{name, std::move(shape)});
    };
    add("vae/post_quant_conv.weight", {64, 64, 1, 1});
    add("vae/post_quant_conv.bias", {64});
    add("vae/decoder.conv_in.weight", {1152, 64, 3, 3});
    add("vae/decoder.conv_in.bias", {1152});
    for (int mid = 0; mid < 2; ++mid) {
        const std::string p = "vae/decoder.mid_block.resnets." + std::to_string(mid) + ".";
        add(p + "conv1.weight", {1152, 1152, 3, 3});
        add(p + "conv1.bias", {1152});
        add(p + "conv2.weight", {1152, 1152, 3, 3});
        add(p + "conv2.bias", {1152});
        add(p + "norm1.gamma", {1152, 1, 1, 1});
        add(p + "norm2.gamma", {1152, 1, 1, 1});
    }
    const std::string attn = "vae/decoder.mid_block.attentions.0.";
    add(attn + "to_qkv.weight", {3 * 1152, 1152, 1, 1});
    add(attn + "to_qkv.bias", {3 * 1152});
    add(attn + "proj.weight", {1152, 1152, 1, 1});
    add(attn + "proj.bias", {1152});
    add(attn + "norm.gamma", {1152, 1, 1});
    const struct { std::uint64_t cin, cout; bool up; } blocks[5] = {
        {1152, 1152, true}, {1152, 1152, true}, {1152, 576, true}, {576, 288, true}, {288, 144, false},
    };
    for (int b = 0; b < 5; ++b) {
        const std::string p    = "vae/decoder.up_blocks." + std::to_string(b) + ".";
        const std::uint64_t cin = blocks[b].cin, cout = blocks[b].cout;
        for (int r = 0; r < 3; ++r) {
            const std::string rp  = p + "resnets." + std::to_string(r) + ".";
            const std::uint64_t rin = r == 0 ? cin : cout;
            add(rp + "conv1.weight", {cout, rin, 3, 3});
            add(rp + "conv1.bias", {cout});
            add(rp + "conv2.weight", {cout, cout, 3, 3});
            add(rp + "conv2.bias", {cout});
            add(rp + "norm1.gamma", {rin, 1, 1, 1});
            add(rp + "norm2.gamma", {cout, 1, 1, 1});
            if (r == 0 && cin != cout) {
                add(rp + "conv_shortcut.weight", {cout, cin, 1, 1});
                add(rp + "conv_shortcut.bias", {cout});
            }
        }
        if (blocks[b].up) {
            add(p + "upsampler.resample.1.weight", {cout, cout, 3, 3});
            add(p + "upsampler.resample.1.bias", {cout});
        }
    }
    add("vae/decoder.norm_out.gamma", {144, 1, 1, 1});
    add("vae/decoder.conv_out.weight", {4, 144, 3, 3});
    add("vae/decoder.conv_out.bias", {4});
    return rows;
}

qi::VaeWeights bind_the_stored_rows(const std::vector<Row>& rows) {
    qi::VaeWeights weights;
    for (const auto& row : rows) {
        artifact::ParameterReference ref;
        ref.name     = row.name;
        ref.shape    = row.shape;
        ref.residency = artifact::Residency::Streamed;
        weights.by_name.insert_or_assign(row.name, std::move(ref));
    }
    return weights;
}

// Any reader verdict, spelled the way the suite reports it: the accepted case names no value, so the
// expected's T does not matter -- both std::expected<void, std::string> (the topology check) and
// std::expected<VaeTopology, std::string> (the reader) print their refusal text and nothing else.
template <typename T>
std::string accepted_or(const std::expected<T, std::string>& result) {
    return result.has_value() ? std::string("accepted") : result.error();
}

// ---- the reader: every structural claim is read from the artifact or refused by name ----

void expect_refuses(nlohmann::json document, const std::string& name,
                    const std::vector<std::string>& needles) {
    const auto result = qi::read_vae_topology(document);
    if (result.has_value()) {
        check_detail(false, name, "the reader ACCEPTED a document it must refuse");
        return;
    }
    bool named = true;
    for (const std::string& needle : needles) {
        if (result.error().find(needle) == std::string::npos) named = false;
    }
    check_detail(named, name, "refusal: " + result.error());
}

void expect_reads(nlohmann::json document, const std::string& name) {
    const auto result = qi::read_vae_topology(document);
    check_detail(result.has_value(), name, accepted_or(result));
}

void gate_reader() {
    expect_reads(qt::qwen_image_2_1_vae_document(), "the committed artifact document is accepted");

    // A key the decoder topology reads must be PRESENT: no compiled fallback stands in for it.
    {
        nlohmann::json j = qt::qwen_image_2_1_vae_document();
        j.erase("decoder_base_dim");
        expect_refuses(j, "a config without decoder_base_dim is refused", {"decoder_base_dim"});
    }
    {
        nlohmann::json j = qt::qwen_image_2_1_vae_document();
        j.erase("dim_mult");
        expect_refuses(j, "a config without dim_mult is refused", {"dim_mult"});
    }
    {
        nlohmann::json j = qt::qwen_image_2_1_vae_document();
        j.erase("z_dim");
        expect_refuses(j, "a config without z_dim is refused", {"z_dim"});
    }
    {
        nlohmann::json j = qt::qwen_image_2_1_vae_document();
        j.erase("scale_factor_spatial");
        expect_refuses(j, "a config without scale_factor_spatial is refused", {"scale_factor_spatial"});
    }
    // No coercion across JSON types: 144.0 is a different claim from 144.
    {
        nlohmann::json j = qt::qwen_image_2_1_vae_document();
        j["decoder_base_dim"] = 144.0;
        expect_refuses(j, "a float decoder_base_dim is refused", {"decoder_base_dim", "JSON integer"});
    }
    {
        nlohmann::json j = qt::qwen_image_2_1_vae_document();
        j["dim_mult"]           = {1.0, 2, 4, 8, 8};
        expect_refuses(j, "a float entry in dim_mult is refused", {"dim_mult"});
    }
    {
        nlohmann::json j = qt::qwen_image_2_1_vae_document();
        j["dim_mult"]           = {1, 2, 4, 8}; // four levels: the graph runs five up blocks
        expect_refuses(j, "a four-level dim_mult is refused", {"dim_mult"});
    }
    {
        nlohmann::json j = qt::qwen_image_2_1_vae_document();
        j["dim_mult"]           = {1, 2, 4, 8, 0};
        expect_refuses(j, "a zero-wide level in dim_mult is refused", {"dim_mult"});
    }
    // The PRODUCT of two separately-plausible declarations is what the stage widths are, and the widths
    // are stored and priced as std::uint32_t. Each factor passes its own kVaeLargestPlausibleValue bound
    // (below 2^20), so nothing but a bound on the product itself keeps a hand-authored document from
    // narrowing a gigabyte-wide stage to a truncated width the ledger then charges for free. These two
    // rows are the reviewer's failure case driven through the SAME production reader the shipped document
    // goes through: the first multiplies to exactly 2^32 (which would truncate to 0 -- a zero-channel
    // stage admitted at zero cost), the second to 2^36 at a different stage index, so the guard is shown
    // to bound every stage and not just the deepest one.
    {
        nlohmann::json j = qt::qwen_image_2_1_vae_document();
        j["decoder_base_dim"]   = 524288; // 2^19, under its own bound
        j["dim_mult"]           = {1, 2, 4, 8, 8192}; // 2^13, under its own bound
        expect_refuses(j, "a stage width whose product exceeds UINT32_MAX is refused naming both keys",
                       {"decoder_base_dim", "dim_mult", "4294967296", "524288", "8192", "4294967295"});
    }
    {
        nlohmann::json j = qt::qwen_image_2_1_vae_document();
        j["decoder_base_dim"]   = 262144; // 2^18
        j["dim_mult"]           = {1, 2, 4, 262144, 8};
        expect_refuses(j, "the same overflow at another stage index is refused naming that stage",
                       {"decoder_base_dim", "dim_mult", "68719476736", "[3]", "up stage 1"});
    }
    // Structural claims the graph implements unconditionally are VERIFIED, never adopted: a family whose
    // chain differs from the one image_pipeline.cu runs is refused instead of drawn with another
    // family's structure.
    {
        nlohmann::json j = qt::qwen_image_2_1_vae_document();
        j["num_res_blocks"]     = 3;
        expect_refuses(j, "a chain with a different resnet count is refused", {"num_res_blocks", "3"});
    }
    {
        nlohmann::json j = qt::qwen_image_2_1_vae_document();
        j["scale_factor_spatial"] = 8; // three doublings, while the chain declares four
        expect_refuses(j, "a spatial downscale the up chain cannot produce is refused",
                       {"scale_factor_spatial", "8"});
    }
    {
        nlohmann::json j = qt::qwen_image_2_1_vae_document();
        j["scale_factor_spatial"] = 12;
        expect_refuses(j, "a non-power-of-two spatial downscale is refused", {"scale_factor_spatial"});
    }
    {
        nlohmann::json j = qt::qwen_image_2_1_vae_document();
        j["temperal_downsample"]  = {false, false, false, false};
        expect_refuses(j, "a different temporal scheme is refused", {"temperal_downsample"});
    }
    {
        nlohmann::json j = qt::qwen_image_2_1_vae_document();
        j["is_residual"]          = false;
        expect_refuses(j, "a non-residual block is refused", {"is_residual", "false"});
    }
    {
        nlohmann::json j = qt::qwen_image_2_1_vae_document();
        j["attn_scales"]          = {0.5};
        expect_refuses(j, "attention outside the mid block is refused", {"attn_scales"});
    }
    // The per-channel latent statistics ARE the latent width: an artifact that disagrees with itself is
    // refused naming both numbers.
    {
        nlohmann::json j = qt::qwen_image_2_1_vae_document();
        j["latents_mean"]         = nlohmann::json::array();
        for (int i = 0; i < 32; ++i) j["latents_mean"].push_back(0.0);
        expect_refuses(j, "a denormalization pair that is not z_dim wide is refused",
                       {"latents_mean", "64", "32"});
    }
    {
        nlohmann::json j = qt::qwen_image_2_1_vae_document();
        j["latents_std"]          = nlohmann::json::array();
        for (int i = 0; i < 64; ++i) j["latents_std"].push_back("3.2");
        expect_refuses(j, "a string in latents_std is refused", {"latents_std"});
    }
    // A family that declares a DIFFERENT latent width is expressible: the width is read, not pinned.
    {
        nlohmann::json j = qt::qwen_image_2_1_vae_document();
        j["z_dim"]                = 32;
        for (const char* key : {"latents_mean", "latents_std"}) {
            nlohmann::json values = nlohmann::json::array();
            for (int i = 0; i < 32; ++i) values.push_back(1.0);
            j[key] = values;
        }
        const auto result = qi::read_vae_topology(j);
        check(result.has_value() && result->latent_channels == 32 && result->denorm_elements == 32,
              "a family declaring a 32-wide latent reads as one");
    }
    // The calibration the reader must NOT invent: it leaves the family's allowance unset so the binding
    // stamps it, and check_vae_topology refuses an unset one.
    {
        const auto result = qi::read_vae_topology(qt::qwen_image_2_1_vae_document());
        check(result.has_value() && result->weight_resident_bytes == 0,
              "the reader leaves the code-owned calibration unset");
    }
}

// ---- the ledger follows the topology ----

void gate_ledger() {
    const qi::VaeTopology shipped = qt::qwen_image_2_1_topology();
    expect_shipped(shipped, "the shipped artifact document");

    const auto shipped_need = qi::image_device_need(request_with(shipped));
    std::printf("certified 768x768 text32 ref0: %llu bytes (vae_peak %llu MiB)\n",
                static_cast<unsigned long long>(shipped_need.total()),
                static_cast<unsigned long long>(shipped_need.vae_peak / qi::kImageMiB));
    check(shipped_need.total() == kCertifiedShippedTotal,
          "the shipped family's certified total did not move");

    // The certified route's own decoder high-water, spelled as the arithmetic the shipped stages are:
    // the {576, 288} up block is the widest set the graph holds live, three planes at its 288 output
    // channels on the side its four predecessors reach (768), two bytes an element. If the ledger still
    // charged max(channels_in, channels_out) this would read 1944 MiB, and if it read a retired header
    // array it would read whatever that array held; neither is available to it any more.
    constexpr std::uint64_t kShippedVaePeakBytes = 3ULL * 288ULL * 768ULL * 768ULL * 2ULL;
    check(shipped_need.vae_peak == kShippedVaePeakBytes && kShippedVaePeakBytes == 1019215872ULL,
          "the shipped decoder's high-water is the certified 972 MiB, computed from its own stages");

    // A second family is expressible: 24 latent channels, 16x downscale, a different decoder width -- and
    // four up blocks instead of this build's five, which is the plan's own block, kept verbatim. If the
    // ledger still read a header array this row could not move, and the seam would not be real: it walks
    // the vector it is handed, so a different chain is a different number by construction.
    qi::VaeTopology foreign;
    foreign.latent_channels   = 24;
    foreign.spatial_downscale = 16;
    foreign.mid_stage         = qi::VaeStage{1536, 1536, 5, false};
    foreign.up_stages         = {{1536, 1536, 3, true}, {1536, 768, 3, true},
                                 {768, 384, 3, true}, {384, 192, 3, false}};
    foreign.final_channels    = 192 + 4;
    foreign.denorm_elements   = 64;
    foreign.weight_resident_bytes = 512ULL * qi::kImageMiB;

    const auto other = qi::image_device_need(request_with(foreign));
    std::printf("foreign 768x768 text32 ref0: %llu bytes (vae_peak %llu MiB)\n",
                static_cast<unsigned long long>(other.total()),
                static_cast<unsigned long long>(other.vae_peak / qi::kImageMiB));
    // The plan's line was `check(other.vae_peak > shipped.vae_peak, ...)`. It is kept, but against a
    // family that is actually heavier; this one is not, and asserting it here would mean writing a
    // number that is false. The plan's block gives the foreign family FOUR up blocks, so its chain stops
    // doubling after the third: on the same 768^2 canvas it never reaches the pixel side the shipped
    // five-block chain does, and its widest live set is the {768, 384} block at 384^2 -- 324 MiB against
    // the shipped family's 972 MiB. The property the check exists to prove is that the ledger prices the
    // chain it is handed instead of a compile-time table, and that is what these two checks pin, exactly:
    // the foreign high-water is its own stages' arithmetic, and it is not the shipped family's.
    constexpr std::uint64_t kForeignVaePeakBytes = 3ULL * 384ULL * 384ULL * 384ULL * 2ULL;
    check(other.vae_peak == kForeignVaePeakBytes && kForeignVaePeakBytes == 339738624ULL,
          "the foreign family's decoder high-water is what ITS stages produce: three planes at its "
          "{768, 384} block's 384 output channels on the 384^2 side its three upsamplers reach");
    check(other.vae_peak != shipped_need.vae_peak, "the ledger follows the topology, not a literal");
    // Same seam on the resident half: the bill carries the family's own calibration, not the shipped
    // 3072 MiB, and the only difference from the certified row is the calibration it declares.
    check(other.resident_weights != shipped_need.resident_weights &&
              other.resident_weights ==
                  shipped_need.resident_weights - (3072ULL - 512ULL) * qi::kImageMiB,
          "the resident half of the bill follows the topology's calibration, not a compiled allowance");
    check(shipped_need.total() == kCertifiedShippedTotal, "the shipped family's certified total did not move");

    // The plan's `other.vae_peak > shipped.vae_peak`, carried by a family that satisfies it: the same
    // five-block scheme this build runs, at a 1.5x wider base (216 instead of 144, so 1728/1728/864/432/
    // 216 -- the same dim_mult [1, 2, 4, 8, 8] shape the shipped artifact declares). A heavier decoder
    // must raise the reservation, and it does: its {864, 432} block at the full 768^2 side is 1458 MiB.
    qi::VaeTopology heavier;
    heavier.latent_channels   = 24;
    heavier.spatial_downscale = 16;
    heavier.mid_stage         = qi::VaeStage{1728, 1728, 5, false};
    heavier.up_stages         = {{1728, 1728, 3, true}, {1728, 1728, 3, true}, {1728, 864, 3, true},
                                 {864, 432, 3, true}, {432, 216, 3, false}};
    heavier.final_channels    = 216 + 4;
    heavier.denorm_elements   = 24;
    heavier.weight_resident_bytes = 512ULL * qi::kImageMiB;
    const auto heavier_need = qi::image_device_need(request_with(heavier));
    std::printf("heavier 768x768 text32 ref0: %llu bytes (vae_peak %llu MiB)\n",
                static_cast<unsigned long long>(heavier_need.total()),
                static_cast<unsigned long long>(heavier_need.vae_peak / qi::kImageMiB));
    constexpr std::uint64_t kHeavierVaePeakBytes = 3ULL * 432ULL * 768ULL * 768ULL * 2ULL;
    check(heavier_need.vae_peak > shipped_need.vae_peak &&
              heavier_need.vae_peak == kHeavierVaePeakBytes && kHeavierVaePeakBytes == 1528823808ULL,
          "the ledger follows the topology, not a literal: a wider decoder is priced wider than the "
          "shipped family's certified 972 MiB");

    // Every term the ledger charges for the decoder comes out of the topology it was handed, one field at
    // a time: the final conv_out's width is the last term's multiplicand, and a stage that does not
    // double takes its side down with it.
    qi::VaeTopology wide_final = shipped;
    wide_final.final_channels = shipped.final_channels + 100000U;
    const auto wide_final_need = qi::image_device_need(request_with(wide_final));
    check(wide_final_need.vae_peak ==
              static_cast<std::uint64_t>(wide_final.final_channels) * 768ULL * 768ULL * 2ULL &&
              wide_final_need.vae_peak > shipped_need.vae_peak,
          "the final conv's planes follow the topology too: widening conv_out past the widest stage makes "
          "it the high-water, at its own channel count on the full pixel canvas");
    qi::VaeTopology one_doubling = shipped;
    one_doubling.up_stages[3].doubles_side = false; // the third upsampler is switched off
    const auto one_doubling_need = qi::image_device_need(request_with(one_doubling));
    check(one_doubling_need.vae_peak == 3ULL * 576ULL * 384ULL * 384ULL * 2ULL &&
              one_doubling_need.vae_peak == 509607936ULL,
          "the decoder's high-water follows the stages the graph would double: with one upsample missing "
          "the chain never reaches the 768^2 side, and its widest live set drops to the {1152, 576} block "
          "at 384^2");
}

// ---- check_vae_topology: the topology against the bound weight's actual shape ----

void gate_bound_shapes() {
    const qi::VaeTopology shipped = qt::qwen_image_2_1_topology();

    // The artifact's stored rows agree with the topology derived from the same artifact's config.
    const auto accepted = qi::check_vae_topology(shipped, bind_the_stored_rows(shipped_decoder_rows()));
    check_detail(accepted.has_value(), "the bound decoder shapes agree with the artifact-derived topology",
                 accepted_or(accepted));

    // A stage whose bound conv is not the topology's width stops the binding, naming the role, the
    // stage index and both shapes.
    {
        auto rows = shipped_decoder_rows();
        for (auto& row : rows) {
            if (row.name == "vae/decoder.up_blocks.2.resnets.1.conv2.weight") row.shape[0] = 1152;
        }
        const auto refused = qi::check_vae_topology(shipped, bind_the_stored_rows(rows));
        check_detail(!refused.has_value() &&
                         refused.error().find("up_blocks.2") != std::string::npos &&
                         refused.error().find("resnets.1") != std::string::npos,
                     "a deviating up stage is refused naming its index",
                     refused.has_value() ? "the checker ACCEPTED a deviating weight" : refused.error());
    }
    {
        auto rows = shipped_decoder_rows();
        for (auto& row : rows) {
            if (row.name == "vae/decoder.conv_in.weight") row.shape[1] = 16;
        }
        const auto refused = qi::check_vae_topology(shipped, bind_the_stored_rows(rows));
        check_detail(!refused.has_value() && refused.error().find("conv_in") != std::string::npos,
                     "a decoder input that is not the latent width is refused",
                     refused.has_value() ? "the checker ACCEPTED a deviating weight" : refused.error());
    }
    {
        auto rows = shipped_decoder_rows();
        for (auto& row : rows) {
            if (row.name == "vae/decoder.norm_out.gamma") row.shape[0] = 1152;
        }
        const auto refused = qi::check_vae_topology(shipped, bind_the_stored_rows(rows));
        check_detail(!refused.has_value() && refused.error().find("norm_out") != std::string::npos,
                     "a final norm that is not the last stage's width is refused",
                     refused.has_value() ? "the checker ACCEPTED a deviating weight" : refused.error());
    }
    // A role the artifact no longer stores is refused by name -- at bind time, not mid-decode.
    {
        auto rows = shipped_decoder_rows();
        rows.erase(std::remove_if(rows.begin(), rows.end(),
                                  [](const Row& r) {
                                      return r.name == "vae/decoder.up_blocks.4.resnets.0.conv_shortcut.weight";
                                  }),
                   rows.end());
        const auto refused = qi::check_vae_topology(shipped, bind_the_stored_rows(rows));
        check_detail(!refused.has_value() &&
                         refused.error().find("conv_shortcut") != std::string::npos &&
                         refused.error().find("up_blocks.4") != std::string::npos,
                     "a missing decoder role is refused naming it",
                     refused.has_value() ? "the checker ACCEPTED an incomplete chain" : refused.error());
    }
    // The other direction: another family's topology does not fit the shipped family's weights, and the
    // FIRST deviation is reported with its name and both shapes -- not quietly repaired with a compiled
    // default. The family is the ledger gate's heavier one (a five-block scheme, which is what the graph
    // runs, at 1.5x the widths, 24 latent channels and its own 24-element statistics so that the check
    // gets past the self-consistency rules and lands on a real convolution). Its first operand is
    // post_quant_conv, which the shipped artifact stores as [64, 64, 1, 1] and a 24-channel latent needs
    // as [24, 24, 1, 1] -- the refusal has to say which role, what it found and what it wanted.
    {
        qi::VaeTopology foreign;
        foreign.latent_channels       = 24;
        foreign.spatial_downscale     = 16;
        foreign.mid_stage             = qi::VaeStage{1728, 1728, 5, false};
        foreign.up_stages             = {{1728, 1728, 3, true}, {1728, 1728, 3, true}, {1728, 864, 3, true},
                                         {864, 432, 3, true}, {432, 216, 3, false}};
        foreign.final_channels        = 216 + 4;
        foreign.denorm_elements       = 24;
        foreign.weight_resident_bytes = 512ULL * qi::kImageMiB;
        const auto refused = qi::check_vae_topology(foreign, bind_the_stored_rows(shipped_decoder_rows()));
        check_detail(!refused.has_value() &&
                         refused.error().find("vae/post_quant_conv.weight") != std::string::npos &&
                         refused.error().find("[24, 24, 1, 1]") != std::string::npos &&
                         refused.error().find("[64, 64, 1, 1]") != std::string::npos,
                     "another family's topology is refused against the shipped weights, naming the first "
                     "role and both shapes",
                     refused.has_value() ? "the checker ACCEPTED another family" : refused.error());
    }
    // The plan's four-up-block family is refused too, and for the structural reason: image_pipeline.cu's
    // chain runs exactly kQwenImageVaeUpBlocks up blocks, so a shorter decoder has resnets and resamplers
    // this build never calls and weights it cannot name.
    {
        qi::VaeTopology four_blocks;
        four_blocks.latent_channels       = 24;
        four_blocks.spatial_downscale     = 16;
        four_blocks.mid_stage             = qi::VaeStage{1536, 1536, 5, false};
        four_blocks.up_stages             = {{1536, 1536, 3, true}, {1536, 768, 3, true},
                                             {768, 384, 3, true}, {384, 192, 3, false}};
        four_blocks.final_channels        = 192 + 4;
        four_blocks.denorm_elements       = 24;
        four_blocks.weight_resident_bytes = 512ULL * qi::kImageMiB;
        const auto refused = qi::check_vae_topology(four_blocks, bind_the_stored_rows(shipped_decoder_rows()));
        check_detail(!refused.has_value() &&
                         refused.error().find("up blocks") != std::string::npos &&
                         refused.error().find("4") != std::string::npos,
                     "a decoder with a different number of up blocks is refused naming both counts",
                     refused.has_value() ? "the checker ACCEPTED a shorter chain" : refused.error());
    }
    // An unread or unset topology cannot be admitted: the ledger would be charging for a decoder nobody
    // bound.
    {
        const auto refused = qi::check_vae_topology(qi::VaeTopology{}, bind_the_stored_rows(shipped_decoder_rows()));
        check_detail(!refused.has_value(), "an unset topology is refused",
                     refused.has_value() ? "the checker ACCEPTED an unset topology" : refused.error());
    }
    {
        qi::VaeTopology unstamped = shipped;
        unstamped.weight_resident_bytes = 0;
        const auto refused = qi::check_vae_topology(unstamped, bind_the_stored_rows(shipped_decoder_rows()));
        check_detail(!refused.has_value() &&
                         refused.error().find("weight_resident_bytes") != std::string::npos,
                     "a family that never stamped its calibration is refused",
                     refused.has_value() ? "the checker ACCEPTED an unstamped calibration" : refused.error());
    }
    {
        qi::VaeTopology wrong_floor = shipped;
        wrong_floor.denorm_elements = 32;
        const auto refused = qi::check_vae_topology(wrong_floor, bind_the_stored_rows(shipped_decoder_rows()));
        check_detail(!refused.has_value(), "a denormalization width that disagrees with the latent width is refused",
                     refused.has_value() ? "the checker ACCEPTED a disagreement" : refused.error());
    }
}

// ---- the live artifact through the production reader (fails closed when the artifact is absent) ----

void gate_real_artifact() {
    // No quiet skip. The registration helper does grant this host-only entry a SKIP_RETURN_CODE of 77
    // (tests/cmake/InfernixTests.cmake:32-36), but that code counts only when the test EXITS 77, and this
    // gate no longer does that for a missing artifact: ctest suppresses stdout on success, so a run that
    // exited 77 without ever opening the shipped .infernix container reported a clean pass with the two
    // pins below unexecuted, which is precisely the blindness the certified VAE
    // topology exists to remove. The harness that certifies the image lane names the artifact through the
    // environment and existence-checks it, so an unset or absent INFERNIX_TEST_ARTIFACT here is a
    // misconfigured run rather than a legitimate host-only run, and it FAILS. The precedent is the
    // sibling suite's artifact gate (tests/models/qwen_image/test_image_artifact_constants.cpp), and the
    // fixture-only gates in main() ran before this point, so a host without the container still gets the
    // reader's and the ledger's verdicts -- it simply cannot also claim the shipped artifact was verified.
    const char* env = std::getenv("INFERNIX_TEST_ARTIFACT");
    if (env == nullptr || env[0] == '\0') {
        check_detail(false, "the artifact-backed gates have an artifact to open",
                     "INFERNIX_TEST_ARTIFACT is not set, so the live container's decoder topology was "
                     "never read -- this suite does not pass while skipping its real work");
        return;
    }
    std::ifstream probe(env, std::ios::binary);
    if (!probe) {
        check_detail(false, "the artifact-backed gates can open the artifact they were named",
                     std::string("INFERNIX_TEST_ARTIFACT names ") + env +
                         ", which is not a readable file, so the live decoder topology was never read");
        return;
    }
    probe.close();

    // Host-only metadata read: the reader opens the container's header lazily; nothing here touches a
    // device or starts a server.
    // Braces, not parentheses: Reader's one-argument constructor would otherwise be parsed as a function
    // declaration and this would not be a reader at all.
    const infernix::artifact::Reader reader{std::filesystem::path(env)};
    const infernix::artifact::Directory& directory = reader.directory();
    if (directory.components.find("vae") == directory.components.end()) {
        // An artifact packed before the converter carried the VAE config must NOT bind: refusing it is
        // the point of the change, so no job is decoded with a compiled decoder geometry.
        std::printf("ok: the live artifact carries no vae component -- the strict reader must refuse it\n");
        return;
    }
    const auto live = qi::read_vae_topology(directory.component("vae").config);
    if (!live.has_value()) {
        check_detail(false, "the live artifact's vae component is accepted", live.error());
        return;
    }
    // Stamp the resident calibration the way the binding does before gating: read_vae_topology leaves the
    // field zero because it is a measurement of THIS build, not a claim of the checkpoint's
    // (artifact_constants.h:147-152), and bind_image_vae is what sets it
    // (src/models/qwen_image/image_weights.cpp:640). Gating the raw reader output instead asked the
    // artifact to declare a number only this build can produce, and the certified 3072 MiB check failed
    // on a topology that was never meant to carry it.
    qi::VaeTopology live_stamped = *live;
    live_stamped.weight_resident_bytes = qi::kQwenImageWeightResidentBytes;
    expect_shipped(live_stamped, "the live artifact");
    const qi::VaeTopology committed = qt::qwen_image_2_1_topology();
    check(live->latent_channels == committed.latent_channels &&
              live->spatial_downscale == committed.spatial_downscale &&
              live->mid_stage.channels_in == committed.mid_stage.channels_in &&
              live->mid_stage.channels_out == committed.mid_stage.channels_out &&
              live->final_channels == committed.final_channels &&
              live->denorm_elements == committed.denorm_elements &&
              live->up_stages == committed.up_stages,
          "the committed document and the live artifact declare one decoder topology");
}

} // namespace

int main() {
    try {
        gate_reader();
        gate_ledger();
        gate_bound_shapes();
        gate_real_artifact();
    } catch (const std::exception& e) {
        std::printf("FAIL: uncaught exception: %s\n", e.what());
        ++g_failures;
    }
    if (g_failures != 0) {
        std::printf("IMAGE_VAE_TOPOLOGY_TEST: FAIL (%d check(s))\n", g_failures);
        return 1;
    }
    std::printf("IMAGE_VAE_TOPOLOGY_TEST: PASS\n");
    return 0;
}
