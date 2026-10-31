// M9, THE SHIP GATE: instruction-guided image edit, graded on the picture rather than on a tensor.
//
// Everything before this test compares rows -- the conditioned tower against the checkpoint's own
// forward, one denoising step against a torch dump, the ledger against the pipeline's allocator. None
// of it can say whether the EDIT INSTRUCTION was followed: a route that ignores the prompt entirely
// and reproduces the reference (the failure that started this work, image-edit/evidence/attempt*.png)
// passes every row comparison it is given, because the reference rows are exactly the rows it is
// supposed to carry. This test therefore asks only the question the user asks: did the picture change
// the way the sentence said, and did the thing the sentence said to keep stay?
//
// The fixture is synthetic and built in memory (no photograph, nothing committed): a solid
// RGB(20,40,200) field with a centred 96x96 RGB(220,30,30) square. Two instructions run on it, and
// the assertions are pixel statistics alone -- no VLM, no OCR, no judge:
//   not-a-copy    the output differs from the reference by a wide margin (MAE and SSIM), so a copy
//                 cannot pass; its message names the defect in the words the report needs,
//   subject held  the centred square is still the red it was, within the BF16/VAE round trip,
//   instruction   prompt A turns the field outside the square red, prompt C turns it green -- two
//                 different, mutually exclusive effects, so a generic "changed a lot" cannot pass
//                 either. WHICH instructions are graded here is a measured choice: the reference
//                 pipeline's own pictures are scored by image-edit\ref_edit_behavior.py with these
//                 same numbers, and at 512^2 / 20 steps / CFG 1 with seed 7 the reference itself fails
//                 "Add a solid green bar across the top edge" (mae 12.36, band green 125.7) and "along
//                 the bottom edge" (mae 14.45) -- against the engine's 13.30 / 112.8 on the first of
//                 them. An instruction the checkpoint does not obey is not something the engine may be
//                 failed for (the plan's own premise), so the ship gate asks for the two recolourings
//                 the reference demonstrably performs; the bar cases stay in the control as documented
//                 negatives, printed and not counted.
// The route is the production one end to end: the same host renderers, packer, vision tower,
// conditioned text tower, VAE encoder and DiT loop the serving layer drives (serve/image_text_tower,
// serve/reference_image, models/qwen_image run_image_job), with the seed, steps and guidance the plan
// fixes. It runs the real pipeline on the real GPU, so it must not overlap a live serve.
//
// This is the gate for the `reference_image` capability: the flag stays false until this CTest is
// green on a named artifact, and a red run keeps it false.
//
// Skips (77) without INFERNIX_TEST_ARTIFACT or INFERNIX_QWENVL_PROCESSOR, or when either is absent;
// returns 2 with no CUDA device.

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "models/qwen_image/image_device_util.h"
#include "models/qwen_image/image_job.h"
#include "models/qwen_image/image_pipeline.h"
#include "models/qwen_image/image_weights.h"
#include "models/qwen_image/noise.h"
#include "serve/image_text_tower.h"
#include "serve/reference_image.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::artifact;
using namespace infernix::models::qwen_image;
namespace srv = infernix::serve;

// The plan's fixture, fixed. The reference is 256x256 -- a multiple of 32, so the vision grid is
// exact (8x8 = 64 tokens) and the x4 invariant ties it to the VAE grid (16x16 = 256 tokens) without
// any resize ambiguity -- and the output is the square the chat route actually serves.
constexpr std::uint32_t kRefSide   = 256;
constexpr std::uint32_t kSubject   = 96;   // the centred square, in reference pixels
constexpr std::uint32_t kOutSide   = 512;
constexpr std::uint32_t kSteps     = 20;
constexpr std::uint64_t kSeed      = 7;
constexpr float       kCfgScale  = 1.0F;   // single forward: the instruction must carry it alone

constexpr int kBg[3] = {20, 40, 200};      // the field: blue
constexpr int kSub[3] = {220, 30, 30};     // the subject: red

// The gates, from the plan. They are wide on purpose: this measures semantics (did the instruction
// happen), not precision, and a precision question is the other tests' job.
constexpr double kMinMae        = 25.0;    // 0..255 scale, mean over channels and pixels
constexpr double kMaxSsim       = 0.98;
constexpr double kHoldTolerance = 40.0;    // per channel, around (220,30,30)

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::string env_or(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    return (value == nullptr || std::string_view(value).empty()) ? fallback : std::string(value);
}

// ---- the fixture -------------------------------------------------------------------------------

// Row-major RGB8 of the synthetic reference: the field everywhere, the square in the middle.
std::vector<std::uint8_t> fixture_rgb() {
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(kRefSide) * kRefSide * 3ULL);
    const std::uint32_t lo = (kRefSide - kSubject) / 2ULL;
    const std::uint32_t hi = lo + kSubject;
    for (std::uint32_t y = 0; y < kRefSide; ++y) {
        for (std::uint32_t x = 0; x < kRefSide; ++x) {
            const bool subject = x >= lo && x < hi && y >= lo && y < hi;
            const int*   c     = subject ? kSub : kBg;
            const std::size_t i = (static_cast<std::size_t>(y) * kRefSide + x) * 3ULL;
            rgb[i]     = static_cast<std::uint8_t>(c[0]);
            rgb[i + 1] = static_cast<std::uint8_t>(c[1]);
            rgb[i + 2] = static_cast<std::uint8_t>(c[2]);
        }
    }
    return rgb;
}

// Both planes from one fixture, exactly as prepare_reference_planes would have produced them for an
// opaque source: the VAE plane carries the raw alpha as its fourth channel (never composited), the
// vision plane is the row-major RGB8 copy the Qwen3-VL tower was trained on.
srv::ReferencePlanes fixture_planes() {
    const std::vector<std::uint8_t> rgb = fixture_rgb();
    srv::ReferencePlanes planes;
    planes.vae.height_px    = kRefSide;
    planes.vae.width_px     = kRefSide;
    planes.vision.height_px = kRefSide;
    planes.vision.width_px  = kRefSide;
    planes.vision.rgb       = rgb;
    const std::size_t plane  = static_cast<std::size_t>(kRefSide) * kRefSide;
    planes.vae.pixels.assign(4ULL * plane, 0.0F);
    for (std::size_t p = 0; p < plane; ++p) {
        for (int c = 0; c < 3; ++c) {
            // The reference pipeline's normalization: x / 127.5 - 1.
            planes.vae.pixels[static_cast<std::size_t>(c) * plane + p] =
                static_cast<float>(rgb[p * 3ULL + static_cast<std::size_t>(c)]) / 127.5F - 1.0F;
        }
        planes.vae.pixels[3ULL * plane + p] = 1.0F; // opaque alpha, raw
    }
    return planes;
}

// ---- reading the picture back -------------------------------------------------------------------

// GeneratedImage::channels is [4, height, width] channel-major in the VAE's own [-1,1].
double channel_byte(const GeneratedImage& img, int c, std::uint32_t x, std::uint32_t y) {
    const std::size_t plane = static_cast<std::size_t>(img.height) * img.width;
    const double      v     = img.channels[static_cast<std::size_t>(c) * plane +
                                         static_cast<std::size_t>(y) * img.width + x];
    return (std::clamp(v, -1.0, 1.0) + 1.0) * 127.5;
}

struct Rect {
    std::uint32_t x0, y0, x1, y1; // half-open, output coordinates
};

// The mean of one channel over a rect, on the 0..255 scale.
double mean_byte(const GeneratedImage& img, int c, const Rect& r) {
    double sum = 0.0;
    std::size_t n = 0;
    for (std::uint32_t y = r.y0; y < r.y1; ++y) {
        for (std::uint32_t x = r.x0; x < r.x1; ++x) {
            sum += channel_byte(img, c, x, y);
            ++n;
        }
    }
    return n == 0ULL ? 0.0 : sum / static_cast<double>(n);
}

// The fixture resampled nearest into the output grid. The source is piecewise constant and the scale
// is exact (256 -> 512), so nearest is the same mapping the pipeline's resize performs; the copy
// detector compares against this rather than against a resized PNG it would have to invent.
double fixture_byte_at(std::uint32_t out_x, std::uint32_t out_y, int c) {
    const std::uint32_t x = static_cast<std::uint32_t>(static_cast<double>(out_x) * kRefSide / kOutSide);
    const std::uint32_t y = static_cast<std::uint32_t>(static_cast<double>(out_y) * kRefSide / kOutSide);
    const std::uint32_t lo = (kRefSide - kSubject) / 2ULL;
    const bool subject = x >= lo && x < lo + kSubject && y >= lo && y < lo + kSubject;
    return static_cast<double>((subject ? kSub : kBg)[c]);
}

// Mean absolute error of the output against that resampled reference, and a small local SSIM.
double mae_vs_reference(const GeneratedImage& img) {
    double sum = 0.0;
    for (std::uint32_t y = 0; y < img.height; ++y) {
        for (std::uint32_t x = 0; x < img.width; ++x) {
            for (int c = 0; c < 3; ++c) {
                sum += std::abs(channel_byte(img, c, x, y) - fixture_byte_at(x, y, c));
            }
        }
    }
    return sum / (static_cast<double>(img.height) * img.width * 3.0);
}

// A test-local SSIM: 8x8 windows, luminance, the standard stabilization constants. It is an
// approximation -- no Gaussian weighting, no multiscale -- and it is only asked to tell a
// near-copy (>= 0.999) from a re-rendered picture, which any windowed form does.
double ssim_vs_reference(const GeneratedImage& img) {
    constexpr int   kWin = 8;
    const std::size_t columns = img.width / kWin;
    const std::size_t rows    = img.height / kWin;
    if (columns == 0ULL || rows == 0ULL) { return 0.0; }
    const double c1 = 6.5025, c2 = 58.5225; // (0.01*255)^2, (0.03*255)^2
    double total = 0.0;
    std::size_t windows = 0;
    for (std::size_t wy = 0; wy < rows; ++wy) {
        for (std::size_t wx = 0; wx < columns; ++wx) {
            double sa = 0.0, sb = 0.0, saa = 0.0, sbb = 0.0, sab = 0.0;
            for (int dy = 0; dy < kWin; ++dy) {
                for (int dx = 0; dx < kWin; ++dx) {
                    const std::uint32_t x = static_cast<std::uint32_t>(wx * kWin + dx);
                    const std::uint32_t y = static_cast<std::uint32_t>(wy * kWin + dy);
                    double a = 0.0, b = 0.0;
                    for (int c = 0; c < 3; ++c) {
                        // Rec.601 luma: the structural term cares about brightness, not hue.
                        static constexpr double kW[3] = {0.299, 0.587, 0.114};
                        a += kW[c] * channel_byte(img, c, x, y);
                        b += kW[c] * fixture_byte_at(x, y, c);
                    }
                    sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
                }
            }
            const double n     = static_cast<double>(kWin * kWin);
            const double mu_a  = sa / n, mu_b = sb / n;
            const double var_a = saa / n - mu_a * mu_a;
            const double var_b = sbb / n - mu_b * mu_b;
            const double cov   = sab / n - mu_a * mu_b;
            total += ((2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)) /
                     ((mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2));
            ++windows;
        }
    }
    return windows == 0ULL ? 0.0 : total / static_cast<double>(windows);
}

// ---- one instruction, graded ---------------------------------------------------------------------------------

struct Instruction {
    const char* what;
    const char* prompt;
    // Which pixels assertion (3) reads. Named per instruction instead of recovered by comparing the
    // prompt string, so a case cannot silently fall into the wrong branch by editing its wording.
    const char* probe;
};

void grade(const GeneratedImage& out, const Instruction& ins, std::vector<std::string>& failures) {
    require(out.height == kOutSide && out.width == kOutSide,
            std::string(ins.what) + ": the job returned the wrong geometry");

    const double scale = static_cast<double>(kOutSide) / kRefSide;
    const std::uint32_t lo = static_cast<std::uint32_t>((kRefSide - kSubject) / 2ULL * scale);
    const std::uint32_t hi = lo + static_cast<std::uint32_t>(kSubject * scale);
    // 8 px of margin keeps interpolation and the VAE's own edge response out of the subject window.
    const Rect subject{lo + 8U, lo + 8U, hi - 8U, hi - 8U};
    const Rect outside{4U, 4U, kOutSide - 4U, kOutSide - 4U};

    // (1) not a copy.
    const double mae  = mae_vs_reference(out);
    const double ssim = ssim_vs_reference(out);
    std::cout << "  " << ins.what << ": mae=" << mae << " ssim=" << ssim << "\n";
    if (!(mae > kMinMae && ssim < kMaxSsim)) {
        failures.push_back(std::string(ins.what) +
                           ": output is a copy of the reference; the edit instruction was ignored");
    }

    // (2) the subject the instruction told the model to keep, or that the fixture simply holds.
    double held = 0.0;
    for (int c = 0; c < 3; ++c) {
        const double got   = mean_byte(out, c, subject);
        const double delta = std::abs(got - static_cast<double>(kSub[c]));
        held = std::max(held, delta);
        std::cout << "  " << ins.what << ": subject channel " << c << " mean=" << got
                  << " want=" << kSub[c] << "\n";
    }
    if (!(held <= kHoldTolerance)) {
        failures.push_back(std::string(ins.what) + ": the centred square moved " +
                           std::to_string(held) + " levels from the reference subject");
    }

    // (3) the instruction itself, in the pixels it names. The two probes are the two recolourings the
    // reference control performs (its field red=234.2/blue=9.9 and green=180.7/red=56.1/blue=35.7);
    // the band probe the plan also named is gone, because the reference does not satisfy it either --
    // see the header. Both probes read the SAME two rects: everything outside a frame around the
    // square, minus a 12 px margin where the square's own anti-aliasing reaches.
    Rect field{outside.x0, outside.y0, outside.x1, lo - 12U}; // above the square
    Rect right{hi + 12U, outside.y0, outside.x1, outside.y1}; // right of it
    const double red  = (mean_byte(out, 0, field) + mean_byte(out, 0, right)) / 2.0;
    const double green = (mean_byte(out, 1, field) + mean_byte(out, 1, right)) / 2.0;
    const double blue = (mean_byte(out, 2, field) + mean_byte(out, 2, right)) / 2.0;

    if (std::string_view(ins.probe) == std::string_view("field_red")) {
        std::cout << "  " << ins.what << ": field red=" << red << " blue=" << blue << "\n";
        if (!(red > 180.0 && blue < 80.0)) {
            failures.push_back(std::string(ins.what) + ": the background is not red "
                               "(red=" + std::to_string(red) + ", blue=" + std::to_string(blue) + ")");
        }
    } else if (std::string_view(ins.probe) == std::string_view("field_green")) {
        std::cout << "  " << ins.what << ": field green=" << green << " red=" << red
                  << " blue=" << blue << "\n";
        if (!(green > 150.0 && red < 80.0 && blue < 80.0)) {
            failures.push_back(std::string(ins.what) + ": the background is not green (green=" +
                               std::to_string(green) + ", red=" + std::to_string(red) +
                               ", blue=" + std::to_string(blue) + ")");
        }
    } else {
        // A probe nobody asked for is a bug in the table, not a grading decision: refusing here is
        // what keeps a typo in instructions[] from grading nothing and reporting a pass.
        failures.push_back(std::string(ins.what) + ": unknown probe '" + ins.probe + "'");
    }
}

} // namespace

int main() {
    try {
        const std::string artifact_path  = env_or("INFERNIX_TEST_ARTIFACT", "");
        const std::string processor_dir  = env_or("INFERNIX_QWENVL_PROCESSOR", "");
        if (artifact_path.empty() || processor_dir.empty()) {
            std::cout << "INFERNIX_TEST_ARTIFACT / INFERNIX_QWENVL_PROCESSOR unset; skipping the "
                         "behavioural edit gate\n";
            return 77;
        }
        std::error_code ec;
        if (!std::filesystem::exists(artifact_path, ec) || !std::filesystem::exists(processor_dir, ec)) {
            std::cout << "artifact or processor absent; skipping the behavioural edit gate\n";
            return 77;
        }
        if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
            std::cout << "no CUDA device; the behavioural edit gate is runnable\n";
            return 2;
        }

        Reader reader(artifact_path);
        Binder binder(reader);
        DiTWeights dit = bind_dit(binder);
        VaeWeights vae = bind_image_vae(binder);
        const VaeDenorm denorm = read_vae_denorm(binder);
        (void)std::move(binder).finish();

        // The same host halves the serving layer uses, so the instruction reaches the DiT through the
        // route it will really travel: the raw ti2i template, the genuine tokenizer, the processor's
        // pad expansion, the vision tower, the conditioned 36-layer forward.
        const auto render      = srv::make_tower_reference_prompt_renderer(processor_dir);
        const auto vision_rows = srv::make_tower_vision_rows_provider(artifact_path);
        const auto conditioned = srv::make_tower_conditioned_embeds_provider(artifact_path);

        const srv::ReferencePlanes planes = fixture_planes();
        const std::uint64_t        vision_tokens = srv::vision_token_count(kRefSide, kRefSide);
        require(vision_tokens * 4ULL == (static_cast<std::uint64_t>(kRefSide) / 16ULL) *
                                           (kRefSide / 16ULL),
                "the fixture breaks the x4 vision/VAE invariant");

        const ImageGeometry geom = ImageGeometry::derive(kOutSide, kOutSide);
        const Schedule      sched = ImageJob{kOutSide, kOutSide, kSteps}.schedule(SchedulerConfig{});

        // The reference latents do not depend on the instruction, but the encode is re-run per job so
        // each job owns its own ledger window and the byte-zero gate attributes a leak to one prompt.
        // The two instructions the reference control proved it can carry out at these settings; the
        // provenance of this choice (and the numbers of the two it cannot) is in this file's header.
        const std::vector<Instruction> instructions = {
            {"A background to red",
             "Change the background to solid red. Keep the red square in the center.", "field_red"},
            {"C background to green",
             "Change the background to solid green. Keep the red square in the center.", "field_green"},
        };

        require(image_device_bytes_live() == 0, "the ledger holds bytes before any job");
        std::vector<std::string> failures;
        for (const auto& ins : instructions) {
            const auto rendered = render(ins.prompt, {vision_tokens});
            const auto vision   = vision_rows(planes.vision);
            const auto embeds   = conditioned(rendered, vision);
            // image_pad_mask is one entry per KEPT ROW of the whole post-drop sequence, 1 on the rows
            // that carry the picture (image_text_tower.h: "[S - drop_rows]"), so its length belongs
            // against the rendered token count. The picture gate is the pad count below. Comparing the
            // length with vision_tokens made every job die here before a single pixel was generated
            // (blackout 36: "kept 93 rows for 64 vision tokens" -- 93 = 29 text rows + 64 pads).
            require(embeds.image_pad_mask.size() == rendered.tokens(),
                    std::string(ins.what) + ": the conditioned tower kept " +
                        std::to_string(embeds.image_pad_mask.size()) + " mask rows for " +
                        std::to_string(rendered.tokens()) + " rendered tokens");
            std::size_t pads = 0;
            for (const std::uint8_t m : embeds.image_pad_mask) { pads += m; }
            require(pads == vision_tokens, std::string(ins.what) + ": only " + std::to_string(pads) +
                                               " of " + std::to_string(vision_tokens) +
                                               " kept rows carry the picture");

            EncodedReference reference;
            {
                ReferenceEncodeInputs in;
                in.pixels       = planes.vae.pixels;
                in.latents_mean = denorm.mean;
                in.latents_std  = denorm.std;
                in.height_px    = planes.vae.height_px;
                in.width_px     = planes.vae.width_px;
                cudaStream_t encode_stream = nullptr;
                if (cudaStreamCreate(&encode_stream) != cudaSuccess) {
                    throw std::runtime_error("no encode stream");
                }
                reference = encode_reference_image(reader, vae, in, encode_stream);
                cudaStreamDestroy(encode_stream);
            }
            require(image_device_bytes_live() == 0,
                    std::string(ins.what) + ": the VAE encode left image device bytes live");
            require(reference.height_lat == kRefSide / 16U && reference.width_lat == kRefSide / 16U,
                    std::string(ins.what) + ": the reference encoded to the wrong latent grid");

            ImageJobInputs inputs;
            inputs.prompt_embeds          = embeds.prompt_embeds;
            inputs.prompt_image_pad_mask  = embeds.image_pad_mask;
            inputs.ref_latents            = reference.latents;
            inputs.ref_grids.emplace_back(reference.height_lat, reference.width_lat);
            inputs.timesteps     = sched.timesteps;
            inputs.sigmas        = sched.sigmas;
            inputs.latents_mean  = denorm.mean;
            inputs.latents_std   = denorm.std;
            inputs.height_px     = geom.height;
            inputs.width_px      = geom.width;
            inputs.true_cfg_scale = kCfgScale;
            {
                std::vector<float> noise(static_cast<std::size_t>(geom.tokens) * dit.dims.image_in);
                fill_normal(kSeed, noise);
                inputs.init_latent = bits_of(noise);
            }

            cudaStream_t stream = nullptr;
            if (cudaStreamCreate(&stream) != cudaSuccess) { throw std::runtime_error("no image stream"); }
            GeneratedImage out;
            try {
                out = run_image_job(reader, dit, vae, inputs, stream);
            } catch (...) {
                cudaStreamDestroy(stream);
                require(image_device_bytes_live() == 0,
                        std::string(ins.what) + ": the refused job left image device bytes live");
                throw;
            }
            cudaStreamDestroy(stream);
            require(image_device_bytes_live() == 0,
                    std::string(ins.what) + ": the job left image device bytes live");

            grade(out, ins, failures);
        }

        for (const auto& f : failures) { std::cout << "  FAIL " << f << "\n"; }
        if (!failures.empty()) {
            std::cout << "EDIT_BEHAVIOR_FAIL " << failures.front() << "\n";
            return 1;
        }
        std::cout << "EDIT_BEHAVIOR_OK\n";

        // ---- the manual look (plan Task 10 step 5): opt-in, outside the gate ----------------------
        // Nothing below is a CTest assertion. It renders ONE REFERENCE THE CALLER NAMES through the
        // identical route -- the same prepare_reference_planes the HTTP route runs, so the decode, the
        // aspect snap to the 32-grid and the bilinear resize are the shipped ones, not a fixture -- so
        // a person can look at the picture a real edit produced. It runs only when INFERNIX_EDIT_PORTRAIT
        // is a readable file AND INFERNIX_EDIT_OUTDIR is set, so the gate that ships the capability is
        // byte-for-byte the run above; the photograph never enters the repository.
        const std::string portrait = env_or("INFERNIX_EDIT_PORTRAIT", "");
        const std::string out_dir  = env_or("INFERNIX_EDIT_OUTDIR", "");
        if (!portrait.empty() && !out_dir.empty()) {
            std::ifstream in(portrait, std::ios::binary);
            const std::vector<std::uint8_t> bytes{(std::istreambuf_iterator<char>(in)),
                                                  std::istreambuf_iterator<char>()};
            require(in.good() || !bytes.empty(), "portrait: the named file is empty or unreadable");
            static constexpr char kB64[] =
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            std::string b64;
            b64.reserve((bytes.size() + 2ULL) / 3ULL * 4ULL);
            for (std::size_t i = 0; i < bytes.size(); i += 3ULL) {
                const auto at = [&](std::size_t k) {
                    return i + k < bytes.size() ? static_cast<std::uint32_t>(bytes[i + k]) : 0U;
                };
                const std::uint32_t v = (at(0) << 16U) | (at(1) << 8U) | at(2);
                b64.push_back(kB64[(v >> 18U) & 63U]);
                b64.push_back(kB64[(v >> 12U) & 63U]);
                b64.push_back(i + 1ULL < bytes.size() ? kB64[(v >> 6U) & 63U] : '=');
                b64.push_back(i + 2ULL < bytes.size() ? kB64[v & 63U] : '=');
            }

            const srv::ReferencePlanes real = srv::prepare_reference_planes(b64);
            const std::uint64_t ref_vision =
                srv::vision_token_count(real.vision.height_px, real.vision.width_px);
            const std::string prompt = env_or("INFERNIX_EDIT_PROMPT",
                                              "Change the background to solid green. Keep the person "
                                              "standing in the center unchanged.");
            std::cout << "PORTRAIT source=" << portrait << " plane=" << real.vae.width_px << "x"
                      << real.vae.height_px << " vision=" << ref_vision << " prompt=" << prompt << "\n";

            const auto rendered = render(prompt, {ref_vision});
            const auto vision   = vision_rows(real.vision);
            const auto embeds   = conditioned(rendered, vision);
            std::size_t pads = 0;
            for (const std::uint8_t m : embeds.image_pad_mask) { pads += m; }
            require(pads == ref_vision,
                    "portrait: only " + std::to_string(pads) + " of " + std::to_string(ref_vision) +
                        " kept rows carry the picture");

            EncodedReference reference;
            {
                ReferenceEncodeInputs enc_in;
                enc_in.pixels       = real.vae.pixels;
                enc_in.latents_mean = denorm.mean;
                enc_in.latents_std  = denorm.std;
                enc_in.height_px    = real.vae.height_px;
                enc_in.width_px     = real.vae.width_px;
                cudaStream_t encode_stream = nullptr;
                if (cudaStreamCreate(&encode_stream) != cudaSuccess) {
                    throw std::runtime_error("portrait: no encode stream");
                }
                reference = encode_reference_image(reader, vae, enc_in, encode_stream);
                cudaStreamDestroy(encode_stream);
            }
            require(image_device_bytes_live() == 0,
                    "portrait: the VAE encode left image device bytes live");

            const ImageGeometry pgeom = ImageGeometry::derive(real.vae.height_px, real.vae.width_px);
            const Schedule psched =
                ImageJob{pgeom.height, pgeom.width, kSteps}.schedule(SchedulerConfig{});
            ImageJobInputs inputs;
            inputs.prompt_embeds         = embeds.prompt_embeds;
            inputs.prompt_image_pad_mask = embeds.image_pad_mask;
            inputs.ref_latents           = reference.latents;
            inputs.ref_grids.emplace_back(reference.height_lat, reference.width_lat);
            inputs.timesteps    = psched.timesteps;
            inputs.sigmas       = psched.sigmas;
            inputs.latents_mean = denorm.mean;
            inputs.latents_std  = denorm.std;
            inputs.height_px    = pgeom.height;
            inputs.width_px     = pgeom.width;
            inputs.true_cfg_scale = kCfgScale;
            {
                std::vector<float> noise(static_cast<std::size_t>(pgeom.tokens) * dit.dims.image_in);
                fill_normal(kSeed, noise);
                inputs.init_latent = bits_of(noise);
            }
            cudaStream_t stream = nullptr;
            if (cudaStreamCreate(&stream) != cudaSuccess) {
                throw std::runtime_error("portrait: no image stream");
            }
            GeneratedImage out;
            try {
                out = run_image_job(reader, dit, vae, inputs, stream);
            } catch (...) {
                cudaStreamDestroy(stream);
                require(image_device_bytes_live() == 0,
                        "portrait: the refused job left image device bytes live");
                throw;
            }
            cudaStreamDestroy(stream);
            require(image_device_bytes_live() == 0,
                    "portrait: the job left image device bytes live");

            const std::string ppm = out_dir + "\\portrait_edit.ppm";
            std::ofstream file(ppm, std::ios::binary);
            require(static_cast<bool>(file), "portrait: cannot open " + ppm);
            file << "P6\n" << out.width << ' ' << out.height << "\n255\n";
            std::array<char, 3> rgb{};
            for (std::uint32_t y = 0; y < out.height; ++y) {
                for (std::uint32_t x = 0; x < out.width; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        rgb[static_cast<std::size_t>(c)] =
                            static_cast<char>(std::lround(channel_byte(out, c, x, y)));
                    }
                    file.write(rgb.data(), 3);
                }
            }
            file.close();
            require(static_cast<bool>(file), "portrait: the write of " + ppm + " failed");
            std::cout << "PORTRAIT_EDIT_WRITTEN " << ppm << " " << out.width << "x" << out.height
                      << " steps=" << kSteps << " seed=" << kSeed << " cfg=" << kCfgScale << "\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cout << "EDIT_BEHAVIOR_FAIL " << error.what() << "\n";
        return 1;
    }
}
