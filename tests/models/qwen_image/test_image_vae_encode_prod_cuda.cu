// GPU production-route qualification of the #75 reference-image ENCODER: encode_reference_image()
// runs the real Qwen-Image-2.1 VAE encoder chain on device (BF16) for the exact reference-prep
// geometry the serving route will feed it, and its packed [tokens,64] BF16 latents are compared
// against an independent diffusers FP32 oracle of the same graph (image-spike/oracle/
// dump_ref_encode.py: AutoencoderKLQwenImage21.encode -> latent_dist.mode -> latents_mean/std
// normalize -> token-major pack, run on the artifact's vae/ subfolder).
//
// Two dumps, both from the same deterministic 700x460 RGBA source:
//   oracle_ref_encode/       -- pipeline geometry: calculate_dimensions(1024^2, AR) -> 1248x832
//                               pixels -> 52x78 latent -> 4056 tokens (the real job shape)
//   oracle_ref_encode_small/ -- small geometry 160x96 -> 6x10 -> 60 tokens (localizes an op defect
//                               that a full-geometry pass would hide, and runs everywhere)
//
// Tolerance (a-priori, fixed before the first run, never narrowed to pass): rel-Frobenius < 5e-2 on
// the packed latents for BOTH dumps. This matches the VAE-decoder production precedent: a ~125-op
// deep BF16 device chain vs FP32, whose measured error sits at O(1e-3..1e-2); a real op/layout
// defect is O(0.1..10). Skips (77) without INFERNIX_TEST_ARTIFACT / INFERNIX_REF_ENCODE_ORACLE;
// returns 2 (runnable) with no CUDA device.

#include "artifact/reader.h"
#include "models/qwen_image/image_pipeline.h"
#include "models/qwen_image/image_weights.h"
#include "models/qwen_image/qwen_image_oracle_ref.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace qo = qioracle;
namespace qi = infernix::models::qwen_image;
using namespace infernix::artifact;

namespace {

float from_bf16(std::uint16_t h) {
    std::uint32_t x = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

// One geometry leg: read the dump, encode on device, compare packed latents. Returns the rel-Fro.
double run_leg(const std::string& dir, Reader& reader, const qi::VaeWeights& vae,
               std::vector<std::string>& notes) {
    const qo::NpyArray px = qo::read_npy(qo::join(dir, "ref_pixels.npy"));
    if (px.shape.size() != 5 || px.shape[0] != 1 || px.shape[1] != 4 || px.shape[2] != 1) {
        throw std::runtime_error(dir + ": ref_pixels.npy is not [1,4,1,H,W]");
    }
    const std::uint32_t H = static_cast<std::uint32_t>(px.shape[3]);
    const std::uint32_t W = static_cast<std::uint32_t>(px.shape[4]);

    qi::ReferenceEncodeInputs in;
    in.pixels = px.floats();
    in.height_px = H;
    in.width_px = W;
    in.latents_mean = qo::read_npy(qo::join(dir, "latents_mean.npy")).floats();
    in.latents_std = qo::read_npy(qo::join(dir, "latents_std.npy")).floats();
    if (in.latents_mean.size() != 64 || in.latents_std.size() != 64) {
        throw std::runtime_error(dir + ": latents_mean/std are not the 64 production channels");
    }

    const qi::EncodedReference enc = qi::encode_reference_image(reader, vae, in, 0);
    if (enc.height_lat * 16 != H || enc.width_lat * 16 != W) {
        throw std::runtime_error(dir + ": latent grid does not match pixels/16");
    }
    const std::size_t tokens = static_cast<std::size_t>(enc.height_lat) * enc.width_lat;
    if (enc.latents.size() != tokens * 64) {
        throw std::runtime_error(dir + ": packed encoder output is not [tokens,64]");
    }

    const qo::NpyArray ref = qo::read_npy(qo::join(dir, "ref_lat_packed.npy"));
    if (ref.shape.size() != 3 || ref.shape[0] != 1 || ref.shape[1] != tokens || ref.shape[2] != 64) {
        throw std::runtime_error(dir + ": oracle ref_lat_packed.npy is not [1,tokens,64]");
    }
    std::vector<float> got(enc.latents.size());
    for (std::size_t i = 0; i < got.size(); ++i) got[i] = from_bf16(enc.latents[i]);

    char line[192];
    std::snprintf(line, sizeof(line), "%ux%u -> %ux%u latents, %zu tokens", W, H,
                  enc.width_lat, enc.height_lat, tokens);
    notes.push_back(dir + ": " + line);
    return qo::rel_fro(got, ref.floats());
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("INFERNIX_TEST_ARTIFACT");
    const char* oracle_dir = std::getenv("INFERNIX_REF_ENCODE_ORACLE");
    if (artifact_path == nullptr || oracle_dir == nullptr ||
        !std::filesystem::exists(artifact_path)) {
        std::printf("IMAGE_VAE_ENCODE_PROD_TEST: SKIP (artifact or oracle dumps absent)\n");
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("IMAGE_VAE_ENCODE_PROD_TEST: no CUDA device\n");
        return 2;
    }

    const double tol = 5e-2;  // a-priori, never narrowed
    try {
        Reader reader(artifact_path);
        Binder binder(reader);
        const qi::VaeWeights vae = qi::bind_image_vae(binder);

        const std::string base(oracle_dir);
        // INFERNIX_REF_ENCODE_LEG runs one geometry only: the pipeline dump needs a device that is
        // not shared with a live server, the small dump runs anywhere.
        const char* only = std::getenv("INFERNIX_REF_ENCODE_LEG");
        bool ok = true;
        for (const char* sub : {"oracle_ref_encode", "oracle_ref_encode_small"}) {
            const std::string name(sub);
            if (only != nullptr && name != std::string(only) &&
                name != "oracle_ref_encode_" + std::string(only)) {
                std::printf("leg %s: skipped by INFERNIX_REF_ENCODE_LEG\n", sub);
                continue;
            }
            std::vector<std::string> notes;
            const double rel = run_leg(qo::join(base, sub), reader, vae, notes);
            for (const auto& n : notes) std::printf("stage %s rel_fro=%.3e\n", n.c_str(), rel);
            std::fflush(stdout);
            if (!(rel < tol)) ok = false;
        }
        std::printf("IMAGE_VAE_ENCODE_PROD_TEST: %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("IMAGE_VAE_ENCODE_PROD_TEST: EXCEPTION %s\n", e.what());
        return 1;
    }
}
