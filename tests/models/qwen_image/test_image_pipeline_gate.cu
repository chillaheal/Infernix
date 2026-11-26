// Host gate for the production image-job entry (task #13 step 3): bind_image_vae resolves EVERY
// VAE-decoder operand the qualified decode chain names (128 direct-BF16 roles) against the REAL
// artifact's Directory bindings, and ONE Binder carrying the DiT plus the VAE finishes to a plan
// with zero device and pinned bytes -- the job's exact open-time contract. The VAE contract is
// asserted structurally: conv shortcuts exist exactly at the channel-changing first resnets (up
// blocks 2-4), none at blocks 0-1, the artifact's time_conv tensors are deliberately NOT bound
// (dupup3d replaces them), and the largest bound object is the 1152 x 1152 x 3 x 3 conv plane.
// Taking the address of run_image_job links the whole CUDA pipeline entry (image_pipeline.cu) into
// infernix_qwen_image_runtime and proves its header is consumable by host callers. A .cu so the
// entry header's cuda_runtime.h include is assured. The entry is LINKED, never called. Skips (77)
// without INFERNIX_TEST_ARTIFACT / when absent. No GPU, no server.
#include "artifact/reader.h"
#include "models/qwen_image/image_pipeline.h"
#include "models/qwen_image/image_weights.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace infernix;
using namespace infernix::artifact;
using namespace infernix::models::qwen_image;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

void require_eq(std::uint64_t actual, std::uint64_t expected, const std::string& what) {
    if (actual != expected) {
        throw std::runtime_error(what + ": got " + std::to_string(actual) + ", expected " +
                                 std::to_string(expected));
    }
}

} // namespace

int main() {
    try {
        const char* env = std::getenv("INFERNIX_TEST_ARTIFACT");
        if (env == nullptr || std::string_view(env).empty()) {
            std::cout << "INFERNIX_TEST_ARTIFACT unset; skipping image pipeline gate\n";
            return 77;
        }
        const std::filesystem::path path(env);
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            std::cout << "artifact absent; skipping image pipeline gate: " << env << "\n";
            return 77;
        }

        const Reader reader(path);
        Binder binder(reader);

        const VaeWeights vae = bind_image_vae(binder);
        // 128 decoder roles + 98 encoder/quant_conv roles (reference-image path, #75). The
        // artifact's 6 extra encoder rows are time_convs, deliberately unbound like the decoder's.
        require_eq(vae.by_name.size(), 226, "VAE role count");
        require(vae.by_name.count("vae/post_quant_conv.weight") == 1, "post_quant_conv.weight unbound");
        require(vae.by_name.count("vae/decoder.conv_out.weight") == 1, "conv_out.weight unbound");
        for (int b : {2, 3, 4}) {
            require(vae.by_name.count("vae/decoder.up_blocks." + std::to_string(b) +
                                      ".resnets.0.conv_shortcut.weight") == 1,
                    "conv_shortcut missing where the channel count changes");
        }
        for (int b : {0, 1}) {
            require(vae.by_name.count("vae/decoder.up_blocks." + std::to_string(b) +
                                      ".resnets.0.conv_shortcut.weight") == 0,
                    "conv_shortcut bound where the channel count is kept");
        }
        for (const auto& entry : vae.by_name) {
            const std::string& name = entry.first;
            require(name.find("time_conv") == std::string::npos, "time_conv is not part of the route: " + name);
            require(entry.second.residency == Residency::Streamed, name + ": role is not Streamed");
            const auto geometry = binder.reader().geometry(entry.second.binding.parts.front().object);
            require(geometry.format == QType::BF16, name + ": role representation differs from the recipe");
            require(geometry.high_bytes == 0, name + ": role carries a high plane");
        }
        // The decoder's widest operand: the 1152 x 1152 x 3 x 3 mid-block / first-level conv plane.
        require_eq(vae.max_object_bytes, 1152ULL * 1152 * 3 * 3 * 2, "VAE streaming bound");

        const DiTWeights dit = bind_dit(binder);
        require_eq(dit.blocks.size(), 32, "DiT block count");

        // The entry links (image_pipeline.cu compiled into the runtime library) and is callable.
        auto (*entry)(artifact::Reader&, const DiTWeights&, const VaeWeights&, const ImageJobInputs&,
                      cudaStream_t, ImageJobProgressView*,
                      std::function<void(const std::string&)>) = &run_image_job;
        require(entry != nullptr, "run_image_job entry resolved to null");

        // The #17 ledger is exported and readable with no job run: live and peak start at zero and
        // re-baselining at that value is safe. (The zero-after-job gates run on device in the
        // prod-real cert, success and throw path.)
        require_eq(image_device_bytes_live(), 0, "ledger live bytes before any job");
        require_eq(image_device_bytes_peak(), 0, "ledger peak bytes before any job");
        reset_image_device_peak();
        require_eq(image_device_bytes_live(), 0, "ledger live bytes after re-baseline");

        // The job opens its weights with one streamed-only plan: DiT roles + VAE roles, nothing placed.
        const MaterializationPlan plan = std::move(binder).finish();
        require(plan.source == &reader, "plan source is not the artifact Reader");
        require(plan.device_objects.empty() && plan.device_capacity_bytes == 0,
                "image job plan places bytes on device");
        require(plan.pinned_objects.empty() && plan.pinned_capacity_bytes == 0,
                "image job plan pins host bytes");
        // 9 img_in/top-level + 32*9 blocks + 128 VAE-decoder + 98 VAE-encoder/quant_conv roles
        // (#75). The artifact's 6 encoder time_conv objects are unbound and therefore absent here too.
        require_eq(plan.streamed_objects.size(), 9 + 32 * 9 + 128 + 98, "combined streamed object count");

        std::cout << "image pipeline gate passed: vae roles=" << vae.by_name.size()
                  << " vae_max=" << vae.max_object_bytes << " streamed=" << plan.streamed_objects.size()
                  << " device=0 pinned=0 entry=linked\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "image pipeline gate FAILED: " << error.what() << "\n";
        return 1;
    }
}
