// infernix-image: on-demand image generation sidecar (IMAGE_ADR section 0b). This revision loads and
// reports a Qwen-Image-2.1 v3 artifact's topology; the DiT/scheduler/VAE pipeline attaches here as
// it lands. It is a separate executable so it never forces infernix-serve to relink.
#include "artifact/reader.h"

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using namespace infernix;
using namespace infernix::artifact;

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: infernix-image <artifact.infernix>\n";
        return 2;
    }
    try {
        const Reader reader(argv[1]);
        const Directory& directory = reader.directory();

        std::size_t tensors = 0;
        std::size_t resources = 0;
        std::size_t fp8_row_scale = 0;
        for (const auto& object : directory.objects) {
            if (const auto* tensor = std::get_if<TensorObject>(&object)) {
                ++tensors;
                if (tensor->format == "fp8_e4m3fn_row_bf16") {
                    ++fp8_row_scale;
                }
            } else {
                ++resources;
            }
        }

        std::cout << "infernix-image artifact loaded\n"
                  << "  bytes=" << reader.file_bytes() << "\n"
                  << "  components=" << directory.components.size() << "\n";
        for (const auto& [name, component] : directory.components) {
            std::cout << "    " << name << " resources=" << component.resources.size() << "\n";
        }
        std::cout << "  objects=" << directory.objects.size() << " tensors=" << tensors
                  << " resources=" << resources << " fp8_rowscale=" << fp8_row_scale << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "infernix-image failed: " << error.what() << "\n";
        return 1;
    }
}
