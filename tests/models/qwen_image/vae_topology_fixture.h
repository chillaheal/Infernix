// The VAE decoder topology the shipped image family runs on, read from the COMMITTED artifact document
// (tests/convert/fixtures/qwen_image_vae_config.json -- the reference checkpoint's own vae/config.json,
// which the converter packs verbatim as the artifact's "vae" component, tools/convert/qwen_image.py:
// records["vae"] = {"config": _load_config(root / "vae" / "config.json")}) through the PRODUCTION
// reader (src/models/qwen_image/artifact_constants.cpp), not from literals in a test.
//
// Why: the point of the change is that the decoder's stage and channel data comes from the artifact and
// that the VRAM ledger and the CUDA decode graph read the SAME copy of it. A guard carrying its own
// copy of the channel numbers would keep passing on hand-written values while the shipped artifact said
// something else -- the exact failure this lane closes -- and would never notice the ledger and the
// graph disagreeing.
//
// The per-family resident calibration (VaeTopology::weight_resident_bytes) is NOT the reader's: it is
// the code-owned measurement the binding stamps. This helper stamps it exactly the way bind_image_vae
// does, so a guard can never certify a route that could not be bound.
//
// Header-only, host tests only: the reader returns std::expected, which this tree compiles per target at
// C++23 (the CUDA lane stays at C++20 and does not use this header). The target must be registered with
// NEEDS_SOURCE_DIR so the committed document is found from the source tree.

#pragma once

#include "models/qwen_image/artifact_constants.h"
#include "models/qwen_image/image_weights.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace infernix::tests::qwen_image {

// The one path a guard may take a VAE topology from. Never a fallback value: an unreadable or refusing
// document is a hard failure of the guard, because a guard that silently drew its own decoder geometry
// would certify nothing.
[[nodiscard]] inline infernix::models::qwen_image::VaeTopology qwen_image_2_1_topology() {
#ifndef INFERNIX_SOURCE_DIR
    throw std::runtime_error(
        "the VAE topology comes from the committed artifact document, which this target cannot locate: "
        "register it with NEEDS_SOURCE_DIR");
#else
    const std::string path =
        std::string(INFERNIX_SOURCE_DIR) + "/tests/convert/fixtures/qwen_image_vae_config.json";
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open the committed VAE document " + path +
                                 ", which the decoder's topology is read from");
    }
    std::ostringstream text;
    text << in.rdbuf();
    const nlohmann::json document = nlohmann::json::parse(text.str(), nullptr, /*allow_exceptions=*/false);
    if (document.is_discarded()) {
        throw std::runtime_error("the committed VAE document " + path + " is not valid JSON");
    }
    const auto topology = infernix::models::qwen_image::read_vae_topology(document);
    if (!topology.has_value()) {
        throw std::runtime_error("the committed VAE document is refused by the reader: " + topology.error());
    }
    infernix::models::qwen_image::VaeTopology out = *topology;
    out.weight_resident_bytes = infernix::models::qwen_image::kQwenImageWeightResidentBytes;
    return out;
#endif
}

// The committed document itself, for a guard that must hand the reader a mutated copy of what ships.
[[nodiscard]] inline nlohmann::json qwen_image_2_1_vae_document() {
#ifndef INFERNIX_SOURCE_DIR
    throw std::runtime_error("the VAE document cannot be located without NEEDS_SOURCE_DIR");
#else
    const std::string path =
        std::string(INFERNIX_SOURCE_DIR) + "/tests/convert/fixtures/qwen_image_vae_config.json";
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open the committed VAE document " + path);
    }
    std::ostringstream text;
    text << in.rdbuf();
    nlohmann::json document = nlohmann::json::parse(text.str(), nullptr, /*allow_exceptions=*/false);
    if (document.is_discarded()) {
        throw std::runtime_error("the committed VAE document " + path + " is not valid JSON");
    }
    return document;
#endif
}

} // namespace infernix::tests::qwen_image
