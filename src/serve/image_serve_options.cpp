#include "serve/image_serve_options.h"

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>

namespace infernix::serve {
namespace {

// What --image-* take after the artifact path: every one of them takes a value and none
// repeats. The argument loop and --config of serve/serve_options.cpp accept nothing
// outside its own table plus these entries, which is how a config file gets exactly the
// command line's options.
constexpr std::array<product::ConfigOption, 4> kImageServeOptionArities{
    product::ConfigOption{"--image-artifact"},
    product::ConfigOption{"--image-text-processor"},
    product::ConfigOption{"--image-prompt-enhancer-url"},
    product::ConfigOption{"--image-retention-days"},
};

} // namespace

const product::ConfigOption* image_option_arity(std::string_view name) noexcept {
    for (const product::ConfigOption& option : kImageServeOptionArities) {
        if (option.name == name) { return &option; }
    }
    return nullptr;
}

std::string image_help_text() {
    // Byte for byte the block infernix-serve has always printed at this point of its
    // --help, so the option text and the arity table stay one contract.
    return
           "  --image-artifact PATH      .infernix image artifact for the synchronous\n"
           "                             POST /v1/images/generations (default: the route\n"
           "                             answers 503 image_generation_not_configured)\n"
           "  --image-text-processor PATH\n"
           "                             HF Qwen3-VL processor directory (tokenizer.json,\n"
           "                             tokenizer_config.json, chat_template.jinja) that\n"
           "                             turns the raw prompt into the tower conditioning\n"
           "                             (default: the route stays 503 not_configured)\n"
           "  --image-prompt-enhancer-url HOST:PORT\n"
           "                             loopback address of the Qwen/Qwen-Image-2.1-PE-I2I\n"
           "                             worker (tools/prompt_enhancer/pe_worker.py) that\n"
           "                             rewrites an edit instruction before the DiT sees\n"
           "                             it; accept 127.0.0.1 or localhost, and without it\n"
           "                             prompt_enhance answers 503 unavailable\n"
           "  --image-retention-days N   days a generated picture stays on disk, which is\n"
           "                             how long its /v1/images/<id> URL answers; after\n"
           "                             that the file is swept and the URL is 404\n"
           "                             (default 14; 0 keeps every picture forever)\n";
}

bool parse_image_option(const std::string& arg,
                        ServeOptions& options,
                        const std::function<const char*(const char*)>& require_value,
                        const std::function<int(const char*, const char*)>&
                            parse_nonnegative_int) {
    // The lane's own options: the order the command line has always accepted them in.
    if (arg == "--image-artifact") {
        options.image_artifact_path = require_value("--image-artifact");
        if (options.image_artifact_path.empty()) {
            throw std::invalid_argument("--image-artifact must not be empty");
        }
    } else if (arg == "--image-retention-days") {
        options.image_retention_days = parse_nonnegative_int(
            require_value("--image-retention-days"), "image-retention-days");
        if (options.image_retention_days > 3650) {
            throw std::invalid_argument("--image-retention-days must be at most 3650");
        }
    } else if (arg == "--image-text-processor") {
        options.image_text_processor_path = require_value("--image-text-processor");
        if (options.image_text_processor_path.empty()) {
            throw std::invalid_argument("--image-text-processor must not be empty");
        }
    } else if (arg == "--image-prompt-enhancer-url") {
        options.image_prompt_enhancer_url = require_value("--image-prompt-enhancer-url");
        // Loopback only, and only the two literal spellings of it: the sidecar receives this
        // route's raw pictures and asks the card for about 20 GB, so a routable address is a
        // misconfiguration rather than a deployment choice. Resolving a name to decide is not the
        // option parser's job, so no name other than these two is accepted.
        std::string host = options.image_prompt_enhancer_url;
        if (const auto scheme = host.find("://"); scheme != std::string::npos) {
            host = host.substr(scheme + 3);
        }
        const auto colon = host.rfind(':');
        const std::string name = colon == std::string::npos ? host : host.substr(0, colon);
        if (options.image_prompt_enhancer_url.empty() || colon == std::string::npos ||
            host.find('/') != std::string::npos ||
            (name != "127.0.0.1" && name != "localhost")) {
            throw std::invalid_argument(
                "--image-prompt-enhancer-url must be 127.0.0.1:PORT or localhost:PORT");
        }
    } else {
        return false;
    }
    return true;
}

} // namespace infernix::serve
