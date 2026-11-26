#pragma once

#include "product/config/config_file.h"
#include "serve/serve_options.h"

#include <functional>
#include <string>
#include <string_view>

namespace infernix::serve {

// The serving options of the image lane: the four --image-* flags, their --help block and their
// arity live in this file (serve/image_serve_options.cpp), so an upstream change to
// serve/serve_options.cpp never has to carry image text or image rules. serve_options.cpp reaches
// all three through the typed calls below; there is no registry and no name-driven dispatch.

// The arity entry of an image option, or nullptr when `name` is not one of them. serve_options.cpp
// consults this after its own table, so --image-* keep the command line's arity in --config too.
const product::ConfigOption* image_option_arity(std::string_view name) noexcept;

// The image options' lines of infernix-serve's --help exactly where they are printed: inside
// NETWORKING & RESOURCES, between --response-store-max-mib and --log-stats-interval-ms.
std::string image_help_text();

// Reads one --image-* option into `options`. True when `arg` was one of them and is now parsed,
// false when the caller continues with its own options. `require_value` takes the option's value
// from the caller's argument list (it advances the caller's own cursor) and `parse_nonnegative_int`
// is the caller's integer parser, so both keep their exact diagnostics. A rejected value throws
// std::invalid_argument with the message the option has always answered with.
bool parse_image_option(const std::string& arg,
                        ServeOptions& options,
                        const std::function<const char*(const char*)>& require_value,
                        const std::function<int(const char*, const char*)>& parse_nonnegative_int);

} // namespace infernix::serve
