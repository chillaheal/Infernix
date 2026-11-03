// POST /v1/images/generations and POST /v1/images/edits: the synchronous image routes. Wire parsing
// and the model value-equality gate run before the ImageGenerationService, which owns the interlude,
// the job lifecycle and the failure mapping; the response is the OpenAI images object.
//
// The two routes are one handler with one flag: they differ in exactly one rule (an edit requires its
// reference), and everything else - parse order, the model gate, the 503 until the pipeline is
// attached, the response shape - must not be able to drift between them. The multipart half of the
// edit contract is translated here, into the same JSON document the JSON route parses, so a validation
// rule cannot exist on one wire shape and be missing from the other.
#include "serve/http_server.h"

#include "serve/chat_image_tool.h" // prompt_enhance_word: the same word the client-facing attempt uses
#include "serve/http_transport.h"
#include "serve/image_generation_service.h"
#include "serve/openai_common.h"
#include "serve/openai_images.h"
#include "serve/request_validation.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <exception>
#include <string>
#include <span>
#include <vector>
#include <string_view>
#include <system_error>

namespace infernix::serve {
namespace {

[[nodiscard]] bool is_unsigned_integer(std::string_view text) {
    return !text.empty() &&
           std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; });
}

// The file part's bytes are the picture; the parser speaks base64, which is also what the JSON route
// takes, so the two shapes converge here rather than two pages deeper in.
[[nodiscard]] std::string bytes_to_base64(const std::string& bytes) {
    return base64_encode_bytes(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
}

} // namespace

RequestJson edit_body_from_multipart(const MultipartEditParts& parts) {
    RequestJson body = RequestJson::object();

    // Every reference the form carries, in the order it arrived. The two halves of a multipart body
    // are separate containers in the wire library, so the order is text parts first and then file
    // parts; what has to survive is the SET and each picture's own bytes, which is what the model
    // reads. Collapsing them to the last one -- which is what a one-slot route did -- is a picture
    // quietly dropped, and a client that sent two did not ask for one.
    std::vector<std::string> references;

    // Text parts, in the order the client sent them. A repeated name is the client's ambiguity and the
    // first occurrence wins - the same choice a JSON object makes when it cannot hold two keys.
    for (const auto& [name, value] : parts.fields) {
        if (body.contains(name)) { continue; }
        if (name == "image") {
            // Some clients post the reference as a text field holding the base64 rather than as a file.
            // It is the same picture; refusing the shape because it arrived in the other half of the
            // form would be a wire rule with no reason behind it.
            if (references.size() >= kMaximumImageReferences) {
                bad_request("image accepts at most " +
                                std::to_string(kMaximumImageReferences) + " reference images",
                            "image", "image_count_invalid");
            }
            references.push_back(value);
            continue;
        }
        // Numbers stay numbers, so a client that sends steps=abc gets the ordinary value error from the
        // parser instead of a type error that hides what it meant.
        if ((name == "n" || name == "seed" || name == "steps") && is_unsigned_integer(value)) {
            std::uint64_t parsed = 0;
            const auto result    = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (result.ec == std::errc{} && result.ptr == value.data() + value.size()) {
                body[name] = parsed;
                continue;
            }
        }
        if (name == "true_cfg_scale") {
            try {
                std::size_t used   = 0;
                const double scale = std::stod(value, &used);
                if (used == value.size()) {
                    body[name] = scale;
                    continue;
                }
            } catch (const std::exception&) {
                // Not a number: fall through and let the parser report it as one.
            }
        }
        body[name] = value;
    }

    // File parts. `image` is a reference. `mask` gets its own error rather than being ignored: a
    // masked edit answered with an unmasked picture is the dropped-parameter failure this module's key
    // whitelist exists to prevent, and there is no mask input in this pipeline to honor it with.
    for (const auto& [name, bytes] : parts.files) {
        if (name == "mask") {
            bad_request("this image pipeline has no mask input; edit by instruction", "mask",
                        "image_mask_not_supported");
        }
        if (name != "image") {
            bad_request("unsupported file field: " + name, name, "image_unsupported_file_field");
        }
        if (references.size() >= kMaximumImageReferences) {
            bad_request("image accepts at most " +
                            std::to_string(kMaximumImageReferences) + " reference images",
                        "image", "image_count_invalid");
        }
        references.push_back(bytes_to_base64(bytes));
    }

    // One reference goes out as the plain string a single-picture client sends; more goes out as the
    // family's list form. Both are shapes the JSON route already parses, so the two wire shapes stay
    // one contract rather than two rules that can drift. No reference at all is left ABSENT on
    // purpose: the "this route is an edit" refusal belongs to parse_openai_images_edit_request, which
    // gives it its own error code.
    if (references.size() == 1) {
        body["image"] = references.front();
    } else if (!references.empty()) {
        RequestJson list = RequestJson::array();
        for (const std::string& one : references) { list.push_back(one); }
        body["image"] = std::move(list);
    }
    return body;
}

RequestJson HttpServer::run_image_job(const OpenAIImagesRequest& request,
                                      std::string_view artifact_url_base, std::string_view source,
                                      std::string_view protocol, std::string correlation) {
    ImageJobLogRecord record;
    record.source       = std::string(source);
    record.protocol     = std::string(protocol);
    record.correlation  = std::move(correlation);
    record.model        = request.model;
    record.size         = std::to_string(request.width) + "x" + std::to_string(request.height);
    record.steps        = request.steps;
    record.seed         = request.seed;
    record.image_count  = request.images.size();
    record.prompt_bytes = request.prompt.size();
    // Sizes only. A reference is the user's photograph and up to 16 MB of base64; a log line is not
    // where it belongs, and it is exactly what a log outlives a request by.
    record.image_bytes.reserve(request.images.size());
    for (const std::string& reference : request.images) { record.image_bytes.push_back(reference.size()); }
    record.prompt_enhance = prompt_enhance_word(request.prompt_enhance);

    const auto started = std::chrono::steady_clock::now();
    const auto elapsed = [started] {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - started)
            .count();
    };
    try {
        RequestJson response = images_->generate(request, unix_time_now(), artifact_url_base);
        record.outcome     = "succeeded";
        record.duration_ms = elapsed();
        if (const auto field = response.find("prompt_enhance_outcome");
            field != response.end() && field->is_string()) {
            record.enhancer_outcome = field->get<std::string>();
        }
        // The OpenAI images object names a picture by URL alone, so the stored id comes back off the
        // URL's last segment -- the same reading the chat tool does, from the same builder.
        if (const auto data = response.find("data"); data != response.end() && data->is_array() &&
                                                     !data->empty()) {
            if (const auto url = data->front().find("url");
                url != data->front().end() && url->is_string()) {
                const std::string value = url->get<std::string>();
                const auto slash        = value.find_last_of('/');
                if (slash != std::string::npos) { record.job_id = value.substr(slash + 1); }
            }
        }
        request_jsonl_.write_image_job(record);
        return response;
    } catch (const ApiException& exception) {
        record.duration_ms = elapsed();
        record.code        = exception.error().code;
        record.message     = exception.error().message;
        record.http_status = exception.error().status;
        record.outcome     = exception.error().status >= 500 ? "failed" : "refused";
        record.stage       = "service";
        request_jsonl_.write_image_job(record);
        throw;
    } catch (const std::exception& exception) {
        record.duration_ms = elapsed();
        record.outcome     = "failed";
        record.code        = "image_generation_failed";
        record.message     = exception.what();
        record.stage       = "service";
        request_jsonl_.write_image_job(record);
        throw;
    }
}

void HttpServer::handle_images_generations(const httplib::Request& req, httplib::Response& res) {
    handle_images(req, res, /*require_reference=*/false);
}

void HttpServer::handle_images_edits(const httplib::Request& req, httplib::Response& res) {
    handle_images(req, res, /*require_reference=*/true);
}

void HttpServer::handle_images(const httplib::Request& req, httplib::Response& res,
                               bool require_reference) {
    OpenAIImagesRequest request;
    const std::string_view route = require_reference ? "/v1/images/edits" : "/v1/images/generations";
    try {
        if (require_reference && req.is_multipart_form_data()) {
            MultipartEditParts parts;
            for (const auto& field : req.form.fields) {
                parts.fields.emplace_back(field.first, field.second.content);
            }
            for (const auto& file : req.form.files) {
                parts.files.emplace_back(file.first, file.second.content);
            }
            request = parse_openai_images_edit_request(edit_body_from_multipart(parts));
        } else if (require_reference) {
            request = parse_openai_images_edit_request(parse_json_body(req));
        } else {
            request = parse_openai_images_request(parse_json_body(req));
        }
        // Either public id names this process: the image resource's own id, or the chat id every
        // client already sends (one process, one routeable image pipeline).
        validate_openai_model(request.model, {public_model_id_, image_model_id()});
    } catch (const ApiException& exception) {
        // Refused before the service ever saw it, which is the one failure run_image_job cannot
        // witness -- and the one the operator asked about most, because a canvas the parser will not
        // take (image_size_invalid) disappears without a trace while the picture never gets drawn.
        ImageJobLogRecord record;
        record.source   = "images_route";
        record.protocol = std::string(route);
        record.outcome  = exception.error().status >= 500 ? "failed" : "refused";
        record.stage    = "parse";
        record.code     = exception.error().code;
        record.message  = exception.error().message;
        record.http_status = exception.error().status;
        request_jsonl_.write_image_job(record);
        write_openai_error(res, exception.error());
        return;
    }
    if (images_ == nullptr) {
        const ApiError not_configured{.status  = 503,
                                      .type    = "server_error",
                                      .message = "the image pipeline is not configured",
                                      .code    = "image_generation_not_configured"};
        ImageJobLogRecord record;
        record.source      = "images_route";
        record.protocol    = std::string(route);
        record.model       = request.model;
        record.size        = std::to_string(request.width) + "x" + std::to_string(request.height);
        record.steps       = request.steps;
        record.seed        = request.seed;
        record.image_count = request.images.size();
        record.outcome     = "not_configured";
        record.stage       = "configured";
        record.code        = not_configured.code;
        record.message     = not_configured.message;
        record.http_status = not_configured.status;
        request_jsonl_.write_image_job(record);
        write_openai_error(res, not_configured);
        return;
    }
    try {
        const RequestJson response =
            run_image_job(request, artifact_url_base(req, options_), "images_route", route);
        res.set_content(response.dump(-1, ' ', false, RequestJson::error_handler_t::replace),
                        "application/json");
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
    }
}

} // namespace infernix::serve
