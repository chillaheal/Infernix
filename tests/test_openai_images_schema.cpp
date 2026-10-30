#include "serve/openai_images.h"
#include "serve/request.h"
#include "serve/request_json.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Json = infernix::serve::RequestJson;
using namespace infernix::serve;

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

template <typename Function>
ApiError api_error(Function&& function) {
    try {
        function();
    } catch (const ApiException& exception) { return exception.error(); }
    return ApiError{.status = 0, .message = "no exception"};
}

Json base_request() { return Json{{"model", "qwen-image"}, {"prompt", "a red cube"}}; }

OpenAIImagesRequest parse(Json body) { return parse_openai_images_request(body); }

int test_minimal_and_defaults() {
    int failures = 0;
    const OpenAIImagesRequest request = parse(base_request());
    failures += check(request.model == "qwen-image", "model captured");
    failures += check(request.prompt == "a red cube", "prompt captured");
    failures += check(request.width == 768 && request.height == 768, "size defaults to 768x768");
    failures += check(!request.seed.has_value(), "seed absent by default");
    failures += check(request.user.empty(), "user empty when absent");

    const ApiError no_object = api_error([&] { parse(Json::array()); });
    failures += check(no_object.status == 400, "non-object body is a 400");

    const ApiError no_model = api_error([&] { parse(Json{{"prompt", "x"}}); });
    failures += check(no_model.param == "model" && no_model.status == 400, "missing model rejected");

    const ApiError no_prompt = api_error([&] { parse(Json{{"model", "qwen-image"}}); });
    failures += check(no_prompt.param == "prompt", "missing prompt rejected");

    const ApiError empty_prompt = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", ""}});
    });
    failures += check(empty_prompt.param == "prompt", "empty prompt rejected");

    Json null_optional = base_request();
    null_optional["seed"]  = nullptr;
    null_optional["size"]  = nullptr;
    null_optional["n"]     = nullptr;
    const OpenAIImagesRequest defaults = parse(null_optional);
    failures += check(!defaults.seed.has_value() && defaults.width == 768,
                      "null optional keys fall back to defaults");
    return failures;
}

int test_prompt_bound() {
    int failures = 0;
    const ApiError too_long = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", std::string(kMaximumImagePromptBytes + 1, 'a')}});
    });
    failures += check(too_long.code == "string_above_max_length" && too_long.param == "prompt",
                      "over-long prompt rejected with OpenAI code");

    const OpenAIImagesRequest edge =
        parse(Json{{"model", "qwen-image"}, {"prompt", std::string(kMaximumImagePromptBytes, 'a')}});
    failures += check(edge.prompt.size() == kMaximumImagePromptBytes, "max-length prompt accepted");
    return failures;
}

int test_size() {
    int failures              = 0;
    const OpenAIImagesRequest square = parse(Json{{"model", "qwen-image"},
                                                  {"prompt", "x"},
                                                  {"size", "512x512"}});
    failures += check(square.width == 512 && square.height == 512, "square size parsed");

    const OpenAIImagesRequest wide = parse(Json{{"model", "qwen-image"},
                                                {"prompt", "x"},
                                                {"size", "1024x768"}});
    failures += check(wide.width == 1024 && wide.height == 768, "non-square size parsed");

    const ApiError no_x =
        api_error([&] { parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"size", "1024"}}); });
    failures += check(no_x.code == "image_size_invalid" && no_x.param == "size",
                      "size without separator rejected");

    const ApiError not_mult =
        api_error([&] { parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"size", "100x100"}}); });
    failures += check(not_mult.code == "image_size_invalid", "size not multiple of 32 rejected");

    const ApiError too_big = api_error(
        [&] { parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"size", "4096x4096"}}); });
    failures += check(too_big.code == "image_size_invalid", "oversized dimension rejected");

    const ApiError too_small =
        api_error([&] { parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"size", "16x16"}}); });
    failures += check(too_small.code == "image_size_invalid", "undersized dimension rejected");

    const ApiError junk =
        api_error([&] { parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"size", "axb"}}); });
    failures += check(junk.code == "image_size_invalid", "non-numeric size rejected");
    return failures;
}

int test_n_and_response_format() {
    int failures = 0;
    const OpenAIImagesRequest one = parse(Json{{"model", "qwen-image"},
                                               {"prompt", "x"},
                                               {"n", 1},
                                               {"response_format", "b64_json"},
                                               {"user", "u-1"}});
    failures += check(one.user == "u-1", "user captured");

    const ApiError many =
        api_error([&] { parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"n", 2}}); });
    failures += check(many.code == "image_n_not_supported" && many.param == "n",
                      "n greater than one rejected");

    const ApiError zero_n =
        api_error([&] { parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"n", 0}}); });
    failures += check(zero_n.param == "n", "non-positive n rejected");

    const OpenAIImagesRequest url = parse(Json{{"model", "qwen-image"},
                                               {"prompt", "x"},
                                               {"response_format", "url"}});
    failures += check(url.response_format == ImageResponseFormat::Url, "url response_format accepted");
    failures += check(one.response_format == ImageResponseFormat::Base64,
                      "explicit b64_json is the base64 form");
    failures += check(parse(base_request()).response_format == ImageResponseFormat::Base64,
                      "absent response_format is the base64 form");

    const ApiError bad_format = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"response_format", "svg"}});
    });
    failures += check(bad_format.code == "response_format_not_supported",
                      "unknown response_format rejected");
    return failures;
}

int test_seed_and_unknown_keys() {
    int failures = 0;
    const OpenAIImagesRequest seeded =
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"seed", 12345}});
    failures += check(seeded.seed && *seeded.seed == 12345, "seed parsed");

    const ApiError negative =
        api_error([&] { parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"seed", -5}}); });
    failures += check(negative.param == "seed", "negative seed rejected");

    for (std::string_view key : {"quality", "style", "background", "input_fidelity", "moderation"}) {
        Json body = base_request();
        body[std::string(key)] = "whatever";
        const ApiError error   = api_error([&] { parse(body); });
        failures += check(error.code == "unsupported_parameter" && error.param == key,
                          std::string("unsupported ") + std::string(key) + " rejected");
    }
    return failures;
}

int test_steps() {
    int failures = 0;
    const OpenAIImagesRequest defaulted = parse(base_request());
    failures += check(defaulted.steps == 40, "steps defaults to 40");

    const OpenAIImagesRequest few =
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"steps", 8}});
    failures += check(few.steps == 8, "explicit steps captured");

    const OpenAIImagesRequest floor =
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"steps", kMinimumImageSteps}});
    failures += check(floor.steps == kMinimumImageSteps, "minimum steps accepted");

    const OpenAIImagesRequest ceil =
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"steps", kMaximumImageSteps}});
    failures += check(ceil.steps == kMaximumImageSteps, "maximum steps accepted");

    // The schedule divides by (steps - 1), so a single step is rejected before it reaches the pipeline.
    const ApiError one =
        api_error([&] { parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"steps", 1}}); });
    failures += check(one.code == "image_steps_invalid" && one.param == "steps",
                      "steps below the floor rejected");

    const ApiError zero =
        api_error([&] { parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"steps", 0}}); });
    failures += check(zero.param == "steps", "zero steps rejected");

    const ApiError over = api_error(
        [&] { parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"steps", 101}}); });
    failures += check(over.code == "image_steps_invalid", "steps above the ceiling rejected");

    const ApiError negative =
        api_error([&] { parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"steps", -1}}); });
    failures += check(negative.param == "steps", "negative steps rejected");

    const ApiError not_int =
        api_error([&] { parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"steps", 10.5}}); });
    failures += check(not_int.param == "steps", "non-integer steps rejected");
    return failures;
}

int test_reference_image_field() {
    int failures = 0;
    const OpenAIImagesRequest absent = parse(base_request());
    failures += check(absent.image.empty(), "image absent by default");

    const OpenAIImagesRequest bare =
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"image", "aGVsbG8="}});
    failures += check(bare.image == "aGVsbG8=", "bare base64 captured");

    const OpenAIImagesRequest uri = parse(Json{{"model", "qwen-image"},
                                               {"prompt", "x"},
                                               {"image", "data:image/png;base64,aGVsbG8="}});
    failures += check(uri.image == "aGVsbG8=", "data: URI prefix stripped");

    // The family's LIST form, accepted at this build's cap of one (openai_images.h
    // kMaximumImageReferences): the shape is legal upstream, so it parses here; what the route can
    // actually draw is the ledger's decision, and asking for more than that is a count error rather
    // than a silently dropped picture. Every content rule below the shape is the string's own.
    const OpenAIImagesRequest one =
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"image", Json::array({"aGVsbG8="})}});
    failures += check(one.image == "aGVsbG8=", "array of one captured like the string form");

    const OpenAIImagesRequest one_uri =
        parse(Json{{"model", "qwen-image"},
                   {"prompt", "x"},
                   {"image", Json::array({"data:image/png;base64,aGVsbG8="})}});
    failures += check(one_uri.image == "aGVsbG8=", "array of one strips the data: URI prefix");

    Json too_many = Json::array();
    for (std::size_t i = 0; i <= kMaximumImageReferences; ++i) too_many.push_back("aGVsbG8=");
    const ApiError over = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"image", too_many}});
    });
    failures += check(over.status == 400 && over.code == "image_count_invalid" && over.param == "image",
                      "a reference past the cap is refused, never silently dropped");

    const ApiError none = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"image", Json::array()}});
    });
    failures += check(none.code == "image_count_invalid", "an empty array is a count error, not a 500");

    const ApiError blank = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"image", ""}});
    });
    failures += check(blank.code == "image_count_invalid", "an empty image payload is a count error");

    const ApiError array_item = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"image", Json::array({7})}});
    });
    failures += check(array_item.code == "image_count_invalid", "a non-string array item is a count error");

    const ApiError url_in_array = api_error([&] {
        parse(Json{{"model", "qwen-image"},
                   {"prompt", "x"},
                   {"image", Json::array({"https://example.com/a.png"})}});
    });
    failures += check(url_in_array.code == "image_url_not_supported",
                      "the URL rule applies inside the array form too (no server-side fetch)");

    Json null_image = base_request();
    null_image["image"] = nullptr;
    failures += check(parse(null_image).image.empty(), "null image falls back to text-to-image");

    const ApiError not_string =
        api_error([&] { parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"image", 7}}); });
    failures += check(not_string.status == 400 && not_string.param == "image",
                      "non-string image rejected");

    const ApiError url = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"image", "https://example.com/a.png"}});
    });
    failures += check(url.code == "image_url_not_supported" && url.param == "image",
                      "image URL rejected, never fetched");

    const ApiError url_http = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"image", "http://example.com/a.png"}});
    });
    failures += check(url_http.code == "image_url_not_supported", "plain-http image URL rejected");

    const ApiError not_b64 = api_error([&] {
        parse(Json{{"model", "qwen-image"},
                   {"prompt", "x"},
                   {"image", "data:image/png,%89PNG"}});
    });
    failures += check(not_b64.code == "image_data_uri_not_base64" && not_b64.param == "image",
                      "non-base64 data: URI rejected");

    const std::size_t limit   = (kMaximumImageInputBytes / 3) * 4 + 4;
    const ApiError oversized = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"image", std::string(limit + 1, 'A')}});
    });
    failures += check(oversized.code == "image_too_large" && oversized.param == "image",
                      "over-long base64 rejected before decode");

    const OpenAIImagesRequest edge = parse(
        Json{{"model", "qwen-image"}, {"prompt", "x"}, {"image", std::string(limit, 'A')}});
    failures += check(edge.image.size() == limit, "base64 at the pre-gate boundary accepted");
    return failures;
}

int test_true_cfg_fields() {
    int failures = 0;

    const OpenAIImagesRequest plain = parse(Json{{"model", "qwen-image"}, {"prompt", "x"}});
    failures += check(plain.true_cfg_scale == 1.0 && plain.negative_prompt.empty(),
                      "no CFG fields -> single forward at scale 1");

    const OpenAIImagesRequest steered = parse(Json{{"model", "qwen-image"},
                                                   {"prompt", "x"},
                                                   {"true_cfg_scale", 3.5},
                                                   {"negative_prompt", "blurry"}});
    failures += check(steered.true_cfg_scale == 3.5 && steered.negative_prompt == "blurry",
                      "true_cfg_scale + negative_prompt parsed");

    const OpenAIImagesRequest boundary = parse(Json{{"model", "qwen-image"},
                                                    {"prompt", "x"},
                                                    {"true_cfg_scale", kMaximumImageCFGScale},
                                                    {"negative_prompt", "b"}});
    failures += check(boundary.true_cfg_scale == kMaximumImageCFGScale,
                      "the top of the CFG range is accepted");

    // Steering without something to steer away from would silently mean "no steering".
    const ApiError no_negative = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"true_cfg_scale", 4}});
    });
    failures += check(no_negative.code == "image_negative_prompt_required" &&
                          no_negative.param == "negative_prompt",
                      "true_cfg_scale above 1 without negative_prompt rejected");

    // Scale 1 asks for no steering, so a negative prompt there is accepted and simply unused.
    const OpenAIImagesRequest unused = parse(
        Json{{"model", "qwen-image"}, {"prompt", "x"}, {"true_cfg_scale", 1}, {"negative_prompt", "b"}});
    failures += check(unused.true_cfg_scale == 1.0 && unused.negative_prompt == "b",
                      "negative_prompt with scale 1 accepted (no second forward)");

    for (const double out_of_range : {0.5, 20.5}) {
        const ApiError bad = api_error([&] {
            parse(Json{{"model", "qwen-image"},
                       {"prompt", "x"},
                       {"true_cfg_scale", out_of_range},
                       {"negative_prompt", "b"}});
        });
        failures += check(bad.code == "image_cfg_scale_invalid" && bad.param == "true_cfg_scale",
                          "CFG scale outside [1, 20] rejected: " + std::to_string(out_of_range));
    }

    const ApiError not_number = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"true_cfg_scale", "3.5"}});
    });
    failures += check(not_number.status == 400 && not_number.param == "true_cfg_scale",
                      "non-numeric true_cfg_scale rejected");

    const ApiError negative_not_string = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"negative_prompt", 3}});
    });
    failures += check(negative_not_string.status == 400 && negative_not_string.param == "negative_prompt",
                      "non-string negative_prompt rejected");

    const ApiError negative_too_long = api_error([&] {
        parse(Json{{"model", "qwen-image"},
                   {"prompt", "x"},
                   {"true_cfg_scale", 2},
                   {"negative_prompt", std::string(kMaximumImagePromptBytes + 1, 'a')}});
    });
    failures += check(negative_too_long.code == "string_above_max_length" &&
                          negative_too_long.param == "negative_prompt",
                      "over-long negative_prompt rejected at the prompt bound");
    return failures;
}

int test_response_builder() {
    int failures = 0;
    const RequestJson with_revision =
        make_openai_image_response(1700000000, {"QUJD", "REVG"}, "a nicer prompt");
    failures += check(with_revision.at("created").get<std::int64_t>() == 1700000000,
                      "created set");
    failures += check(with_revision.at("data").size() == 2, "one entry per image");
    failures += check(with_revision.at("data")[0].at("b64_json").get<std::string>() == "QUJD",
                      "b64_json emitted");
    failures += check(with_revision.at("data")[0].at("revised_prompt").get<std::string>() ==
                          "a nicer prompt",
                      "revised_prompt emitted when present");

    const RequestJson plain = make_openai_image_response(1, {"AAA"});
    failures += check(!plain.at("data")[0].contains("revised_prompt"),
                      "revised_prompt omitted when absent");

    const RequestJson empty = make_openai_image_response(2, {});
    failures += check(empty.at("data").is_array() && empty.at("data").empty(),
                      "empty image list yields empty data array");
    return failures;
}

int test_url_response_and_artifact_id() {
    int failures          = 0;
    const std::string url = "http://127.0.0.1:8080/v1/images/img-1700000000-1";

    const RequestJson delivered = make_openai_image_url_response(1700000000, {url});
    failures += check(delivered.at("created").get<std::int64_t>() == 1700000000,
                      "url response carries created");
    failures += check(delivered.at("data")[0].at("url").get<std::string>() == url,
                      "url is handed back verbatim");
    failures += check(!delivered.at("data")[0].contains("b64_json"),
                      "url response carries no inline payload");
    failures += check(make_openai_image_url_response(3, {}).at("data").empty(),
                      "the url form keeps the empty-data shape");

    // The artifact id is the only name a GET may resolve, so any other shape must be refused rather
    // than normalised: an id that could contain a separator would become a path.
    for (const std::string_view bad : {"",
                                       "img-1",
                                       "img-1-",
                                       "img-1-2-3",
                                       "img-a-1",
                                       "img-1-2.png",
                                       "img-1-2/../x",
                                       "img- 1-2",
                                       "images/img-1-2",
                                       "GENERATIONS"}) {
        failures += check(!is_image_artifact_id(bad), "non-artifact id refused: [" + std::string(bad) + "]");
    }
    for (const std::string_view good : {"img-1-1", "img-0-0", "img-1770000000-42"}) {
        failures += check(is_image_artifact_id(good), "artifact id accepted: " + std::string(good));
    }

    failures += check(image_model_id_from_artifact("out/Qwen-Image-2.1-FP8-v3.infernix") ==
                          "Qwen-Image-2.1-FP8-v3",
                      "the artifact stem becomes the image model id");
    failures += check(image_model_id_from_artifact("/data/models/x.infernix") == "x",
                      "the id comes from the file name, not its directory");
    failures += check(image_model_id_from_artifact("").empty(), "no artifact means no id");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_minimal_and_defaults();
    failures += test_prompt_bound();
    failures += test_size();
    failures += test_n_and_response_format();
    failures += test_seed_and_unknown_keys();
    failures += test_steps();
    failures += test_reference_image_field();
    failures += test_true_cfg_fields();
    failures += test_response_builder();
    failures += test_url_response_and_artifact_id();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
