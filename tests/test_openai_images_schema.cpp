#include "serve/openai_images.h"
#include "serve/request.h"
#include "serve/request_json.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>
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
    failures += check(absent.images.empty(), "image absent by default");

    const OpenAIImagesRequest bare =
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"image", "aGVsbG8="}});
    failures += check(bare.images.size() == 1ULL && bare.images[0] == "aGVsbG8=", "bare base64 captured");

    const OpenAIImagesRequest uri = parse(Json{{"model", "qwen-image"},
                                               {"prompt", "x"},
                                               {"image", "data:image/png;base64,aGVsbG8="}});
    failures += check(uri.images.size() == 1ULL && uri.images[0] == "aGVsbG8=", "data: URI prefix stripped");

    // The family's LIST form, accepted at this build's cap of one (openai_images.h
    // kMaximumImageReferences): the shape is legal upstream, so it parses here; what the route can
    // actually draw is the ledger's decision, and asking for more than that is a count error rather
    // than a silently dropped picture. Every content rule below the shape is the string's own.
    const OpenAIImagesRequest one =
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"image", Json::array({"aGVsbG8="})}});
    failures += check(one.images.size() == 1ULL && one.images[0] == "aGVsbG8=", "array of one captured like the string form");

    const OpenAIImagesRequest one_uri =
        parse(Json{{"model", "qwen-image"},
                   {"prompt", "x"},
                   {"image", Json::array({"data:image/png;base64,aGVsbG8="})}});
    failures += check(one_uri.images.size() == 1ULL && one_uri.images[0] == "aGVsbG8=", "array of one strips the data: URI prefix");

    Json too_many = Json::array();
    for (std::size_t i = 0; i <= kMaximumImageReferences; ++i) too_many.push_back("aGVsbG8=");
    const ApiError over = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"image", too_many}});
    });
    failures += check(over.status == 400 && over.code == "image_count_invalid" && over.param == "image",
                      "a reference past the cap is refused, never silently dropped");

    // More than one reference is what the family is for -- every one of them conditions the same
    // forward -- so the parse keeps them ALL, in the order the client sent them, and strips each
    // one's own data: URI prefix. Dropping or merging here would be the quietest kind of wrong.
    const OpenAIImagesRequest pair =
        parse(Json{{"model", "qwen-image"},
                   {"prompt", "x"},
                   {"image", Json::array({"YWJj", "data:image/png;base64,ZGVm"})}});
    failures += check(pair.images.size() == 2ULL && pair.images[0] == "YWJj" &&
                          pair.images[1] == "ZGVm",
                      "two references are both kept, in the order they were sent");

    // `ratio_follow` is a 1-based name for one of the pictures this request carries. Naming a fourth
    // in a two-picture request is a client mistake; following "whichever one we have" instead would
    // answer a request nobody made, so it is refused rather than clamped.
    const OpenAIImagesRequest follows_second =
        parse(Json{{"model", "qwen-image"},
                   {"prompt", "x"},
                   {"image", Json::array({"YWJj", "ZGVm"})},
                   {"ratio_follow", "<image2>"}});
    failures += check(follows_second.ratio_follow &&
                          *follows_second.ratio_follow == "<image2>",
                      "ratio_follow may name any reference the request carries");

    const ApiError past = api_error([&] {
        parse(Json{{"model", "qwen-image"},
                   {"prompt", "x"},
                   {"image", Json::array({"YWJj", "ZGVm"})},
                   {"ratio_follow", "<image3>"}});
    });
    failures += check(past.status == 400 && past.code == "image_ratio_follow_unsupported" &&
                          past.param == "ratio_follow",
                      "a ratio_follow naming a picture nobody sent is refused, not clamped");

    const ApiError no_brackets = api_error([&] {
        parse(Json{{"model", "qwen-image"},
                   {"prompt", "x"},
                   {"image", "YWJj"},
                   {"ratio_follow", "image1"}});
    });
    failures += check(no_brackets.code == "image_ratio_follow_unsupported",
                      "a tag without its angle brackets is the same refusal");

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
    failures += check(parse(null_image).images.empty(), "null image falls back to text-to-image");

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
    failures += check(edge.images.size() == 1ULL && edge.images[0].size() == limit, "base64 at the pre-gate boundary accepted");
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

int test_prompt_enhance_field() {
    int failures = 0;

    // The enhancer switch is a real parameter of the family, so the parse accepts it and the
    // SERVICE decides whether it can honor it. Refusing the key at the parse would tell a client
    // that the parameter does not exist, which is a different (and less true) answer than
    // "this build has no rewriter installed" -- and the route's whole whitelist exists so a
    // dropped parameter is never mistaken for an honored one.
    const OpenAIImagesRequest absent = parse(base_request());
    failures += check(absent.prompt_enhance == PromptEnhanceMode::Off,
                      "prompt_enhance is off unless asked for");

    const OpenAIImagesRequest on =
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"prompt_enhance", true}});
    failures += check(on.prompt_enhance == PromptEnhanceMode::On,
                      "prompt_enhance true is captured, not refused at the parse");

    const OpenAIImagesRequest off =
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"prompt_enhance", false}});
    failures += check(off.prompt_enhance == PromptEnhanceMode::Off,
                      "prompt_enhance false is captured");

    // The third value is a string, not another boolean: a client that would rather not decide per
    // request sends "auto" and lets the server's rule answer (image_generation_service.cpp
    // wants_prompt_enhancement). The parse accepts it without looking at the instruction, because
    // what auto runs on is a property of the whole request and of what the card can yield.
    const OpenAIImagesRequest automatic =
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"prompt_enhance", "auto"}});
    failures += check(automatic.prompt_enhance == PromptEnhanceMode::Auto,
                      "prompt_enhance \"auto\" is captured, not refused at the parse");

    const ApiError not_bool = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"prompt_enhance", "yes"}});
    });
    failures += check(not_bool.status == 400 && not_bool.param == "prompt_enhance",
                      "prompt_enhance has to be a boolean or \"auto\"");

    const ApiError not_a_switch = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"prompt_enhance", 1}});
    });
    failures += check(not_a_switch.status == 400 && not_a_switch.param == "prompt_enhance",
                      "a number is not a switch either");
    return failures;
}


int test_canvas_keys() {
    int failures = 0;

    const OpenAIImagesRequest none = parse(base_request());
    failures += check(!none.size_requested && !none.wh_ratio && !none.ratio_follow &&
                          none.width == 768 && none.height == 768,
                      "no canvas key: the 768x768 default stands");

    const OpenAIImagesRequest size =
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"size", "544x608"}});
    failures += check(size.size_requested && size.width == 544 && size.height == 608,
                      "an explicit size is recorded as the canvas");

    const OpenAIImagesRequest ratio =
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"wh_ratio", "16:9"}});
    failures += check(ratio.wh_ratio == "16:9" && !ratio.size_requested, "wh_ratio captured");

    const OpenAIImagesRequest follow = parse(Json{{"model", "qwen-image"},
                                                 {"prompt", "x"},
                                                 {"image", "YWJj"},
                                                 {"ratio_follow", "<image1>"}});
    failures += check(follow.ratio_follow == "<image1>",
                      "ratio_follow <image1> names the one reference and is accepted");

    // Exactly one of the three. A precedence would let a client that sent two believe the one it
    // cared about was honored, which is worse than a refusal.
    const ApiError size_and_ratio = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"size", "544x608"}, {"wh_ratio", "16:9"}});
    });
    failures += check(size_and_ratio.status == 400 && size_and_ratio.code == "image_size_conflict",
                      "size plus wh_ratio is refused, not ordered");

    const ApiError size_and_follow = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"size", "544x608"},
                   {"ratio_follow", "<image1>"}});
    });
    failures += check(size_and_follow.code == "image_size_conflict", "size plus ratio_follow refused");

    const ApiError ratio_and_follow = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"wh_ratio", "16:9"},
                   {"ratio_follow", "<image1>"}});
    });
    failures += check(ratio_and_follow.code == "image_size_conflict",
                      "wh_ratio plus ratio_follow refused");

    const ApiError wrong_separator = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"wh_ratio", "16x9"}});
    });
    failures += check(wrong_separator.code == "image_ratio_invalid" &&
                          wrong_separator.param == "wh_ratio",
                      "wh_ratio needs a colon");

    const ApiError zero_side = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"wh_ratio", "0:16"}});
    });
    failures += check(zero_side.code == "image_ratio_invalid", "a zero ratio side is refused");

    const ApiError extreme = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"wh_ratio", "1:200"}});
    });
    failures += check(extreme.code == "image_ratio_invalid", "an aspect past 1:64 is refused");

    // A tag is a NAME, and a name has to resolve: the index is checked against the references the
    // request actually carries. Naming a second picture in a one-picture request, or naming any
    // picture in a request with none, both answer a request nobody made -- and a ratio_follow that
    // is silently ignored is the dropped-parameter failure this module's key whitelist exists to
    // prevent, so neither is a fallback to "follow whatever we have".
    const ApiError second = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"image", "YWJj"},
                   {"ratio_follow", "<image2>"}});
    });
    failures += check(second.code == "image_ratio_follow_unsupported",
                      "ratio_follow naming a picture the request does not carry is refused");

    const ApiError orphan = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"},
                   {"ratio_follow", "<image1>"}});
    });
    failures += check(orphan.code == "image_ratio_follow_unsupported",
                      "ratio_follow with no reference at all is refused, not ignored");

    const ApiError untagged = api_error([&] {
        parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"ratio_follow", "image1"}});
    });
    failures += check(untagged.code == "image_ratio_follow_unsupported",
                      "ratio_follow needs the engine's own <image1> tag");
    return failures;
}

int test_canvas_geometry() {
    int failures = 0;
    auto divisible = [](std::uint32_t v) { return v % 32 == 0 && v >= kMinimumImageDimension; };

    // width/height are in/out, exactly as the service calls it: the request's own fields go in and the
    // canvas comes back out, so "an explicit size is honored" is the early return rather than a branch
    // every caller has to remember.
    const auto canvas = [](OpenAIImagesRequest request, std::uint32_t ref_w, std::uint32_t ref_h) {
        resolve_canvas_geometry(request, ref_w, ref_h, request.width, request.height);
        return std::pair<std::uint32_t, std::uint32_t>{request.width, request.height};
    };

    // A portrait reference must produce a portrait canvas. This is the whole point of the package: the
    // square default conditioned the model on an aspect its target grid did not have.
    const auto [w, h] = canvas(parse(base_request()), 1000, 1500);
    failures += check(h > w && divisible(w) && divisible(h),
                      "a portrait reference gets a portrait canvas");
    const double area = static_cast<double>(w) * static_cast<double>(h);
    failures += check(area > kCanvasTargetArea * 0.8 && area < kCanvasTargetArea * 1.25,
                      "the aspect is filled to the configured area, not grown by it");
    failures += check(std::abs(static_cast<double>(w) / static_cast<double>(h) - 1000.0 / 1500.0) < 0.05,
                      "the reference aspect is preserved to within one snap");

    const auto [lw, lh] = canvas(parse(base_request()), 1600, 900);
    failures += check(lw > lh && divisible(lw) && divisible(lh),
                      "a landscape reference gets a landscape canvas");

    // The same picture, drawn at a size on request, keeps that size: the reference never overrides an
    // order.
    {
        OpenAIImagesRequest ordered  = parse(base_request());
        ordered.width               = 544;
        ordered.height              = 608;
        ordered.size_requested      = true;
        const auto [ow, oh] = canvas(ordered, 1000, 1500);
        failures += check(ow == 544 && oh == 608, "an explicit size is not touched by the reference");
    }

    {
        OpenAIImagesRequest named = parse(base_request());
        named.wh_ratio            = "2:1";
        const auto [nw, nh] = canvas(named, 0, 0);
        failures += check(std::abs(static_cast<double>(nw) / static_cast<double>(nh) - 2.0) < 0.06 &&
                              divisible(nw) && divisible(nh),
                          "wh_ratio holds its ratio at the working area");
    }

    {
        const auto [pw, ph] = canvas(parse(base_request()), 0, 0);
        failures += check(pw == 768 && ph == 768, "text-to-image with no aspect keeps 768x768");
    }

    // An extreme aspect is clamped into the drawable band rather than emitting a degenerate grid.
    const auto [xw, xh] = canvas(parse(base_request()), 4000, 32);
    failures += check(xw <= kMaximumImageDimension && xh >= kMinimumImageDimension && divisible(xw) &&
                          divisible(xh),
                      "an extreme reference aspect stays inside the dimension limits");
    return failures;
}

} // namespace

// The edit route: the generations validation, plus the one rule that a reference must be
// there, plus the multipart form an OpenAI SDK posts translated into the same document.
int test_edit_route() {
    int failures = 0;
    const auto edit_parse = [](Json body) { return parse_openai_images_edit_request(body); };

    const ApiError none = api_error([&] { edit_parse(base_request()); });
    failures += check(none.status == 400 && none.param == "image" &&
                          none.code == "image_reference_required",
                      "an edit with no reference is refused, not drawn from text");
    const ApiError blank = api_error([&] {
        edit_parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"image", ""}});
    });
    failures += check(blank.status == 400 && blank.code == "image_count_invalid",
                      "an empty payload is the generations count error, before the edit rule");

    const OpenAIImagesRequest edited =
        edit_parse(Json{{"model", "qwen-image"}, {"prompt", "make it an owl"},
                        {"image", "YWJj"}});
    failures += check(edited.images.size() == 1ULL && edited.images[0] == "YWJj", "the reference survives the edit parse");
    failures += check(edited.prompt == "make it an owl", "the instruction survives it too");

    const OpenAIImagesRequest wrapped = edit_parse(Json{{"model", "qwen-image"},
                                                     {"prompt", "x"},
                                                     {"image", Json::array({"YWJj"})}});
    failures += check(wrapped.images.size() == 1ULL && wrapped.images[0] == "YWJj", "the family's array-of-one form edits too");

    // A reference does not make a client's other mistakes legal: the canvas rules still apply.
    const ApiError bad_size = api_error([&] {
        edit_parse(Json{{"model", "qwen-image"}, {"prompt", "x"}, {"image", "YWJj"},
                        {"size", "777x777"}});
    });
    failures += check(bad_size.status == 400 && bad_size.code == "image_size_invalid",
                      "an edit validates its canvas exactly like a generation does");

    // ---- the multipart form an OpenAI SDK posts ----------------------------------------
    MultipartEditParts parts;
    parts.fields.emplace_back("model", "qwen-image");
    parts.fields.emplace_back("prompt", "tan the face");
    parts.fields.emplace_back("negative_prompt", "blurry");
    parts.fields.emplace_back("steps", "30");
    parts.fields.emplace_back("seed", "7");
    parts.fields.emplace_back("true_cfg_scale", "1.5");
    parts.files.emplace_back("image", std::string("abc"));
    const Json built = edit_body_from_multipart(parts);
    failures += check(built.at("image") == "YWJj", "the file part becomes the base64 reference");
    failures += check(built.at("steps").is_number() && built.at("steps") == 30 &&
                          built.at("seed").is_number() && built.at("seed") == 7 &&
                          built.at("true_cfg_scale").is_number(),
                      "numeric text parts stay numbers, so the ordinary range checks apply");
    const OpenAIImagesRequest from_form = edit_parse(built);
    failures += check(from_form.steps == 30U && from_form.images.size() == 1ULL && from_form.images[0] == "YWJj" &&
                          from_form.seed.has_value() && *from_form.seed == 7ULL &&
                          from_form.true_cfg_scale > 1.0,
                      "the multipart document parses as an edit request through the shared path");

    // Two pictures in the form are BOTH references -- the family's point, and the opposite of
    // the quiet drop a route that only had one slot used to perform. Past the cap it is a count error
    // that says so, again rather than a picture nobody asked to be discarded.
    MultipartEditParts two;
    two.files.emplace_back("image", std::string("a"));
    two.fields.emplace_back("model", "qwen-image");
    two.fields.emplace_back("prompt", "x");
    two.files.emplace_back("image", std::string("b"));
    const Json both_images = edit_body_from_multipart(two);
    failures += check(both_images.at("image").is_array() &&
                          both_images.at("image").size() == 2ULL &&
                          both_images.at("image")[0] == "YQ==" &&
                          both_images.at("image")[1] == "Yg==",
                      "two file parts are two references, not one that won");
    failures += check(edit_parse(both_images).images.size() == 2ULL,
                      "and both survive the shared parse");

    MultipartEditParts mixed;
    mixed.fields.emplace_back("image", "YWJj");
    mixed.fields.emplace_back("model", "qwen-image");
    mixed.fields.emplace_back("prompt", "x");
    mixed.files.emplace_back("image", std::string("abc"));
    failures += check(edit_parse(edit_body_from_multipart(mixed)).images.size() == 2ULL,
                      "a text reference and a file reference are both references");

    MultipartEditParts over;
    over.fields.emplace_back("model", "qwen-image");
    over.fields.emplace_back("prompt", "x");
    for (std::size_t i = 0; i <= kMaximumImageReferences; ++i) {
        over.files.emplace_back("image", std::string("a"));
    }
    const ApiError too_many = api_error([&] { (void)edit_body_from_multipart(over); });
    failures += check(too_many.status == 400 && too_many.code == "image_count_invalid" &&
                          too_many.param == "image",
                      "a form past the cap is refused, never trimmed");
    MultipartEditParts masked;
    masked.files.emplace_back("mask", std::string("abc"));
    const ApiError mask = api_error([&] { (void)edit_body_from_multipart(masked); });
    failures += check(mask.code == "image_mask_not_supported" && mask.param == "mask",
                      "a mask is refused by name, never ignored");

    MultipartEditParts odd;
    odd.files.emplace_back("picture", std::string("abc"));
    const ApiError other = api_error([&] { (void)edit_body_from_multipart(odd); });
    failures += check(other.code == "image_unsupported_file_field",
                      "an unknown file part is refused, not silently dropped");

    MultipartEditParts text_ref;
    text_ref.fields.emplace_back("image", "YWJj");
    failures += check(edit_body_from_multipart(text_ref).at("image") == "YWJj",
                      "a base64 text part named image is the reference too");

    MultipartEditParts dup;
    dup.fields.emplace_back("prompt", "first");
    dup.fields.emplace_back("prompt", "second");
    failures += check(edit_body_from_multipart(dup).at("prompt") == "first",
                      "a repeated text part is the client's ambiguity, and the first wins");

    MultipartEditParts bad_steps;
    bad_steps.fields.emplace_back("model", "qwen-image");
    bad_steps.fields.emplace_back("prompt", "x");
    bad_steps.fields.emplace_back("steps", "abc");
    const ApiError steps = api_error([&] { edit_parse(edit_body_from_multipart(bad_steps)); });
    failures += check(steps.status == 400 && steps.param == "steps",
                      "a non-numeric steps is the value error, not a type error");

    MultipartEditParts empty;
    empty.fields.emplace_back("model", "qwen-image");
    empty.fields.emplace_back("prompt", "x");
    const ApiError no_ref = api_error([&] { edit_parse(edit_body_from_multipart(empty)); });
    failures += check(no_ref.code == "image_reference_required",
                      "a form with no reference at all is the edit route's own refusal");
    return failures;
}

int main() {
    int failures = 0;
    failures += test_minimal_and_defaults();
    failures += test_prompt_bound();
    failures += test_size();
    failures += test_n_and_response_format();
    failures += test_seed_and_unknown_keys();
    failures += test_steps();
    failures += test_reference_image_field();
    failures += test_edit_route();
    failures += test_prompt_enhance_field();
    failures += test_canvas_keys();
    failures += test_canvas_geometry();
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
