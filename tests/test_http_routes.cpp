#include "infernix/ops/attn_dit_joint.h"
#include "serve/chat_image_tool.h"
#include "serve/http_server.h"
#include "serve/image_generation_service.h"
#include "serve/openai_images.h"
#include "serve/request.h"
#include "serve/request_json.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <iostream>
#include <string>
#include <thread>

namespace {

using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    using infernix::serve::api_route_pattern;
    using infernix::serve::canonical_api_path;

    // An Anthropic SDK given a base URL ending in /v1 adds one more /v1; only that one is removed.
    failures += check(canonical_api_path("/v1/v1/messages") == "/v1/messages" &&
                          canonical_api_path("/v1/v1/models/qwen") == "/v1/models/qwen",
                      "a doubled /v1 prefix must name the /v1 endpoint");
    failures += check(canonical_api_path("/v1/messages") == "/v1/messages" &&
                          canonical_api_path("/v1/v1/v1/messages") == "/v1/v1/messages" &&
                          canonical_api_path("/v1/v1") == "/v1/v1" &&
                          canonical_api_path("/health") == "/health",
                      "only a single doubled /v1 prefix may be removed");

    // Errors on a doubled path keep the envelope and request-id header of the endpoint it names.
    infernix::serve::ServeOptions options;
    httplib::Request missing_messages;
    missing_messages.path = "/v1/v1/messages/missing";
    httplib::Response missing_response;
    missing_response.status = 404;
    const auto missing_result =
        infernix::serve::handle_unrendered_http_error(options, missing_messages, missing_response);
    failures += check(missing_result == httplib::Server::HandlerResponse::Handled &&
                          !missing_response.body.empty() &&
                          Json::parse(missing_response.body).at("type") == "error" &&
                          missing_response.has_header("request-id") &&
                          !missing_response.has_header("x-request-id"),
                      "a doubled Anthropic path must keep the Anthropic error envelope");
    httplib::Request oversized_responses;
    oversized_responses.path = "/v1/v1/responses";
    httplib::Response oversized_response;
    oversized_response.status = 413;
    const auto oversized_result = infernix::serve::handle_unrendered_http_error(
        options, oversized_responses, oversized_response);
    failures += check(oversized_result == httplib::Server::HandlerResponse::Handled &&
                          Json::parse(oversized_response.body).at("error").at("code") ==
                              "request_too_large" &&
                          oversized_response.has_header("x-request-id"),
                      "a doubled OpenAI path must keep the OpenAI error envelope");

    // Route patterns reach the handler under either prefix and keep the endpoint's own captures.
    httplib::Server server;
    server.Post(api_route_pattern("/messages"),
                [](const httplib::Request&, httplib::Response& res) {
                    res.set_content("messages", "text/plain");
                });
    server.Get(api_route_pattern(R"(/models/(.+))"),
               [](const httplib::Request& req, httplib::Response& res) {
                   const std::string id = req.matches.size() > 1 ? req.matches[1].str() : "";
                   res.set_content(id, "text/plain");
               });
    // The artifact capture excludes separators on purpose: a model lookup's greedy (.+) would hand the
    // handler a multi-segment name for a file.
    server.Get(api_route_pattern(R"(/images/([^/]+))"),
               [](const httplib::Request& req, httplib::Response& res) {
                   const std::string id = req.matches.size() > 1 ? req.matches[1].str() : "";
                   res.set_content(id, "text/plain");
               });
    // The edit route is a POST of its own, not a flag on the artifact GET: forgetting the
    // reference has to be a refusal, and a refusal needs a handler of its own.
    server.Post(api_route_pattern("/images/edits"),
                [](const httplib::Request&, httplib::Response& res) {
                    res.set_content("edits", "text/plain");
                });
    const int port = server.bind_to_any_port("127.0.0.1");
    failures += check(port > 0, "test server could not bind a loopback port");
    if (port > 0) {
        std::thread listener([&server] { server.listen_after_bind(); });
        server.wait_until_ready();
        httplib::Client client("127.0.0.1", port);
        const auto post_status = [&client](const char* path) {
            const httplib::Result result = client.Post(path, "{}", "application/json");
            return result ? result->status : -1;
        };
        failures += check(post_status("/v1/messages") == 200 &&
                              post_status("/v1/v1/messages") == 200,
                          "the messages route must answer under /v1 and /v1/v1");
        failures += check(post_status("/messages") == 404 &&
                              post_status("/v1/v1/v1/messages") == 404 &&
                              post_status("/v2/v1/messages") == 404,
                          "only the /v1 and single doubled /v1 prefixes may reach a route");
        failures += check(post_status("/v1/images/edits") == 200 &&
                              post_status("/v1/v1/images/edits") == 200 &&
                              post_status("/images/edits") == 404,
                          "the edit route answers under /v1 and one doubled /v1, and nowhere else");
        const httplib::Result model = client.Get("/v1/v1/models/qwen3.8-27b");
        failures += check(model && model->status == 200 && model->body == "qwen3.8-27b",
                          "a doubled prefix must not shift the route's capture groups");
        const httplib::Result artifact = client.Get("/v1/v1/images/img-1770000000-1");
        failures += check(artifact && artifact->status == 200 && artifact->body == "img-1770000000-1",
                          "the artifact route answers under both prefixes with its own id");
        const httplib::Result nested = client.Get("/v1/images/generations/x");
        failures += check(!nested || nested->status == 404,
                          "a name with a separator never reaches the artifact handler");
        server.stop();
        listener.join();
    }

    // The capability advertisement is read off the options, so it can never promise a surface the
    // process does not have.
    const Json plain =
        Json::parse(infernix::serve::make_capabilities_json(options, "qwen3.8-27b", {}));
    failures += check(plain.at("object") == "infernix.capabilities" &&
                          plain.at("chat_model") == "qwen3.8-27b" && plain.at("image_generation").is_null(),
                      "without the image options the capability says so instead of guessing");
    infernix::serve::ServeOptions image_options;
    image_options.enable_vision = true;
    const Json configured = Json::parse(
        infernix::serve::make_capabilities_json(image_options, "qwen3.8-27b", "Qwen-Image-2.1-FP8-v3"));
    failures += check(configured.at("vision_input") == true &&
                          configured.at("image_generation").at("model") == "Qwen-Image-2.1-FP8-v3" &&
                          configured.at("image_generation").at("artifact_endpoint") ==
                              "/v1/images/<id>" &&
                          configured.at("image_generation").at("response_formats") ==
                              Json::array({"b64_json", "url"}) &&
                          configured.at("image_generation").at("automatic_from_chat") == true &&
                          configured.at("image_generation").at("chat_tool") == "generate_image" &&
                          configured.at("image_generation").at("chat_max_images_per_request") == 4 &&
                          configured.at("image_generation").at("reference_image") == true,
                      "the image capability reports its id, artifact route, formats, the chat tool a "
                      "client would otherwise have to run itself, and the edit capability the behavioral "
                      "gate certified");
    // The prompt bound travels in the capability document too: a client that reads only it can
    // write a long instruction knowing both what survives and what the failure looks like.
    failures += check(configured.at("image_generation").at("prompt").at("max_bytes") ==
                          static_cast<int>(infernix::serve::kMaximumImagePromptBytes) &&
                          configured.at("image_generation").at("prompt").at("truncated") == false &&
                          configured.at("image_generation").at("prompt").at("too_long")
                                  .get<std::string>().find("image_prompt_too_long") != std::string::npos,
                      "the capability states the prompt byte bound, that nothing is truncated, and the "
                      "error a too-long prompt produces");
    // The other half of sizing, and the half a client cannot infer: the ceiling the canvas shares with
    // the words belongs to the model, not to the card, so freeing VRAM never raises it. Without this the
    // two limits look like one and a client keeps asking the machine for memory it cannot spend.
    failures += check(
        configured.at("image_generation").at("canvas").at("joint_sequence").at("ceiling_tokens") ==
            static_cast<int>(infernix::ops::kAttnDitJointMaxTokens) &&
            configured.at("image_generation").at("canvas").at("joint_sequence").at("counts")
                    .get<std::string>().find("(W/16)x(H/16)") != std::string::npos,
        "the capability names the joint sequence ceiling and how a canvas is counted against it");
    // Steering needs TWO fields and the document used to name only one of them: a client that read
    // nothing else could ask for true_cfg_scale 3.5 and get a 400 it had no way to predict. The
    // refusal's own code travels here so the pair is discoverable, not just the bound.
    failures += check(
        configured.at("image_generation").at("negative_prompt").at("type") == "string" &&
            configured.at("image_generation").at("negative_prompt").at("max_bytes") ==
                static_cast<int>(infernix::serve::kMaximumImagePromptBytes) &&
            configured.at("image_generation").at("negative_prompt").at("required_when") ==
                "true_cfg_scale above 1" &&
            configured.at("image_generation").at("negative_prompt").at("omitted_with_scale_above_1")
                    .get<std::string>().find("image_negative_prompt_required") != std::string::npos,
        "the capability names the negative prompt, its bound, when it is required, and the refusal "
        "a steer without it produces");
    // A keepalive that cannot ask the server whether a picture is in flight reads a chat probe queued
    // behind that picture as death and taskkills a job that was succeeding (measured 2026-11-01 22:30).
    // The two /health fields that settle it are advertised here so no client needs a private protocol
    // to find them -- and so a rename of either field fails this check instead of a supervisor.
    failures += check(
        configured.at("image_generation").at("health").at("endpoint") == "/health" &&
            configured.at("image_generation").at("health").at("busy_field") ==
                "image_jobs.in_flight" &&
            configured.at("image_generation").at("health").at("age_field") ==
                "image_jobs.oldest_age_seconds" &&
            configured.at("image_generation").at("health").at("meaning").get<std::string>().find(
                "drawing") != std::string::npos,
        "the capability tells a supervisor which /health fields separate a drawing engine from a dead "
        "one");
    failures += check(plain.at("endpoints").at("images_generations") == "/v1/images/generations" &&
                          configured.at("endpoints").at("chat_completions") == "/v1/chat/completions" &&
                          configured.at("endpoints").at("messages") == "/v1/messages",
                      "capabilities name the routes a client can actually post to");
    failures += check(plain.at("image_edit").is_null(),
                      "no image pipeline means no edit surface either");
    // The edit block is the answer to the question a fresh client actually asks: how do I change
    // this picture. It names the route, the required reference, and both refusals, so no file
    // and no prior conversation is needed to get an edit right.
    const Json edit = configured.at("image_edit");
    failures += check(edit.at("endpoint") == "/v1/images/edits" &&
                          edit.at("reference_required") == true &&
                          edit.at("accepts_url_reference") == false &&
                          edit.at("content_types") ==
                              Json::array({"application/json", "multipart/form-data"}) &&
                          edit.at("missing_reference").get<std::string>()
                              .find("image_reference_required") != std::string::npos &&
                          edit.at("mask").get<std::string>()
                              .find("image_mask_not_supported") != std::string::npos &&
                          edit.at("chat_tool_image_argument").is_array() &&
                              edit.at("chat_tool_image_argument").size() == 4ULL &&
                              edit.at("chat_tool_image_argument").dump().find("<image1>") !=
                                  std::string::npos &&
                              edit.at("chat_tool_image_argument").dump().find("reference_max") !=
                                  std::string::npos,
                      "the edit capability names the route, the required reference, both refusals and the "
                      "chat tool's four image forms -- the list among them, counted against reference_max "
                      "rather than a number repeated here");
    failures += check(configured.at("endpoints").at("images_edits") == "/v1/images/edits",
                      "the endpoints list names the edit route too");

    // The decide route is registered whether or not an image pipeline is configured (http_server.cpp
    // :616-622 posts both /decide and /systemone unconditionally), so its block belongs in the plain
    // document too -- a client talking to a text-only server has the same right to learn the
    // vocabulary. What it most needs, and what nothing said before, is WHICH of its failures is a
    // status and which is an answer: the route answers a cause three ways, and a caller that cannot
    // tell them either retries a request it must rewrite or trusts a run that produced nothing.
    for (const Json* document : {&plain, &configured}) {
        const Json decide = document->at("decide");
        failures += check(decide.at("endpoint") == "/v1/decide" &&
                              decide.at("alias").get<std::string>().find("/v1/systemone") !=
                                  std::string::npos &&
                              decide.at("request").at("question_types") ==
                                  Json::array({"noul", "choice", "score", "number", "scalar", "point",
                                               "box"}) &&
                              decide.at("request").at("question_fields").size() == 6ULL,
                          "the decide block names the route, its alias, the seven question kinds and the "
                          "six fields a question may carry -- on the plain document as much as on the "
                          "image one, because the route is served either way");
        const Json tiers = decide.at("refusals");
        failures += check(tiers.at("bad_request_400").size() == 17ULL &&
                              tiers.at("in_answer_200").size() == 5ULL &&
                              tiers.at("run_499_or_500").size() == 5ULL,
                          "the three tiers together carry every code the route answers with (17 + 5 + 5 = "
                          "the 27 the probe harvests from the decide sources)");
        // too_many_steps is the one a shape-based reading gets wrong: it travels as an ApiError's .code,
        // which is how the engine faults look, but its own .status is 400 (decide.cpp:659-663) because
        // the request asked for a schedule wider than the engine serves. Filed with the faults it would
        // tell a client to send the same bytes again; sending them again is exactly what cannot work.
        failures += check(tiers.at("bad_request_400").contains("too_many_steps") &&
                              !tiers.at("run_499_or_500").contains("too_many_steps") &&
                              tiers.at("bad_request_400").at("too_many_steps").get<std::string>()
                                  .find("not the server's fault") != std::string::npos,
                          "too_many_steps sits with the 400s a caller must fix, not with the faults worth "
                          "resending");
        // answer_mass is the field a client is most likely to invent a threshold for, since it looks
        // like a score. It is a report: nothing in the serve compares it to anything.
        failures += check(decide.at("answer").at("answer_mass").get<std::string>().find("not a gate") !=
                                  std::string::npos &&
                              decide.at("error_answer").get<std::string>().find("INSIDE the 200") !=
                                  std::string::npos,
                          "the block says answer_mass is a report rather than a gate, and that a failed "
                          "question's error rides inside the 200");
    }

    // P10: the same document also has to say that NOTHING between this server and the model
    // rewrites the client's words, and how to write them instead. A fresh session of any client
    // gets the shipped enhancer's own rules from the server it is talking to, not from a file it
    // was told to read -- which is the whole point of the capability document.
    failures += check(edit.at("prompt_enhancer").at("available") == false &&
                          edit.at("prompt_enhancer").at("field") == "prompt_enhance" &&
                          edit.at("prompt_enhancer").at("model") ==
                              "Qwen/Qwen-Image-2.1-PE-I2I" &&
                          edit.at("prompt_enhancer").at("when_true").get<std::string>()
                              .find("image_prompt_enhancer_unavailable") != std::string::npos,
                      "the edit capability names the enhancer field, the model that is missing and "
                      "the refusal it earns");
    // The third value of that same field has to be discoverable too, or a client that never read a
    // file cannot use it: the accepted values, and what the server does with "auto" when there is no
    // rewriter at all. This test builds exactly that build, so the honest answer is "draws what you
    // wrote" -- not a refusal the client would have to discover by sending one.
    failures += check(edit.at("prompt_enhancer").at("values").is_array() &&
                          edit.at("prompt_enhancer").at("values").size() == 3ULL &&
                          edit.at("prompt_enhancer").at("when_auto").get<std::string>()
                              .find("does not refuse") != std::string::npos,
                      "the edit capability lists prompt_enhance's three values and what auto answers "
                      "with no rewriter installed");
    // The enhancer outcome the document advertises has to be the outcome the code writes. The document
    // builds its list by CALLING to_string, so comparing against to_string again is the drift guard, and
    // the size guard catches an enumerator added to the enum without a word reaching the client.
    failures += check(
        edit.at("prompt_enhancer").at("reported_as").at("field") == "prompt_enhance_outcome" &&
            edit.at("prompt_enhancer").at("reported_as").at("values") ==
                Json::array({infernix::serve::to_string(infernix::serve::EnhancerOutcome::Off),
                             infernix::serve::to_string(infernix::serve::EnhancerOutcome::SkippedNoReference),
                             infernix::serve::to_string(infernix::serve::EnhancerOutcome::SkippedAlreadyRewritten),
                             infernix::serve::to_string(infernix::serve::EnhancerOutcome::SkippedNoRewriter),
                             infernix::serve::to_string(infernix::serve::EnhancerOutcome::SkippedNoDevice),
                             infernix::serve::to_string(infernix::serve::EnhancerOutcome::Applied)}) &&
            edit.at("prompt_enhancer").at("reported_as").at("values").size() == 6ULL &&
            edit.at("prompt_enhancer").at("reported_as").at("on").is_array() &&
            edit.at("prompt_enhancer").at("reported_as").at("on").size() == 3ULL,
        "the capability names the enhancer-outcome field, the three surfaces it appears on, and exactly "
        "the values the service can write");
    // Every check above builds the build WITHOUT a rewriter. That is exactly how a false when_auto
    // survived: the configured branch -- the one this machine actually runs, with the sidecar up -- has
    // never been read by a test. Build it here and hold it to the same contract, so the sentence a client
    // on this machine actually gets is the sentence that is checked.
    infernix::serve::ServeOptions enhanced_options;
    enhanced_options.enable_vision = true;
    enhanced_options.image_prompt_enhancer_url = "http://127.0.0.1:8899";
    const Json enhanced = Json::parse(infernix::serve::make_capabilities_json(
        enhanced_options, "qwen3.8-27b", "Qwen-Image-2.1-FP8-v3"));
    const Json enhancer = enhanced.at("image_edit").at("prompt_enhancer");
    failures += check(enhancer.at("available") == true &&
                          enhancer.at("when_true").get<std::string>()
                              .find("image_prompt_enhancer_unavailable") != std::string::npos &&
                          enhancer.at("when_auto").get<std::string>().find("never a 503") !=
                              std::string::npos &&
                          enhanced.at("image_edit").at("short_instruction").get<std::string>()
                              .find("auto") != std::string::npos &&
                          enhanced.at("image_edit").at("short_instruction").get<std::string>()
                              .find("Absent or false") != std::string::npos,
                      "with a rewriter configured the document says so, keeps true's 503, keeps auto's "
                      "no-refusal promise, and tells the client that auto rewrites a short instruction too "
                      "and that an absent field is no rewrite");
    // The block also has to say WHICH route takes the field, and that is true in both builds. The chat
    // tool is the path a client with no files reaches for first, and it carries the argument now: the
    // guard reads the tool's own schema rather than the sentence, so the schema has to OFFER the switch
    // and teach the two rules the document states (auto needs a reference the instruction does not
    // already address as <imageN>; true refuses when no rewriter runs). A document that promises what the
    // schema does not offer, or a schema that drifts from the document, fails here.
    const std::string chat_schema =
        infernix::serve::make_chat_image_tool_definition().input_schema_json;
    failures += check(enhanced.at("image_edit").at("prompt_enhancer").at("via").get<std::string>()
                              .find("chat") != std::string::npos &&
                          edit.at("prompt_enhancer").at("via").get<std::string>()
                              .find("prompt_enhance") != std::string::npos &&
                          chat_schema.find("prompt_enhance") != std::string::npos &&
                          chat_schema.find("JSON LIST") != std::string::npos &&
                          chat_schema.find("<imageN>") != std::string::npos &&
                          chat_schema.find("image_prompt_enhancer_unavailable") != std::string::npos &&
                          chat_schema.find("true_cfg_scale") != std::string::npos,
                      "the document names the routes that take prompt_enhance, the chat tool among them, "
                      "and the chat tool's own schema offers the argument and states the same two rules");
    failures += check(
        edit.at("short_instruction").get<std::string>().find("not sufficient") !=
            std::string::npos,
        "a client is told a one-phrase instruction is not enough in this build");
    // The two refusals that belong to the tool rather than to the route, named. They are never an HTTP
    // status -- the call comes back as the tool's error result -- so no status list would ever have carried
    // them, and a client that has to tell its own per-request budget from a request the route rejected
    // cannot look either of them up anywhere else.
    const std::string tool_refusals =
        configured.at("image_generation").at("chat_tool_refusals").dump();
    failures += check(tool_refusals.find("image_generation_limit_reached") != std::string::npos &&
                          tool_refusals.find("image_arguments_invalid") != std::string::npos &&
                          tool_refusals.find("JSON object") != std::string::npos,
                      "the document names the tool's own two refusals and what the second one means");
    // The 503s and 500s have to be separable by NAME, which is the whole reason a status code is not
    // enough: one of them wants a smaller canvas, one wants the card left alone, one wants a retry and one
    // must never be retried as it stands. A client that cannot tell these apart burns the card it shares.
    const std::string route_refusals =
        configured.at("image_generation").at("refusals").dump();
    failures += check(route_refusals.find("image_engine_degraded") != std::string::npos &&
                          route_refusals.find("image_interlude_failed") != std::string::npos &&
                          route_refusals.find("image_interlude_unsupported") != std::string::npos &&
                          route_refusals.find("image_store_active_full") != std::string::npos &&
                          route_refusals.find("image_generation_failed") != std::string::npos &&
                          route_refusals.find("image_artifact_failed") != std::string::npos &&
                          route_refusals.find("image_request_unservable") != std::string::npos &&
                          route_refusals.find("image_conditioning_inconsistent") != std::string::npos,
                      "every answer that is not the client's request to fix is named, so the four kinds of "
                      "retry are tellable without a single line of this repository");
    const Json guidance = edit.at("instruction");
    failures += check(guidance.is_array() && guidance.size() >= 10ULL &&
                          guidance.dump().find("<image1>") != std::string::npos &&
                          guidance.dump().find("ratio_follow") != std::string::npos,
                      "the capability carries the enhancer's own rules for writing an instruction");

    // The document now carries the refusal vocabulary, so that vocabulary has to be the parsers' own.
    // Every code below is PRODUCED by driving the real parser into it -- not copied from the source --
    // and then required to appear in the document. A code renamed in openai_images.cpp without the
    // document following, or a rule documented without its code, fails here. That is the drift the
    // refusals blocks exist to prevent, and it is why the codes are driven rather than asserted as text.
    using Request = infernix::serve::OpenAIImagesRequest;
    using Body = infernix::serve::RequestJson;
    const auto refusal_code = [](const std::function<Request()>& call) {
        try {
            (void)call();
        } catch (const infernix::serve::ApiException& exception) {
            return exception.error().code;
        } catch (const std::exception&) {
            return std::string{"<wrong exception>"};
        }
        return std::string{"<accepted, no refusal>"};
    };
    const auto image_body = [](Body extra) {
        Body body = {{"model", "Qwen-Image-2.1-FP8-v4"},
                     {"prompt", "a lighthouse at dusk"},
                     {"size", "768x768"}};
        for (Body::iterator it = extra.begin(); it != extra.end(); ++it) {
            if (it.value().is_null()) { body.erase(it.key()); } else { body[it.key()] = it.value(); }
        }
        return body;
    };
    // The reference is the family's LIST form, so the refusal is about the COUNT and the cap has to be
    // the parser's own constant, never a number written into this file: build one past it and one AT it,
    // the pair that proves the route takes many pictures and refuses only the overflow.
    const auto reference_list = [](std::size_t count) {
        Body list = Body::array();
        for (std::size_t i = 0; i < count; ++i) { list.push_back("AA=="); }
        return list;
    };
    const struct { const char* what; Body body; const char* code; } refused[] = {
        {"a size that is not a multiple of 32", image_body({{"size", "100x100"}}), "image_size_invalid"},
        {"a malformed wh_ratio", image_body({{"size", nullptr}, {"wh_ratio", "0:1"}}),
         "image_ratio_invalid"},
        {"a ratio_follow that names no sent reference",
         image_body({{"size", nullptr}, {"ratio_follow", "image1"}}), "image_ratio_follow_unsupported"},
        {"two canvas keys at once", image_body({{"wh_ratio", "16:9"}}), "image_size_conflict"},
        {"steps outside the served range", image_body({{"steps", 0}}), "image_steps_invalid"},
        {"a guidance scale outside the range", image_body({{"true_cfg_scale", 0.5}}),
         "image_cfg_scale_invalid"},
        {"n other than one", image_body({{"n", 2}}), "image_n_not_supported"},
        {"one reference past the served cap",
         image_body({{"image", reference_list(infernix::serve::kMaximumImageReferences + 1)}}),
         "image_count_invalid"},
        {"a reference given as a URL", image_body({{"image", "https://example.com/a.png"}}),
         "image_url_not_supported"},
        {"a data: URI that carries no base64", image_body({{"image", "data:image/png,AAAA"}}),
         "image_data_uri_not_base64"},
    };
    for (const auto& c : refused) {
        const std::string produced = refusal_code([&c] {
            return infernix::serve::parse_openai_images_request(c.body);
        });
        failures += check(produced == c.code,
                          (std::string("the parser refuses ") + c.what + " with " + c.code +
                           ", and not with " + produced).c_str());
        failures += check(configured.dump().find(c.code) != std::string::npos,
                          (std::string("the capability document names ") + c.code +
                           " so a client can look it up").c_str());
    }
    // The edit route's own refusal, and the two the parsers cannot reach because they are the device's
    // answer rather than the request's: those are advertised too, and a client has to be able to tell a
    // retry from a rewrite of the request.
    failures += check(refusal_code([] {
                          return infernix::serve::parse_openai_images_edit_request(
                              Body{{"model", "Qwen-Image-2.1-FP8-v4"},
                                   {"prompt", "make the ground green"},
                                   {"size", "768x768"}});
                      }) == "image_reference_required" &&
                        configured.dump().find("image_reference_required") != std::string::npos,
                    "the edit route's missing-reference refusal is the parser's and is documented");
    const std::string advertised = configured.dump();
    // The positive control for the count refusal: exactly the cap is ACCEPTED, so the refusal above is
    // the overflow and not a blanket ban on the list form. A "fix" that re-bans multi-reference to make
    // an old sentence true would fail here, and the document has to carry the same number the parser
    // enforces -- an advertised cap of its own is a second truth a client would trust.
    failures += check(
        refusal_code([&] {
            return infernix::serve::parse_openai_images_request(
                image_body({{"image", reference_list(infernix::serve::kMaximumImageReferences)}}));
        }) == "<accepted, no refusal>" &&
            advertised.find(std::to_string(infernix::serve::kMaximumImageReferences)) !=
                std::string::npos,
        "the route takes the full list of references and the document advertises that same cap");
    failures += check(advertised.find("image_interlude_in_flight") != std::string::npos &&
                          advertised.find("image_vram_insufficient") != std::string::npos,
                      "the document separates 'retry the same request' from 'this canvas cannot fit'");

    if (failures == 0) { std::cout << "http routes tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
