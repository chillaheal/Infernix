#include "serve/http_server.h"

#include <nlohmann/json.hpp>

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
                          configured.at("image_generation").at("chat_max_images_per_request") == 4,
                      "the image capability reports its id, artifact route, formats and the chat tool a "
                      "client would otherwise have to run itself");
    failures += check(plain.at("endpoints").at("images_generations") == "/v1/images/generations" &&
                          configured.at("endpoints").at("chat_completions") == "/v1/chat/completions" &&
                          configured.at("endpoints").at("messages") == "/v1/messages",
                      "capabilities name the routes a client can actually post to");

    if (failures == 0) { std::cout << "http routes tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
