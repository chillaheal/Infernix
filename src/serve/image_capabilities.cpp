#include "serve/image_capabilities.h"

#include "infernix/ops/image_joint_limits.h"
#include "serve/chat_image_tool.h"
#include "serve/image_generation_service.h"
#include "serve/openai_images.h"

#include <nlohmann/json.hpp>

#include <string>
#include <string_view>
#include <utility>

namespace infernix::serve {

void add_image_capabilities(nlohmann::json& payload, const ServeOptions& options,
                            std::string_view image_model_id) {
    // The document's own type, named the way make_capabilities_json names it (http_server.cpp:660).
    // Everything below this line, up to the four keys this function adds at the end, is copied byte
    // for byte out of that function -- literals and comments alike, because the comments are the
    // reason each string says what it says. _p43_verify_caps_seam.py re-checks the copy against
    // HEAD's file line by line.
    using Json = nlohmann::json;
    Json image = Json(nullptr);
    if (!image_model_id.empty()) {
        image = Json{{"model", image_model_id},
                     {"endpoint", "/v1/images/generations"},
                     {"artifact_endpoint", std::string(kImageArtifactRoutePrefix) + "<id>"},
                     // How to tell a drawing engine from a dead one without a private protocol. A
                     // supervisor that only asks whether the chat model answers reads a picture's
                     // queue as death and kills the job: /health names the two fields that say
                     // otherwise, so the client that reads only this document can hold off healing
                     // without knowing anything about interludes.
                     {"health",
                      Json{{"endpoint", "/health"},
                           {"busy_field", "image_jobs.in_flight"},
                           {"age_field", "image_jobs.oldest_age_seconds"},
                           {"meaning", "an in_flight above 0 means the card is drawing a picture, not "
                                       "hung: the chat routes resume once the interlude restores the LLM"}}},
                     {"response_formats", Json::array({"b64_json", "url"})},
                     {"dimensions",
                      Json{{"min", kMinimumImageDimension}, {"max", kMaximumImageDimension}}},
                     {"steps", Json{{"min", kMinimumImageSteps}, {"max", kMaximumImageSteps}}},
                     {"true_cfg_scale",
                      Json{{"min", kMinimumImageCFGScale}, {"max", kMaximumImageCFGScale}}},
                     // The other half of steering, named so a client that reads nothing but this
                     // document can find it: scale above 1 means a second DiT forward against this
                     // text, and without it the steer would silently mean nothing, so the pair is
                     // refused at parse (openai_images.cpp:321) instead of degraded in the engine.
                     {"negative_prompt",
                      Json{{"type", "string"},
                           {"max_bytes", kMaximumImagePromptBytes},
                           {"required_when", "true_cfg_scale above 1"},
                           {"omitted_with_scale_above_1",
                            "HTTP 400 image_negative_prompt_required"}}},
                     // Reference-image conditioning runs on the diffusion pipeline's own vision tower,
                     // which sees only the picture being edited; --vision is a separate surface.
                     // TRUE because the behavioural gate (tests/models/qwen_image/
                     // test_image_edit_behavior.cu) is green: it renders the M9 fixture twice, once per
                     // instruction, and grades the PICTURES -- not-a-copy (MAE/SSIM vs the reference),
                     // subject held, instruction happened. Advertising true is exactly the claim that
                     // gate exists to earn; it was false while every edit came back a copy.
                     {"reference_image", true},
                     // How the picture is sized, in the client's terms. Naming the three keys and the
                     // reference-following default here is what lets a client that reads nothing but
                     // this document know that an edit of a portrait should come back portrait: the
                     // canvas is the thing a fixed square default used to get wrong.
                     {"canvas", Json{{"keys", Json::array({"size", "wh_ratio", "ratio_follow"})},
                                     {"give_at_most_one", true},
                                     // What the card gets to decide about a canvas the client named. The
                                     // admission need is a function of the canvas, so the 503 that used to
                                     // answer "this size does not fit today" was a smaller picture away
                                     // from fitting -- and the client's own model found that size one round
                                     // and two minutes later. The server now walks its own 32-px grid down
                                     // (longer side first, so the chosen aspect survives) and draws the
                                     // largest canvas the card can hold, saying which one it drew. Only
                                     // when not even the floor fits does the 503 reach a client, with the
                                     // same two numbers it always carried.
                                     {"vram_downgrade",
                                      "a canvas this process cannot hold is drawn at the largest size that "
                                      "fits instead of refused; the answer then names both, in data[0].size "
                                      "(drawn) and data[0].size_asked (sent), and the same pair rides on "
                                      "image_tool_attempts[].result"},
                                     {"size", "WIDTHxHEIGHT, multiples of 32 in [32,2048]"},
                                     {"wh_ratio", "\"W:H\" held at the working area, e.g. \"16:9\""},
                                     {"ratio_follow", "\"<image1>\": the canvas keeps the reference's aspect"},
                                     // The one ceiling a canvas can break that is NOT about memory: the DiT
                                     // serves one attention sequence over the text rows and the picture
                                     // tokens together, and that range belongs to the model. It is published
                                     // here so a client sizes to it instead of discovering it as a 400 -- and
                                     // so nobody reaches for free VRAM to raise it, because freeing memory
                                     // changes nothing: 2048x2048 draws from words and is refused as an edit.
                                     {"joint_sequence",
                                      Json{{"ceiling_tokens", ops::kImageJointMaxTokens},
                                           {"counts", "the canvas costs (W/16)x(H/16) tokens, every "
                                                      "reference triple its vision tokens, and every text "
                                                      "row one token; the sum must stay under the ceiling"},
                                           {"measured", "2048x2048 fits a draw from words alone; with one "
                                                        "reference the largest edit measured to draw here "
                                                        "is 1792x1792"},
                                           {"ask_the_server", "the refusal names the largest canvas that "
                                                              "fits this prompt and reference -- take that "
                                                              "one rather than shrinking by guess"}}},
                                     // The codes each rule above earns, beside the rule. A client that read
                                     // the rule and still guessed wrong is told WHICH rule it broke only if
                                     // the code is next to the text that describes it -- otherwise the
                                     // document teaches the format and the wire answers in a vocabulary the
                                     // client has to already know.
                                     {"refusals",
                                      Json{{"size_not_a_multiple_of_32_or_out_of_range",
                                           "HTTP 400 image_size_invalid"},
                                           {"wh_ratio_malformed_or_outside_the_supported_span",
                                            "HTTP 400 image_ratio_invalid"},
                                           {"ratio_follow_not_an_imageN_tag_naming_a_sent_reference",
                                            "HTTP 400 image_ratio_follow_unsupported"},
                                           {"more_than_one_of_the_three_keys",
                                            "HTTP 400 image_size_conflict: the keys are mutually "
                                            "exclusive, name at most one"}}},
                                     {"default_with_reference", "the reference's own aspect"},
                                     {"default_without_reference", "768x768"}}},
                     // What a prompt may be. The byte bound is what one request may carry; the whole
                     // text reaches the model, nothing is truncated, and the model's own limit is the
                     // joint sequence -- which depends on the canvas, so it is answered per request (400
                     // image_prompt_too_long names both numbers and the largest canvas that fits). A
                     // client that reads only this document
                     // can therefore write a long instruction without guessing whether it will survive.
                     {"prompt", Json{{"max_bytes", kMaximumImagePromptBytes},
                                     {"truncated", false},
                                     {"too_long", "HTTP 400 image_prompt_too_long, naming the joint "
                                                  "sequence, its ceiling and the largest canvas that "
                                                  "fits this prompt -- ask that one instead of guessing"}}},
                     // The rest of the refusals one of these two routes can answer, split the only way a
                     // client can act on: the 400s are the client's own request to fix, the 429 is worth
                     // retrying as it stands, and the 503 is a card that cannot hold this canvas -- which
                     // is a smaller picture away, not a reason to give up. Without this block a client has
                     // to have met the answer before to know which of the three it is holding.
                     {"refusals",
                      Json{{"steps_out_of_range", "HTTP 400 image_steps_invalid"},
                           {"true_cfg_scale_out_of_range", "HTTP 400 image_cfg_scale_invalid"},
                           {"n_other_than_one", "HTTP 400 image_n_not_supported"},
                           // The route takes the family's LIST form: one to kMaximumImageReferences
                           // pictures, each one conditioning the forward. The number is written from the
                           // constant the parser enforces, so the document can never advertise a cap the
                           // route does not have -- it used to promise "exactly one picture", which the
                           // parser has not enforced since the list form landed.
                           {"reference_count_out_of_range",
                            "HTTP 400 image_count_invalid: the reference takes between 1 and " +
                                std::to_string(kMaximumImageReferences) +
                                " base64 pictures, each of them conditioning the draw"},
                           {"reference_given_as_a_url",
                            "HTTP 400 image_url_not_supported: base64 only, never a URL"},
                           {"reference_data_uri_without_base64",
                            "HTTP 400 image_data_uri_not_base64: a data: URI is accepted only when it "
                            "carries base64"},
                           {"reference_over_the_decoding_budget", "HTTP 400 image_too_large"},
                           {"reference_not_a_decodable_png_or_jpeg", "HTTP 400 image_decode_failed"},
                           {"multipart_file_field_other_than_image",
                            "HTTP 400 image_unsupported_file_field"},
                           {"another_draw_holds_the_card",
                            "HTTP 429 image_interlude_in_flight: retry the same request"},
                           {"the_llm_never_reached_a_safe_point",
                            "HTTP 429 image_interlude_busy: the card was never handed over because the chat "
                            "was still running when the handoff was attempted; nothing moved and nothing "
                            "began, so retry the SAME request at the same canvas rather than editing it"},
                           {"the_yield_cannot_cover_this_canvas",
                            "HTTP 503 image_vram_insufficient: the message names the bytes needed and the "
                            "bytes the model can yield, and says in its own words whether a smaller canvas "
                            "would have lowered that need -- below roughly 1024 it does not, because the "
                            "need is then dominated by what the pipeline keeps resident whatever the "
                            "picture's size, and shrinking into it costs a request per step"},
                           // Three answers that come from outside the request, so nothing the client sent
                           // predicts them -- which is exactly when a name pays: an unknown id is final, an
                           // unconfigured pipeline is the operator's to fix, and a rewriter that answered
                           // badly is one switch away from being drawn without.
                           {"artifact_or_job_id_unknown",
                            "HTTP 404 image_not_found: no job and no picture carries this id, so polling it "
                            "again will not help"},
                           {"pipeline_not_configured",
                            "HTTP 503 image_generation_not_configured: this process has no image model, so "
                            "no request can be drawn until one is loaded"},
                           {"enhancer_answered_badly",
                            "HTTP 500 image_prompt_enhancer_failed: the prompt rewriter answered an error "
                            "status or something that is not JSON; the same request with prompt_enhance "
                            "false goes straight to the draw"},
                           // The rest are the answers that are nobody's request to fix, and the document
                           // gives them anyway because the one thing a caller can still do is choose
                           // BETWEEN them: wait, walk away, or stop spending the card on a retry that
                           // reproduces. Without these names 503 and 500 are one blob and a client cannot
                           // tell the canvas that wants to be smaller (above) from the engine that wants
                           // to be left alone, nor a draw that simply failed from a kernel that will refuse
                           // the same geometry forever.
                           {"the_card_is_not_healthy_right_now",
                            "HTTP 503 image_engine_degraded: the draw failed and the text model could not "
                            "be confirmed restored, so nothing about this request is wrong -- leave the "
                            "card alone and come back later"},
                           {"the_handover_failed_before_the_draw",
                            "HTTP 503 image_interlude_failed: the card could not be taken or given back "
                            "before the draw began, so the same request may go through once it is free"},
                           {"this_model_has_no_handover_to_give",
                            "HTTP 503 image_interlude_unsupported: the loaded image model supports no "
                            "interlude, so a request that needs one will never be served here and retrying "
                            "changes nothing"},
                           {"every_job_slot_is_busy",
                            "HTTP 503 image_store_active_full: this process holds no free job slot, which is "
                            "the store's own half of the overload policy -- the same request is worth "
                            "retrying"},
                           {"the_draw_simply_failed",
                            "HTTP 500 image_generation_failed: the draw ran, the text model was restored, "
                            "and the picture just did not come, so a retry is fair"},
                           {"a_weight_could_not_be_bound",
                            "HTTP 500 image_artifact_failed: a weight could not be bound or streamed, which "
                            "is the artifact lane rather than this request"},
                           {"a_shape_the_admission_missed",
                            "HTTP 500 image_request_unservable: a kernel refused the geometry it was handed, "
                            "which is the server's to fix -- the same request reproduces it, so do not "
                            "retry it as it stands"},
                           {"a_reference_that_disagrees_with_itself",
                            "HTTP 500 image_conditioning_inconsistent: a prepared reference carries a vision "
                            "token count that contradicts its own latent size, and the same picture "
                            "reproduces it"},
                           {"a_job_the_serve_cannot_price",
                            "HTTP 500 image_need_unpriceable: the server could not say what the job's own "
                            "VRAM reservation is -- an edit whose reference planes it cannot hand the vision "
                            "ledger, or a plane above the ceiling the ledger enforces. It is the server's to "
                            "fix, the same request reproduces it, and no smaller canvas changes the answer "
                            "for an edit that already names its references"}}},
                     // A chat request to this process is given the tool below and the picture is
                     // drawn inside that same request, so a client needs no tool call of its own.
                     {"automatic_from_chat", true},
                     {"chat_tool", std::string(kChatImageToolName)},
                     {"chat_max_images_per_request", kMaximumChatImageGenerations},
                     // A picture a chat request attaches goes through the same parsing on all three
                     // protocols, and the one field of it this server cannot honour is a downscale: only
                     // auto and high exist here, so a client that sent "low" is told which field to drop
                     // instead of being left with a bare 400 and a picture it believes it sent.
                     {"chat_picture_detail",
                      "a picture part may ask for detail \"auto\" or \"high\"; anything else is HTTP 400 "
                      "image_detail_not_supported"},
                     // What became of each picture a chat request asked for, in the server's own words
                     // rather than the model's account of it. The model is shown the tool error and still
                     // answers the user in prose -- and invented a cause for it here (a VRAM figure, an
                     // unload it never attempted). This is the half a client could not get at all: the
                     // refusals left no trace in a log and none on the wire.
                     {"chat_tool_attempts",
                      Json{{"field", "image_tool_attempts"},
                           {"on", Json::array({"the chat.completion body of POST /v1/chat/completions",
                                               "the terminal chunk of a streamed chat completion, or its "
                                               "usage chunk when stream_options.include_usage is set",
                                               "the Anthropic message object, and the streamed message_delta",
                                               "the Responses response body, including the streamed "
                                               "response.completed"})},
                           {"present_when", "this request asked for at least one picture, and absent "
                                            "otherwise -- a plain chat answer stays the exact bytes it was "
                                            "before the field existed, so a client must not expect it"},
                           {"entries", "one per picture asked for, even when several tool calls shared a "
                                       "round: {round, outcome, tool_call_id, request, result|error}"},
                           {"outcome", Json{{"succeeded", "result carries id, url and "
                                                          "prompt_enhance_outcome",
                                             "refused", "the request never reached the model: error carries "
                                                       "the route's own code, message and http_status, which "
                                                       "is what to show the user",
                                             "failed", "the draw itself broke inside the server (HTTP 500 "
                                                       "from the pipeline), which a retry may clear"}}},
                           {"request", "what the tool call asked for: size, steps, image_count, and seed "
                                       "and prompt_enhance when the call set them"},
                           {"why", "a client can say which picture failed and why without taking the "
                                   "model's word for it. The same facts per picture also reach the "
                                   "operator's requests.jsonl as an image_job event, refused draws "
                                   "included, with the SAME outcome word for the same draw -- a 503 "
                                   "reads as refused in both, so a failure a client never reports is "
                                   "still readable afterwards, and never as a server that broke"}}},
                     // A chat edit goes through the very same parser as the routes, so its refusals ARE
                     // image_generation.refusals -- the tool hands the model back the route's own code, and
                     // the model retries smaller or waits. Only these two names come from the tool itself
                     // rather than from the route, and a client reading only this document is the one who
                     // needs to know which is which: the first is a per-request budget, the second means
                     // the arguments never were a request.
                     {"chat_tool_refusals",
                      Json{{"more_pictures_than_one_request_may_draw",
                            "image_generation_limit_reached: one chat request draws at most " +
                                std::to_string(kMaximumChatImageGenerations) + " pictures, the rest of the "
                                "calls are refused as tool errors"},
                           {"arguments_not_readable",
                            "image_arguments_invalid: the tool call is not a JSON object, so no route "
                            "refusal can apply to it"}}}};
    }
    // The edit surface, named on its own. image_generation says a picture can be drawn; this says
    // how to change one you already have, which is the question a fresh client actually asks. Every
    // field is the generations one except the reference, so the block points there instead of
    // restating geometry and step bounds a change would then have to make twice.
    // Whether this process was started with the enhancer's sidecar address. "Configured" is the honest
    // word the document can answer without asking the sidecar: a capabilities request is a cheap read,
    // and a health probe of another process does not belong in it. A configured sidecar that has since
    // stopped answering still answers the request with the documented 503 below, so a client that acts
    // on this field and gets that answer has lost nothing but a round trip.
    const bool enhancer_configured = !options.image_prompt_enhancer_url.empty();
    Json edit = Json(nullptr);
    if (!image_model_id.empty()) {
        edit = Json{{"model", image_model_id},
                     {"endpoint", "/v1/images/edits"},
                     {"fields", "every field /v1/images/generations takes"},
                     {"content_types", Json::array({"application/json", "multipart/form-data"})},
                     {"reference_required", true},
                     {"reference_max", kMaximumImageReferences},
                     {"reference_formats",
                      Json::array({"base64 in the image field (a data: URI prefix is accepted)",
                                   "a multipart file part named image",
                                   "a multipart text field named image holding the base64"})},
                     {"accepts_url_reference", false},
                     // The chat route reaches the same edit with no wire work at all: the tool the server
                     // injects into every chat request takes these forms of image, one of them or a list of
                     // up to reference_max -- the same cap this block states two lines above, because the
                     // chat tool resolves each element and hands the route the list the route already took.
                     // The http routes test reads the tool's own schema for this, so the pair cannot drift.
                     {"chat_tool_image_argument",
                      Json::array({"a job id img-<created>-<sequence> this server drew",
                                   "that picture's own URL",
                                   "attachment:N of the chat request",
                                   "a JSON list of 1..reference_max of those forms, which is what lets an "
                                   "instruction address them as <image1>, <image2>, ... in the order given"})},
                     {"missing_reference",
                      "HTTP 400 image_reference_required -- this route never draws a picture "
                      "nobody attached"},
                     {"mask", "not supported: HTTP 400 image_mask_not_supported"},
                     {"prompt_enhancer",
                      Json{{"available", enhancer_configured},
                           {"model", std::string(kImagePromptEnhancerModel)},
                           {"field", "prompt_enhance"},
                           // Which route the field belongs to, said out loud. The tool this server injects
                           // into every chat request -- the path a fresh client reaches for first, because
                           // it needs no base64 and no second call -- carries the same argument: its
                           // arguments go through the images route's own parser, so an edit drawn inside a
                           // chat request can be rewritten too. The document and the tool schema are held
                           // to each other by the http routes test, so neither can advertise a switch the
                           // other does not have.
                           {"via", "the images routes (/v1/images/generations and /v1/images/edits) and the "
                                   "chat tool this server injects into every chat request: put prompt_enhance "
                                   "in the tool call and it reaches the same parser the routes use. Absent is "
                                   "no rewrite, so a chat edit is drawn from the words exactly as written "
                                   "unless the call asks"},
                           {"values", Json::array({"true", "false", "auto"})},
                           {"when_auto",
                            enhancer_configured
                                ? "the server decides per request: the rewrite runs when the request "
                                  "carries a reference picture and the instruction is not already a "
                                  "rewriter's own answer (one that addresses its inputs as <imageN>), and "
                                  "is skipped otherwise -- and also when no rewriter is reachable or the "
                                  "card cannot be had for one, which for auto is a skip and never a 503, "
                                  "because auto asked the server to choose"
                                : "the picture is drawn from the instruction exactly as written: this "
                                  "build has no rewriter to run, and auto does not refuse over it -- only "
                                  "true gets the 503"},
                           {"when_true",
                            enhancer_configured
                                ? "the server rewrites the instruction with the enhancer before drawing; its "
                                  "rewritten_prompt is what reaches the model, and its wh_ratio or ratio_follow "
                                  "decides the canvas only where this request named no size of its own. HTTP "
                                  "503 image_prompt_enhancer_unavailable if the sidecar has stopped answering, "
                                  "500 image_prompt_enhancer_bad_answer if its answer breaks the contract below"
                                : "HTTP 503 image_prompt_enhancer_unavailable -- no rewriter is running in this "
                                  "build (serve --image-prompt-enhancer-url), so the client is the rewriter"},
                           {"answers",
                            "rewritten_prompt, wh_ratio, ratio_follow -- the two size fields are the "
                            "same ones the images routes take, mutually exclusive, and rewritten_prompt carries no "
                            "resolution or ratio of its own"},
                           // The answer to when_auto, per job. The strings come from the code that writes
                           // them (to_string) rather than from a list here, because a capability document
                           // that keeps its own copy is a document that can advertise an outcome the
                           // server never produces.
                           {"reported_as",
                            Json{{"field", "prompt_enhance_outcome"},
                                 {"on", Json::array({"the successful POST response body",
                                                     "the job's status object, which GET /v1/images/<id> "
                                                     "serves while no picture file exists (an in-flight job, "
                                                     "or one that failed)",
                                                     "the stored record"})},
                                 {"values", Json::array({to_string(EnhancerOutcome::Off),
                                                          to_string(EnhancerOutcome::SkippedNoReference),
                                                          to_string(EnhancerOutcome::SkippedAlreadyRewritten),
                                                          to_string(EnhancerOutcome::SkippedNoRewriter),
                                                          to_string(EnhancerOutcome::SkippedNoDevice),
                                                          to_string(EnhancerOutcome::Applied)})},
                                 {"meaning", "which decision this job got; every skip_* value is legitimate for "
                                             "auto, and only skipped_no_rewriter and skipped_no_device point at "
                                             "the deployment rather than at the request"}}}}},
                     // Not just true: auto rewrites too, whenever its own rule is met, and a client told
                     // only about true pays the rewriter on requests that do not need it -- or believes a
                     // short instruction is impossible under auto. Absent is spelled out because the
                     // request's own default is the third answer and the client cannot guess it.
                     {"short_instruction",
                      enhancer_configured
                          ? "acceptable when prompt_enhance is true, and also when it is the string auto and "
                            "the request carries a reference picture the instruction does not already address "
                            "as <imageN>: the server rewrites it first. Absent or false is no rewrite at all "
                            "(the request default is false), so on those requests the words reach the model "
                            "exactly as written and the instruction below is what carries the edit"
                          : "not sufficient in this build: nothing between the client and the model rewrites it, "
                            "so a one-phrase instruction reaches the diffusion model exactly as written"},
                     {"instruction",
                      Json::array({
                          "lead with the operation, not a description of the finished picture",
                          "name the target concretely, and push the named attribute to a strong and unmistakable "
                              "degree",
                          "hold everything else at input fidelity: say what stays by type, position and role, never "
                              "by describing its appearance",
                          "one blanket preservation clause beats a checklist -- describing the things you meant to keep "
                              "is what makes them drift",
                          "point identity at the reference image instead of describing features in words",
                          "with more than one reference, address them as <image1>, <image2>, ... and say which one is "
                              "the canvas and which supply material; with one reference, refer to it naturally",
                          "put no resolution or aspect ratio in the instruction: size travels only in size, wh_ratio "
                              "or ratio_follow",
                          "commit to the exact characters of any text that will be rendered into the picture, in the "
                              "image's own dominant language",
                          "state what must be preserved affirmatively, not as a prohibition",
                          "describe only what is visibly there -- a preserved element named at a higher abstraction "
                              "is safer than an invented specific",
                          "ask for nothing that was not requested, and say enough about any newly exposed region that "
                              "the result stays physically coherent",
                      })},
                     {"canvas_default", "the reference's own aspect (see image_generation.canvas)"}};
    }
    // The rewriter is not an edit-only argument: prompt_enhance is parsed by the images routes'
    // shared parser, so a draw with no reference takes it too. A client that reads only
    // image_generation could not find the switch at all -- it had to guess that a field named under
    // image_edit also exists one level up. Copied out of the edit block rather than written twice,
    // so the two blocks cannot advertise different rules for the same argument.
    if (!image.is_null() && !edit.is_null()) { image["prompt_enhancer"] = edit["prompt_enhancer"]; }
    // Both blocks always join the document, under the two names it has always published them by: a
    // null block is the lane's own way of reporting a process with no image model, so a client that
    // reads only this document still learns which routes exist and which answers 503. The endpoints
    // table is upstream's and keeps its three keys; the lane adds exactly these two.
    payload["image_generation"] = std::move(image);
    payload["image_edit"] = std::move(edit);
    payload["endpoints"]["images_generations"] = "/v1/images/generations";
    payload["endpoints"]["images_edits"] = "/v1/images/edits";
}

} // namespace infernix::serve
