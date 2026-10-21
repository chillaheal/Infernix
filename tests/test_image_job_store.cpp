#include "serve/image_job_store.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

namespace {

using namespace infernix::serve;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

StoredImageJob job(std::string id) {
    StoredImageJob value;
    value.id      = std::move(id);
    value.state   = ImageJobState::Queued;
    value.created = 1700000000;
    value.request = nlohmann::json{
        {"model", "qwen-image-2.1"}, {"prompt", "a cat"}, {"size", "1024x1024"}};
    return value;
}

nlohmann::json err(const std::string& code) {
    return nlohmann::json{{"code", code}, {"message", "boom"}};
}

std::string transition_code(ImageJobStore& store, const std::string& id, const JobUpdate& update) {
    try {
        (void)store.advance(id, update);
        return {};
    } catch (const ApiException& exception) {
        return exception.error().code;
    }
}

int test_admission_idempotency() {
    ImageJobStore store(4, 1ULL << 20);
    store.put(job("img_a"));

    int failures = 0;
    std::string code;
    try {
        store.put(job("img_a"));
    } catch (const ApiException& exception) { code = exception.error().code; }
    failures += check(code == "image_job_already_exists", "duplicate admission is rejected, never replaced");
    failures += check(store.size() == 1, "rejected admission does not duplicate the record");
    const auto kept = store.get("img_a");
    failures += check(kept && kept->state == ImageJobState::Queued, "the original record survives a retry");
    return failures;
}

int test_lifecycle_walk_and_guards() {
    ImageJobStore store(8, 1ULL << 20);
    store.put(job("img_run"));

    int failures = 0;
    failures += check(transition_code(store, "img_run", {ImageJobState::Restoring, 10}) ==
                          "image_job_transition_rejected",
                      "a step cannot skip WaitingForFrontier");
    failures += check(transition_code(store, "missing", {ImageJobState::WaitingForFrontier, 10}) ==
                          "image_job_transition_rejected",
                      "advancing an unknown id is rejected");

    auto record = store.advance("img_run", {ImageJobState::WaitingForFrontier, 11});
    failures += check(record && record->state == ImageJobState::WaitingForFrontier && record->finished == 0,
                      "the frontier step moves and stays non-terminal");
    record = store.advance("img_run", {ImageJobState::Running, 12});
    failures += check(record && record->state == ImageJobState::Running, "running moves");
    record = store.advance("img_run", {ImageJobState::Restoring, 13});
    failures += check(record && record->state == ImageJobState::Restoring, "restoring moves");
    record = store.advance("img_run", {ImageJobState::Succeeded, 14, {}, {},
                                       nlohmann::json{{"created", 14}, {"data", nlohmann::json::array()}}});
    failures += check(record && record->state == ImageJobState::Succeeded && record->finished == 14 &&
                          !record->response.is_null(),
                      "success publishes only after restoration and stamps finished");
    failures += check(transition_code(store, "img_run", {ImageJobState::Succeeded, 15}) ==
                          "image_job_transition_rejected",
                      "a terminal state is absorbing (a late writer cannot resurrect the job)");

    store.put(job("img_err"));
    record = store.advance("img_err", {ImageJobState::WaitingForFrontier, 20});
    record = store.advance("img_err", {ImageJobState::Failed, 21, err("cuda_launch_failed")});
    failures += check(record && record->state == ImageJobState::Failed && record->finished == 21 &&
                          record->error != nullptr && !record->error.is_null(),
                      "job-failed-but-healthy records the pipeline error");
    failures += check(transition_code(store, "img_err", {ImageJobState::Running, 22, err("late")}) ==
                          "image_job_transition_rejected",
                      "an error cannot be written after the job went terminal");
    store.put(job("img_err2"));
    failures += check(transition_code(store, "img_err2", {ImageJobState::WaitingForFrontier, 30, err("x")}) ==
                          "image_job_transition_rejected",
                      "an error is legal only into a failed state");
    return failures;
}

int test_degraded_keeps_both_errors() {
    ImageJobStore store(8, 1ULL << 20);
    store.put(job("img_deg"));
    (void)store.advance("img_deg", {ImageJobState::WaitingForFrontier, 1});
    (void)store.advance("img_deg", {ImageJobState::Running, 2});
    (void)store.advance("img_deg", {ImageJobState::Restoring, 3});
    const auto record = store.advance("img_deg",
                                      {ImageJobState::FailedEngineDegraded, 4, err("dit_step_threw"),
                                       err("kv_restore_failed")});

    int failures = 0;
    failures += check(record && record->state == ImageJobState::FailedEngineDegraded && record->finished == 4,
                      "uncertain restoration lands in failed_engine_degraded");
    failures += check(record && record->error != nullptr && !record->error.is_null() &&
                          record->restoration_error != nullptr && !record->restoration_error.is_null(),
                      "the store keeps BOTH the pipeline and the restoration failure");
    failures += check(transition_code(store, "img_deg",
                                      {ImageJobState::Failed, 5, {}, err("only one")}) ==
                          "image_job_transition_rejected",
                      "a restoration error belongs only to failed_engine_degraded");
    return failures;
}

int test_cancel_is_check_and_set() {
    ImageJobStore store(8, 1ULL << 20);
    store.put(job("img_c"));
    (void)store.advance("img_c", {ImageJobState::WaitingForFrontier, 1});
    (void)store.advance("img_c", {ImageJobState::Running, 2});

    int failures = 0;
    failures += check(store.cancel("img_c", 3), "cancelling a running job flips it once");
    failures += check(!store.cancel("img_c", 4), "a second cancel is a no-op");
    const auto record = store.get("img_c");
    failures += check(record && record->state == ImageJobState::Cancelled && record->finished == 3,
                      "the cancel winner stamped finished (the runtime polls this and unwinds)");
    failures += check(transition_code(store, "img_c", {ImageJobState::Restoring, 5}) ==
                          "image_job_transition_rejected",
                      "the losing runtime writer sees the rejection, never overwrites Cancelled");
    failures += check(!store.cancel("img_missing", 6), "cancelling an unknown id is a no-op");
    return failures;
}

int test_eviction_only_evicts_terminal() {
    ImageJobStore store(2, 1ULL << 20);
    store.put(job("img_act1"));
    store.put(job("img_act2"));

    int failures       = 0;
    std::string code;
    try {
        store.put(job("img_act3"));
    } catch (const ApiException& exception) { code = exception.error().code; }
    failures += check(code == "image_store_active_full", "admission is refused when every stored job is active");
    failures += check(store.size() == 2 && store.get("img_act1") && store.get("img_act2"),
                      "the refused admission left both active jobs untouched");

    (void)store.advance("img_act2", {ImageJobState::WaitingForFrontier, 10});
    (void)store.advance("img_act2", {ImageJobState::Running, 11});
    (void)store.advance("img_act2", {ImageJobState::Restoring, 12});
    (void)store.advance("img_act2", {ImageJobState::Succeeded, 13});
    store.put(job("img_act3"));  // evicts the terminal img_act2, never the older active img_act1
    failures += check(store.get("img_act2") == nullptr, "the terminal LRU victim was evicted");
    failures += check(store.get("img_act1") != nullptr, "an active job is never evicted, even at the LRU tail");
    failures += check(store.get("img_act3") != nullptr, "the admitted job is stored");
    return failures;
}

int test_prune_expired_touches_only_terminals() {
    ImageJobStore store(8, 1ULL << 20);
    store.put(job("img_done"));
    (void)store.advance("img_done", {ImageJobState::WaitingForFrontier, 100});
    (void)store.advance("img_done", {ImageJobState::Failed, 1000, err("prompt_rejected")});
    store.put(job("img_live"));

    int failures = 0;
    failures += check(store.prune_expired(1600, 600) == 0, "a terminal job is kept until strictly past its TTL");
    failures += check(store.prune_expired(1601, 600) == 1, "a terminal job past its TTL is pruned");
    failures += check(store.get("img_done") == nullptr && store.get("img_live") != nullptr,
                      "pruning never touches a non-terminal job");
    return failures;
}

int test_lru_recency_and_delete() {
    ImageJobStore store(2, 1ULL << 20);
    store.put(job("img_1"));
    store.put(job("img_2"));
    (void)store.advance("img_1", {ImageJobState::WaitingForFrontier, 1});
    (void)store.advance("img_2", {ImageJobState::WaitingForFrontier, 2});
    (void)store.advance("img_1", {ImageJobState::Failed, 3, err("x")});
    (void)store.advance("img_2", {ImageJobState::Failed, 4, err("y")});
    (void)store.get("img_1");  // img_2 becomes the least-recently-used terminal entry.
    store.put(job("img_3"));

    int failures = 0;
    failures += check(store.get("img_1") != nullptr, "get refreshes LRU recency for the eviction order");
    failures += check(store.get("img_2") == nullptr, "least-recent terminal job evicted");
    failures += check(store.get("img_3") != nullptr, "new job retained");
    failures += check(store.size() == 2 && store.bytes() != 0, "store reports bounded usage");
    failures += check(store.erase("img_1"), "stored job deleted");
    failures += check(!store.erase("img_1") && store.get("img_1") == nullptr,
                      "deleted job is no longer addressable");
    failures += check(std::string(state_to_string(ImageJobState::FailedEngineDegraded)) ==
                          "failed_engine_degraded",
                      "state names match the lifecycle contract");
    return failures;
}

int test_oversized_record() {
    ImageJobStore store(4, 256);
    StoredImageJob large = job("img_large");
    large.request        = nlohmann::json{{"prompt", std::string(1024, 'x')}};
    std::string code;
    try {
        store.put(std::move(large));
    } catch (const ApiException& exception) { code = exception.error().code; }

    int failures = 0;
    failures += check(code == "image_store_capacity_exceeded",
                      "oversized job fails deterministically");
    failures += check(store.size() == 0, "oversized insertion does not mutate store");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_admission_idempotency();
    failures += test_lifecycle_walk_and_guards();
    failures += test_degraded_keeps_both_errors();
    failures += test_cancel_is_check_and_set();
    failures += test_eviction_only_evicts_terminal();
    failures += test_prune_expired_touches_only_terminals();
    failures += test_lru_recency_and_delete();
    failures += test_oversized_record();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
