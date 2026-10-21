#pragma once

// Process-local bounded storage for Qwen-Image job records and their OpenAI image response bodies.
// The lifecycle state (Queued/WaitingForFrontier/Running/Restoring/Succeeded/Failed/Cancelled/
// FailedEngineDegraded) is host bookkeeping only: this store owns no Program, device buffer or VRAM.
// The image runtime owns execution and releases its own device memory at idle (#17). Lifecycle moves
// only through advance()/cancel(), which enforce the transition table and stamp `finished`; put() is
// admission only (duplicate ids are rejected). Budget eviction and prune_expired() touch terminal
// jobs only, so an active job can never be silently dropped. get() refreshes LRU recency; returned
// immutable records remain valid if a later insert evicts or prunes their public entry.

#include "serve/image_job_progress.h"
#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace infernix::serve {

enum class ImageJobState { Queued, WaitingForFrontier, Running, Restoring, Succeeded, Failed, Cancelled, FailedEngineDegraded };

[[nodiscard]] bool is_terminal(ImageJobState state) noexcept;

// The only legal lifecycle moves. A terminal state is absorbing: late writers (a runtime finishing
// after the caller already cancelled) must observe the rejection instead of resurrecting the job.
[[nodiscard]] bool can_transition(ImageJobState from, ImageJobState to) noexcept;

[[nodiscard]] const char* state_to_string(ImageJobState state) noexcept;

struct StoredImageJob {
    std::string id;
    ImageJobState state = ImageJobState::Queued;
    std::int64_t created  = 0;
    std::int64_t finished = 0;            // stamped at the terminal transition; 0 while non-terminal
    nlohmann::json request;               // validated caller params (prompt, size, n, ...); no device data
    nlohmann::json response;              // the OpenAI images response object once Succeeded
    nlohmann::json error;                 // the pipeline failure (Failed / FailedEngineDegraded)
    nlohmann::json restoration_error;     // FailedEngineDegraded keeps BOTH failures, never one overwriting the other
    // The job's live progress, shared with the record: constness of the record protects the
    // terminal fields only, the atomics inside stay writable by the job thread. Null for records
    // admitted without an execution (a rejected request).
    std::shared_ptr<ImageJobProgress> progress;
};

// One validated lifecycle move. `now` is caller-supplied (epoch seconds) so tests and the serving
// clock stay deterministic; it is stamped into `finished` when `to` is terminal.
struct JobUpdate {
    ImageJobState to = ImageJobState::Succeeded;
    std::int64_t now = 0;
    nlohmann::json error;
    nlohmann::json restoration_error;
    nlohmann::json response;
};

class ImageJobStore {
public:
    ImageJobStore(std::size_t max_records, std::size_t max_bytes);

    // get() refreshes LRU recency. A record stays valid if another request later evicts or prunes
    // its entry, because callers hold a shared_ptr to the immutable value.
    std::shared_ptr<const StoredImageJob> get(const std::string& id);

    // Admission: inserts a fresh record. A single job larger than max_bytes throws ApiException with
    // code image_store_capacity_exceeded; an existing id throws code image_job_already_exists (the
    // idempotency policy: retries re-discover the job through get(), they never silently replace it).
    // Budget eviction evicts TERMINAL LRU victims only; if every record is active the insert throws
    // code image_store_active_full (the store-level half of the overload policy).
    void put(StoredImageJob job);

    // One guarded transition; throws ApiException code image_job_transition_rejected when the id is
    // unknown, `to` is illegal from the current state, or the update carries an error on a state
    // that never holds one. Returns the stored record after the move.
    std::shared_ptr<const StoredImageJob> advance(const std::string& id, const JobUpdate& update);

    // Race-safe cancellation: one check-and-set under the store lock flips any non-terminal job to
    // Cancelled and stamps finished. Returns true only for the call that performed the flip; the
    // runtime polls the record and unwinds through its normal cleanup path.
    bool cancel(const std::string& id, std::int64_t now);

    // Drops terminal records whose finished + ttl_seconds is older than `now`; never touches a
    // non-terminal job. Returns the number pruned. Serving calls this from its housekeeping tick.
    std::size_t prune_expired(std::int64_t now, std::int64_t ttl_seconds);

    bool erase(const std::string& id);

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t bytes() const;

private:
    struct Entry {
        std::shared_ptr<const StoredImageJob> job;
        std::list<std::string>::iterator lru;
        std::size_t envelope_bytes = 0;
    };

    void erase_locked(const std::string& id);
    void replace_locked(std::shared_ptr<const StoredImageJob> owned, std::size_t envelope_bytes);
    void evict_terminal_locked(const std::string& protect_id);

    std::size_t max_records_ = 0;
    std::size_t max_bytes_   = 0;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Entry> records_;
    std::list<std::string> lru_;
    std::size_t current_bytes_ = 0;
};

} // namespace infernix::serve
