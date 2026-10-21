#include "serve/image_job_store.h"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infernix::serve {
namespace {

std::size_t record_envelope_bytes(const StoredImageJob& record) {
    std::size_t bytes = sizeof(StoredImageJob) + record.id.size();
    if (!record.request.is_null())           { bytes += record.request.dump().size(); }
    if (!record.response.is_null())          { bytes += record.response.dump().size(); }
    if (!record.error.is_null())             { bytes += record.error.dump().size(); }
    if (!record.restoration_error.is_null()) { bytes += record.restoration_error.dump().size(); }
    return bytes;
}

std::size_t checked_add(std::size_t left, std::size_t right, const char* label) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error(label);
    }
    return left + right;
}

[[noreturn]] void throw_api(int status, const char* type, const char* code, const std::string& message) {
    ApiError error;
    error.status  = status;
    error.type    = type;
    error.code    = code;
    error.message = message;
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_store_capacity() {
    throw_api(500, "server_error", "image_store_capacity_exceeded",
              "image job exceeds the configured local image store capacity");
}

[[noreturn]] void reject_transition(const std::string& reason) {
    throw_api(409, "invalid_request_error", "image_job_transition_rejected", reason);
}

} // namespace

bool is_terminal(ImageJobState state) noexcept {
    return state == ImageJobState::Succeeded || state == ImageJobState::Failed ||
           state == ImageJobState::Cancelled || state == ImageJobState::FailedEngineDegraded;
}

bool can_transition(ImageJobState from, ImageJobState to) noexcept {
    switch (from) {
        case ImageJobState::Queued:
            return to == ImageJobState::WaitingForFrontier || to == ImageJobState::Failed || to == ImageJobState::Cancelled;
        case ImageJobState::WaitingForFrontier:
            return to == ImageJobState::Running || to == ImageJobState::Failed || to == ImageJobState::Cancelled;
        case ImageJobState::Running:
            return to == ImageJobState::Restoring || to == ImageJobState::Failed || to == ImageJobState::Cancelled;
        case ImageJobState::Restoring:
            return to == ImageJobState::Succeeded || to == ImageJobState::Failed ||
                   to == ImageJobState::FailedEngineDegraded;
        default:
            return false; // terminal states are absorbing
    }
}

const char* state_to_string(ImageJobState state) noexcept {
    switch (state) {
        case ImageJobState::Queued:               return "queued";
        case ImageJobState::WaitingForFrontier:   return "waiting_for_frontier";
        case ImageJobState::Running:              return "running";
        case ImageJobState::Restoring:            return "restoring";
        case ImageJobState::Succeeded:            return "succeeded";
        case ImageJobState::Failed:               return "failed";
        case ImageJobState::Cancelled:            return "cancelled";
        case ImageJobState::FailedEngineDegraded: return "failed_engine_degraded";
    }
    return "unknown";
}

ImageJobStore::ImageJobStore(std::size_t max_records, std::size_t max_bytes)
    : max_records_(max_records), max_bytes_(max_bytes) {
    if (max_records_ == 0 || max_bytes_ == 0) {
        throw std::invalid_argument("image store limits must be positive");
    }
}

std::shared_ptr<const StoredImageJob> ImageJobStore::get(const std::string& id) {
    std::lock_guard lock(mutex_);
    const auto found = records_.find(id);
    if (found == records_.end()) { return {}; }
    lru_.splice(lru_.begin(), lru_, found->second.lru);
    return found->second.job;
}

void ImageJobStore::put(StoredImageJob job) {
    if (job.id.empty()) {
        throw std::invalid_argument("stored image job must have an id");
    }
    const std::size_t envelope_bytes = record_envelope_bytes(job);
    if (envelope_bytes > max_bytes_) { throw_store_capacity(); }

    auto owned = std::make_shared<const StoredImageJob>(std::move(job));
    std::lock_guard lock(mutex_);
    if (records_.contains(owned->id)) {
        // The idempotency policy: a retry re-discovers the job through get(); it never silently replaces it.
        throw_api(409, "invalid_request_error", "image_job_already_exists",
                  "image job id " + owned->id + " already exists in the store");
    }
    while (records_.size() + 1 > max_records_ || current_bytes_ + envelope_bytes > max_bytes_) {
        evict_terminal_locked(/*protect_id=*/{});
    }
    lru_.push_front(owned->id);
    try {
        records_.emplace(owned->id, Entry{owned, lru_.begin(), envelope_bytes});
    } catch (...) {
        lru_.pop_front();
        throw;
    }
    try {
        current_bytes_ = checked_add(current_bytes_, envelope_bytes,
                                     "image store byte accounting overflowed");
    } catch (...) {
        records_.erase(owned->id);
        lru_.pop_front();
        throw;
    }
}

std::shared_ptr<const StoredImageJob> ImageJobStore::advance(const std::string& id, const JobUpdate& update) {
    std::lock_guard lock(mutex_);
    const auto found = records_.find(id);
    if (found == records_.end()) {
        reject_transition("image job " + id + " is not in the store");
    }
    const StoredImageJob& current = *found->second.job;
    if (!can_transition(current.state, update.to)) {
        reject_transition(std::string("image job ") + id + " cannot move from " + state_to_string(current.state) +
                          " to " + state_to_string(update.to));
    }
    if (!update.error.is_null() && update.to != ImageJobState::Failed &&
        update.to != ImageJobState::FailedEngineDegraded) {
        reject_transition("image job " + id + " may carry an error only into a failed state");
    }
    if (!update.restoration_error.is_null() && update.to != ImageJobState::FailedEngineDegraded) {
        reject_transition("image job " + id + " may carry a restoration error only into failed_engine_degraded");
    }

    StoredImageJob next = current;
    next.state = update.to;
    if (!update.error.is_null())             { next.error = update.error; }
    if (!update.restoration_error.is_null()) { next.restoration_error = update.restoration_error; }
    if (!update.response.is_null())          { next.response = update.response; }
    if (is_terminal(update.to))              { next.finished = update.now; }

    const std::size_t envelope_bytes = record_envelope_bytes(next);
    if (envelope_bytes > max_bytes_) { throw_store_capacity(); }

    auto owned = std::make_shared<const StoredImageJob>(std::move(next));
    replace_locked(std::move(owned), envelope_bytes);
    return records_.at(id).job;
}

bool ImageJobStore::cancel(const std::string& id, std::int64_t now) {
    std::lock_guard lock(mutex_);
    const auto found = records_.find(id);
    if (found == records_.end() || is_terminal(found->second.job->state)) { return false; }
    StoredImageJob next = *found->second.job;
    next.state    = ImageJobState::Cancelled;
    next.finished = now;
    const std::size_t envelope_bytes = record_envelope_bytes(next);
    if (envelope_bytes > max_bytes_) { throw_store_capacity(); }

    auto owned = std::make_shared<const StoredImageJob>(std::move(next));
    replace_locked(std::move(owned), envelope_bytes);
    return true;
}

std::size_t ImageJobStore::prune_expired(std::int64_t now, std::int64_t ttl_seconds) {
    std::lock_guard lock(mutex_);
    std::vector<std::string> expired;
    for (const auto& [id, entry] : records_) {
        if (is_terminal(entry.job->state) && entry.job->finished != 0 &&
            entry.job->finished + ttl_seconds < now) {
            expired.push_back(id);
        }
    }
    for (const auto& id : expired) { erase_locked(id); }
    return expired.size();
}

bool ImageJobStore::erase(const std::string& id) {
    std::lock_guard lock(mutex_);
    if (!records_.contains(id)) { return false; }
    erase_locked(id);
    return true;
}

std::size_t ImageJobStore::size() const {
    std::lock_guard lock(mutex_);
    return records_.size();
}

std::size_t ImageJobStore::bytes() const {
    std::lock_guard lock(mutex_);
    return current_bytes_;
}

void ImageJobStore::replace_locked(std::shared_ptr<const StoredImageJob> owned, std::size_t envelope_bytes) {
    const std::string id = owned->id;
    erase_locked(id);
    lru_.push_front(id);
    try {
        records_.emplace(id, Entry{std::move(owned), lru_.begin(), envelope_bytes});
    } catch (...) {
        lru_.pop_front();
        throw;
    }
    try {
        current_bytes_ = checked_add(current_bytes_, envelope_bytes,
                                     "image store byte accounting overflowed");
    } catch (...) {
        records_.erase(id);
        lru_.pop_front();
        throw;
    }
}

void ImageJobStore::erase_locked(const std::string& id) {
    const auto found = records_.find(id);
    if (found == records_.end()) { return; }
    if (current_bytes_ < found->second.envelope_bytes) {
        throw std::logic_error("image envelope byte accounting underflowed");
    }
    current_bytes_ -= found->second.envelope_bytes;
    lru_.erase(found->second.lru);
    records_.erase(found);
}

void ImageJobStore::evict_terminal_locked(const std::string& protect_id) {
    for (auto victim = lru_.end(); victim != lru_.begin();) {
        --victim;
        if (*victim == protect_id) { continue; }
        const auto entry = records_.find(*victim);
        if (entry != records_.end() && is_terminal(entry->second.job->state)) {
            erase_locked(*victim);
            return;
        }
    }
    throw_api(503, "server_error", "image_store_active_full",
              "the local image store is at capacity and every stored job is still active");
}

} // namespace infernix::serve
