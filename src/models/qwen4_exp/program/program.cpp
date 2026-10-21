#include "models/qwen4_exp/program/program_impl.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>

namespace infernix::models::qwen4_exp {

RequestBasePlan::RequestBasePlan(std::unique_ptr<detail::BasePlanImpl> impl) noexcept : impl_(std::move(impl)) {}
RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept            = default;
RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;
RequestBasePlan::~RequestBasePlan()                                     = default;

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept { return impl_->summary; }

ResumeState::ResumeState(std::unique_ptr<detail::ResumeStateImpl> impl) noexcept : impl_(std::move(impl)) {}
ResumeState::ResumeState(ResumeState&&) noexcept            = default;
ResumeState& ResumeState::operator=(ResumeState&&) noexcept = default;
ResumeState::~ResumeState()                                 = default;
std::uint32_t ResumeState::frontier() const noexcept { return impl_ ? impl_->target : 0U; }

Program::Program(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options)
    : impl_(std::make_unique<detail::ProgramImpl>(parameters, device, std::move(options))) {
    detail::ProgramImpl::set_active(impl_.get()); // fully built: safe for the coexistence test to reach
}

Program::~Program() noexcept {
    detail::ProgramImpl::set_active(nullptr);
    impl_.reset();
}

ProgramDevicePlan Program::plan_device(const ProgramOptions& options, const Config& config,
                                       std::uint32_t public_tokens) {
    return detail::ProgramImpl::plan_device(options, config, static_cast<std::int32_t>(public_tokens)).bytes;
}

const ProgramDevicePlan& Program::device_plan() const noexcept { return impl_->device_layout().bytes; }

const VramSizing& Program::vram_sizing() const noexcept { return impl_->vram_sizing(); }

void Program::set_maintenance_waker(std::function<void()> waker) { impl_->set_maintenance_waker(std::move(waker)); }

void Program::maintain() { impl_->maintain_all(); }

bool Program::submit_door_open() const noexcept { return impl_->submit_open(); }

bool Program::interlude_degraded() const noexcept { return impl_->interlude_degraded(); }

bool Program::interlude_preempt_outstanding() const noexcept { return impl_->interlude_preempt_outstanding(); }

std::uint64_t Program::interlude_preempt_epoch() const noexcept { return impl_->interlude_preempt_epoch(); }

void Program::signal_interlude_quiesced(std::uint64_t epoch) noexcept { impl_->signal_interlude_quiesced(epoch); }

bool Program::interlude_yieldable() noexcept { return impl_->interlude_yieldable(); }

HybridPrefixCacheStats Program::hybrid_stats() const noexcept { return impl_->prefix_stats(); }

RequestBasePlan Program::plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(std::move(prompt), options);
}

bool Program::isolated_request_feasible(const RequestBasePlan& base) const noexcept { return impl_->feasible(base); }

std::vector<SourceCandidate> Program::hybrid_sources(const RequestBasePlan& base, std::uint32_t maximum_frontier) {
    return impl_->hybrid_sources(base, maximum_frontier);
}

BindingReservation Program::start_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                          const SourceCandidate& source, ResumeState* resume,
                                          ExecutionUnitKind, std::uint32_t) {
    const runtime::ResourceReservation reserved = impl_->start_binding(base, lane, source, resume);
    BindingReservation out;
    out.reserved = static_cast<bool>(reserved);
    out.shortage = reserved.shortage;
    return out;
}

ContextProgress Program::poll_context(runtime::CancellationFlagView cancellation) {
    return impl_->poll_context(cancellation);
}

bool Program::has_context_transaction() const noexcept { return impl_->in_transaction(); }

bool Program::context_blocks(SequenceHandle sequence) const noexcept { return impl_->context_blocks(sequence); }

bool Program::hybrid_reclaim(runtime::ContextResourceUsage shortage) { return impl_->hybrid_reclaim(shortage); }
std::optional<std::uint32_t> Program::hybrid_prefetch(const RequestBasePlan& base) { return impl_->hybrid_prefetch(base); }
std::uint32_t Program::hybrid_prefetch_room() const noexcept { return impl_->hybrid_prefetch_room(); }
void Program::hybrid_hold_queue(std::span<const RequestBasePlan* const> queue) { impl_->hybrid_hold_queue(queue); }
std::uint64_t Program::hybrid_cache_epoch() const noexcept { return impl_->hybrid_cache_epoch(); }

runtime::ResourceReservation Program::reserve_units(std::span<const ExecutionUnit> units) {
    return impl_->reserve_units(units);
}

bool Program::start_pause(SequenceHandle sequence, runtime::ExecutionTiming*) {
    return impl_->start_pause(sequence);
}

ReplayProgress Program::advance_replay(SequenceHandle sequence, runtime::ExecutionTiming*) {
    return impl_->advance_replay(sequence);
}

PrefillProgress Program::advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming*,
                                         runtime::TokenMaskProvider* masks) {
    return impl_->advance_prefill(sequence, masks);
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming*, runtime::TokenMaskProvider* masks) {
    return impl_->decode(sequences, budgets, masks);
}

runtime::ExecutionTiming Program::append_forced_tokens(std::span<const SequenceHandle> sequences,
                                                       std::span<const TokenId> row_major_tokens,
                                                       std::uint32_t row_stride,
                                                       std::span<const std::optional<std::uint32_t>>,
                                                       runtime::ExecutionTiming*) {
    return impl_->append_forced(sequences, row_major_tokens, row_stride);
}

CommitResult Program::commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation, runtime::ExecutionTiming*) {
    return impl_->commit(std::move(pending), decisions);
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept { return impl_->discard(std::move(pending)); }

FinishResult Program::finish(SequenceHandle sequence) noexcept { return impl_->finish(sequence); }

AbortResult Program::abort(SequenceHandle sequence) noexcept { return impl_->abort(sequence); }

void Program::fail_all_cleanup() noexcept { impl_->release_all(); }

bool Program::recover_after_failure() noexcept { return impl_->recover_after_failure(); }

void Program::shutdown_cleanup() noexcept {
    impl_->report_cache_layers();
    impl_->release_all();
    impl_->save_expert_state();
    impl_->save_prefix_cache_for_shutdown();
}

PrefixCachePersistence Program::attach_prefix_cache_file(const std::filesystem::path& path, std::string fingerprint,
                                                         const StartupObserver& observer) {
    return impl_->attach_prefix_cache_file(path, std::move(fingerprint), observer);
}

std::optional<PrefixCachePersistence> Program::prefix_shutdown_save() const { return impl_->prefix_shutdown_save(); }

ImageYieldProbe Program::probe_image_yield(std::uint64_t target_bytes, std::uint64_t margin_bytes) const {
    return impl_->probe_image_yield(target_bytes, margin_bytes);
}

std::uint64_t Program::run_image_interlude(std::uint64_t target_bytes, std::uint64_t margin_bytes,
                                           const std::function<bool(const ImageInterludeContext&)>& job,
                                           std::uint64_t drain_deadline_ms) {
    // Single-flight: one interlude per Program. The serving layer's bounded FIFO queues in front of
    // this gate; a caller that reaches it concurrently is told so, never silently serialized.
    if (!impl_->interlude_mu_.try_lock()) {
        throw std::runtime_error("image interlude: another interlude is already in flight");
    }
    const std::lock_guard<std::mutex> flight(impl_->interlude_mu_, std::adopt_lock);

    // Reach a YIELDABLE frontier in ONE attempt: begin_image_quiesce closes the door, drains within
    // `drain_deadline_ms`, and checks the LLM is not holding the residency ring. The door stays CLOSED for
    // the whole handoff -- reopening between attempts would admit new LLM work exactly where the handoff
    // must be bounded. A false return means begin_image_quiesce cleanly unwound (door reopened, nothing
    // moved), so we fail ONLY this image job: the LLM keeps serving (never Degraded).
    if (!impl_->begin_image_quiesce(drain_deadline_ms)) {
        throw std::runtime_error("image interlude: no yieldable LLM frontier within the deadline; "
                                 "the image job was rejected and the LLM resumed");
    }

    detail::ProgramImpl::ImageYieldLease lease;
    try {
        lease = impl_->yield_for_image(target_bytes, margin_bytes);
    } catch (...) {
        // yield_for_image is transactional: it unwound its own pools back to the pre-yield snapshot, so the
        // Program is still cleanly Quiesced and the image leased nothing. Reopen the door and reject ONLY
        // this image job -- the LLM keeps serving. An image request must never take the engine down; park
        // Degraded only if even reopening proves the device state unrestorable.
        std::string yield_message = "the image yield failed";
        try {
            std::rethrow_exception(std::current_exception());
        } catch (const std::exception& failure) {
            yield_message = failure.what();
        }
        try {
            impl_->end_image_quiesce();
        } catch (const std::exception& reopen_failure) {
            impl_->mark_interlude_degraded();
            throw std::runtime_error("image interlude: " + yield_message + "; the engine could not be reopened (" +
                                     reopen_failure.what() + "); the Program is Degraded until process restart");
        }
        throw; // recovered cleanly: only this image job is rejected, the LLM resumed
    }

    // The explicit, reportable interlude exit: returns "" once the restoration is CONFIRMED and the
    // door reopened; otherwise parks the Program Degraded (door closed, process = recovery unit)
    // and returns the failure text to fold into the thrown error.
    auto restore_and_reopen = [&]() -> std::string {
        try {
            impl_->release_from_yield(lease);
        } catch (const std::exception& failure) {
            impl_->mark_interlude_degraded();
            return std::string("restoration failed: ") + failure.what();
        }
        try {
            impl_->end_image_quiesce();
            return {};
        } catch (const std::exception& failure) {
            impl_->mark_interlude_degraded();
            return std::string("the submit door did not reopen: ") + failure.what();
        }
    };

    cudaStream_t image_stream = nullptr;
    // Blocking stream (not cudaStreamNonBlocking): the legacy default stream orders against it, so
    // image work the job submits on either route stays sequenced without cross-stream events.
    if (cudaStreamCreate(&image_stream) != cudaSuccess) {
        const std::string restoration = restore_and_reopen();
        throw std::runtime_error("image interlude: the image stream could not be created; " +
                                 (restoration.empty() ? std::string("the LLM pools were restored") : restoration));
    }
    ImageInterludeContext context{lease.freed_bytes, image_stream};
    try {
        job(context);
    } catch (...) {
        cudaStreamDestroy(image_stream);
        std::string job_message = "non-standard job exception";
        try {
            std::rethrow_exception(std::current_exception());
        } catch (const std::exception& failure) {
            job_message = failure.what();
        }
        const std::string restoration = restore_and_reopen();
        if (restoration.empty()) {
            throw; // job failed but healthy: the job's own exception propagates, the LLM resumed
        }
        // The store of record keeps BOTH failures; so does the thrown error.
        throw std::runtime_error("image interlude job failed: " + job_message + "; " + restoration +
                                 "; the Program is Degraded until process restart");
    }
    cudaStreamDestroy(image_stream);
    // Success is never published before the restoration is confirmed.
    if (const std::string restoration = restore_and_reopen(); !restoration.empty()) {
        throw std::runtime_error("image interlude job completed but: " + restoration +
                                 "; the Program is Degraded until process restart");
    }
    return lease.freed_bytes;
}

std::optional<PrefixCachePersistence> Program::save_prefix_cache_now(const CancellationView& abandoned) {
    return impl_->save_prefix_cache_now(abandoned);
}

ScoreResult Program::causal_score(PreparedPrompt&& prompt, std::uint32_t first_target, const ScoreOptions& options) {
    return impl_->causal_score(std::move(prompt), first_target, options);
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept { return impl_->usage(); }

MemorySummary Program::memory_summary() const noexcept { return impl_->memory(); }

namespace {
std::atomic<std::uint32_t> g_restore_fault{0}; // consumed by ProgramImpl::take_image_restore_fault
} // namespace

namespace detail {
ProgramImpl* ProgramImpl::active_ = nullptr; // the live Program, registered/cleared by Program

bool ProgramImpl::take_image_restore_fault() noexcept {
    std::uint32_t left = g_restore_fault.load(std::memory_order_relaxed);
    while (left != 0) {
        if (g_restore_fault.compare_exchange_weak(left, left - 1)) { return left == 1; }
    }
    return false;
}
} // namespace detail

namespace testing {

namespace {
std::atomic<std::uint32_t> g_expert_fault{0};
} // namespace

void set_image_restore_fault(std::uint32_t checks) noexcept { g_restore_fault.store(checks); }

void set_expert_fault(std::uint32_t checks) noexcept { g_expert_fault.store(checks); }

bool take_expert_fault() noexcept {
    std::uint32_t left = g_expert_fault.load(std::memory_order_relaxed);
    while (left != 0) {
        if (g_expert_fault.compare_exchange_weak(left, left - 1)) { return left == 1; }
    }
    return false;
}

bool has_image_yield_program() noexcept { return detail::ProgramImpl::has_active(); }

std::uint64_t yield_image_vram(std::uint64_t target_bytes, std::uint64_t margin_bytes, std::uintptr_t& lease) {
    lease = 0;
    if (!detail::ProgramImpl::has_active()) { return 0; }
    return detail::ProgramImpl::yield_testing(target_bytes, margin_bytes, &lease);
}

void release_image_vram(std::uintptr_t lease) {
    if (lease != 0) { detail::ProgramImpl::release_testing(lease); }
}

std::uint64_t begin_image_quiesce() {
    if (!detail::ProgramImpl::has_active()) { return 0; }
    return detail::ProgramImpl::begin_quiesce_testing();
}

void end_image_quiesce() {
    if (detail::ProgramImpl::has_active()) { detail::ProgramImpl::end_quiesce_testing(); }
}

int image_interlude_state() noexcept {
    return detail::ProgramImpl::has_active() ? detail::ProgramImpl::state_testing() : -1;
}

bool image_interlude_graphs_inert() noexcept {
    return detail::ProgramImpl::has_active() && detail::ProgramImpl::graphs_inert_testing();
}

bool image_interlude_submit_open() noexcept {
    return !detail::ProgramImpl::has_active() || detail::ProgramImpl::submit_open_testing();
}

} // namespace testing

} // namespace infernix::models::qwen4_exp
