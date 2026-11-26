#include "models/qwen4_exp/program/program_impl.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
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
ExpertRamFill Program::expert_ram_fill() const noexcept { return impl_->expert_ram_fill(); }
std::optional<ExpertCacheTotals> Program::expert_cache_totals() const noexcept { return impl_->expert_cache_totals(); }
MemoryUsage Program::memory_usage() const noexcept { return impl_->memory_usage(); }

void Program::set_maintenance_waker(std::function<void()> waker) { impl_->set_maintenance_waker(std::move(waker)); }

// The Engine worker's side of the image-interlude handshake (the submit door and the preemption ack). The
// common EngineCore reaches these only through the kProgramInterlude contract test, so a model family with
// no interlude (Inkling 2) never sees them; they are thin forwards, and the worker must never need the
// Program's execution lock to make an ack -- which is why the door is an atomic, not a mutex-guarded read.
bool Program::submit_door_open() const noexcept { return impl_->submit_open(); }

bool Program::interlude_degraded() const noexcept { return impl_->interlude_degraded(); }

bool Program::interlude_preempt_outstanding() const noexcept { return impl_->interlude_preempt_outstanding(); }

std::uint64_t Program::interlude_preempt_epoch() const noexcept { return impl_->interlude_preempt_epoch(); }

void Program::signal_interlude_quiesced(std::uint64_t epoch) noexcept { impl_->signal_interlude_quiesced(epoch); }

bool Program::interlude_yieldable() const noexcept { return impl_->interlude_yieldable(); }

// Maintenance is a device-doing entry (VRAM retarget, expert-state save, RAM tiers, SSD write-back): it
// counts for the interlude drain exactly like the execution entries, so an image yield never unmaps the
// pools under a resize that is already in flight.
// A verification round's expert-cache update may wait for the next round's proposal (ProgramImpl::DeferredRound), and
// a DFlash2 verification's commit waits on the device behind a host gate until commit() (ProgramImpl::CommitGate):
// the entry points that can touch the expert cache, launch a forward or report on it release the gate with no
// committed column and settle the update first; decode settles it while its proposal drafts, commit defers it, and
// planning and accessors leave both.
void Program::maintain() {
    detail::ProgramImpl::LlmEntryScope entry{*impl_};
    impl_->settle_deferred();
    impl_->apply_vram_target(true);
    impl_->maintain_expert_state();
    impl_->maintain_ram(true);
    impl_->maintain_storage();
}

// It settles a deferred round only when the SSD expert tier resizes (ProgramImpl::quiet_expert_tier).
void Program::maintain_busy() {
    detail::ProgramImpl::LlmEntryScope entry{*impl_}; // device-doing: counted for the interlude drain
    impl_->maintain_ram(false);
}

void Program::start_ram_resize(std::uint64_t reserve_bytes, bool trim_host_tier) {
    impl_->start_ram_resize(reserve_bytes, trim_host_tier);
}

HybridPrefixCacheStats Program::hybrid_stats() const noexcept { return impl_->prefix_stats(); }

RequestBasePlan Program::plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(std::move(prompt), options);
}

std::vector<SourceCandidate> Program::hybrid_sources(const RequestBasePlan& base, const ResumeState* resume) {
    return impl_->hybrid_sources(base, resume);
}

BindingReservation Program::start_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                          const SourceCandidate& source, ResumeState* resume,
                                          ExecutionUnitKind, std::uint32_t) {
    impl_->settle_deferred();
    return impl_->start_binding(base, lane, source, resume);
}

ContextProgress Program::poll_context(runtime::CancellationFlagView cancellation) {
    impl_->settle_deferred();
    return impl_->poll_context(cancellation);
}

bool Program::has_context_transaction() const noexcept { return impl_->in_transaction(); }

bool Program::context_blocks(SequenceHandle sequence) const noexcept { return impl_->context_blocks(sequence); }

bool Program::hybrid_reclaim(runtime::ContextResourceUsage shortage) {
    impl_->settle_deferred();
    return impl_->hybrid_reclaim(shortage);
}
std::optional<std::uint32_t> Program::hybrid_prefetch(const RequestBasePlan& base, bool spare_only) {
    // PORT(ported tree): the body lives in prefix/prefix_program.cpp, outside the files this port owns, so
    // the entry is taken here: a restore batch enqueues transfers that must retire before an image yield
    // unmaps the pools they land in.
    detail::ProgramImpl::LlmEntryScope entry{*impl_};
    impl_->settle_deferred();
    return impl_->hybrid_prefetch(base, spare_only);
}
std::uint32_t Program::hybrid_prefetch_room() const noexcept { return impl_->hybrid_prefetch_room(); }
void Program::hybrid_hold_queue(std::span<const RequestBasePlan* const> queue,
                                std::span<const ResumeState* const> resumes) {
    impl_->hybrid_hold_queue(queue, resumes);
}
std::uint64_t Program::hybrid_cache_epoch() const noexcept { return impl_->hybrid_cache_epoch(); }

bool Program::hybrid_poll_storage() { return impl_->hybrid_poll_storage(); }

bool Program::hybrid_source_pending() const noexcept { return impl_->hybrid_source_pending(); }

bool Program::hybrid_storage_due() const noexcept { return impl_->hybrid_storage_due(); }
std::optional<std::chrono::steady_clock::time_point> Program::hybrid_storage_retry_at() const noexcept {
    return impl_->hybrid_storage_retry_at();
}
std::uint32_t Program::hybrid_resume_grid() const noexcept { return impl_->hybrid_resume_grid(); }

runtime::ResourceReservation Program::reserve_units(std::span<const ExecutionUnit> units) {
    return impl_->reserve_units(units);
}

bool Program::start_pause(SequenceHandle sequence, runtime::ExecutionTiming*) {
    impl_->settle_deferred();
    return impl_->start_pause(sequence);
}

std::uint32_t Program::pause_replay_tokens(SequenceHandle sequence) const noexcept {
    return impl_->pause_replay_tokens(sequence);
}

ReplayProgress Program::advance_replay(SequenceHandle sequence, runtime::ExecutionTiming*) {
    impl_->settle_deferred();
    return impl_->advance_replay(sequence);
}

PrefillProgress Program::advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming*,
                                         runtime::TokenMaskProvider* masks) {
    impl_->settle_deferred();
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
    impl_->settle_deferred();
    return impl_->append_forced(sequences, row_major_tokens, row_stride);
}

CommitResult Program::commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation, runtime::ExecutionTiming*) {
    return impl_->commit(std::move(pending), decisions);
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept { return impl_->discard(std::move(pending)); }

FinishResult Program::finish(SequenceHandle sequence) noexcept {
    impl_->settle_deferred_or_drop();
    return impl_->finish(sequence);
}

bool Program::can_finish_held(SequenceHandle sequence) const noexcept { return impl_->can_finish_held(sequence); }

FinishResult Program::finish_held(SequenceHandle sequence) noexcept {
    impl_->settle_deferred_or_drop();
    return impl_->finish_held(sequence);
}

void Program::release_finished(SequenceHandle sequence) noexcept { impl_->release_finished(sequence); }

AbortResult Program::abort(SequenceHandle sequence) noexcept {
    impl_->settle_deferred_or_drop();
    return impl_->abort(sequence);
}

void Program::fail_all_cleanup() noexcept { impl_->release_all(); }

bool Program::recover_after_failure() noexcept { return impl_->recover_after_failure(); }

void Program::shutdown_cleanup() noexcept {
    impl_->settle_deferred_or_drop();
    impl_->report_cache_layers();
    impl_->release_all();
    impl_->save_expert_state();
    impl_->save_prefix_cache_for_shutdown();
}

runtime::prefix_cache::PersistResult Program::attach_prefix_cache_file(const std::filesystem::path& path, std::string fingerprint,
                                                         const StartupObserver& observer) {
    impl_->settle_deferred();
    return impl_->attach_prefix_cache_file(path, std::move(fingerprint), observer);
}

std::optional<runtime::prefix_cache::PersistResult> Program::prefix_shutdown_save() const { return impl_->prefix_shutdown_save(); }

ImageYieldProbe Program::probe_image_yield(std::uint64_t target_bytes, std::uint64_t margin_bytes) const {
    return impl_->probe_image_yield(target_bytes, margin_bytes);
}

std::uint64_t Program::run_image_interlude(std::uint64_t target_bytes, std::uint64_t margin_bytes,
                                           const std::function<bool(const ImageInterludeContext&)>& job,
                                           std::uint64_t drain_deadline_ms) {
    // Single-flight: one interlude per Program, and nothing waits behind this gate -- a caller that
    // reaches it concurrently is told so, never silently serialized, and the serving layer turns that
    // into 429 image_interlude_in_flight for the request.
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
    // A cudaStream_t takes the current device of the thread that CREATES it, and this thread is
    // whichever serving worker asked for a picture: src/core/device.cu binds a DeviceContext to the
    // thread that builds it and nothing else, so this worker had no chosen device at all. The lane's
    // Bufs allocate with cudaMallocAsync, which the runtime documents as allocating "from the memory
    // pool associated with the stream's device", so a stream left on the worker's default would put the
    // image job's bytes -- and the LLM's restore, which goes looking for them with plain cudaMalloc --
    // on a device the engine never picked. The engine's own DeviceContext is the source of that choice
    // (the Engine built it from EngineOptions::device); the thread is put straight back afterwards.
    const int engine_device = impl_->device_.device;
    int       thread_device = -1;
    // False until the cudaGetDevice below has actually read thread_device. Without it a failed probe would
    // leave the sentinel -1 in place and teardown would report a bogus "cudaSetDevice (the serving worker
    // back to device -1)" beside the real cause; there is no thread binding to put back when none was read.
    bool thread_known = false;

    // cudaStreamDestroy is as device-bound as cudaStreamCreate, so the teardown binds the same way and
    // never leaves this worker thread switched. Best effort and never silent: if the bind to the engine's
    // device itself fails the stream is still destroyed (on whatever device this thread holds) rather than
    // skipped, because a leaked stream strands bytes the lane's single account of the card can no longer
    // name -- and every refusal comes back as text the caller folds into the interlude error.
    auto destroy_image_stream = [&]() -> std::string {
        if (image_stream == nullptr) { return {}; }
        std::string report;
        bool        ours = false; // did a call below produce a sticky error that is OURS to consume?
        const cudaError_t bind = cudaSetDevice(engine_device);
        if (bind != cudaSuccess) {
            ours   = true;
            report = "cudaSetDevice (the engine's device " + std::to_string(engine_device) +
                     " for the teardown): " + cudaGetErrorString(bind);
        }
        if (const cudaError_t destroyed = cudaStreamDestroy(image_stream); destroyed != cudaSuccess) {
            ours = true;
            if (!report.empty()) { report += "; "; }
            report += std::string("cudaStreamDestroy (the image stream of device ") +
                      std::to_string(engine_device) + "): " + cudaGetErrorString(destroyed);
        }
        // Only tried when this thread was actually moved AND its original device was read: otherwise it was
        // never switched by us and the worker's binding is nobody else's to change. The thread_known test
        // keeps a never-read sentinel out of the message rather than asking CUDA about device -1.
        if (bind == cudaSuccess && thread_known) {
            if (const cudaError_t back = cudaSetDevice(thread_device); back != cudaSuccess) {
                ours = true;
                if (!report.empty()) { report += "; "; }
                report += "cudaSetDevice (the serving worker back to device " + std::to_string(thread_device) +
                          "): " + cudaGetErrorString(back);
            }
        }
        if (ours) { (void)cudaGetLastError(); } // all of it is in the text returned
        image_stream = nullptr;
        return report;
    };

    // Every device call here is an INTERLUDE failure, never an exception that reaches the server's
    // worker thread: the stream (if any) is destroyed, the yield is unwound through restore_and_reopen
    // -- which parks the Program Degraded when the restoration is uncertain -- and the folded reason
    // becomes the interlude error the caller reports. The refusal's own CUDA flag is consumed because
    // it is already in the text thrown, and a sticky one would be charged to the next unrelated call.
    auto interlude_device_failure = [&](const char* call, cudaError_t error) {
        const std::string teardown    = destroy_image_stream();
        (void)cudaGetLastError(); // the caller's failing call raised it, and it is in the text thrown below
        const std::string restoration = restore_and_reopen();
        std::string tail = restoration.empty() ? std::string("the LLM pools were restored") : restoration;
        if (!teardown.empty()) { tail = teardown + "; " + tail; } // folded, never dropped
        throw std::runtime_error("image interlude: " + std::string(call) + " on the engine's device " +
                                 std::to_string(engine_device) + " failed (" + cudaGetErrorString(error) +
                                 "); " + tail);
    };

    if (const cudaError_t probe = cudaGetDevice(&thread_device); probe != cudaSuccess) {
        interlude_device_failure("reading the calling thread's current device (cudaGetDevice)", probe);
    }
    thread_known = true; // thread_device now holds the device this worker actually stood on
    // Blocking stream (not cudaStreamNonBlocking): the legacy default stream orders against it, so
    // image work the job submits on either route stays sequenced without cross-stream events.
    if (const cudaError_t bind = cudaSetDevice(engine_device); bind != cudaSuccess) {
        interlude_device_failure("binding the creating thread (cudaSetDevice)", bind);
    }
    if (const cudaError_t created = cudaStreamCreate(&image_stream); created != cudaSuccess) {
        interlude_device_failure("cudaStreamCreate", created);
    }
    if (const cudaError_t back = cudaSetDevice(thread_device); back != cudaSuccess) {
        // The stream exists on the engine's device; handing it back before refusing keeps the lane's
        // single account of the card clean, and the worker's binding is what could not be restored.
        interlude_device_failure("restoring the calling thread's device (cudaSetDevice)", back);
    }
    ImageInterludeContext context{lease.freed_bytes, lease.device_free_bytes, image_stream};
    try {
        job(context);
    } catch (...) {
        const std::string teardown = destroy_image_stream();
        std::string job_message = "non-standard job exception";
        try {
            std::rethrow_exception(std::current_exception());
        } catch (const std::exception& failure) {
            job_message = failure.what();
        }
        const std::string restoration = restore_and_reopen();
        if (restoration.empty()) {
            if (teardown.empty()) {
                throw; // job failed but healthy: the job's own exception propagates, the LLM resumed
            }
            // The LLM resumed, so this is not a Degraded parking and the job's exception still stands --
            // but a stream or a worker binding the lane could not hand back is not the job's fault either,
            // so it is reported beside the job's text instead of being swallowed by the rethrow.
            throw std::runtime_error("image interlude job failed: " + job_message + "; " + teardown +
                                     "; the LLM pools were restored and the Program is Running");
        }
        // The store of record keeps BOTH failures; so does the thrown error, and the teardown's too.
        throw std::runtime_error("image interlude job failed: " + job_message + "; " + restoration +
                                 (teardown.empty() ? std::string() : "; " + teardown) +
                                 "; the Program is Degraded until process restart");
    }
    const std::string teardown = destroy_image_stream();
    // Success is never published before the restoration is confirmed.
    if (const std::string restoration = restore_and_reopen(); !restoration.empty()) {
        throw std::runtime_error("image interlude job completed but: " +
                                 (teardown.empty() ? restoration : teardown + "; " + restoration) +
                                 "; the Program is Degraded until process restart");
    }
    if (!teardown.empty()) {
        // The job succeeded and the LLM is whole again, so the only doubt left is the lane's own teardown.
        throw std::runtime_error("image interlude job completed but: " + teardown +
                                 "; the LLM pools were restored and the Program is Running");
    }
    return lease.freed_bytes;
}

std::optional<runtime::prefix_cache::PersistResult> Program::save_prefix_cache_now(const CancellationView& abandoned) {
    impl_->settle_deferred();
    return impl_->save_prefix_cache_now(abandoned);
}

ScoreResult Program::causal_score(PreparedPrompt&& prompt, std::uint32_t first_target, const ScoreOptions& options) {
    // PORT(ported tree): the body lives in score.cpp, outside the files this port owns, so the entry is
    // taken here: offline CausalScoring runs the same forwards on the same streams as generation, and an
    // interlude must not unmap the pools under one.
    detail::ProgramImpl::LlmEntryScope entry{*impl_};
    impl_->settle_deferred();
    return impl_->causal_score(std::move(prompt), first_target, options);
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept { return impl_->usage(); }

MemorySummary Program::memory_summary() const noexcept { return impl_->memory(); }

namespace {
std::atomic<std::uint32_t> g_restore_fault{0}; // consumed by ProgramImpl::take_image_restore_fault
std::atomic<std::uint32_t> g_kv_remap_fault{0}; // consumed by ProgramImpl::take_kv_remap_fault
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

bool ProgramImpl::take_kv_remap_fault() noexcept {
    std::uint32_t left = g_kv_remap_fault.load(std::memory_order_relaxed);
    while (left != 0) {
        if (g_kv_remap_fault.compare_exchange_weak(left, left - 1)) { return left == 1; }
    }
    return false;
}
} // namespace detail

namespace testing {

void set_image_restore_fault(std::uint32_t checks) noexcept { g_restore_fault.store(checks); }

void set_kv_remap_fault(std::uint32_t checks) noexcept { g_kv_remap_fault.store(checks); }

std::uint32_t kv_backed_pages() noexcept {
    return detail::ProgramImpl::has_active() ? detail::ProgramImpl::kv_backed_pages_testing() : 0;
}

void kv_hold_pages(std::uint32_t pages) {
    if (!detail::ProgramImpl::has_active()) { throw std::runtime_error("Qwen4Exp: no Program is loaded to hold KV"); }
    detail::ProgramImpl::kv_hold_testing(pages);
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
