# Infernix: implementation plan for a RAM + SSD prefix cache

**Status:** proposal for a testable first implementation  
**Scope:** `infernix-serve`, initially Qwen3.8-Flash-Next (`Qwen4ExpForCausalLM`)  
**Non-goals:** DFlash2 work, changing model math, changing token/KV/state semantics, or replacing the current RAM cache in the first release.

## 1. Goal

Add an optional SSD backing tier to the existing prefix cache so that the server can trade some prefix-hit latency for a larger retained working set and more RAM available to the expert cache and the operating system.

Proposed command line:

```text
# Existing behaviour: preserve the current RAM-only cache behaviour
infernix-serve ... --prefix-mode ram --host-context-mib 4096

# Experimental tiered mode: 1 GiB RAM + up to 32 GiB of SSD-backed cache
infernix-serve ... \
  --prefix-mode hybrid \
  --host-context-mib 1024 \
  --prefix-ssd-mib 32768 \
  --prefix-ssd-file "D:/infernix-cache/prefix-ssd"
```

`--host-context-mib` remains the **RAM-tier budget**. `--prefix-ssd-mib` is an additional SSD-tier capacity, not a replacement for the RAM budget. Use MiB consistently in CLI/config; make the help text explicit that the SSD size is a logical live-data budget and disk usage may temporarily exceed it during segment compaction.

Keep `ram` as the default for backward compatibility. The `ram` mode must preserve today's behaviour, defaults, cache persistence semantics, and performance within measurement noise.

### Naming caution

The codebase already calls its token-prefix + recurrent-state cache the *Hybrid Prefix Cache* / `ContextCacheMode::Hybrid`. The new CLI `--prefix-mode ram|hybrid` describes **storage tiers**, not that existing cache algorithm. Use a separate type such as `PrefixStorageMode`; do not reuse or overload `ContextCacheMode`.

## 2. Current implementation and implications

Relevant repository documentation/code areas:

- `docs/qwen3_8-flash-next.md`, section **Prefix cache**: current `--host-context-mib` budget, cache snapshot size, `--prefix-cache-file`, and `--prefix-cache-save-mins` behaviour.
- `docs/maintainer/hybrid-prefix-cache-spec.md`: cache index and eviction decisions, state/KV ownership, write-through, persistence, and the Qwen4Exp binding.
- `src/runtime/prefix_cache/`: model-agnostic prefix index, block tree, snapshots, cache-cost model and eviction policy.
- `src/models/qwen4_exp/program/prefix/`: Flash-Next-specific host pool, snapshot/state image handling, restore, write-through and cache persistence.
- Core KV/state storage and transfer helpers: reuse their supported record-addressed copies where possible instead of inventing a second tensor-copy implementation.
- CLI/config parsing in the serving app: locate the existing `--host-context-mib`, `--prefix-cache-file` and `--prefix-cache-save-mins` definitions and follow the same option-registration path.

The current Flash-Next RAM pool holds **both** KV blocks and recurrent-state snapshots; it is not a snapshot-only cache. The model guide documents snapshots at approximately **116 MB each**. As an upper-bound illustration, 1 GiB divided by 116 MB is fewer than ten snapshot-equivalents, before allowing for KV blocks, allocator overhead and other cache objects. Likewise, 32 GiB of SSD is at most roughly 280 snapshot-equivalents before storing any KV. Actual retained prefixes will be fewer because the state snapshots and KV blocks both have to be available along a reusable path.

The current `--prefix-cache-file` is a persistence/checkpoint mechanism for the RAM host tier: it saves/restores a whole cache image and periodic saves rewrite that file. It is **not** an on-demand, per-object SSD tier. Do not treat simply increasing that file as equivalent to the feature proposed here.

## 3. Architecture

Keep the current prefix index and its semantic correctness rules. Add an SSD storage backend below the index / Flash-Next physical binding.

```text
Prompt / token path
       |
       v
 Existing PrefixIndex + cost model
       |
       +--------------------------+
       |                          |
       v                          v
 Device cache                 RAM host tier
 (unchanged)                  pinned, bounded by
                              --host-context-mib
                                   |
                             promotion / eviction
                                   |
                                   v
                            SSD backing tier
                            --prefix-ssd-mib
                            --prefix-ssd-file
```

**The index stays in RAM; tensor payloads do not need to.** Cache metadata should be small compared with payloads. The SSD tier backs KV blocks and state snapshots that have been evicted from the RAM pool. A logical prefix can therefore be represented by a mixture of Device-resident, RAM-resident and SSD-resident objects.

For the first version, an SSD hit should load objects to the normal pinned RAM pool and then use the existing RAM-to-Device restore path. This keeps the Device restore kernels and cache correctness machinery unchanged. Direct SSD-to-GPU / GPUDirect Storage, mmap-based serving, compressed payloads and custom GPU restore kernels are explicitly later optimizations, not prerequisites for the experiment.

### 3.1 Separate logical identity from physical residency

A cache node/object is logically present even if its payload is not in RAM. Track at least:

- whether the object is resident in Device, RAM, SSD, or more than one tier;
- SSD record/segment location and payload length;
- live refs/pins, restore pins, and in-flight I/O refs;
- object generation/version and I/O state;
- last-use/hit counters and estimated recomputation value.

Suggested I/O/residency states (exact representation is implementation-dependent):

```text
RAM_ONLY
SSD_WRITE_PENDING
RAM_AND_SSD
SSD_ONLY
SSD_READ_PENDING
RAM_LOADING
INVALID / EVICTED
```

Never make a new SSD record visible to lookups until its payload and commit record have been written successfully. Never free/reuse a RAM destination while an SSD read, H2D restore or another operation still owns it.

### 3.2 Storage ownership

The existing cache index remains the authority for whether a token path and state snapshot can be reused. The SSD store is only a payload store; it must not independently create prefix hits or make a prefix reusable without the existing complete state/KV requirements.

Preserve current rules for committed tokens, block-aligned snapshots, MTP/speculative committed frontiers, vision/media keys, request cancellation, queue holds, and the model-specific state image. A missing or corrupt SSD object is a cache miss, never permission to resume from incomplete state.

## 4. CLI and configuration

### 4.1 First-release options

| Option | Meaning | Proposed behaviour |
|---|---|---|
| `--prefix-mode <ram|hybrid>` | RAM-only or RAM + SSD backing tier | Default `ram`; no behaviour change in this mode |
| `--host-context-mib N` | RAM host pool | Existing meaning; in `hybrid`, this is the hot RAM budget |
| `--prefix-ssd-mib N` | Maximum live SSD payload budget | Required and positive in `hybrid` mode |
| `--prefix-ssd-file PATH` | SSD store base path/directory | Required in `hybrid`; must be on writable local storage for the recommended test |

Keep the first implementation deliberately small: no independent read-thread, staging-size or cache-policy CLI flags until measurements show a need. Use bounded, internal defaults and report them in startup logs.

### 4.2 Validation

- `ram`: follow existing defaults and behaviour exactly. If SSD-only flags are supplied in `ram` mode, reject them with a clear error rather than silently ignoring them.
- `hybrid`: require `--prefix-ssd-mib > 0`, a valid SSD path, and a positive RAM budget. Fail early if the path is a directory/file of the wrong kind, the parent does not exist, free space is clearly insufficient, or the store cannot be created.
- Do not silently fall back to RAM-only mode after an SSD I/O error; log the error and degrade the affected cache entries to misses. Serving may continue if correctness is preserved.
- Keep `--no-prefix-reuse` behaviour as it is. It disables prefix reuse rather than turning SSD into a standalone cache.
- In the first release, reject `--prefix-cache-file` / `--prefix-cache-save-mins` together with `--prefix-mode hybrid`, with a clear explanation that `--prefix-ssd-file` is the active backing store. This avoids two competing persistence mechanisms. A later release may migrate the old save/restore option to an import/export or compatibility layer.

### 4.3 Startup logging

Print a clear cache budget and configuration line, for example:

```text
prefix cache: mode=hybrid ram=1024 MiB ssd_live_limit=32768 MiB
prefix SSD: path=... segment_size=256 MiB async_io=yes
prefix SSD: index=... objects=... live_bytes=... recovered=...
```

Also report the current free space and whether the path looks like local storage when that can be determined reliably. Do not label a NAS path as local NVMe based only on its drive letter.

## 5. SSD data format

### 5.1 Design principles

Use a format that is:

1. Exact: raw serialized bytes, no quantization or lossy compression.
2. Versioned and self-describing enough to reject incompatible data.
3. Append-friendly: cache writes do not rewrite 32 GiB for every update.
4. Recoverable: truncated writes or crashes produce misses, not wrong output.
5. Batchable: adjacent reads/writes can be coalesced where practical.
6. Independent of process pointers, allocator addresses and the in-memory slab layout.

Do not write a raw dump of the pinned RAM pool. That would couple the cache to runtime-specific slab IDs, padding, pointer-like metadata and allocation geometry.

### 5.2 Logical records

Represent persisted payloads as typed records, at minimum:

- `KV_BLOCK`: one reusable logical KV block, serialized in a deterministic component order across KV planes/layers and any enabled draft-KV component;
- `STATE_IMAGE`: one complete recurrent/model state snapshot at a valid cache frontier;
- `TAIL` or equivalent snapshot-tail payload when required by the current representation.

A snapshot record must include every state component required for exact resume, not merely the large recurrent matrix. Prefer storing the required state bundle as one logical object even if the physical file record contains several components. Use a small descriptor table for component offsets/lengths.

### 5.3 Physical layout (recommended v1)

Use append-only, fixed-size **segment files** (for example, 256 MiB each) under the configured store base path, plus a compact metadata index and journal. Keep records 4 KiB-aligned; store headers and payloads in a deterministic little-endian format. The exact segment size should be a constant initially, not a user-facing knob.

A record header should include:

- magic + format version + record type;
- record/header length and payload length;
- object key / cache-object identity;
- model/artifact fingerprint;
- KV dtype/geometry and relevant state-schema identifier;
- sequence/block position or snapshot frontier where applicable;
- record generation/version;
- payload checksum and header checksum.

A trailing commit marker/checksum (or equivalent journal commit protocol) distinguishes a complete record from an interrupted write. An incomplete record is never published in the index.

The index maps a stable object key to segment ID, offset, length, generation, type and last-use/value metadata. On disk, references must use stable keys/offsets, never pointers or process-local `NodeId`s alone.

### 5.4 Stable identity and compatibility

The SSD store must be invalidated or entries rejected when any relevant compatibility field differs, including:

- target artifact/model fingerprint (prefer a real artifact identity or manifest fingerprint, not only the filename);
- model architecture and state-image schema/version;
- KV dtype/format, geometry and enabled draft-state/KV components;
- cache record format version and relevant runtime/cache ABI version;
- any positional/media identity needed by the existing token-path key, including vision content and positional effects.

Continue to use the existing exact token path plus media extras for prefix identity. A hash is a lookup accelerator, not proof that unrelated tokens are identical. On load, validate record metadata and checksums before use. A mismatch or failed validation must be treated as a cache miss and the record can be removed lazily.

### 5.5 Index recovery and crash safety

Use a small checkpointed index plus an append-only metadata journal so startup need not checksum or read all payload bytes in a 32 GiB store.

- Write payload + record commit first; publish its index entry only after successful completion.
- Journal insert, replacement, eviction and segment metadata changes.
- Write an updated index checkpoint to a temporary file, flush it, and atomically rename it; then rotate/truncate the covered journal.
- On startup, load a compatible index checkpoint and replay only valid journal entries. Verify payload records lazily on first load. An invalid/truncated tail is ignored.
- On orderly shutdown, flush outstanding write jobs, write the index checkpoint, and close handles cleanly. Cache correctness must not depend on a clean shutdown.

If implementation time demands an even smaller first spike, it may start with a non-persistent per-process store, but that should be an explicitly temporary experiment. Do not silently ship a design that scans and hashes the full 32 GiB on every start.

## 6. I/O path and latency

### 6.1 Bounded asynchronous I/O

Do not do blocking SSD reads/writes on the main serving/Engine worker. Add a small bounded I/O service with a completion queue that the scheduler/cache can poll.

- Writes are submitted after the source cache object is stable and no writer can mutate it.
- Reads target a reserved RAM destination or a bounded staging buffer; then populate the existing pinned RAM slabs.
- After an SSD read finishes, the existing restore path performs the RAM-to-Device copies.
- An awaiting request is marked restore-pending and yields to the scheduler while I/O runs; other requests should continue progressing.
- Use one bounded writer queue and a small number of read workers initially. More parallel I/O is not automatically faster: it can contend with expert reads, OS paging and NAS traffic.
- Batch adjacent component reads/writes and coalesce record operations when it is straightforward.
- Use platform abstractions for Windows and Linux. The first implementation can use a worker thread plus overlapped/asynchronous file operations supported by the platform abstraction; avoid requiring `io_uring` or platform-specific direct-storage APIs to get the feature working.

### 6.2 RAM usage

The feature must not accidentally add another 116 MiB or larger temporary buffer for every in-flight snapshot. Use a bounded staging pool (small fixed number of buffers, initially around 16–64 MiB total) or direct/chunked reads into known destination spans where the storage interface permits it. Report staging memory in the RAM ledger.

Since the existing Host pool is pinned for GPU transfer, avoid treating an arbitrary OS file mapping as if it were pinned host memory. `mmap`/memory mapping may be an optional experiment, but it is not the correctness-critical read path for v1.

### 6.3 Read benefit vs recomputation

Before choosing an SSD-only prefix candidate, estimate:

```text
T_ssd_restore = bytes_to_read / measured_effective_read_bandwidth
                + queue/dispatch latency
                + scatter/copy cost
                + normal RAM-to-Device restore cost

T_saved = estimated prefill time avoided by resuming at that frontier
```

Prefer the deepest reusable candidate whose saved prefill time clearly exceeds its SSD restore cost. If the SSD candidate is not worthwhile, use a shallower RAM/Device candidate or recompute. Use the existing cache cost model as the starting point and add SSD-read costs rather than introducing a separate incompatible selection algorithm.

For very long contexts, a cache hit may require many KV blocks in addition to the ~116 MB state snapshot. Track total bytes to load, not just the snapshot size.

## 7. Promotion, demotion and eviction

Keep the first policy explainable and measured. Reuse the current index's refcount/pin rules and GDSF/value model where possible; don't build a new general-purpose database cache policy from scratch.

### 7.1 Writes / demotion

1. New cacheable objects first follow the existing RAM write-through path.
2. If the SSD tier is enabled and the object is judged worth retaining, enqueue its serialization/write to SSD in the background.
3. While the write is pending, the RAM copy remains authoritative and cannot be evicted if it is the only valid copy.
4. After the SSD record is committed, publish its location and mark it as SSD-backed.
5. If RAM pressure requires eviction before the SSD write completes, either wait for the relevant asynchronous write completion without blocking the serving thread, or skip that object's demotion and let normal cache eviction/recomputation happen. Never evict the last valid copy.
6. If the SSD tier is full, evict low-value SSD objects/segments or reject new writes. Never block a request indefinitely to make room.

Only persist objects that are useful to a reusable prefix path. Avoid spending SSD bandwidth and write endurance on dead KV blocks that no retained snapshot can resume through. Preserve the existing superseded-snapshot and dead-KV knowledge from the prefix index.

### 7.2 Reads / promotion

- The first SSD hit should be eligible for a direct SSD-to-RAM restore, but don't automatically promote every one-time hit to long-lived RAM residency; otherwise a scan of unique prompts pollutes the hot tier.
- Promote objects to RAM when their reuse value/frequency is high enough or when the active request needs them and RAM capacity is available.
- If RAM is full, evict an unpinned RAM object only if it remains recoverable from SSD, or after a committed SSD write. Otherwise apply the existing normal eviction choice.
- Keep in-flight reads pinned until the request's restore has safely completed.
- When one request references a cache object, prevent concurrent eviction/compaction from invalidating its record until the reader releases its I/O pin.

A reasonable initial heuristic is “write worthwhile reusable objects to SSD; retain the hottest/frequent objects in RAM; let the existing cost model estimate value.” Keep the thresholds internal for the first benchmark rather than adding several premature command-line controls.

### 7.3 Capacity and compaction

Append-only records leave dead bytes after object eviction. Segment files simplify sequential I/O and deletion, but partial segments need compaction.

- Treat `--prefix-ssd-mib` as maximum **live payload** in v1; document that physical usage includes metadata, alignment and temporary compaction space.
- Delete a segment when none of its records are live or pinned.
- When garbage crosses a threshold (for example, >50% of the store or when allocation requires it), compact live records in a background/low-priority job, atomically update index locations, and only then delete old segments.
- Keep at least one-segment headroom for rotation and document the required free disk space.
- If compaction cannot proceed because there is not enough disk space, stop admitting new SSD records and keep serving from RAM/recompute; do not fail inference.

## 8. Correctness and error handling

Treat the SSD as a disposable cache, not a source of truth.

- A missing, malformed, truncated or checksum-failing record is a cache miss. Invalidate it and any affected snapshot path whose required KV/state pieces are no longer available.
- If a read fails after a request started restoring, release all read pins and temporary allocations; resume via a shallower safe hit or cold prefill/replay.
- Do not publish partially loaded state to the live request or index.
- Do not reuse buffers/records while an asynchronous operation is in flight.
- On write failure, keep the RAM copy if possible and mark the object as not SSD-backed.
- On index/journal mismatch, ignore affected entries and log a warning; the server must still start and serve.
- Preserve current exactness requirements for committed state, recurrent-state snapshots, KV pages, vision/media identity and cache paths.
- Cache hits must never cause a crash, wrong token, invalid state, or unrecoverable request. A cache miss or slower prefill is an acceptable fallback.

## 9. Implementation phases

### Phase 0 — Baseline and source audit

- [ ] Locate every use of `--host-context-mib`, the Flash-Next host pool allocator, snapshot image serialization, KV host copies and `--prefix-cache-file` persistence.
- [ ] Record the current RAM-only defaults and add a reproducible test command/replay.
- [ ] Add or expose baseline counters: cache hits by object/tier, cached prompt tokens, state snapshots, RAM live/pinned bytes, SSD bytes read/written, restore time, prefill time, TTFT and decode tok/s.
- [ ] Confirm whether the pinned Host pool can release capacity when reduced from 4096 to 1024 MiB, and exactly which RAM ledger entries shrink.

**Exit:** `ram` baseline can be reproduced and the memory freed by reducing the pool is measured, not assumed.

### Phase 1 — Introduce the storage abstraction without SSD behaviour changes

- [ ] Add `PrefixStorageMode { Ram, Hybrid }` as a separate setting.
- [ ] Route existing RAM host allocations/lookups through a small `PrefixPayloadStore`/tier interface while preserving code paths and layout.
- [ ] Keep `ram` mode byte-for-byte equivalent at the behaviour level.
- [ ] Add unit tests proving the abstraction preserves lookup, release, pins and eviction behaviour in RAM mode.

**Exit:** no measurable RAM-mode regression; no actual SSD I/O yet.

### Phase 2 — SSD record store in isolation

- [ ] Implement aligned segment creation, record header/payload/footer, checksum, metadata index and journal.
- [ ] Implement write/commit/read/invalidate/segment deletion and index recovery independent of model execution.
- [ ] Add crash/truncation and incompatible-version tests.
- [ ] Implement the bounded asynchronous I/O queue, cancellation/lifetime rules and test-only injected I/O errors.

**Exit:** storage round-trips records; after a simulated crash, only fully committed compatible records are returned.

### Phase 3 — Wire Flash-Next cache objects to SSD

- [ ] Add residency metadata for SSD-backed KV blocks and state snapshots in the Flash-Next prefix binding.
- [ ] Reuse existing `PrefixIndex` identity, pins, supersession, GDSF/value estimates and dead-KV accounting.
- [ ] Implement RAM-to-SSD serialization and SSD-to-RAM restoration for both KV and complete state images.
- [ ] Introduce asynchronous restore-pending handling so the main serving worker is not blocked on storage I/O.
- [ ] Add cost-aware hit selection: compare estimated SSD restore cost with prefill avoided.
- [ ] Start with a simple promotion policy; do not add mmap or direct-to-GPU restore yet.
- [ ] Wire `--prefix-mode`, `--prefix-ssd-mib` and `--prefix-ssd-file`, plus help text and startup ledger.

**Exit:** hybrid mode produces correct cached and uncached outputs; RAM mode is unchanged; SSD I/O failure falls back to recomputation.

### Phase 4 — Persistence and resource hardening

- [ ] Checkpoint the SSD index and replay its journal on startup.
- [ ] Add segment reclamation and background compaction.
- [ ] Add backpressure/throttling so cache writes do not compete excessively with model/expert reads.
- [ ] Ensure process shutdown flushes pending writes within a bounded policy; dirty entries may be discarded safely if shutdown is forced.
- [ ] Document store versioning, cache invalidation and expected disk-space headroom.

**Exit:** repeated stop/start preserves valid SSD cache entries; corrupt or stale entries become misses; disk usage remains bounded by the configured policy plus documented compaction headroom.

### Phase 5 — Performance tuning

Only after correctness passes:

- [ ] Tune segment size, queue depth and staging budget.
- [ ] Add optional sequential/read-ahead behaviour for adjacent KV blocks if traces show it helps.
- [ ] Consider layer-wise/pipelined SSD-to-RAM-to-Device restore if the full-object path is a measured bottleneck.
- [ ] Consider mmap or direct-storage alternatives only as separately benchmarked backends.
- [ ] Tune admission/promotion using observed reuse value, bytes and restore latency.

## 10. Test plan

### 10.1 Unit and storage tests

- [ ] Store/load each record type and compare raw bytes exactly.
- [ ] Alignment, empty/oversized records, segment rollover and capacity accounting.
- [ ] Header/payload checksum failure, truncated write, truncated journal, corrupt index and incompatible metadata.
- [ ] Eviction while record is pinned or being read; compaction while another request references a record.
- [ ] Failed read/write and cancellation while I/O is pending.
- [ ] Queue saturation/backpressure; no unbounded RAM growth.

### 10.2 Model-level correctness tests

For the same prompts and model artifact, compare `ram` vs `hybrid` on:

- [ ] cold prefix / no cache hit;
- [ ] repeated exact prefix;
- [ ] reused system + tools prefix and multi-turn agent continuation;
- [ ] branch/divergence before and after a stored snapshot;
- [ ] prompts long enough to require many KV blocks;
- [ ] concurrent requests and queue holds;
- [ ] request cancellation / abort during prefill;
- [ ] MTP/speculative-enabled path supported by the current Flash-Next setup;
- [ ] vision/media prompts if prefix cache supports them for the chosen artifact.

With deterministic single-request greedy generation, cached and uncached outputs should be identical wherever the existing cache mode promises exactness. Test token IDs and the first-token logits/state where the test harness supports it. In all cases, a cache error must safely fall back rather than resume with incomplete state.

### 10.3 Benchmark matrix

Run the same replay/seed set and report median plus p95, preferably at least three repetitions:

| Test | RAM tier | SSD tier | Purpose |
|---|---:|---:|---|
| A — baseline | 4096 MiB | off | Current performance reference |
| B — RAM budget only | 1024 MiB | off | Isolates the RAM reduction cost/benefit |
| C — hybrid | 1024 MiB | 8192 MiB | Small SSD test; validates the full path |
| D — hybrid target | 1024 MiB | 32768 MiB | Proposed configuration |
| E — larger RAM control | 4096 MiB | 32768 MiB | Separates tiering benefit from simply increasing cache capacity |

Run each with:

1. cold start / cold cache;
2. warm repeated-prefix agent workload;
3. mostly unique prompts (cache-hostile control);
4. NAS path only after local NVMe has been tested and measured.

Collect:

- prefix tokens served from cache (% and absolute count);
- hit counts/bytes by Device, RAM and SSD;
- SSD read/write bytes, IOPS, queue latency and p50/p95 per-hit restore latency;
- TTFT p50/p95 and end-to-end agent wall time;
- cold prefill tokens/s and decode tokens/s;
- RAM committed, pinned and available; expert RAM residency/hit rate and expert read latency;
- SSD live bytes, dead bytes, compaction/write amplification and restart restore time.

Do not report only aggregate hit rate; one huge slow SSD hit can increase hit rate but worsen latency.

## 11. Proposed acceptance criteria

Treat these as initial engineering gates, not measured results:

- `ram` mode: unchanged behavior and within ±1% of baseline throughput in repeated controlled runs.
- Correctness: no invalid state, crash or output discrepancy attributable to the tiered cache; failures degrade to misses.
- Cold path: prefill/decode within ±1% of the same 1 GiB RAM configuration with SSD disabled, except for measurable write-through overhead that must be reported.
- RAM target: at least approximately 2.5–3 GiB less host pool reservation when using 1 GiB rather than 4 GiB. Report actual free RAM after the full model/expert allocation; do not assume the whole 3 GiB becomes available to experts.
- Warm agent workload: improve end-to-end wall time or TTFT over the 1 GiB RAM-only run, and ideally match or beat the 4 GiB RAM baseline while keeping the smaller pinned RAM budget.
- Cache-hostile workload: no unbounded stalls or material decode regression caused by SSD write/read activity.
- Resilience: malformed/corrupt cache data is discarded safely; restart recovery does not read/hash the full SSD payload store eagerly.

Do not call the feature a win unless the **1 GiB RAM + SSD** setup is compared both to current 4 GiB RAM and to the 1 GiB RAM-only control. The 4 GiB + SSD control shows whether the larger working set, rather than freed RAM, is producing the gain.

## 12. Main risks and mitigations

| Risk | Mitigation |
|---|---|
| SSD hit is slower than prefill avoided | Use restore-cost estimation and allow shallower hit/recompute |
| 116 MB state snapshots dominate bytes | Value-aware admission/eviction; retain high-reuse snapshots; measure snapshot vs KV bytes separately |
| SSD reads compete with expert reads | Bounded asynchronous workers, low-priority/batched writes, backpressure |
| RAM staging erases memory savings | Small shared staging pool or direct chunked reads to pinned destinations; show it in RAM ledger |
| Dirty/incomplete writes become a false hit | Commit record + checksums + publish only after write completion |
| Cache compaction races with active restores | I/O refs/pins; atomically switch index locations; delete old segment only after all readers release it |
| Disk wear / write amplification | Do not write dead/low-value objects; batch writes; segment compaction; counters for total bytes written |
| Cache key/version collision | Artifact fingerprint + schema/config identifiers + exact existing token/media path identity |
| Existing persistence flags conflict | Keep `ram` unchanged; reject ambiguous old persistence options in hybrid v1 with a clear message |
| NAS is latency-limited | Validate first on local NVMe; measure NAS separately; don't assume sequential bandwidth predicts random-hit latency |

## 13. Recommended first PR

Keep the first PR narrow enough to review:

1. Add `--prefix-mode ram|hybrid`, `--prefix-ssd-mib`, and `--prefix-ssd-file` parsing/validation.
2. Add the versioned segmented SSD record store and isolated unit tests.
3. Add SSD residency/backing for **one object type first** (prefer the complete state snapshot, because it is large and easy to reason about), but do not declare the feature useful until the required KV blocks can also be restored from SSD.
4. Add KV-block records and integrate asynchronous restores.
5. Add hybrid cache correctness tests plus counters before any cache-policy tuning.

An alternative first spike is to keep the existing RAM cache unchanged and make only SSD persistence/readback work behind an experimental flag; that validates the file format and I/O path, but by itself it does **not** solve the RAM pressure problem. The first useful end-to-end milestone is the ability to evict a valid KV/state object from RAM, later restore it from SSD, and resume the exact prefix safely.

## 14. Definition of done

The implementation is ready for wider testing when:

- `ram` remains backwards compatible;
- `hybrid` adds a bounded, observable SSD tier while reducing the pinned RAM pool as configured;
- SSD-backed snapshots **and their required KV path** can be restored on demand;
- SSD read/write/compaction is asynchronous relative to the serving worker;
- all read/write/corruption failure paths degrade to a cache miss, not bad model state;
- results are published for the benchmark matrix in Section 10.3;
- docs explain the new flags, storage lifecycle, disk-space headroom and the distinction from `--prefix-cache-file`.

## Repository references

- [Infernix repository](https://github.com/Wallawalla47/Infernix)
- [Flash-Next guide — Prefix cache and RAM budget](https://github.com/Wallawalla47/Infernix/blob/main/docs/qwen3_8-flash-next.md#prefix-cache)
- [Hybrid prefix-cache implementation/design spec](https://github.com/Wallawalla47/Infernix/blob/main/docs/maintainer/hybrid-prefix-cache-spec.md)
