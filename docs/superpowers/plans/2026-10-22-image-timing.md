# Image timing (bundle B) implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans (or subagent-driven-development) to
> work this plan task by task. Steg är ikryssningsbara (`- [ ]`). Varje task slutar i något som kan köras och
> mätas, inte i en halvfärdig refactoring.

**Goal:** sänk LLM-pausen och total tid per bild enligt `docs/superpowers/specs/2026-10-22-image-timing-design.md`,
med en mätt baslinje före någon förändring.

**Architecture:** fas 0 ger talen; fas 1 flyttar host-arbetet ut ur interludet; fas 2 överlappar blockvikternas
H2D med beräkningen; fas 3 (och 3b) håller block boende och flyttar Euler-steget till enheten.

## Global constraints

- Tal och vägar kopieras från verktygsresultat; inga gissade filnamn eller flagnamn.
- Varje fas redovisar **både** LLM-paus (ms) och total tid per bild (ms), före och efter, p50/p95, kall och
  varm, per upplösning i ledger-matrisen (512, 768, 768+ref, 768+CFG, 1024). Medelvärde ensam räknas inte.
- Bundle A:s certer får inte bli sämre: `RESULT ledger=0 prod=0 failures=0 healthy=True` i
  `image-spike/blackout_bundle_a.ps1` efter varje fas.
- Numerik: fas 0-3 oförändrad. Fas 3b och eventuell kärnsammanslagning körs som fast path,
  `--disable-fast-paths` bit-exakt, kvalificerad enligt `docs/maintainer/op-development.md`.
- Bygg endast test/bench-mål (`cmake --build build --target <namn> -j 8` i synlig terminal); länka aldrig appen
  -- den levande `infernix-serve.exe` håller den filen.
- Certer och mätprogram kräver GPU:n ensam: svartfönster, och servern återstartas frisk efteråt.

---

### Task 1: mätprogrammet (fas 0)

**Files:**
- Modify: `bench/CMakeLists.txt` (registrering, följ befintliga måls mönster)
- Create: `bench/<namn tas ur bench/CMakeLists.txt:s egen namnstil>`
- Modify: `bench/README.md`

**Interfaces:**
- Consumes: `ImageGenerationService`/pipeline-vägen och `image_device_bytes_peak()`/`image_device_bytes_live()`.
- Produces: en tabell (per upplösning: paus-ms, total-ms, per-block kopierade byte, H2D-ms och GB/s,
  kernel-ms, sync-/väntetid, tower-, VAE-, D2H- och CPU-delar, peak inkl. mempool `ReservedMemCurrent`) som
  task 3-8 jämför mot.

- [ ] **Step 1:** Läs `bench/CMakeLists.txt` och ett befintligt bench-mål; kopiera mål-namnstil, register-
  makro och hur artifact-validering görs. Skriv inget filnamn innan det står i den filen.
- [ ] **Step 2:** Mät med CUDA-events runt varje segment och `cudaMemPoolGetAttribute(...ReservedMemCurrent)`
  plus `cudaMemGetInfo` fria bytes före/efter; tid per API-anrop separat från GPU-tid (annars ser en synkron
  `cudaMemcpy` ut som sin egen väntetid).
- [ ] **Step 3:** kör i Release (`build` config Release, samma artifact `out/Qwen-Image-2.1-FP8-v2.infernix`)
  i svartfönster; skriv tabellen till en fil under `image-spike/` och till stdout.
- [ ] **Step 4:** Bygg målet och kör det en gång; verifiera att raderna innehåller båda huvudtalen per
  upplösning annars är mätningen värdelös.
- [ ] **Step 5:** Commit: `bench: time every image stage per resolution`

### Task 2: baslinjen blir en del av specen

- [ ] **Step 1:** Klistra tabellen in i avsnitt 8 i `docs/superpowers/specs/2026-10-22-image-timing-design.md`
  och skriv vilken körning (datum, config, commit) den kommer från.
- [ ] **Step 2:** Kontrollera mot A:s ledger-siffror att uppmätt peak ligger under `image_device_need().total()`
  för samma inputs (annars är något av mätningen eller admittaren fel, och det är den frågan som ska lösas
  nu, inte senare).
- [ ] **Step 3:** **Beslutspunkt:** om tabellen visar att överföringarna klart dominerar och certifierat
  minne finns över, kör task 5 (resident budget) före task 4 finjustering -- det undantaget står i specen.
- [ ] **Step 4:** Commit: `docs: record the image timing baseline`

### Task 3: host-arbetet ut ur pausen (fas 1)

**Files:** Modify `src/serve/image_generation_service.cpp` (PNG `:215`, base64 `:308`, disk `:315-331` och
anropet `:496-508`), `src/serve/chat_image_tool.cpp` om svagskontraktet rörts, `tests/` för serven.

- [ ] **Step 1:** Säkerställ att interludet avslutas efter sista nödvändiga D2H och att kodning/base64/skriv
  sker efteråt; räkna paustiden som skillnaden (samma mått som task 1).
- [ ] **Step 2:** Behåll avtalet: fel i kodning eller skriv ger samma HTTP-svar som idag och annonsera ingen
  URL som aldrig skrevs (posten publiceras efter att filen finns).
- [ ] **Step 3:** Kör `infernix_image_generation_service_test` och `infernix_chat_image_tool_test`
  (`ctest --test-dir build -C Debug -R "image_generation_service|chat_image_tool"`); gröna.
- [ ] **Step 4:** Mät om (task 1:s program) i svartfönster; redovisa paus före/efter per upplösning.
- [ ] **Step 5:** A:s tre certer gröna. Commit: `perf(serve): finish image encoding and storage after the interlude`

### Task 4: överlappa blockvikternas överföring (fas 2)

**Files:** Modify `src/models/qwen_image/image_pipeline.cu` (`upload_streamed`-anropen `:513-523`, `h2d`,
`Buf`), `src/models/qwen_image/image_weights.{h,cpp}` (host-cache och dess allokring), ev.
`src/models/qwen_image/image_device_need.h` (+1 block peak).

- [ ] **Step 1:** Innan kod: bekräfta i `cudaDeviceGetAttribute` vad enheten klarar (`asyncEngineCount`) och
  skriv in talet i commitmeddelandet; utan copy-engine är hela tasken värdelös som överlappning.
- [ ] **Step 2:** Gör block-cachen pinned (`cudaHostAlloc`), pröva `cudaHostAllocWriteCombined` som en
  mätt variant och behåll den som vinner; motivera i commit:en (WC är långsamt för CPU-läsningar).
- [ ] **Step 3:** Egen non-blocking copy-stream med event-beroenden; ladda block `i+1` medan `i` räknar. Allt
  arbete måste bli explicit stream-styrt annars suger legacy default stream ihop överlappningen (det gör den
  idag, se `image_pipeline.h:12-15`).
- [ ] **Step 4:** Numeriken får inte röra sig: kör prod-certen (`infernix_qwen4_exp_image_interlude_prod_real_test`)
  och jämför `FINAL_LATENT`/`FINAL_IMAGE` mot dess egna trösklar; ett flyttat tal är ett fel här, inte en
  acceptabel skillnad.
- [ ] **Step 5:** Mät om; redovisa andelen H2D som faktiskt döljs av beräkning. Om den är nära noll, stoppa
  och rapportera varför i stället för att bygga vidare.
- [ ] **Step 6:** A:s tre certer gröna. Commit: `perf(image): overlap DiT block uploads with the block forward`

### Task 5: resident blockbudget (fas 3)

- [ ] **Step 1:** Beräkna budgeten ur det yieldade beloppet och `image_block_bytes(dims)`
  (`src/models/qwen_image/image_device_need.h:39-42`); budgeten räknas **före** yield, aldrig under, så att
  ingen storlek admitteras med ett löfte som inte kan infrias.
- [ ] **Step 2:** Håll de valda blocken boende över alla steg; resten streamas med task 4:s prefetch.
- [ ] **Step 3:** Räkna om `image_device_need` och `kImageWeightResidentBytes`; dokumentera var talet kommer
  från i koden (kommentaren på `:27-34` är auktoriteten idag).
- [ ] **Step 4:** Kör `infernix_image_need_ledger_real_test` på hela matrisen; kravet är att predictions
  aldrig hamnar under uppmätt peak och att varje storlek som rymdes förut ryms nu. Nya 503 måste kunna
  förklaras av en verkligt högre peak, inte av en slarvig marginal.
- [ ] **Step 5:** Mät om (paus + total); redovisa minnespriset explicit: peak före/efter per upplösning.
- [ ] **Step 6:** A:s tre certer gröna. Commit: `perf(image): keep as many DiT blocks resident as the yield allows`

### Task 6: Euler-steget på enheten (fas 3b)

**Files:** Modify `src/models/qwen_image/image_pipeline.cu:538-598`, ev. nytt op i `src/ops/`.

- [ ] **Step 1:** Flytta `new_lat = lat + (sigma_next - sigma) * noise` till enheten och låt latenten bo kvar
  mellan stegen; ta bort `velocity D2H`, `latent D2H`, `h2d` och de två extra `cudaStreamSynchronize`.
- [ ] **Step 2:** Jämför bit mot bit mot dagens routing över hela matrisen. Bit-exakt: behåll, ingen ny
  kvalifikation behövs. Avvikelse: kör som fast path med naiv FP32/FP64-orakel enligt
  `docs/maintainer/op-development.md`, och registrera bara om den är mätt snabbare.
- [ ] **Step 3:** Mät om; säg explicit hur mycket av vinsten som är överföringar och hur mycket som är
  synkroniseringar.
- [ ] **Step 4:** A:s tre certer gröna. Commit: `perf(image): step the flow-match latent on the device`

### Task 7: avslut och dokumentation

- [ ] **Step 1:** Uppdatera `docs/serving.md` om något synligt tal eller svar flyttat (t.ex. att stora
  storlekar nu 503:ar tidigare, eller att pausten blivit kortare).
- [ ] **Step 2:** Skriv slutliga talen i specens avsnitt 8 (baslinje, efter varje fas, sämsta fall).
- [ ] **Step 3:** Uppdatera `.agent/STATE.md`; ta bort denna planfilen när arbetet är klart (AGENTS.md:
  tillfälliga planer ska inte ligga kvar).
- [ ] **Step 4:** Deploybeslutet är användarens: föreslå svartfönster-deploy av `infernix-serve.exe` och kör
  bara på ett ja.
