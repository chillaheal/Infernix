# Bildvagen som familjesom och statisk plugin: design

Datum: 2026-11-14. Gren: `image-generation`. Godkand av anvandaren under denna session
("jag litar pa att du gor detta bra"), med tillag: smoke tests langs vagen, och sa fa
omstarter av Infernix som mojligt.

Underlag: `docs/research/2026-11-14-image-plugin-grok-review-verified.md` (extern
granskning verifierad mot koden, med `path:line` for varje pavstande) och
`.agent/p40_code_facts.md` (verbatim kodfakta fran samma traed).

## Goal

Gora bildvagen (generering och redigering) sa att (1) de verifierade felen blir
omojliga att bega om, (2) en bildfamilj beskrivs som data plus tre funktioner istallet
for literaler spridda over femton filer, och (3) ett nytt Infernix-drop ar en rebase av
en patch med fyra avgransade ytor istallet for en omskrivning.

## Architecture

Bilden ritas redan idag i ett VRAM-interlude dar LLM:n ar urstand att kora, och
`Engine::run_image_interlude` ar den befintliga eldvaggen
(`include/infernix/engine.h:141-166`). Designen flyttar inte pa den mekanismen. Den
flyttar *beslutet om vad en bild ar* ur serve-lagret och in i en familjbinding som
valjs av artifactet, och flyttar sedan hela den koden ut ur upstreams blast radius bakom
ett enda CMake-alternativ.

Tre principer halls isar och ar var for sig testbara:

1. **En sanningskalla per fakta.** Ett joint-tak, en VAE-topologi, en scheduler, en
   rope-konfiguration, en pixelmultiple. Idag finns samma tal och samma listor
   kopierade (tre kopior av 20480, tva kopior av VAE-kanallistan), och det ar dar
   fulen bor.
2. **Familjen valjer tre funktioner, inte tio hookar.** `prepare` (host only),
   `need` (byte), `execute` (inuti yieldet). Allt annat ar data.
3. **Upstream är ingen ABI.** Var diff mot upstream far bara vara fyra ytor:
   device-lease pa Engine, worker-ack i `engine_core.h`, yield-koden i en egen fil vi
   ager, och en serve-hook per protokoll. Allt annat ligger i `plugins/image/`.

## Non-goals

- Att fa **MiniMax H3** att kora. Det ar ett senare, separat arbete (anvandarens ord).
  H3 används endast som specificeringsunderlag och som form pa acceptanstestet.
- Prestanda som inte behoven for sommen: host-Euler-loopen
  (`src/models/qwen_image/image_pipeline.cu:1014-1066`) ar den storsta enskilda visten
  men ar ett eget uppdrag.
- LoRA (P15), `n > 1` (single-flight ar avsiktligt), prefix-KV som portabilitet, och
  all dynamisk plugin-loader (ingen `.so`, ingen C-vtable, inget versionsnummer innan
  det finns en andra konsument - kompilering ar kompatibilitetskontrollen).
- **Ingen plugin-upptackt och ingen strangstyrd korning.** Traedets egen
  `AGENTS.md` forbjuder uttryckligen "generic model graphs, family base classes, plugin
  discovery, string-driven execution, hidden device allocation, runtime weight
  repacking, or placeholders for hypothetical targets without a product requirement",
  och den regeln styr. "Plugin" i denna design betyder darfor alltid **en statisk
  modul bakom ett byggalternativ med en explicit valjspunkt**, aldrig en registry,
  aldrig en strang som styr korningen, och aldrig enFamiljetyp som finns bara for att
  en framtida modell kanske dyker upp.
- Att flytta LLM:ens egen kod eller att gora `Engine` variantagnostisk utover det som
  kravs for att niva laset.

## Global constraints (galler varje task)

- Gren `image-generation`. Commits med `-c user.name="Ian Ranson"
  -c user.email="ian.ranson@users.noreply.github.com"`. Push endast pa begaran.
- `image-edit/` hamnar aldrig i en commit.
- Aldrig en andra serve bredvid den levande. Detta session körs av :8888.
- Host-tester foredras. GPU- och `_real`-tester och all device-link-bygg som kraever att
  korten ar fri körs i **ett enda** avslutande offline-fonster tillsammans med
  deploy och omstart. Mal: en omstart for hela arbetet.
- Alla tester i traedet ar handskrivna `int main()` med `check(bool, label)` och skriver
  ut `NAMN: PASS`; de är inte gtest.
- Bygg kommandot for **den har** traeden (de befintliga `image-edit\run_host_*.bat`
  cd:ar till en annan traed och far inte ateranvandas som de ar):
  vcvars64 fran `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat`,
  `VCPKG_ROOT=C:\vcpkg`, `VCPKG_TARGET_TRIPLET=x64-windows`,
  `CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4`,
  `cd /d "C:\Users\Micke\Documents\Kodprojekt\Infernix - Flash next\Infernix"`,
  `cmake --build build --config Release --target <t>`,
  `ctest --test-dir build -C Release -R "<regex>" --output-on-failure`.
- ASCII i all ny kod och alla nya skript.
- Traedets egen `AGENTS.md` binder, och star over nagon metod fran en skill:
  **commits skapas endast nar anvandaren ber om det**, med Conventional Commit-ämnen
  (`fix:`, `refactor:`, `test:`, `build:` ...). Inga bakatkompatibla alias eller
  transitions-grenar. Ingen plugin-upptackt eller strangstyrd korning. Ingen dolt
  enhetsminne.
- Innan nagot roer Engine/Program-gransen (WP4): las `docs/maintainer/engine-architecture.md`.
  Innan nagot roer ett Op: las `docs/maintainer/op-development.md`. Bygg- och
  testkommandon: `tests/README.md` (inte `image-edit\run_host_*.bat`, som bygger en
  annan traed).
- OpenAI/Anthropic-protokoll som annonseras ar ett externt kontrakt: en andring i
  bild-schemat uppdaterar drabbade schematester och serving-dokumentationen i samma
  commit.
- Numeriska andringar kravera en oberoende oracle for det berorda Op:et; den har designen
  ska inte andra nagra bitar, och det bevakas av att `infernix_image_need_ledger_real_test`
  och de certifierade baseline-siffrorna ar oforandrade.
- Python körs med `py -3.11` (Windows), inte `python`.

## File structure

Nya filer vi ager (inga upstream-filer flyttas hit, bara var kod):

| Fil | Ansvar |
|---|---|
| `include/infernix/ops/image_joint_limits.h` | det enda joint-taket och valideringen; `kImageJointMaxTokens`, `image_joint_tokens_ok()`, `image_joint_range_message()` |
| `include/infernix/image/family.h` | `ImageFamily`, `PrepareFn`, `NeedFn`, `ExecuteFn`, `FamilyConstants`, registry, refusaltyper |
| `src/runtime/expert_offload/device_lease.cpp` | definitionerna av `probe_image_yield`/`yield_for_image`/`release_from_yield`/`begin_quiesce`/`begin_image_quiesce`/`rollback_failed_yield`, flyttade ur `program_impl.h` |
| `plugins/image/CMakeLists.txt` | `add_library(infernix_image STATIC ...)`; den enda byggrutten for hela vagen |
| `plugins/image/families/qwen_image_2_1.cpp` | familjens data + prepare/need/execute, delegerar till den befintliga pipeline-koden |
| `plugins/image/serve/image_module.cpp` | implementerar `ImageModule`: routeer, hook, capabilities-bidrag |

Modifierade upstream-filer (var hela diff, max): `src/ops/rope_c64/rope_c64.cpp`,
`src/ops/dit_adaln/dit_adaln.cpp`, `src/ops/attn_dit_joint/attn_dit_joint.cpp`,
`include/infernix/ops/{attn_dit_joint,dit_adaln,rope_c64}.h`,
`src/models/qwen_image/image_device_need.h`, `src/models/qwen_image/image_pipeline.cu`,
`src/models/qwen_image/image_weights.cpp`, `src/models/qwen_image/rope.cpp`,
`src/models/qwen_image/scheduler.cpp`,
`src/models/qwen4_exp/program/program.cpp`, `src/models/qwen4_exp/program/program_impl.h`,
`src/models/qwen4_exp/program/program.h`, `src/runtime/engine/engine.cpp`,
`src/runtime/engine/engine_core.h`, `include/infernix/engine.h`,
`src/serve/image_generation_service.{h,cpp}`, `src/serve/openai_images.{h,cpp}`,
`src/serve/reference_image.cpp`, `src/serve/image_job_store.{h,cpp}`,
`src/serve/http_server.{h,cpp}`, `src/serve/openai_{chat,responses}_http.cpp`,
`src/serve/anthropic_messages_http.cpp`, `src/serve/chat_image_tool.{h,cpp}`,
`src/serve/CMakeLists.txt`, `src/models/CMakeLists.txt`, `src/ops/CMakeLists.txt`,
`CMakeLists.txt`, testens registrationsfiler.

---

## WP1 - fail-closed

### 1. Ett joint-tak

`include/infernix/ops/image_joint_limits.h` agrar det enda talet:

```cpp
#pragma once
#include <cstdint>
#include <string>
namespace infernix::ops {
// The one joint-sequence ceiling. Attention, rope_c64, dit_adaln, admission, the chat
// tool schema and /v1/capabilities all read THIS symbol.
inline constexpr std::int32_t kImageJointMaxTokens = 20480;
[[nodiscard]] constexpr bool image_joint_tokens_ok(std::int32_t tokens) noexcept {
    return tokens >= 1 && tokens <= kImageJointMaxTokens;
}
[[nodiscard]] inline std::string image_joint_range_message(std::string_view op) {
    return std::string(op) + ": S must be 1.." + std::to_string(kImageJointMaxTokens);
}
} // namespace infernix::ops
```

Det gamla namnet `kAttnDitJointMaxTokens` (`include/infernix/ops/attn_dit_joint.h:15`)
**tas bort**, inte aliaskes - traedets regel är att egna API:er inte behaller
bakatkompatibilitet ("remove superseded aliases, fallbacks, transition branches, and
their tests within the affected contract"). Alla lasare uppdateras i samma commit:
`attn_dit_joint.cpp:16-17`, `http_server.cpp` som publicerar
`canvas.joint_sequence.ceiling_tokens`, och `chat_image_tool.cpp` som bygger schemat.
De tre opsen
byter sina egna literaler mot `image_joint_tokens_ok(...)` +
`throw std::invalid_argument(image_joint_range_message("rope_c64"))`.
`chat_image_tool.cpp:23` byter det utskrivna "20480" mot att schemat byggs fran
konstanten i `make_chat_image_tool_definition()`.

### 2. Enhet och mempool

`src/models/qwen4_exp/program/program.cpp:224-232` far ett `cudaSetDevice` for den
enhet engine:nsactually arbetar pa innan `cudaStreamCreate`, och
`image_pipeline.cu:418-424` laser av enheten fran streamen, lagrar den tidigare
treskeln och aterstaller den i scope-destruktorn. Ingen hardcoded `0` kvarstar.

### 3. Enhancer-grinden lattar pa levererat fri

`src/serve/image_generation_service.cpp:934-935` och `:946-947` byter
`context.freed_bytes + fit.device_free_bytes` (proven fore quiesce) mot
`context.device_free_bytes`, samma som bildvagen redan använder pa `:1324-1326`.

### 4. Jobbrecordet säger sanningen

`JobUpdate` (`src/serve/image_job_store.h:660-666`) far ett `nlohmann::json request;`
och `advance()` tillampar det nar det inte ar null. Efter innerwalkens krympning
(`image_generation_service.cpp:1324-1341`) skrivs `effective.width/height` tillbaka till
recorden, sa att `GET /v1/images/<id>` visar samma kanad som wire-svaret. Kommentaren
`:1284-1285` rattas.

### 5. Artifact-deklarerade konstanter, vgra vid bindning

`SchedulerConfig` (`src/models/qwen_image/scheduler.h:16-24`) och DiT-rope
(`rope.cpp:17` axes `{16,56,56}`, theta `10000.0`) reads fran artifactets metadata,
samma vag som text-tornet redan läser M-RoPE. Okand schema (t.ex. en `sample_sigmas`)
eller saknade nyckel = typad refusar vid bindning, aldrig en tyst default. Readerns
ingngar ar en ren funktion over ett metadata-map sa att den ar host-testbar utan
artifact.

## WP2 - en var av VAE-topologin

`VaeStage`/`kVaeMidStage`/`kVaeUpStages`/`kVaeFinalChannels`
(`src/models/qwen_image/image_device_need.h:729-746`) och
`kVisionWorstReferencePatches` (`:758`) och `kImageWeightResidentBytes` (`:791`) blir
falt i en `VaeTopology` som byggs av bindingen och konsumeras av bade graferna
(`image_pipeline.cu`) och ledaren. `image_device_need()` far topologin som parameter.
Runtime-assert jamforer kanallistorna med de faktiska viktparametrarnas former och
vagrart bindningen om de skiljer sig.

Detta ar inte en stilfraga: MiniMax-H3-Image har 24 latentkanaler, 16x spatial
downsampling och en 24-lags **ViT-decoder** - inte en CNN up-stack. Utan WP2 ar vara
VAE-kedja en vagg.

## WP3 - familjesommen

```cpp
// include/infernix/image/family.h  (public; den enda headern en familj behover kanna till)
namespace infernix::image {
struct FamilyConstants {
    std::uint32_t pixel_multiple   = 0;   // var fjordet must be a multiple of
    std::uint32_t vae_downscale    = 0;   // VAE spatial compression
    std::uint32_t latent_channels  = 0;
    std::uint32_t latent_patch     = 0;   // DiT patchify over the latent
    std::uint32_t max_references   = 0;
    std::uint64_t weight_resident_bytes = 0;
    std::size_t   joint_ceiling_tokens  = 0;
    bool          cfg_supported  = true;
    // template/pad-token/vision mean+std kommer fran artifactet, inte fran denna struct
};
struct PrepareResult { /* tokenantal, referensplaner, geometry, need-underlag */ };
using PrepareFn = std::function<PrepareResult(const PreparedRequest&, PrepareSink&)>; // host only
using NeedFn    = std::function<std::uint64_t(const PrepareResult&)>;
using ExecuteFn = std::function<FinishedPicture(const PrepareResult&, const ImageInterludeContext&,
                                                const std::shared_ptr<ImageJobProgress>&)>;
struct ImageFamily {
    std::string_view name;
    FamilyConstants  constants;
    PrepareFn prepare;  // far inte round-tripa till enheten
    NeedFn    need;     // far inte gissa en areafaktor
    ExecuteFn execute;  // ingen HTTP, inget json
};
} // namespace infernix::image
```

**Valjet explicit, aldrig upptackt.** En familj ar en konkret typ bakom en namngiven
fabrik, och ett enda stalli serve matchar artifactets deklarera arkitektur mot den
fabriken - samma monster som traedet redan foljer for sina Modeller ("Prefer explicit
implementations for supported architectures"):

```cpp
// plugins/image/families/qwen_image_2_1.h
[[nodiscard]] const infernix::image::ImageFamily& qwen_image_2_1_family();
```

```cpp
// det enda staellet som valjer familj. ny familj = en fil hit plus en rad har, ingen
// registry, ingen loop over en lista, ingen strang som styr korningen.
inline std::optional<ImageFamilyKind> parse_image_family(std::string_view name) noexcept {
    if (name == "qwen-image-2-1") { return ImageFamilyKind::QwenImage2_1; }
    return std::nullopt;   // CLI-valideringen gör detta till ett startfel som naemner de tillatna namnen
}
inline const ImageFamily& image_family_for(ImageFamilyKind kind) noexcept {
    switch (kind) {
    case ImageFamilyKind::QwenImage2_1: return qwen_image_2_1_family();
    }
    std::terminate();      // alla enumeratorer hateras ovan
}
```

**Rättelse efter kodkontroll (2026-11-14):** specen hade fran borjan `switch
(descriptor.architecture)`. Artifactet har ingen sadan falt - `artifact::Binder`
(`src/artifact/binder.h:35-70`) erbjuder `resource`/`find_resource`/`host_object` och
bindingen gar pa komponent-configs dar en avvikelse redan vranks vid bind
(`src/models/qwen_image/image_weights.cpp:381`). Darfor kommer valet fran den explicita
CLI-flaggan `--image-family` (kravd nar `--image-artifact` ar satt, ingen tyst default),
och familjens bindning kontrollerar att artifactet faktiskt ar det som namns. Ett nytt
artifact-falt och en omkonvertering behövs inte.

En andra familj som inte finns som produkt (MiniMax H3) skal **inte** ha en plats har.
Den finns bara som ett testfixtures i `tests/` som-kompile-mot
`infernix/image/family.h` och bevisar att granssnittet hallr - ett test av ett publikt
kontrakt, inte en produktplaceholder.

`ImageGenerationService` tar en `const ImageFamily*` i konstruktorn istallet for att
sjalv kanna till `models/qwen_image/*`, och alla tre nämnaren
(`PromptRendererProvider`, `PromptEmbedsProvider`, `ImageJobExecutor`) ersatts av
familjens tre funktioner. Qwen-Image-2.1 blir en avskrift i
`plugins/image/families/qwen_image_2_1.cpp` som delegerar till den befintliga koden -
beteendet ar ofandrat, bara agandet flyttar.

**Acceptanstest (hela poangen).** En andra familj i `tests/models/qwen_image/` som
skiljer sig fran 2.1 pa varje axel MiniMax-H3-Image gor: 24 latentkanaler, 16x VAE, en
icke-CNN-decoder, sampler med per-steg-heads, `cfg_supported = false`, 16 referenser,
annan pixelmultiple. Den far bara inkludera `infernix/image/family.h`. Om den
kompileras och bestar ar sommen verklig; om den behover nagon annan header ar den fejk
och designen har misslyckats.

## WP4 - statisk plugin, fyra ytor

- `option(INFERNIX_IMAGE "Build the image lane" ON)` i rot-`CMakeLists.txt` (bredvid
  `INFERNIX_BUILD_APPS` pa rad 18), och `add_subdirectory(plugins/image)` villkorat av
  den. `src/serve/CMakeLists.txt:20-27` (bildfilerna) och `:47`
  (`infernix_qwen_image infernix_qwen_image_runtime`) forsvinner ur upstream-targetet.
  `src/models/CMakeLists.txt:22` ar registreringspunkten for familjens `sources.cmake`.
  De tre opsen (`src/ops/CMakeLists.txt:17,19,42`) flyttar till ett eget target;
  ingen utanfor bildvagen använder dem (verifierat).
- **Device-lease.** `run_image_interlude`/`probe_image_yield` behaller sina namn (de
  finns, ar publicerade och testade; Grok har rätt att man inte döper om nar man
  flyttar). Ett generiskt namn tillkommer forst nar det finns en andra modellvariant.
  Separat fel som noteras men inte fixeras nu: `src/runtime/engine/engine.cpp:419-439`
  hardkodar `impl_->qwen4->program`.
- **Worker-ack.** Nagra rader i `src/runtime/engine/engine_core.h` dar
  `interlude_preempt_outstanding` redan bor (traiten pa `:169-176` och
  preempt-blocket pa `:2414-2424`); duck-typen maste fortsatt vara uppfylld.
- **Yield.** Definitionerna av de sex functionerna (`program_impl.h:6013-6048,
  6052-6151, 6184-6218, 6304-6313, 6380-6441, 6509-6514`) flyttar till
  `src/runtime/expert_offload/device_lease.cpp`, deklarationerna star kvar i
  `program_impl.h`. Detta ar den andringen som faktiskt gor patchen barbar - filen
  skrivs om veckovis upstream.
- **En serve-hook per protokoll.** Ett `ImageModule`-interface som HttpServer haller
  ett pekare till (`attach_images` tar ett interface istallet for
  `ImageGenerationService&`, `http_server.h:80`, forward-declared `:30`), och tre
  ett-raders anrop - ett i `openai_chat_http.cpp`, ett i `openai_responses_http.cpp`,
  ett i `anthropic_messages_http.cpp` - dar de idag har kopierad bildlogik pa fyra
  stallen vardera (`append_chat_image_tool` -> `outcome_is_chat_image_tool_round` ->
  `run_chat_image_rounds` -> `missing_image_url_suffix`). `make_capabilities_json`
  (`http_server.cpp:657-745`) far bidrag fran modulen sa att den varken kan namna
  20480, Qwen eller `image_vram_insufficient`.

Utan plugin = tom hook, inga routes = upstream beter sig exakt som idag. Bevis: bygg med
`-DINFERNIX_IMAGE=OFF` och kor den icke-bildsviten gron.

## Felhantering

Alla nya refusaler ar typade och nar ett fel som namner vad som saknas, aldrig en tyst
default. `image_joint_range_message()` byter meddelandet fran konstanten sa att det
aldrig kan ljuga. Bindningsfel (scheduler, rope, VAE-topologi mot viktformer) stoppar
artifactet att bindas och serve svarar da dokumenterad 503 med anledningen, samma monster
som de befintliga `image_generation_not_configured`/`image_interlude_unsupported`.
Ingen ny undantagsvag faar na serverns workertrad (p36:s las learned: protocol- och
valideringsfel ar requestfel, aldrig processfel).

## Tester

- Host, utan GPU och utan serve, efter varje task: `infernix_image_generation_service_test`,
  `infernix_openai_images_schema_test`, `infernix_chat_image_tool_test`,
  `infernix_image_job_store_test`, `infernix_http_routes_test`,
  `infernix_image_device_need_test`, `infernix_qwen4_exp_memory_plan_test`.
  Ny: `infernix_image_joint_limits_test` (ops validerar och publicerar ett tal, inte att
  konstanterna ar lika), `infernix_image_scheduler_artifact_test`,
  `infernix_image_family_binding_test` (det explicita valet vgrart okant arkitektur),
  `infernix_image_family_foreign_test` (fixturet som bara kan inkludera den publika
  headern).
- **Oforandrade numeriska varden ar ett krav.** `infernix_image_need_ledger_real_test`
  (12 rows, "the computed need covers every measured peak") och de certifierade
  baseline-avläsningarna (2048x2048 predicted 13869 vs measured 12653; 2016x2016 13548
  vs 12295; 1024x1152 6409 vs 4367; ViT-edit 6163 vs 3634, where 6163 replaced 6010 when
  plan 1 round-2 widened the ViT term to the derived worst-case reference plane, so the
  gate is the measured column and a rising prediction) maste ga exakt som forut
  genom WP1 och WP2 - det ar vara regressionsvakt nar ledaren far sin data utifran.
- GPU/`_real` (endast i det avslutande offline-fonstret): lease/coexistence-testen i
  `tests/models/qwen4_exp/tests.cmake`, `infernix_image_need_ledger_real_test`, och den
  nya enhet-/mempool-testen som andrar enhet 0:s default mempool och difor inte far kora
  bredvid en levande serve.
- **Smoke test efter deploy (ett fonster, en omstart):** `/health` ska besvara med
  `status ok` och `image_jobs`-busyfalt; `/v1/capabilities` ska visa samma
  `joint_sequence.ceiling_tokens` och samma dimensions-/steps-falt som forut; ett
  text-to-image-jobb 512x576 20 steg, ett edit-jobb med en referens, och ett chat-svar
  som kraver `generate_image`-verktyget; och ett chat-svar under ett pagaende drag
  (fallet som tidigare besvarades 503). Bildfiler verifieras med
  `python ~/.Codex/scripts/verify_image.py <path>` och crosscheckas mot DOM/JSON, inte
  mot mitt minne av dem.

## Risiko och vad som kanns osakert

- **Ett enda tak ar inte nok för ever.** H3-Image behover ~16k mål-tokens for 16 MP
  och H3-video ~20k for 6 s 768p. Att gorta talet artifact-deklarerat ar ratt steg nu;
  sjalva kapacitetsfragan (scratch, ledare, O(n2)) ar ett eget uppdrag och maste inte
  losas här.
- **33B BF16-DiT far aldrig in i ett lease pa ~22 GB.** Kvantisering ar en
  forutsattning for H3, och det ar en produktfraga som provet far besvara - inte nagon
  anledning att avvakta sommen.
- **`attach_images` ar den dyra delen, inte pipelinen.** Tre protokoll har redan
  kopierat bildlogiken; att fa bort den ar mindre kod an idag men roar
  `http_server.cpp` och capabilities. Darfor star det sist.
- **Licensen for MiniMax H3** ("MiniMax-H3-Image Community License Agreement", skriftligt
  tillstand via ansokan, Sverige inom rackvidd) ar en grind fore eventuellt H3-arbete,
  inte fore denna refactor.

## Planer

Arbetet delas i tva planer som var for sig levererar fungerande, testbar mjukvara:

1. `docs/superpowers/plans/2026-11-14-image-fail-closed.md` - WP1 + WP2 (basen robust
   forst). Det ar den som tar bort de faktiska felen och bygger de enda sanningskallorna.
2. `docs/superpowers/plans/2026-11-14-image-family-plugin.md` - WP3 + WP4 (sommen och
   pluginet), med samma form pa detalj nivn nar det star for tur, efter att plan 1s
   verkliga besok har avsljott vad familje-headern faktiskt behover innehalla.

## Rattad 2026-11-24 mot upstream f9c2e42c (origin/main, +119 commits sedan specen skrevs)

Underlag: `docs/research/2026-11-24-image-edit-port-assessment.md`.

- "De fyra upstream-ytorna" ar en underskattning. Uppstroms egen familj-in-slapp (Inkling 2) rorde 302
  filer, 134 kodseams, ~27 for minimal chat-tradning, och redigerade andra familjers delade filer.
  Var lane ar smalare (egen `ImageGenerationService`, ingen registry-`Architecture`): **6-8 namngivna
  seams** ar malet. Se Task 0 i planen for hur de hallts.
- Ingen merge-base mot `origin/main` alls, sa tal om "upstream-merge" ersatts av "vendored subtree plus
  `git am`-serie for sommarna".
- Ankar som specen utgick fran finns inte: `execution_mu_`, och worker-ack-namnen
  `request_interlude_preempt`/`wake_llm_worker`/`drain_llm_frontier`/`interlude_preempt_outstanding`.
  `engine.cpp` dispatchar nu fyra variant-karnor (den fjarde `Inkling2Loop`). `ChatTool`/
  `GenerationRequest`/`GenerationOutcome` har flyttat ur `types.h` till `src/serve/request.h:266` och
  `src/serve/generation_service.h:33`.
- Ny blockerare for leaset: `perf(qwen4_exp)` "every expert in RAM" (`a3f78ef8`, `18388f5b`) haller
  experter boende via `residency_->lendable_bytes()`, sa facaden maste ha en uttrycklig
  residens-evictions-override i stallet for att anta att poolen gar att frysa.
