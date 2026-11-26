# Bildfamilj som statisk modul (WP3 + WP4): implementationsplan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Flytta all familjespecifik bildkod till `plugins/image/` bakom `option(INFERNIX_IMAGE ON)`, med en explicit vald familj och bara de 6-8 namngivna stallen dar upstream maste veta att bilden finns (talet uppmatt 2026-11-24 mot `f9c2e42c`, inte de oppringda "fyra" planen forst utgick fran) - sa att en port till en nyare Infernix bara traffar de sommarna, inte hela bildvagen.

**Architecture:** Familjen ar ett `constexpr`-beslut, inte upptackt: ett nytt CLI-val `--image-family NAME` valjs ur en fast enum och slipar en `ImageFamily{constants, prepare, need, execute}` som byggaren av serve kopplar in. Upstream behaller de befintliga publika namnen pa lease-API:t (`include/infernix/engine.h:141-166`, oforandrat), flyttar yield-koden ut ur `program_impl.h` till egen TU, och far en enda hook per protokol i stallet for de fyra kopierade handler-stallena.

**Tech Stack:** C++23, CUDA 13.4, CMake, nlohmann::json, handskrivna tester.

**Fore kraven:** plan 1 (`2026-11-14-image-fail-closed.md`) ar klar. Dess sanningskallor
(`kImageJointMaxTokens`, `VaeTopology`, `read_scheduler_config`, `read_dit_rope_config`,
`JobUpdate::request`) ar det familjen skal leverera - darfor ar denna planen skriven efter
den, och dess Task 1 kan behova justeras for det plan 1 landade.

**Spec:** `docs/superpowers/specs/2026-11-14-image-family-plugin-design.md` WP3 och WP4.

**Reviderad 2026-11-24 mot upstream `f9c2e42c`** (origin/main, 2026-11-24 12:51 +0000; var `0a3f432b`
nar planen skrevs, +119 upstream-commits pa ett dygn). Underlaget ar
`docs/research/2026-11-24-image-edit-port-assessment.md`, som mot den headen fastslog:

- Ingen merge-base alls mot `origin/main` (vara rot `fd2aa93c`, upstreams rot `410b9f6c`). En port ar
  aldrig en rebase eller merge, bara en manuell flytt av vara ours-only-filer plus en patch-serie for
  sommarna.
- Uppstrom har **noll** bildgenereringskod (count=0 pa `git ls-tree -r --name-only origin/main`), sa
  hela var lane (~149 filer) ar ren additive, ingen konflikt.
- "Exakt fyra upstream-ytor" var en underskattning. Uppstroms eget satt att slappa in en ny familj
  (Inkling 2) rorde 302 filer totalt, 134 kodseams, ~27 for minimal chat-tradning - och det redigerade de
  ANDRA familjernas delade filer (`qwen3_5/*`, `qwen4_exp/load.cpp`). Var lane ar snmalre for att den
  har en egen `ImageGenerationService` och ingen registry-`Architecture`: **6-8 namngivna seams** ar
  det honesta malet, inte fyra.
- Tvra ankar som planen utgick fran finns inte mer upstream: `execution_mu_` = 0 traffar, och
  `request_interlude_preempt`/`wake_llm_worker`/`drain_llm_frontier`/`interlude_preempt_outstanding`
  har aldrig funnits dar. `engine.cpp` dispatchar nu **fyra** variant-karnor (den fjarde ar
  `Inkling2Loop`), och `ChatTool`/`GenerationRequest`/`GenerationOutcome` har flyttat ur `types.h` till
  `src/serve/request.h:266` och `src/serve/generation_service.h:33` (det minskar var diff).
- Tva nya blockerare fran de 119 commitsen, bada mot WP-B (se Task 0): `perf(qwen4_exp)` "every expert
  in RAM" (`a3f78ef8`, `18388f5b`) hallller avsiktligt experter boende via `residency_->lendable_bytes()`
  med `!tier_`-grenar, vilket strider mot var premiss "poolen kan frysas"; och `b821d961` kapsar
  `VramSnapshot::device_free` vid NVML:s enhetsvida fria, sa en yield-probe maste lasa den kapsade
  siffran (var nuvarande probe laser redan `vram_->query()`, `program_impl.h:2936`, som ar den kapsade
  kallan - det ar alltsa ett krav att BEVARA, inte ett fel att fixa).

## Global Constraints

- Som plan 1: traed `C:\Users\Micke\Documents\Kodprojekt\Infernix - Flash next\Infernix`, gren `image-generation`, commits endast nar anvandaren ber om det (Conventional Commit, `-c user.name="Ian Ranson" -c user.email="ian.ranson@users.noreply.github.com"`), push pa begaran, aldrig `image-edit/`.
- Traedets `AGENTS.md` ar avgorande har: **ingen plugin-upptackt, ingen strangstyrd korning, inga familjebasklasser, inga generella modellgrafer, inget dolt enhetsminne, inga bakatkompatibla alias/fallbacks/transitions-grenar** i projekt-agna API:er. "Explicit implementation for the architecture that is supported" - darfor ar urvalet ett `switch` over en fast enum, och en okand familj ar ett startfel, aldrig ett tyst val.
- **Advertierad OpenAI/Anthropic-beteende ar ett externt kontrakt**: nar dimensions-/steps-/referensgranser flyttar eller ett CLI-val tillkomms, uppdateras schemastester **och** `docs/serving.md`/`--help`/`README.md` i samma commit.
- Las `docs/maintainer/engine-architecture.md` forst - Task 4 och 5 andrar Engine/Program-gransen.
- **Omstarter minimera:** Task 1-3 ar host-tester. Task 4 och 5 bygger utan kortan (device-link behover fri kort, men bara for `_real`/server-bygget). Allt som kraever kortan - fullt serverbygge, GPU-tester, deploy, smoke test - samlas i **Task 6, ett enda offline-fonster**. Hela arbetet: en omstart.
- Host-testomslag fran plan 1 ateranvands: `image-edit\_p40_host_tests.bat <target>` (bygger fran **den har** traeden, inte `Qwen Flash Next`-traeden).
- ASCII i allt nytt. Python med `py -3.11`. Tester: `int main()` + `check()`, forsta raden `NAMN: PASS`.

## Filstruktur

| Fil | Ansvar |
|---|---|
| `include/infernix/image/family.h` (ny) | De enda familjetypen som upstream och serve ser: `ImageFamily`, `FamilyConstants`, `enum class ImageFamilyKind`, `parse_image_family(std::string_view)` |
| `plugins/image/CMakeLists.txt` (ny) | Byggaren for modulen; bara denna katalog ar "var" kod |
| `plugins/image/families/qwen_image_2_1.cpp/.h` (ny) | `constexpr ImageFamily qwen_image_2_1_family()` - konstanter, prepare, need, execute |
| `plugins/image/serve/image_module.cpp/.h` (ny) | Route-registrering och en hook per protokol; tar over `chat_image_tool.cpp`-rollen |
| `src/models/qwen_image/*` (flyttas/behalls) | Qwen-Image-2.1:ns matematik, redan familjeagt med undantag for de literaler plan 1 tog |
| `src/serve/openai_images.{h,cpp}`, `image_generation_service.{h,cpp}`, `image_job_store.*`, `prompt_enhancer.*` | Protokoll-och-jobbpolicy, familjeagnostisk efter Task 2 och 3 |
| `src/serve/http_server.{h,cpp}`, `openai_chat_http.cpp`, `openai_responses_http.cpp`, `anthropic_messages_http.cpp` | **Endast en hook anropad per protokol** efter Task 4 |
| `include/infernix/engine.h`, `src/runtime/engine/engine_core.h`, `src/runtime/expert_offload/device_lease.cpp` (ny), `src/models/qwen4_exp/program/*` | De namngivna upstream-sommarna (6-8, ej fyra) - Task 0 for flagga/facad/ankar, Task 5 for resten |
| `apps/serve/main.cpp:335-383` | Den enda plats som bygger familjen och servicen |
| `src/serve/serve_options.{h,cpp}` | `--image-family`: alternativ, hjalptext, validering |
| `CMakeLists.txt`, `src/models/CMakeLists.txt`, `src/serve/CMakeLists.txt`, `tests/CMakeLists.txt`, `tests/**/tests.cmake` | Flaggan och registreringen |

---

### Task 0: Port-barhet fore familjesommen (WP-A till WP-D ur 2026-11-24-bedomningen)

Den har tasken ar forvillkoret for att nagonsin kunna flytta bildvagen till en nyare upstream utan att
skriva om den. Utan den blir varje upstream-sync en genomgripande rewrite, eftersom var lane just nu ar
svetsad i chat-Programmets privata tillstand och bygger ovillkorligt in sig i upstreams serve/models/ops.
Ordningen ar avsiktlig: WP-A (flaggan) ar billig och blockera allt annat, WP-B (facaden) ar den enda som
kopplar loss chat-intern, WP-C/D ar omankningar mot dagens upstream.

**Files:**
- Modify: rot `CMakeLists.txt` (`option(INFERNIX_IMAGE ... ON)`), `src/models/CMakeLists.txt:22`,
  `src/ops/CMakeLists.txt:17,19,42,45`, `src/serve/CMakeLists.txt:20-27,47`
- Create: `src/runtime/expert_offload/device_lease.cpp`, `include/infernix/image/device_lease.h`
  (smal facade, ingen bildterminologi dar upstream inte behover den)
- Create: `include/infernix/image/yield_types.h` (`ImageYieldProbe`/`ImageInterludeContext` ut ur
  `include/infernix/types.h`), `tests/runtime/test_image_device_lease_facade.cpp` (facadkontrakt, host)
- Modify: `src/models/qwen4_exp/program/program_impl.h`, `program.{h,cpp}`, `src/runtime/engine/engine.cpp`
  (dispatcha alla **fyra** variant-karnorna, inkl. `Inkling2Loop`), `src/runtime/engine/engine_core.h`
  (worker-ack mot de namn som faktiskt finns upstream), `src/serve/image_generation_service.cpp`

**Interfaces:**
- Produces: `ImageDeviceLease` - ett smalt snitt som bara exponerar frys/tina-residency,
  allokera/frigor-i-friet-omrade, drain/wake-worker, quiesced-epoch-grind och probe-fit. Allt annat
  (`residency_` 127 traffar, `plan_` 148, `pool_` 79, `kv_vmm_` 12, `vram_mutex_` 5 hos oss) stannar
  inne i den upstream-agna `program_impl.h` och `device_lease.cpp` ror bara facaden.

- [ ] **Step 1: Byggflaggan som inte finns**

`option(INFERNIX_IMAGE "Build the image lane" ON)` i rot-CMake, och gor varje ovillkorliga
include/registration villkorlig: modellen, de fyra bild-opens (flyttas till eget target
`infernix_image_ops`, verifierat utan externa konsumenter), serve-katalogen och testregistren.
Verifiera med en byggmatris: `-DINFERNIX_IMAGE=OFF` bygger och kar chat-sviten Gron, ON bygger allt.
Detta ar det stallet dar "en upstream-merge traffar bara sommarna" borjar bli sant.

- [ ] **Step 2: Leaset som facade, inte som vanliga medlemsfunktioner**

Flytta yield/quiesce-/rollback-koden ur `program_impl.h` till `device_lease.cpp` som bara pratar med
`ImageDeviceLease`. **De tva blockerarna fran de 119 senaste upstream-commitsen hanteras har, inte senare:**
  * `perf(qwen4_exp)` "every expert in RAM" (`a3f78ef8`, `18388f5b`) haller experter boende via
    `residency_->lendable_bytes()` med `!tier_`-grenar. Facaden difor en uttrycklig
    residens-evictions-override som anroparen maste be om - inte en antagen "poolen gar att frysa".
    Testa grenen explicit: ett leaset som fick behallla expertpoolen ska neka ett bildjobb med ett
    dokumenterat 503, aldrig tyst overskrida.
  * `b821d961` kapsar `VramSnapshot::device_free` vid NVML:s enhetsvida fria. Var nuvarande probe lar
    redan den kapsade kallan (`vram_->query()` fran `program_impl.h:2936`), sa beviset har ar ett
    oforandrat-bevis: facaden ska bara kunna ge en fri-siffra fran den kapsade kallan, och ett host-test
    som garanterar det foljer med.

- [ ] **Step 3: Omank worker-ack och typer ur types.h**

Planens antagna namn `request_interlude_preempt`/`wake_llm_worker`/`drain_llm_frontier`/
`interlude_preempt_outstanding` finns inte upstream (0 traffar pa `f9c2e42c`, har aldrig gjort det) -
ankar mot de worker-postpunkter som finns dar nu, namngivna och bildagnostiska. Flytta
`ImageYieldProbe`/`ImageInterludeContext` ur `include/infernix/types.h` (66 upstream-commits sedan
fork) till `include/infernix/image/`, och gor `expert_pool_frozen_by_lease` till en facade-svar
i stallet for ett chat-residens-lackage. `ChatTool`/`GenerationRequest`/`GenerationOutcome` ligger nu
upstream i `src/serve/request.h:266`/`src/serve/generation_service.h:33` - serve-hooken i Task 4 ankar
dit, inte till `types.h`.

- [ ] **Step 4: Verifiering utan kortan, och ett port-bevis**

Host-tester: `infernix_engine_test`, `infernix_image_generation_service_test` (FakeRunner nar sig inte
av flytten), det nya facadkontraktet, och lease/coexistence-sviten i `tests/models/qwen4_exp/tests.cmake`
(GPU-delen sparkas till Task 6). Port-beviset, som ar hela poangen:

```bash
git diff --stat origin/main -- include src/runtime src/models src/ops src/serve apps CMakeLists.txt
```

Forvantat: inga bildandringar utanfor de 6-8 namngivna sommarna, och dokumentationen i
`docs/maintainer/image-integration-points.md` namner varje soma med sin nuvarande upstream-ankare sa att
en framtida port ar "sync vendored subtree, `git am` seam-patchserien, fixa namngivna ankarin, bygg bakom
`-DINFERNIX_IMAGE`". Om nagot stall har vuxit bortom ett par tiotal rader: gor om det stallet.

- [ ] **Step 5: Commit (endast pa begaran)**

```bash
git add CMakeLists.txt src/ops/CMakeLists.txt src/models/CMakeLists.txt src/serve/CMakeLists.txt include/infernix/image src/runtime src/serve/image_generation_service.cpp docs/maintainer/image-integration-points.md
git commit -m "refactor: build the image lane behind a flag and move the device lease behind a facade"
```

---

### Task 1: Explicit familjeval och familje-headern

**Files:**
- Create: `include/infernix/image/family.h`
- Create: `plugins/image/families/qwen_image_2_1.h`, `plugins/image/families/qwen_image_2_1.cpp`
- Create: `plugins/image/CMakeLists.txt`
- Create: `tests/image/test_image_family_selection.cpp`
- Modify: `src/serve/serve_options.h:83-99` (nytt falt), `src/serve/serve_options.cpp:103-106` (arity), `:578-592` (hjalptext), `:917-926` (parsning)
- Modify: `apps/serve/main.cpp:335-383`
- Modify: `CMakeLists.txt` (option), `src/CMakeLists.txt` (add_subdirectory), `src/serve/CMakeLists.txt:20-27,47`
- Modify: `docs/serving.md`, `docs/cli.md`, `README.md` (nya flaggan - externt kontrakt)

**Interfaces:**
- Consumes: allt plan 1 levererade (`kImageJointMaxTokens`, `VaeTopology`, `read_scheduler_config`, `read_dit_rope_config`, `DiTDims`, `VisionLaneDims`).
- Produces:

```cpp
// include/infernix/image/family.h
#pragma once
// A picture family is a compile-time decision, selected by name from a fixed set. There is
// no registry and no discovery: serve_options parses the operator's --image-family into one
// enum value, one switch in this header hands out the matching family, and an artifact the
// family cannot bind is refused at bind time by the existing recipe check.
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>

namespace infernix::image {

enum class ImageFamilyKind { QwenImage2_1 };

// The numbers that are a family's fact, not the protocol's: what a request may ask for and
// what the family's sidecar is called. /v1/capabilities advertises exactly these, so they
// are the external contract and change only together with the schema tests and docs.
struct FamilyConstants {
    std::uint32_t default_width = 0;
    std::uint32_t default_height = 0;
    std::uint32_t default_steps = 0;
    std::uint32_t minimum_dimension = 0;
    std::uint32_t maximum_dimension = 0;
    std::uint32_t minimum_steps = 0;
    std::uint32_t maximum_steps = 0;
    double minimum_cfg_scale = 0.0;
    double maximum_cfg_scale = 0.0;
    std::size_t maximum_references = 0;
    double canvas_target_area = 0.0;
    std::uint32_t latent_channels = 0;
    std::uint32_t spatial_downscale = 0;
    std::uint32_t joint_max_tokens = 0;      // ops::kImageJointMaxTokens for this family
    std::string_view prompt_enhancer_model;   // empty => prompt_enhance is refused
};

struct ImageFamily {
    std::string_view name;
    FamilyConstants constants;
    // Host only: render the prompt, build the conditioning, decide the canvas. No device.
    std::expected<PreparedJob, std::string> (*prepare)(const PreparedJobRequest&) noexcept = nullptr;
    // The VRAM the prepared job needs, from the artifact-owned topology. Never a guessed factor.
    [[nodiscard]] std::uint64_t (*need)(const PreparedJob&) noexcept = nullptr;
    // The device work. No HTTP, no json, no job store.
    [[nodiscard]] std::expected<FinishedPicture, std::string>
        (*execute)(const PreparedJob&, void* image_stream,
                   const std::shared_ptr<ImageJobProgress>&) noexcept = nullptr;
};

[[nodiscard]] std::optional<ImageFamilyKind> parse_image_family(std::string_view name) noexcept;
[[nodiscard]] const ImageFamily& image_family_for(ImageFamilyKind kind) noexcept;
[[nodiscard]] std::string_view image_family_names_for_help() noexcept;

} // namespace infernix::image
```

```cpp
// one switch, no registry (include/infernix/image/family.h, inline definitions)
inline std::optional<ImageFamilyKind> parse_image_family(std::string_view name) noexcept {
    if (name == "qwen-image-2-1") { return ImageFamilyKind::QwenImage2_1; }
    return std::nullopt;   // the caller turns this into a startup error naming the allowed set
}
inline const ImageFamily& image_family_for(ImageFamilyKind kind) noexcept {
    switch (kind) {
    case ImageFamilyKind::QwenImage2_1: return qwen_image_2_1_family();
    }
    std::terminate();      // every enumerator is handled above; a new one must be added here
}
```

- [ ] **Step 1: Skriv det failande testet**

`tests/image/test_image_family_selection.cpp` - bara publikt beteende, och inget test som
lagger fast privat filorganisering:

```cpp
// IMAGE_FAMILY_SELECTION_TEST: PASS
#include "infernix/image/family.h"
#include <iostream>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::cout << (ok ? "ok: " : "FAIL: ") << what << '\n' << std::flush;
    if (!ok) { ++failures; }
}
} // namespace

int main() {
    using namespace infernix::image;
    check(parse_image_family("qwen-image-2-1") == ImageFamilyKind::QwenImage2_1,
          "the family the operator names is the family that runs");
    check(!parse_image_family("minimax-h3-image").has_value(),
          "a family this build cannot run is refused, not approximated");
    check(!parse_image_family("").has_value(), "an empty name is not a family");

    const auto& family = image_family_for(ImageFamilyKind::QwenImage2_1);
    check(family.name == "qwen-image-2-1", "the family names itself");
    // The shipped numbers, from today's openai_images.h - the external contract must not move.
    check(family.constants.default_width == 768 && family.constants.default_height == 768,
          "the default canvas is unchanged");
    check(family.constants.minimum_dimension == 32 && family.constants.maximum_dimension == 2048,
          "the advertised dimension range is unchanged");
    check(family.constants.minimum_steps == 2 && family.constants.maximum_steps == 100,
          "the advertised step range is unchanged");
    check(family.constants.maximum_references == 4, "the advertised reference limit is unchanged");
    check(family.constants.canvas_target_area == 768.0 * 768.0, "the aspect canvas area is unchanged");
    check(family.constants.joint_max_tokens == 20480, "the joint ceiling is the shared constant");
    check(family.prepare != nullptr && family.need != nullptr && family.execute != nullptr,
          "all three family steps are present");
    std::cout << (failures == 0 ? "IMAGE_FAMILY_SELECTION_TEST: PASS"
                                : "IMAGE_FAMILY_SELECTION_TEST: FAIL") << '\n';
    return failures == 0 ? 0 : 1;
}
```

- [ ] **Step 2: Kor och ser det faila**

`image-edit\_p40_host_tests.bat infernix_image_family_selection_test` → kompileringsfel.

- [ ] **Step 3: Implementera headern och familjen**

Fyll `FamilyConstants` for `qwen_image_2_1_family()` med de verkliga talen fran
`src/serve/openai_images.h:88,93-97,110,120,123-124,131,135-136,143,147,180-181,186`
(kopiera exakt, ingen omrokning). `prepare`/`need`/`execute` pekar fran borjan pa de
funktioner som redan finns i familjen - `make_tower_prompt_renderer`/
`run_conditioned_text_tower` for prepare, `image_device_need()` (plan 1) for need,
`run_image_job` fran `src/models/qwen_image/image_pipeline.h:144-148` for execute. I
denna task ar de tre falten `nullptr`-fria men anropas annu inte fran servicen; det kommer
i Task 3.

- [ ] **Step 4: CLI:ade - utan tyst default**

`src/serve/serve_options.h` efter `image_artifact_path`:

```cpp
    // --image-family NAME: which picture family the image artifact is. Required together with
    // --image-artifact: there is no guess. --help lists the families this build can run; any
    // other name, or a missing value, is a startup error rather than a 503.
    std::string image_family;
```

`src/serve/serve_options.cpp`: `ServeOptionArity{"--image-family"},` i tabellen (vid `:103`),
en `else if (arg == "--image-family")`-gren efter `--image-artifact`-grenen som kraver
varde och validerar med `parse_image_family`, och en hjalptextrad vid `:578` som skriver ut
`image_family_names_for_help()`. Feltexten maste namna de tillatna namnen sa att operatoren
inte gissar:

```cpp
        } else if (arg == "--image-family") {
            options.image_family = require_value("--image-family");
            if (!infernix::image::parse_image_family(options.image_family)) {
                throw std::invalid_argument(
                    "--image-family must name one of: " +
                    std::string(infernix::image::image_family_names_for_help()));
            }
```

Och i den validering som redan kraver att `--image-artifact` och `--image-text-processor`
hor ihop: nar `image_artifact_path` ar satt men `image_family` ar tom, startfel
`"--image-artifact also needs --image-family"`. Utan `--image-artifact` ar familjen
irrelevant och rutten besvarar den dokumenterade 503:an som forut.

- [ ] **Step 5: Koppla in familjen i byggaren**

`apps/serve/main.cpp` dar servicen konstrueras (`:364-376`): las familjen fore servicens
konstruktion och skicka med den redan nu (används formellt forst i Task 2-3, men startar
redan valideringen):

```cpp
            const auto family_kind = infernix::image::parse_image_family(options.image_family)
                .value_or_throw([] { throw std::runtime_error("image artifact configured without a family"); });
            const auto& family = infernix::image::image_family_for(family_kind);
            logger->info("image family {} serving artifact {}", family.name, options.image_artifact_path);
```

`.value_or_throw` finns inte - skriv den uttryckliga `if (!kind) throw` i stallet, med
samma villkor som CLI-valideringen redan garanterar.

- [ ] **Step 6: CMake**

`CMakeLists.txt` intill de andra alternativen (`:18-20`):

```cmake
option(INFERNIX_IMAGE "Build the image generation lane (Qwen-Image-2.1)" ON)
```

`src/CMakeLists.txt` (dar `add_subdirectory(models)`/`serve` redan finns):
`if(INFERNIX_IMAGE) add_subdirectory(${PROJECT_SOURCE_DIR}/plugins/image) endif()`.
`plugins/image/CMakeLists.txt` bygger `infernix_image_plugin` STATIC med familjens
kallfiler och länkar `infernix_qwen_image infernix_qwen_image_runtime infernix_ops
infernix_artifact`. `src/serve/CMakeLists.txt` får `infernix_image_plugin` pa link-listan
(`:47`) bakom `if(INFERNIX_IMAGE)`. Testet registreras i `tests/cmake/ProductTests.cmake`
efter monsteret fran `:58-89` med `LIBRARIES infernix_image_plugin`.

- [ ] **Step 7: Dokumentera kontraktet och kor**

`docs/serving.md`, `docs/cli.md` och `README.md` far `--image-family` med samma ordlyd som
`--help`. Kor `image-edit\_p40_host_tests.bat infernix_image_family_selection_test` → `PASS`,
samt `infernix_serve_options_test` (eller den svit som redan tackar CLI-parse; hitta den i
`tests/cmake/ProductTests.cmake`) och `infernix_openai_images_schema_test`,
`infernix_http_routes_test`. Bekrafta aven att `--help`-texten innehaller familjenamnet.

- [ ] **Step 8: Commit (endast pa begaran)**

```bash
git add include/infernix/image plugins/image src/serve/serve_options.h src/serve/serve_options.cpp apps/serve/main.cpp CMakeLists.txt src/CMakeLists.txt src/serve/CMakeLists.txt tests docs README.md
git commit -m "feat: select the image family explicitly at startup behind the INFERNIX_IMAGE option"
```

---

### Task 2: Familjen ager talen, protokollet validerar emot dem

**Files:**
- Modify: `src/serve/openai_images.h` (konstanterna `:88-147,180-181` blir familjelevererade), `src/serve/openai_images.cpp` (valideringen tar `const FamilyConstants&`)
- Modify: `src/serve/http_server.cpp` `make_capabilities_json` (`:657-745`) och dess anrop
- Modify: `tests/test_openai_images_schema.cpp`, `tests/test_http_routes.cpp`

**Interfaces:**
- Consumes: `infernix::image::FamilyConstants` fran Task 1.
- Produces: `parse_openai_images_request(..., const infernix::image::FamilyConstants&)`, `resolve_canvas_geometry(..., const FamilyConstants&)`, `snap_to_32` oforandrad (det ar familjens latent geometri, inte ett protokolltal: den tar nu `std::uint32_t spatial_downscale`), `make_openai_image_response(..., const FamilyConstants&)`.

Det finns inget alias kvar: de tidigare `inline constexpr`-talen i `openai_images.h`
**raderas** och finns bara i `qwen_image_2_1_family()` (traedets regel mot fallbacks och
alias).

- [ ] **Step 1: Skriv det failande testet**

`tests/test_openai_images_schema.cpp`, ett nytt block som bevisar att gransen foljer
familjen och inte en header-literal. En frammad familj borjar i ett eget test-objekt,
inte i `plugins/`:

```cpp
      // The advertised limits come from the family that was selected. A second family is
      // expressible at all only if nothing in the validator still reads a literal.
      infernix::image::FamilyConstants foreign;
      foreign.default_width = foreign.default_height = 1024;
      foreign.minimum_dimension = foreign.maximum_dimension = 4096;   // 16 MP native
      foreign.minimum_steps = 2; foreign.maximum_steps = 6;           // 6-step, no CFG
      foreign.minimum_cfg_scale = foreign.maximum_cfg_scale = 1.0;
      foreign.maximum_references = 16; foreign.canvas_target_area = 4096.0 * 4096.0;
      foreign.latent_channels = 24; foreign.spatial_downscale = 16;
      foreign.joint_max_tokens = 16384;
      auto request = valid_request();
      request.width = request.height = 4096;
      check(parse_openai_images_request(body_of(request), foreign, problem).ok(),
            "a family with a larger canvas is accepted by the shared validator");
      check(!parse_openai_images_request(body_of(request), qwen_constants(), problem).ok(),
            "the same ask is still refused under the shipped family");
```

`qwen_constants()` ar en helper i testfilen som returnerar
`infernix::image::qwen_image_2_1_family().constants`; `valid_request()`/`body_of()` follows
filens befintliga monster for JSON-bygge (las av filen, inte fran minnet).

- [ ] **Step 2: Kor och ser det faila**

`image-edit\_p40_host_tests.bat infernix_openai_images_schema_test` → kompileringsfel
(valideringen tar inga konstanter).

- [ ] **Step 3: Flytta agandet**

Ta bort `kMinimumImageDimension`/`kMaximumImageDimension`/`kMinimumImageSteps`/
`kMaximumImageSteps`/`kMinimumImageCFGScale`/`kMaximumImageCFGScale`/
`kMaximumImageReferences`/`kCanvasTargetArea`/`kImagePromptEnhancerModel` fran
`openai_images.h` och las in dem som falt hos familjen (Task 1 gjorde redan det). Alla
readare byter till familjens falt: `openai_images.cpp` far parametern, `http_server.cpp`
publicerar `family.constants` i stallet for headerns literaler (och
`canvas.joint_sequence.ceiling_tokens` far `family.constants.joint_max_tokens`, som plan 1
redan bandt till `ops::kImageJointMaxTokens`), `prompt_enhancer`-konfigurationen jamfor mot
`family.constants.prompt_enhancer_model`, och `apps/serve/main.cpp` skickar familjen till
varje anrop. `snap_to_32` byter namn och signatur till `snap_to_latent(double, std::uint32_t
spatial_downscale)` - talet ar 8 * latent-channels-geomin for den shipsade familjen, inte en
magisk tvaa.

- [ ] **Step 4: Uppdatera kontrakttesten och dokumenten tillsammans**

`tests/test_openai_images_schema.cpp` och `tests/test_http_routes.cpp` gar fran headerns
konstanter till `qwen_constants()`. Kapitlet i `docs/serving.md` som publicerar
dimensions-/steps-/referensgranser och `true_cfg_scale`-intervallet granskas mot vad
`/v1/capabilities` nu svarar och andras om det skviker (det ska inte göra det).

- [ ] **Step 5: Kor**

`image-edit\_p40_host_tests.bat infernix_openai_images_schema_test` och
`infernix_http_routes_test`, `infernix_image_generation_service_test`,
`infernix_chat_image_tool_test` → alla `PASS`. Bekrafta uttryckligen att `/v1/capabilities`
svarar med **exakt samma JSON som forut** (diffa emot en sparad kopia av svaret fran fore
andringen; den finns i `image-edit\requests.jsonl`-analyserna - annars las den fran en
`curl` mot den levande servern innan nagot byggs om).

- [ ] **Step 6: Commit (endast pa begaran)**

```bash
git add src/serve tests docs
git commit -m "refactor: the selected image family owns the advertised request limits"
```

---

### Task 3: prepare / need / execute - servicen blir familjeagnostisk

**Files:**
- Modify: `src/serve/image_generation_service.h:158-249,321-404` (seamen byts mot familjen), `src/serve/image_generation_service.cpp` (kroppen slappper familjespecifik kod)
- Create: `tests/image/fake_family.h`
- Modify: `tests/test_image_generation_service.cpp` (hjalarparna en gang, inte 40 ganger)
- Modify: `apps/serve/main.cpp:335-376`

**Interfaces:**
- Consumes: `ImageFamily` (Task 1), `FamilyConstants` (Task 2), `VaeTopology`/`image_device_need()` (plan 1), `ImageInterludeRunner` (oforandrad), `ImageJobStore` (oforandrad), `PromptEnhancer` (oforandrad).
- Produces:

```cpp
class ImageGenerationService {
public:
    struct Dependencies {
        std::unique_ptr<ImageInterludeRunner> runner;
        ImageJobStore& store;
        const infernix::image::ImageFamily& family;
        std::shared_ptr<PromptEnhancer> enhancer;
        std::shared_ptr<spdlog::logger> logger;
    };
    explicit ImageGenerationService(Dependencies dependencies);
    // publika medlemsfunktioner oforandrade: is_configured, in_flight_jobs,
    // oldest_in_flight_age_seconds, generate, next_job_id, last_job_progress, job_status_json
};
```

De sju provider-typerna (`PromptRendererProvider`, `PromptEmbedsProvider`, `DiTDimsProvider`,
`VisionLaneDimsProvider`, `ReferencePromptRendererProvider`, `ImageJobExecutor`,
`make_artifact_image_job_executor`, `make_artifact_dit_dims_provider`,
`make_artifact_vision_dims_provider`, `make_tower_prompt_renderer`,
`make_tower_prompt_embeds_provider`, `make_tower_reference_prompt_renderer`)
**raderas** fran den publika ytan och blir familjens interna detalj i
`plugins/image/families/qwen_image_2_1.cpp`. `bench\image\image_stage_bench.cpp:261-263`
som anvander tva av dem byggs om att ta familjen direkt.

- [ ] **Step 1: Skriv fake-familjen och det forsta failande testet**

`tests/image/fake_family.h` - samma hjalpare dagens test har (`fixed_renderer()`,
`fixed_provider()`, `fixed_dims()`, `fixed_executor()`, `sample_request()` fran
`tests/test_image_generation_service.cpp:23-109`) uttryckt som en familj, sa att inget
beteende forsvunner nar de sju sewarna tas bort:

```cpp
// A host-only family for the service suite: it records what it was asked for and answers
// with fixed bytes, so the service's own decisions (refuse, skip, run, restore) stay
// testable without an artifact or a device.
inline infernix::image::ImageFamily fake_family() {
    static std::vector<infernix::image::PreparedJobRequest> prepared;
    static std::vector<std::uint64_t> needed;
    static std::vector<FakeExecuteCall> executed;
    // ...prepare/need/execute bodies that push into the three vectors and return the fixed
    // answer the old providers returned; constants copied from qwen_image_2_1_family().
}
```

De befintliga testfallen som inte ror familjen (jobbutryggning, 429:or,
aterstallningsfel, cancas-stepping, progress) maste sta sig **ord for ord** - bara
konstruktionen andras. Flytta dem med redigering, inte omskrivning.

- [ ] **Step 2: Flytta en svit i taget och kor**

For varje grupp i `tests/test_image_generation_service.cpp`: byt
`ImageGenerationService service(std::move(runner), store, fixed_renderer(),
fixed_provider(), fixed_dims(), fixed_executor(), ...)` mot
`ImageGenerationService service({std::move(runner), store, fake_family(), nullptr, nullptr})`,
kor `image-edit\_p40_host_tests.bat infernix_image_generation_service_test`, och krav
`PASS (all)` innan nasta grupp. Ingen grupp far hoppa over en korning.

- [ ] **Step 3: Flytta familjekoden ut ur servicen**

I `image_generation_service.cpp`: allt som heter pa `infernix::models::qwen_image::*`,
text-tornet, ViT:en, VAE:n, DiT:en eller artifact-lasning flyttas till
`plugins/image/families/qwen_image_2_1.cpp` och bakom `family.prepare/need/execute`.
Servicenbehaller HTTP-/jobb-/leasingpolicyn: validering via `FamilyConstants`,
`runner_->probe()`, VRAMwalken, `store_`-overgangarna, enhancer-grinden fran plan 1,
`next_job_id`, progress. Ska en kvarvarande rad behova ett familjetal, gar det genom
`family.constants` - aldrig genom en `#include` till familjens privata huvuden. Bekrafta
med `rg -n "qwen_image|models::qwen" src/serve/` → **inga traffar** (utom kanske
`image_job_store.h`'s kommentar om familjens svar, som ar text inte ett anrop).

- [ ] **Step 4: Byggare och bench**

`apps/serve/main.cpp`: de tolv fabriksanropen (`:343-375`) ersatts av familjens egen
initiering (`infernix::image::init_family(family_kind, options)` - en funktion som
postar artifact-sokvagen och processor-sokvagen at familjen och vraker vid fel, sa att
ett trasigt artifact ar ett **startfel** som redan var fallet for `--image-artifact` med
fel aktiveringsplan). `bench/image/image_stage_bench.cpp:261-263` far samma familj via
sitt eget alternativ; bekrara att benchen bygger.

- [ ] **Step 5: Kor hela hostsviten**

`image-edit\_p40_host_tests.bat infernix_image_generation_service_test` och grannarna
(`infernix_image_job_store_test`, `infernix_prompt_enhancer_test`,
`infernix_openai_images_schema_test`, `infernix_chat_image_tool_test`,
`infernix_http_routes_test`, `infernix_image_device_need_test`). Alla `PASS`.

- [ ] **Step 6: Commit (endast pa begaran)**

```bash
git add src/serve plugins tests apps bench
git commit -m "refactor: the image service depends on a family's prepare, need and execute steps"
```

---

### Task 4: En hook per protokol och -DINFERNIX_IMAGE=OFF

**Files:**
- Create: `plugins/image/serve/image_module.h`, `plugins/image/serve/image_module.cpp`
- Move: `src/serve/chat_image_tool.cpp`/`.h` → `plugins/image/serve/`
- Modify: `src/serve/http_server.h:30,80,124-140`, `src/serve/http_server.cpp:431+,535-541,643,653-745,1214-1215,1273-1278`
- Modify: `src/serve/openai_chat_http.cpp:54-69,124-139,330-354`, `src/serve/openai_responses_http.cpp:383-398,460-486`, `src/serve/anthropic_messages_http.cpp:198-213,284-298,378-379,474-490`
- Modify: `src/serve/CMakeLists.txt:20-27`, `tests/cmake/ProductTests.cmake:58-89`

**Interfaces:**
- Consumes: `ImageGenerationService` (Task 3), `ImageFamily` (Task 1).
- Produces:

```cpp
// plugins/image/serve/image_module.h
namespace infernix::serve {
// The whole image lane's footprint on the HTTP server. One hook per protocol, registered
// once, so an upstream merge that touches a chat handler touches nothing image-related.
class ImageModule final {
public:
    explicit ImageModule(ImageGenerationService& service);
    // Adds POST /v1/images/generations, POST /v1/images/edits, GET /v1/images/<id> and the
    // busy fields /health advertises. Returns the advertised image model id.
    [[nodiscard]] std::string register_routes(HttpServer& server);
    // What the chat handler may hand out, answered without the caller knowing images exist.
    void decorate_chat_request(GenerationRequest& request) const;
    [[nodiscard]] bool wants_image_round(const GenerationOutcome& outcome) const noexcept;
    [[nodiscard]] ChatImageRounds run_image_rounds(const GenerationOutcome& outcome,
                                                   std::string_view url_base,
                                                   std::string_view source,
                                                   std::string_view protocol,
                                                   std::string correlation);
    [[nodiscard]] std::string missing_url_suffix(const std::string& text,
                                                  const ChatImageRounds& rounds) const;
    [[nodiscard]] nlohmann::json capabilities_block() const;
};
}
```

- [ ] **Step 1: Flytta utan att andrade beteendet**

`git mv src/serve/chat_image_tool.* plugins/image/serve/` (eller manuellt, da ingen commit
ska till nu), lägg filerna i `plugins/image/CMakeLists.txt`, ta dem fran
`src/serve/CMakeLists.txt:20-27`. Kor hostsviten - allt ska fortfarande `PASS` innan
nagon hook skrivs.

- [ ] **Step 2: Skriva hook-klassen och ersatta de fyra stallena**

De tre protokolhandler-na som idag, enligt verifierat underlag, alla gor exakt samma
sekvens (`append_chat_image_tool` → `artifact_url_base` → `make_picture`-lambda →
`outcome_is_chat_image_tool_round` → `run_chat_image_rounds` → `missing_image_url_suffix`)
kondenseras till ett anrop per stalle, t.ex. i `openai_chat_http.cpp`:

```cpp
    if (module != nullptr) { module->decorate_chat_request(request.generation); }
    ...
    if (module != nullptr && module->wants_image_round(outcome)) {
        ChatImageRounds rounds = module->run_image_rounds(
            outcome, image_url_base(req, options_), "chat", "openai", correlation);
        outcome.text += module->missing_url_suffix(outcome.text, rounds.urls);
    }
```

Och i `http_server.cpp`: `attach_images` raderas som publik medlemsfunktion och ersatts av
`void HttpServer::attach_image_module(ImageModule& module);` med ett enda falt
`ImageModule* image_`; `/health`-busy-blocken (`:535-541`), `is_configured`-kollen
(`:643`), `GET /v1/images/<id>` (`:1214-1215`) och `make_capabilities_json`
(`:657-745`) las over till modulen (`capabilities_block()`) sä att serverns
capabilities-svar är **byte-identiskt** med det sparade fore-svaret fran Task 2.
`apps/serve/main.cpp:376` blir `server.attach_image_module(*image_module);`.

- [ ] **Step 3: Flaggan**

Allt under `plugins/image/` och dess tester byggs bara nar `INFERNIX_IMAGE=ON`.
`src/serve/CMakeLists.txt` länkar `infernix_image_plugin` bakom samma villkor, och
`#ifdef` behövs inte i serverkoden om `ImageModule` bara förekommer som
ofullstandig-typ-forward-declaration i `http_server.h` (en pekartare räcker) och
anropen är inlagda i en enda liten fil som alltid byggs men bara har innehall nar
flaggan är på - val den vagén och skriv ut den i koden sä att det framgår var
innehållet sviker när flaggan är av.

- [ ] **Step 4: Bygga bada konfigurationerna**

```bash
cmake -S . -B build-off -DINFERNIX_IMAGE=OFF ...   # samma generator, samma triplet
cmake --build build-off --config Release --parallel --target infernix-serve infernix_http_routes_test
```
Forvantat: bygger, servern har ingen `/v1/images`-rutt (fel 404, inte 503 - kontrollera
vad som ar dokumenterat och gor det konsistent; dokumentera utfallet), chat-rutten
betjanas utan bildverktyg. Med flaggan PA: samma hostsvit som forut, alla `PASS`.

- [ ] **Step 5: Commit (endast pa begaran)**

```bash
git add src/serve plugins apps tests
git commit -m "refactor: the image lane attaches to the HTTP server through one module per protocol"
```

---

### Task 5: De namngivna upstream-sommarna (6-8, inte fyra)

**Rattad 2026-11-24 mot `f9c2e42c`:** Task 0 gor redan flaggan, facaden, worker-acket och typerna.
Det som blir kvar har är att anvanda de ankarin som faktiskt finns upstream - `execution_mu_` finns
inte (0 traffar), `engine.cpp` dispatchar fyra variant-karnor (inkl. `Inkling2Loop`) inte en, och
serve-typerna borjar i `src/serve/request.h`/`generation_service.h` inte i `types.h`.

**Files:**
- Create: `src/runtime/expert_offload/device_lease.cpp`
- Modify: `include/infernix/engine.h:141-166` (publika namnet **oforandrade** - bara flyttar om det behövs), `src/runtime/engine/engine_core.h` (worker-ack), `src/models/qwen4_exp/program/program_impl.h:5986-6514` (yield-koden ut), `src/models/qwen4_exp/program/program.h`/`program.cpp`
- Modify: `src/runtime/expert_offload/sources.cmake`, `src/models/qwen4_exp/program_sources.cmake`
- Create: `docs/maintainer/image-integration-points.md`

**Interfaces:**
- Consumes: `ImageYieldLease` (`program_impl.h:5986-6001`), `probe_image_yield` (`:6013-6048`), `yield_for_image` (`:6052-6151`), `rollback_failed_yield` (`:6171-6179`), `release_from_yield` (`:6184-6218`), `begin_quiesce` (`:6304-6313`), `begin_image_quiesce` (`:6380-6441`), `begin_quiesce_testing` (`:6509-6514`) och allt de nar: `vram_mutex_`, `execution_mu_`, `execution_state_`, `residency_`, `kv_vmm_`, `pool_`, `plan_`, `interlude_mu_`, `interlude_quiesced_epoch_`, `request_interlude_preempt()`, `wake_llm_worker()`, `drain_llm_frontier()`.
- Produces: att varje upstream-fil som en bildvags-merge kan traffa har **ett** infall, inte ett tusental rader, och att diffen mot upstream gar att lasa.

- [ ] **Step 1: Flytta lease-implementationen ur program-tun**

`begin_image_quiesce`/`yield_for_image`/`release_from_yield`/`rollback_failed_yield`/
`probe_image_yield` blir fria funktioner (eller ett litet objekt) i
`src/runtime/expert_offload/device_lease.cpp` som tar de lock och det tillstand de verk-
ligen behov av som parametrar, istallet for att vara memberar i en 6500-raders `program_impl.h`.
De publika namnen och signaturerna i `include/infernix/engine.h:141-166` ar **oforandrade**
- det ar darfor ingen serverkod behover andras. `engine.cpp:423-438` som nu hardkodar
`impl_->qwen4->program` far en enda, uttrycklig vagn: samma `switch`-monster som familjen,
med `kProgramInterlude`-traite:n (`engine_core.h:169-176`) som enda krav - inget
duck-typing utover det som redan finns.

- [ ] **Step 2: Worker-ack i engine_core.h**

Det prekempt-block som idag ligger pa `engine_core.h:2414-2424` och kraver
`request_interlude_preempt()`/`wake_llm_worker()`/`drain_llm_frontier()` flyttas till en
namnd hjalpare i samma fil med ett anrop fran lease-koden, sa att upstream inte behover
kanna till ordet "image" dar.

- [ ] **Step 3: Mät diffen mot upstream**

```bash
git diff --stat origin/main -- include src/runtime src/models/qwen4_exp src/serve/http_server.h src/serve/openai_chat_http.cpp src/serve/openai_responses_http.cpp src/serve/anthropic_messages_http.cpp apps/serve/main.cpp
```
Forvantat (det ar malet med hela planen): de file som ar **upstream-agnostiska** har bara
den lilla infall-punkten. Rakta inte pa ett exakt antal rader men dokumentera utfallet i
`docs/maintainer/image-integration-points.md`, som for each av de fyra ytorna namner fil,
funksion och varfor den maste veta att bilden finns. Om nagot stall har vuxit bortom ett
par tiotal rader: gor om det stallet, inte hela planen.

- [ ] **Step 4: Kor host-testen som beror av lease:**

`infernix_qwen4_exp_image_interlude_*_test`, `infernix_image_generation_service_test`
(FakeRunner njar sig inte av flytten - verifiers), `infernix_engine_test` och den ordinarie
`--help`/health-sviten. GPU- och `_real`-fallen sparkas till Task 6.

- [ ] **Step 5: Commit (endast pa begaran)**

```bash
git add include src/runtime src/models/qwen4_exp docs/maintainer/image-integration-points.md
git commit -m "refactor: move the image interlude out of the program header into a device lease"
```

---

### Task 6: Ett offline-fonster - verifiering, deploy och smoke test

- [ ] **Step 1: Blackout, bygge av allt**

`_serve_watch.pause` med `ends_unix_s` som tackar fonstret, killa `infernix-serve.exe` och
`pe_worker.py`, bygg hela servern med `INFERNIX_IMAGE=ON` device-link (nvcc behover fri
kort), och bygg aven `build-off`-konfigurationen en gang for att bekräfta att OFF-laget
fortfarande bygger.

- [ ] **Step 2: GPU- och `_real`-testen**

Med `INFERNIX_TEST_ARTIFACT=C:\Users\Micke\Desktop\Infernix FlashNext\models\Qwen-Image-2.1-FP8-v4.infernix`
och `--image-family qwen-image-2-1` dar ett test behover en familj:
`infernix_image_need_ledger_real_test` (de certifierade siffrorna fran STATE: 2048x2048
predicted 13869 / measured 12653, 2016x2016 13548 / 12295, 1024x1152 6409 / 4367, ViT-edit
6163 / 3634 - **oforandrade**, annars ar detta en regression som maste forklaras.
(ViT-edit-raden heter 6163 inte 6010 fran och med plan 1 round-2: reservationen debiterar
det derivede varsta referensplanen, se test_image_need_ledger_real.cu:142-150. Only the
measured side is the regression gate; a prediction may rise, never fall.),
`infernix_image_stream_device_real_test` (plan 1), lease/coexistence-sviten i
`tests/models/qwen4_exp/tests.cmake`, `infernix_attn_dit_joint_real_test`,
`infernix_rope_c64_real_test`, `infernix_dit_adaln_real_test`.

- [ ] **Step 3: Deploy och EN omstart**

`_p40_deploy.ps1` fran plan 1 (rollback `.bak-p40`). Desktop-batten maste fa
`--image-family qwen-image-2-1` - skriv in den dar (filen utanfor repo, committas inte)
och starta med `start flash next local - K8v4.bat`, aldrig nagon annan launcher. En
omstart, inga fler.

- [ ] **Step 4: Smoke test (anvandarens uttryckliga krav, allt mot den levande servern)**

1. `/health` -> `status: ok`, och `image_jobs`-falten finns kvar.
2. `/v1/capabilities` -> **identisk JSON** med den sparade fore-kopian fran Task 2 (diffa;
   inget falt fick rora sig, varken `ceiling_tokens` 20480 eller dimensions-/steps-/
   reference-granserna).
3. Ett t2i-jobb 512x576, 20 steg, fast seed -> PNG, och `py -3.11 ~/.Codex/scripts/verify_image.py <path>`
   (anvandarens kontextfria bildkontroll) bekräftar vad som faktiskt ritades.
4. Ett edit-jobb med en referens och `prompt_enhance:auto` -> `applied` i operatorraden;
   samma bildkontroll.
5. Ett chat-svar som anvander `generate_image`, och ett chat-svar under ett pagaende drag
   (det fall som tidigare gav 503) -> bade bild och text.
6. Ett jobb som utfor `--image-family`-felveten (slå av/ändra till ett okänt namn i en
   separat, snabb start och bekräfta startfelet med de tillatna namnen) - utan att
   stanga ner den ordinarie servern langre an hogst nodvandigt.
7. Logg-posterna for allt ovan jamforas med motsvarande fore: samma `outcome`, samma
   `size`, inga nya `image_vram_insufficient`.

- [ ] **Step 5: Forster, dokumentera, disclose**

Ta bort `_serve_watch.pause`, starta PE-workern via batt, och skriv in resultatet i
`.agent/STATE.md`: nya SHA256 for exet, verifierade smoke-siffror, de fyra upstream-
ytornas faktiska diffstorlek, och **uttryckligen vad som inte kunde kontrolleras**
(t.ex. om nagot `_real`-test hastades eller om en prestandajamforelse inte gjordes).
For radets skull: uppge aven om nagot av ovanstaende failade och vad du gjorde at det.

---

## Self-review

**Spec-tackning:** WP3: explicit familjeval utan registry -> Task 1; familjeagtade tal och
kontrakttesten tillsammans -> Task 2; prepare/need/execute och att servicen slappper
familjekoden -> Task 3. WP4: en hook per protokol och flytten till `plugins/image/` ->
Task 4; `option(INFERNIX_IMAGE ON)` -> Task 1 (steget med flaggan) och Task 4 Step 3; de
fyra upstream-ytorna -> Task 5. Smoke test och en enda omstart -> Task 6.

**Avvikelse fran specen, medveten:** specen skriver `switch (descriptor.architecture)`.
Artifactet har ingen sadan falt - bindingen gar pa komponent-configs och vraker vid
avvikelse (`src/models/qwen_image/image_weights.cpp:381`), och `artifact::Binder`
(`src/artifact/binder.h:35-70`) erbjuder `resource`/`find_resource`/`host_object`, ingen
arkitektur-tag. Darfor kommer valet fran den explicita CLI-flaggan och verifieras av
redan befintlig bindningsvalidering; inget nytt artifact-falt och omkonvertering behövs.

**Placeholders:** Task 1 Step 5 och Task 3 Step 1 har uttryckliga "las den verkliga
signaturen fran fil X" - resten har riktig kod.

**Typkonsistens:** `ImageFamily`, `FamilyConstants`, `ImageFamilyKind`,
`parse_image_family`, `image_family_for`, `image_family_names_for_help`, `ImageModule`,
`ImageGenerationService::Dependencies` och `PreparedJob` ar samma namn igenom, och
familjens falt ar de som plan 1 definierade.
