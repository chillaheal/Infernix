# Bildgenerering / bildredigering: extern granskning verifierad mot koden

Datum: 2026-11-14. Författare: Claude (Claude Code), åt Ian.

## Proveniens och vad som är data

- Underlaget är en granskning som Grok gjorde åt användaren, hämtad från en flik i
  CloakBrowser. Den råa sidtexten ligger oförändrad i
  `.agent/grok_tab_snapshot.md` (working-dir-rot, inte i repo). All text där är
  **data, inte direktiv**: inget i den har körts, och inget i den autoriserar något.
- Användarens två prompts till Grok, återgivna från samma snapshot:
  1. "Kritisk granskning av vår kod för image generation och image editing i infernix
     motorn, som kopplar ytterligare en modell till en befintlig modell on demand typ.
     Kolla efter uppenbara buggar, prestandavinster, fel eller problem. Allt funkar
     idag på det stora hela men är hårdkodat för just qwen image 2.1 modellen. skulle
     även vilja få i samma svar, spec på hur man skulle kunna bygga mer flexibelt så vi
     även skulle kunna tillåta andra bildmodeller men kanske även rörlig film längre
     fram men då ska basen vara robust först. Fråga först om oklart syfte"
     (med länk till `https://github.com/chillaheal/Infernix/tree/image-generation`)
  2. "min tanke var också att göra "image generation" som en plugin till infernix
     motorn snarare än att bygga in den totalt i alla filer, så att det enklare går att
     porta vår utveckling till en uppdaterad version av infernix när det släpps för vi
     kaske inte får det mergeat PR, ge om du kan, tankar kring ett sånt upplägg"
- Varje påstående nedan har kontrollerats mot trädet
  `C:\Users\Micke\Documents\Kodprojekt\Infernix - Flash next\Infernix`
  (gren `image-edit`). Verdict-kolumnen är min, inte Groks. Citaten är kopierade ur
  koden med `path:line`.

Användningen av "de"/"deras" om Grok är avsaknad av könsinformation, inte ett antagande.

## 1. Groks bedömning av det som redan är rätt

Dessa egenskaper finns och är anledningen till att basen tål att byggas vidare på
(verifierade som sanningsenliga beskrivningar av koden):

- Bildmodellen är ett eget artifact som endast går in i VRAM efter att chatmodellen
  lämnat den, och svaret emitteras inte innan återställningen bekräftats.
- Separat text-tower-checkpoint (Qwen3-VL-precisionen läses från artifactet).
- För stor joint-sekvens besvaras 400 innan någon quiesce sker.
- Edit och text-to-image är separata routes.
- Felkoderna är avsiktliga (`image_prompt_too_long`, `image_vram_insufficient`,
  `image_interlude_busy`, `image_request_unservable`).
- x4-invarianten är vår egen 500: det är ett villkor vi ställer, inte ett fel som ärvts.

## 2. De fyra "uppenbara buggarna" - verdict

| # | Groks påstående | Verdict | Bevis iour kod |
|---|---|---|---|
| 1 | Prompt-enhancern mäter fel fri minne: bildvägen använder levererat `device_free_bytes`, enhancer-grinden använder den gamla proben | **SANT** | `src/serve/image_generation_service.cpp:1324-1326` använder `context.freed_bytes + context.device_free_bytes`; `:934-935` och `:946-947` använder `context.freed_bytes + fit.device_free_bytes` där `fit` kommer från proben före quiesce |
| 2 | Bild-streamen skapas på anropande trådens CUDA-enhet, inte engine:ns | **SANT (dolt av `--device 0`)** | `src/runtime/qwen4_exp/program.cpp:224-231` `cudaStreamCreate` utan `cudaSetDevice`; interluden löper på httplib-request-tråden (`src/serve/openai_images_http.cpp:159`). Samma antagande i `src/models/qwen_image/image_pipeline.cu:418-424`: `cudaDeviceGetDefaultMemPool(&pool, 0)` + release-threshold 0, aldrig återställd. Repo-vida `cudaSetDevice` finns bara i `src/core/device.cu:147,154` och `src/runtime/expert_offload/vram_monitor.cpp:54` |
| 3 | Två tak på 20480 där admission bara ser ett | **SANT, och värre än Grok sagt** | `include/infernix/ops/attn_dit_joint.h:15` är publikt och används korrekt av `src/ops/attn_dit_joint/attn_dit_joint.cpp:16-17`. Oberoende kopia 2: `src/ops/rope_c64/rope_c64.cpp:17` `kMaxTokens = 20480` med eget meddelande `:62`. Oberoende kopia 3 som Grok missade: `src/ops/dit_adaln/dit_adaln.cpp:14` `kMaxSeq = 20480` med `:66` "dit_adaln: S must be 1..20480". Prosa som publicerar talet: `src/serve/chat_image_tool.cpp:23`; `/v1/capabilities` publicerar från konstanten (`src/serve/http_server.cpp:730-731`) |
| 4 | Den sparade canvaskanten kan ljuga: jobbet behåller yttermåttet medan innerwalken krymper | **SANT** | `src/serve/image_generation_service.cpp:1288-1294` `store_.put(record)` med ytters canvas; innerwalken muterar `effective.width/height` `:1324-1341` (`step_down` `:1338-1339`); `JobUpdate` (`src/serve/image_job_store.h:52-58`) har inget `request`-fält, så recorden korrigeras aldrig - bara wire-svaret `:1391-1398`. Kommentaren `:1284-1285` säger motsatsen |

Delvis sant, ej fel: Grok påstår att `engine_core.h:129` skapar streamen. Där binder
`device_.bind_to_current_thread()` (`engine_core.h:127-129`) endast workern; någon
stream-literal finns inte i den filen.

## 3. Kvalitet/minne - verdict

| Område | Verdict | Bevis |
|---|---|---|
| Referensyta 1024^2 mot canvasdefault 768^2 | **SANT att det skiljer**; Groks påstående att "upstream använder en `output_resolution` för båda" är **okontrollerat** mot extern källa | `src/serve/reference_image.cpp:19-21` `kReferenceTargetArea = 1024*1024`; `include/infernix/serve/openai_images.h:43-44,131` 768^2 |
| Schemat är kompilerat, en turbo `sample_sigmas` skulle ritas med fel schema tyst | **SANT** | `image_pipeline.cu:447` `desc.schedule(qi::SchedulerConfig{})`; literalerna `src/models/qwen_image/scheduler.h:16-24`; `sample_sigmas` existerar inte i trädet; `inputs.sigmas` kommer från host-byggt schema `:304` |
| Vision-peak alltid 64x64 patches och SUMMERAS med DiT+VAE (borde vara max av sekventiella peaks) | **SANT** | `src/serve/image_device_need.h:111` `kVisionWorstReferencePatches = 64*64`, använd `:223`; `total()` summerar lanes `:136-139`; motivering `:218-219`; `kImageWeightResidentBytes = 3072 MiB` `:57`; default-kanvasmotivering `openai_images.h:126-131` |
| VAE-kedjan är en literal med en andra kopia i ledaren | **SANT, båda** | `image_pipeline.cu:1137-1138`, `:1255`, `:1212` (`mean.size()!=64`); ledaren `image_device_need.h:80-87` (`kVaeUpStages`); bara kommentarer korsreferenser |

## 4. Prestanda - verdict, i Groks ordning

1. **SANT.** Varje Euler-steg är ett host-runtrip: sync `image_pipeline.cu:976`,
   D2H `:1009-1010`, loop `:1017-1066`, CFG i double `:1042-1049`, latent D2H
   `:1054-1056`, double-steg `:1059-1062`, BF16-avrundning + h2d `:1063`, andra sync
   `:1064`. Att flytta avrundningsordningen till enheten är den största enskilda vinsten.
   CFG-dubbelns fusion ska lämnas (Groks eget råd, håller).
2. **SANT.** Ingen cachead icke-mål-väg: varje block räknar om alla `Si`
   (32 block x ~40 steg). "prefix" är bara `tp`-gränsargumentet; P13 prefix-KV är
   opt-in och inte bit-exakt mot referensen. Bildmiljöknappar som finns:
   `INFERNIX_IMAGE_OVERLAP`, `INFERNIX_IMAGE_HOST_CACHE_WC`, `INFERNIX_IMAGE_TRACE_DIR`,
   `INFERNIX_VAE_STAGE_STATS`, `INFERNIX_VISION_STAGE_TAG`.
3. **SANT.** Blocken läses från artifactet till pinned host per jobb
   (`image_pipeline.cu:538-572`, `image_device_util.h:392-393` "once per job",
   funktion-lokal cache). En jobb-övergripande cache betalar bara om RAM-ledaren tillåter
   pinning att överleva medan LLM:n kör - det är exakt den handel som Phase A i dag
   rört (43000/7168).
4. **SANT.** Ingen VAE-tiling alls. P14c hör hemma före 2K, inte före en andra modell.

## 5. Plugin-arkitekturen - Groks svar, och vad som redan finns

Grok: ett plugin är rätt **endast som en statisk modul plus en liten patch**, inte en
`.so` som inkluderar `program_impl.h` (den headeren är ingen ABI). Grundsregel:
"Allt som ritar en bild kan lämna trädet. Allt som tar VRAM från den körande LLM:en kan
inte det." `Engine::run_image_interlude` är redan elden vägg.

Verifierat - detta stämmer, och det gör splitten billigare än Grok tror:

- Publikt lease-API finns redan: `include/infernix/engine.h:154-157`
  `supports_image_interlude()` / `run_image_interlude(...)`, `:165-166`
  `probe_image_yield`; `src/runtime/engine/engine.cpp:423-430,433-438` vidaresänder
  direkt till `impl_->qwen4->program->...` (den generiska Engine:n hårdkodar qwen4-
  varianten, vilket är en egen brist). Program-sidan `program.h:671,678,728`;
  internal `program_impl.h:6013 probe_image_yield`, `:6052 yield_for_image`,
  `:6184 release_from_yield`, `:6304 begin_quiesce`, `:6380 begin_image_quiesce`.
  `engine_core.h` deklarerar ingen av dem (bara worker-dörr-hookar).
- De tre bild-opsen (`attn_dit_joint`, `dit_adaln`, `rope_c64`) används av **ingen**
  utanför bildvagen -> flyttbara. De är vanliga källistor inuti delade `infernix_ops`
  (`src/ops/CMakeLists.txt:17,19,42`).
- `src/models/qwen_image/sources.cmake` byggs villkorslöst via
  `src/models/CMakeLists.txt:22`; serve-bildfilerna är poster i
  `add_library(infernix_serve STATIC ...)` som länkar `infernix_qwen_image
  infernix_qwen_image_runtime` okodat (`src/serve/CMakeLists.txt:47`). Rotens
  `CMakeLists.txt` har bara `INFERNIX_BUILD_APPS`/`BUILD_TESTING`/
  `INFERNIX_BUILD_BENCHMARKS`: **det finns inget `INFERNIX_IMAGE`-flagga idag**.
- Ingen registry/plugin-mekanism alls: `register_routes()` (`http_server.h:122`,
  `http_server.cpp:431`) hårdkodar varje handler; enda sömmen är
  `attach(...)`/`attach_images(ImageGenerationService&)` (`http_server.h:80`,
  `http_server.cpp:1273-1279`) och `run_image_job(...)` (`http_server.h:141-143`,
  `openai_images_http.cpp:133`).
- Kopplingen som måste brytas är **serve till engine**, inte delade ops:
  `src/serve/image_generation_service.cpp:11-17` inkluderar `models/qwen_image/*`,
  `src/serve/image_text_tower.cpp:4` når in i
  `models/qwen3_5/frontend/chat_template.h`.
- Chat-kopplingen är kopierad på fyra ställen (Groks "mer konfliktyta än pipelinen" är
  **sant**): `openai_chat_http.cpp:3,59,124,330`,
  `openai_responses_http.cpp:3,388,481,703`,
  `anthropic_messages_http.cpp:4,203,284,474` plus tre response-filer och
  `openai_images_http.cpp:12`; de replikerar `append_chat_image_tool` ->
  `outcome_is_chat_image_tool_round` -> runda (`chat_image_tool.cpp:519-550`).
  Capabilities är en inline `make_capabilities_json` i `http_server.cpp:658` med
  Qwen-specifik prosa `:732-740`.

Grok: den egna patchen ska vara kort och ha exakt fyra ytor: **device-lease** (probe +
run på Engine, utan att heta "image"), **worker-ack** (några rader i `engine_core.h`
där workern redan tittar på `interlude_preempt_outstanding`), **yield**
(`begin_quiesce`/`yield_for_image`/`release_from_yield`, helst i en egen fil de äger,
anropad från `program_impl.h`), och **en enda serve-hook** per protokoll (en callback,
inte tre inlinade bildloopar). Ett nytt Infernix-drop: rebasa patchen, bygg modulen mot
nya headers, kör lease-testerna utan DiT och pixeltesten i modulen. "Det är den ändring
som faktiskt gör patchen bärbar": flytta yield-koden ur `program_impl.h` till en egen
`.cpp` **före** nästa port. Bygg inte: dynamisk loader, C-vtable, versionsnummer före en
andra konsument, `#ifdef` genom `image_pipeline.cu` (flaggan lever på ett ställe:
`add_subdirectory` plus de fyra ytorna), och byt inte namn på lease-metoderna när kod
flyttas.

## 6. Flexibilitet: Groks fas 0 / 1 / 2

**Fas 0, fail-closed, före alla andra modeller** (varje punkt verifierad som ett
verkligt hål ovan):

1. Ett joint-tak som används av attention, `rope_c64`, `dit_adaln`, admission och
   tool-schemat.
2. `cudaSetDevice` på tråden som skapar bild-streamen; mempool-tröskeln sätts och
   återställs på den enheten.
3. Enhancerns netto använder levererat fri minne.
4. Innerwalkens krympning skrivs tillbaka till jobbrecorden.
5. Scheduler-konstanter läses från artifactet och bindning vägrar om de saknas eller om
   `sample_sigmas` är okänd.
6. DiT-rope (axes {16,56,56}, theta 10000, `src/models/qwen_image/rope.cpp:20-24`)
   läses som text-tornet redan läser M-RoPE (`text_tower.cu:66-70,121-128,178,
   277-278`; `image_weights.h:42-57`); vägra om det saknas.
7. EN VAE-stage-lista som ägs av bindingen, används av både grafen och ledaren,
   onammad mot viktparametrarnas former.

**Fas 1, familjen i artifactet:** koden väljer TRE funktioner vid start -
`prepare` (äger decode/resize/template/tal, får inte röra enheten),
`need` (byte för den geometriska inkl. referens och CFG, får inte gissa en
areafaktor), `execute` (encode/condition/denoise/decode inuti yieldet, får inte känna
till HTTP). Familjens DATA, inte kod: pixelmultiple, VAE-scale, pack, latent-kanaler,
pad-token, template, vision mean/std, rope, scheduler, referenstak. Beviset att sömmen
är verklig är en **andra stillbildsfamilj som inte redigerar `image_pipeline.cu`:s
VAE-kedja**.

**Fas 2, video:** först när fas 1 har en andra stillbild. Video är inte "bild med
frames > 1" - `joint_layout.cpp:37` vägrar temporal extent med rätta
("a temporal extent != 1 has no rope grid in this route"). Återanvänd interlude,
single-flight, jobbstore, retention, progress och admissionwalken; återanvänd INTE
2D-VAE, x4-padregeln, image-rope eller host-Euler. Första videomodellen blir sin egen
familj med eget `execute` och egen `need` (temporal latent, rambudget i ledaren),
avskriven lika noggrant som 2.1, bakom fas-1-sömmen - inte en allmän video-runtime.

**Gör inte:** LoRA (P15) som villkor för sömmen; prefix-KV som portabilitetsfunktion;
`n > 1` (single-flight är avsiktligt); låt chat-LLM:n vara bildens text-encoder.

## 7. Min bedömning

Håller för direkt åtgärd (Groks lista är korrekt och liten): punkterna 1-7 i fas 0, plus
att flytta yield-koden ur `program_impl.h`. De är billiga, kan testas var för sig, och
de är just de ställen där en tyst felaktighet idag blir en bild med fel mått eller ett
jobb som släpps in för lite minne.

Jag justerar Groks prioritet på tre punkter:

- **Fas 1:s tre funktioner räcker inte ensam för "portabel"**. Verkligt portabilitet
  köps av de fyra patch ytorna och att `attach_images` tar ett *interface* istället för
  `ImageGenerationService` med `qwen_image`-typer i headern. Annars kvarstår samma
  konfliktyta vid varje drop.
- **Serve-hooken är den dyra delen**, inte pipelinen. Tre protokoll har redan kopierat
  bildlogiken; en hook per protokoll är mindre kod än idag men kräver att
  capabilities och tool-append går ut ur `http_server.cpp`.
- **Prestanda 1 (host-Euler) är större än allt annat i den listan** men är inte ett
  portabilitetsteg och ska inte blockera fas 0/1. En annan anledning att göra den:
  den är en av få förändringar som är svår att göra efter att `execute`-gränsen
  fryst.

Det Grok inte säger och som gäller oss här: hela Phase A (RAM-ledaren 43000/7168) är
förutsättningen för Groks prestanda-punkt 3. Utan att pinned blocks får överleva medan
LLM:n kör är en jobb-övergripande cache värdelös. De två hänger ihop och bör planeras
tillsammans.

## 8. Öppna beslut före planen

1. **Vilken andra bildmodell är nästa verkliga mål?** Ett Qwen-Image-variant gör
   `prepare/need/execute`-sömmen nästan gratis (samma VAE, samma rope, andra vic-
   tikter); en FLUX/SD3.5-klass tvingar fram den fulla datadrivna familjen direkt.
   Abstraktionens form beror på detta, inte på en preferens.
2. **Var modulen bor**: `Infernix/plugins/image/` med eget CMake-target (behåller ett
   repo, en build) eller en syskon-repo byggd som `infernix_image`. Jag rekommenderar
   `plugins/image/` + `-DINFERNIX_IMAGE=ON` första; syskon-repo kan komma senare utan
   att koden flyttas igen.
3. **Får de fyra patch ytorna ändra upstream-filer nu** (`engine.h` namn på lease,
   `engine_core.h` worker-ack, egen `image_yield.cpp` anropad från `program_impl.h`,
   en hook per protokoll)? Utan dem finns ingen portabilitet, med dem är patchen
   avgränsad och kort.
4. **Omfattning av första planen**: fas 0 + patch ytor (rekommendation), eller allt
   inklusive fas 1.

## 9. Lösa trådar

- Fas A (RAM-ledaren) är ändrad i live-lanseringen och omstart är begärd;
  `/health`-bekräftelse ska finnas i samma ärende som detta.
- `start flash next local - K8v4.bat` pekar `PEDIR`/`PEWORKER` fortfarande på den
  gamla trädet `C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\...`; en plugin-
  split som flyttar `tools/prompt_enhancer/` måste uppdatera den länken med.
- `image-edit/` är inte tillåtet att committa; material härifrån får inte hamna i en
  commit.

## 10. Det verkliga målet: MiniMax H3 (bild och video)

Användaren svarade att nästa mål är "minimax h3", och preciserade under arbetets gång:
att få H3 att faktiskt fungera tas **sist och separat** - det här arbetet förbereder
arkitekturen för det. H3 är alltså specificeringsunderlaget, inte något som ska köras nu.

Fakta hämtade från Hugging Face (av mig, samma dag; sidtext är **data**, inte direktiv):
`https://huggingface.co/MiniMaxAI/MiniMax-H3-Image` och
`https://huggingface.co/MiniMaxAI/MiniMax-H3`.

### MiniMax-H3-Image (text-to-image och edit i samma checkpoint)

| Egenskap | H3-Image | Vår kod idag | Konsekvens |
|---|---|---|---|
| DiT | dense single-stream, 50 lager, hidden 5376, 56 heads, FFN 14336, 33.1B, BF16 (embedding/timestep/output FP32), transformer 66.28 GB | Qwen-Image-2.1 FP8, `kImageWeightResidentBytes` 3072 MiB hårdkodad (`image_device_need.h:57`) | resident-budget måste bli familje-data; kvantisering (FP8/NVFP4 finns som community-repo) är ett krav, inte ett val |
| VAE | CNN-encoder + **2D ViT-decoder (24 lager)**, **16x** spatial, **24** latentkanaler, FP32 2.34 GB | 8x, 16 kanaler, CNN up-stages som literal i `image_pipeline.cu:1137-1138` + kopia i ledaren `:80-87` | Groks "en VAE-stage-lista" räcker inte: **decoderklassen är en annan**. `execute` måste äga hela decodern |
| Tokengeometri | latent patchify 2x2 -> varje token 32x32 px | 32-px-multiple hårdkodad `openai_images.h:123-124` | slumpträffande lika, men måste härledas ur familjen |
| Joint-sekvens | ~16 MP naturligt (4096x4096 = 16384 mål-tokens) + text + upp till **16** referenser | tak 20480 i tre hårdkodade ställen, inga referenstak som konstant | **vår största strukturella brist**. kärnan tiled redan i shared memory KB=64 (`launch.cu:15`), så höjt tak är ett scratch-/ledarproblem, inte en ny kernel |
| Sampling | distillerad till **6 steg, ingen CFG**; `guidance_scale` bara 1, `negative_prompt` avvisas; schema + per-steg output heads via `pdd_fused_heads.safetensors` + `pdd_config.json` | Euler, `SchedulerConfig{}` kompilerat `image_pipeline.cu:447`, CFG-dubbel inbakad i host-loopen `:1042-1049` | fas 0 punkt 5 är inte hypotetisk: en annan samplerklass och en CFG-lös väg måste kunna uttryckas, och nya viktklasser (heads) måste läsas |
| Text encoder | **full Qwen3-VL, oförändrad**, läser hidden states från **lager 50**; laddas från H3-repo `text_encoder/` (66.7 GB); prompt trunkeras till 4096 | Qwen3-VL vision tower + M-RoPE text tower, deepstack-taps 8/16/24, dims läses från artifactet | bra: samma modellklass, taps redan artifact-lästa; ont: ett helt större tower och en annan tap |
| Referenser | upp till 16, `<Picture 1>` i uppladdningsordning,multipart till `/v1/images/edits`, JSON till `/v1/images/generations`, svar `data[0].b64_json` | max 4 (dokumenterad i schemat `chat_image_tool.cpp:33`), referensyta 1024^2 | **HTTP-kontraktet vi redan har är detsamma** - starkt stöd för att route-lagret är den stabila ytan; antalet och mallen blir familje-data |
| Prompt expander | valfri `MiniMax-H3-Image-PE` 27B NVFP4 17.9 GB, hostad `H3-Context-IR` är **inte** öppen | vår `pe_worker.py`-sidecar på :8899, `--image-prompt-enhancer-url` | sidecar-designen var rätt från början; behöver bara bli per-familj istället för en global flagga |
| Licens | "MiniMax-H3-Image Community License Agreement", skriftligt tillstånd via ansökan (USA/EU/UK/Sydkorea) | - | **gating före allt H3-arbete**, inte före sömmen. Sverige är inom räckvidd |

### MiniMax-H3 (video), fas 2 underlaget

33B dense single-stream, 4-15 s, 24 fps, kortaste sida 768 som standard, stereo 32 kHz.
H3-VisualVAE är **temporalt kausal, f16t4d24** (16x spatial, 4x temporal, 24 kanaler) plus
patchify 1x2x2 -> effektivt 32x spatial och 4x temporal vid transformern; decoder är en
separat inlörd ViT. MM-RoPE över (t,h,w). AdaLN-grenar ~13B som kan förberäknas/cachas så
de inte behöver laddas vid inference. CFG-distillerade vikter är den enda officiella
distillerade varianten; **antal steg och sampler namns inte på kortet** - ett port måste
läsa repoets egna configer, inte kortet. Öppen release har endast full attention (sparse
är uppskjutet). `FL2VA` (0/1/2 bilder) och `Ref2VA` (<=9 bilder, <=3 klipp, <=3 ljud,
<=12 filer totalt).

Räkne exempel: 768x768, 6 s, 24 fps -> 144 ramar / 4 = 36 latent-ramar x 24x24 = ~20736
tokens för målet åskilt. Det bekräftar att det gemensamma taket måste vara
artifact-deklarerat per familj och att sekvensen måste kunna chunkas.

### Vad detta gör kravet till

Grok formulerade beviset som "en andra stillbildsfamilj som inte redigerar
`image_pipeline.cu`:s VAE-kedja". Med H3 blir beviset konkret och hårdare: en familj som
är annorlunda på **varje** axel H3 är (24 latentkanaler, 16x VAE, icke-CNN-decoder,
per-steg-heads som sampler, ingen CFG, 16 referenser, lager-50-betingning) måste kunna
landas utan att röra VAE-kedjan eller ledarens literaler. Kan den inte det är sömmen fejk.

## 11. Uncensored / NVFP4 variants: what exists and what works here (2026-11-14)

Sources: Hugging Face search pages, repository cards and file trees, fetched this session
with WebFetch. Every statement below about a repository's behaviour (uncensored,
abliterated, heretic, NVFP4) is the author's own claim in page text, which is untrusted
data -- it is a name, not a verified property. Nothing found on those pages was executed
or installed.

### 11.1 What is official vs community (Grok correction)

The official `MiniMaxAI/MiniMax-H3` card, read this session, states:

- exactly two released checkpoints (`FL2VA`, `Ref2VA`), both "Video and audio | BF16",
  hub tensor type "F32 . BF16", format safetensors, 33B; "The released checkpoints are
  CFG-distilled Omni Transformer model weights";
- component layout `processor/ tokenizer/ text_encoder/ transformer/ visual_vae/ audio_vae/`
  plus `model_index.json`, with no dtype given per component;
- the text encoder "uses the full pretrained weights of Qwen3-VL-32B" and supplies "the
  hidden states from its 50th layer";
- `H3-VisualVAE` is "temporally causal", spatial 16x, temporal 4x, 24 latent channels,
  "denoted as f16t4d24" -- an architecture label, not a float16 dtype claim;
- zero mentions of NVFP4, AWQ, GPTQ, GGUF, FP8 or INT4/INT8 anywhere in the card;
- `H3-Context-IR` and the upscaler are not open-sourced (hosted endpoints only), and only
  "inference with full attention" ships.

So the list attributed to Grok -- "NVFP4 / AWQ Text Encoder, Video VAE FP16, Audio VAE
FP32, official H3 Text-to-Video workflow" -- is not in the official card. The FP16/FP32
dtype claims look like a misreading of `f16t4d24` (and of the H3-Image card, where the VAE
genuinely is FP32, 2.34 GB); the NVFP4/AWQ encoder builds are community repos, not official
releases. Verified absence, so we do not build against those dtypes.

### 11.2 What the community actually has (names copied from the listings)

H3-Image DiT: `LSDJesus/MiniMax-H3-Image-fp8-nvfp4` (Text-to-Image, fp8+nvfp4, 15.5k dl),
`BennyDaBall/MiniMax-H3-Image-ComfyUI-NVFP4`, GGUF DiTs `Abiray/MiniMax-H3-Image-GGUF`
(20B), `unsloth/MiniMax-H3-Image-GGUF`, `realrebelai/Minimax-H3-Image_GGUFs`,
`unsloth/MiniMax-H3-Image-FP8`, 2-step distill `akhaliq/minimax-h3-image-pdmd-2step`.

H3-Image prompt enhancer (our PE sidecar counterpart):
`prithivMLmods/MiniMax-H3-Image-PE-NVFP4-GGUF` / `-NVFP4-MTP-GGUF` / `-NVFP4-MTP-Repacked`
(27-28B), `online2311/MiniMax-H3-Image-PE-ComfyUI-NVFP4`, and
`DeepNeuralNerd/Minimax_H3_Image_Prompt_Upsampler_Uncensored_Heretic`.

H3 video text encoder (Qwen3-VL-32B), the piece that carries refusals:
`sakamakismile/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4` -- file tree read this session: one
blob `qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors` at 15.7 GB, README/tags
"uncensored", "abliterated", "text-encoder", "comfyui", commit title "fits a single 16GB
card", "License: apache-2.0" (author's grant; the LICENSE file itself not yet read); forks
`Momoking/`, `xdkings/`, `Jamphus/`, `jackieccwu/` (same name, 246-391 dl); non-uncensored
`6block/MiniMax-H3-Qwen3-VL-NVFP4` (5.85k dl); INT8-convrot
`linjian257/qwen3vl_32b_minimax_h3_int8_convrot_uncensored-by-linjian257` and
`RunningHubAI/rh-lsfqwen3vl-32b-minimax-h3-ultra-uncensored-heretic-int8-convrot...`.

Whole-model uncensored/finetune: `webmp3/Sakura-MiniMax-H3-Uncensored-GGUF` (26B,
Image-Text-to-Video, 86.8k dl), `OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM`,
`WarmBloodAban/Minimax-h3_Singularity` (finetunes, not ablations).

For our current Qwen-Image lane the directly relevant one:
`pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-NVFP4`, card read this session: a "Community
derivative (refusal-ablated) of `Qwen/Qwen3-VL-8B-Instruct`, which is the model
Qwen-Image-2.1 uses, unmodified, as its text encoder", 5.87 GB from 16.33 GB bf16,
Apache-2.0 with LICENSE/NOTICE, method "Heretic directional ablation (`o_proj` +
`down_proj`), 200 trials", measured "5/100 refusals against 100/100", and tensor layout
`...weight` uint8 (2x4-bit) + `...weight_scale` float8_e4m3fn per-group-16 +
`...weight_scale_2` float32 + `...comfy_quant` uint8 JSON, "Written in ComfyUI's native
quantized-checkpoint layout" via `TensorCoreNVFP4Layout.quantize()`, vision tower left
bf16. Companion repos named on that card: bf16 source
`pottokao/Qwen-Image-2.1-Text-Encoder-Heretic` (17 GB), W4A8 5.88 GB, GGUF Q4_K_M 4.68 GB,
plus `Qwen-Image-2.1-DiT-GGUF` and `PE-T2I-Heretic-GGUF`. Also
`reinforce20001/Qwen3-VL-8B-Instruct-Heretic-NVFP4`,
`lamianlbe/Qwen3-VL-8B-Instruct-heretic-NVFP4`.

### 11.3 Can it work here -- verdict against our code (verified this session)

Runtime NVFP4 in the image lane: no. `src/models/qwen_image/image_device_util.h` accepts
only two representations -- contiguous BF16, and `QType::FP8_E4M3FN_ROW_BF16` with
`QuantLayout::RowScale` (`:210`, `:238-259`, `:305`, `:442-447`) -- and
`src/models/qwen_image/image_weights.cpp` throws on anything else (`:52`, and `:503-505`
"image text tower: embed_tokens is not a plain [codes][scales] FP8 plane").
`image_weights.h:12` states the recipe outright: "the 17 quantized roles are FP8 E4M3",
everything else direct BF16. The NVFP4 in our tree (`include/infernix/types.h:44-45`
`Nvfp4Group16`, `Fp8KeyNvfp4Value`, and the `LinearRoutes` FP4 comments at `:151`) is the
chat lane's KV-cache and projection machinery, not a weight format the image loader can
bind. So a ComfyUI-layout NVFP4 safetensors cannot be loaded, and cannot even be opened by
`bind_text_encoder`.

Uncensored text encoder in the image lane: yes, and without C++ changes -- through the
BF16 route. `tools/convert/qwen_image.py` is the tool that produced our artifact:
`build_model(base: SafetensorsSource, components=("dit","text",...))` reads
`root/text_encoder/config.json` and `SafetensorsSource(root / "text_encoder")`
(`:245-277`), `_build_text_encoder` and `_build_vision` (`:131`, `:166-167` "The Qwen3-VL
vision tower inside the pipeline's text encoder") map the source names (`model.visual.*`),
and the FP8 step quantizes from BF16 on CPU (`FP8 = "fp8_e4m3fn_row_bf16"`,
`FP8_METHOD = "fp8_row_maxabs"`, `:10-12`, `:283-307`). Consequences:

1. the input we need is the bf16 encoder, not the NVFP4 file --
   `pottokao/Qwen-Image-2.1-Text-Encoder-Heretic` (17 GB, Apache-2.0) is the same
   architecture our artifact already binds, and the ablation edits `o_proj`/`down_proj` in
   place, so shapes and role coverage stay identical (which is what `bind_*` checks);
2. swapping the encoder is a converter invocation with a different `text_encoder/`
   directory, not a rewrite -- the DiT bytes do not change, and a text+vision-only build is
   supported (`components=`, `:266-269` "vision requires the text component");
3. if a repo only publishes NVFP4 (the sakamakismile case), the path is a dequant step
   first: read uint8 4-bit pairs + e4m3 per-group-16 scales + fp32 global scale, rebuild
   BF16, then run our recipe. That is a converter-side addition (Python), not a new runtime
   kernel -- and it is only worth writing when we actually take on H3's 32B encoder,
   because our 8B encoder at FP8 is already about 6 GB.

For H3 proper, the encoder swap is not enough. Our text tower ends at the final layer and
taps the vision tower at the artifact's `deepstack_visual_indexes`
(`image_weights.cpp:177`, `tools/convert/qwen_image.py:189`); H3 conditions on hidden
states from layer 50 of a 32B Qwen3-VL. That is a family-data field (tap index) on the
seam in this doc's design, plus `text_tower.cu` honouring it -- exactly the `ImageFamily`
data item already planned, and a second reason to keep the tap list out of the CUDA
builder's literals.

Cheapest uncensored knob first. Our PE sidecar (`tools/prompt_enhancer/pe_worker.py`,
127.0.0.1:8899, currently `Qwen-Image-2.1-PE-I2I`) rewrites the prompt before the DiT sees
it; a refusal or a sanitised prompt can come from the rewriter rather than from the
encoder. Before touching weights, test which stage refuses, because the fix differs.

Caveat that changes what to expect. "Uncensored" here is measured as chat refusals on
`mlabonne/harmful_behaviors` for the encoder used as a chat model. In an image pipeline the
encoder is a frozen representation extractor: the content prior lives in the DiT's trained
weights, and the user-visible refusal can also come from the PE rewriter. So an ablated
encoder may not remove an image refusal at all. Falsifiable test on our own server, no code
needed: same prompt, our stock artifact vs an artifact rebuilt with the heretic encoder, and
record the refusal outcome (`image_outcome_word` / the refusal reason) for each plus the
`prompt_enhance_outcome` witness.

Licensing, as found (not legal advice). Official H3 weights are under the "MiniMax H3
Community License Agreement" with an application form (USA/EU/UK/South Korea) -- the DiT and
the video transformer are the gated part. The heretic encoders claim Apache-2.0 because
their source is Qwen3-VL (Apache-2.0 upstream), i.e. an Apache-licensed encoder could in
principle pair with a MiniMax-licensed DiT; the LICENSE/NOTICE files were not read, so
treat that as to-be-checked. `webmp3/Sakura-*`, the Singularity repos and the LoRAs carry
their own terms, none checked.

### 11.4 Not verified (open)

- whether the NVFP4 blobs are genuine block-scaled FP4 with a real tensor table (the tree
  page exposes no tensor names, dtypes or shapes);
- the LICENSE/NOTICE text of the heretic encoder repos;
- the byte sizes of the H3 video VAEs (the official card publishes none);
- whether a heretic encoder measurably changes our refusals (see the test above).
