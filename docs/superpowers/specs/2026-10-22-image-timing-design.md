# Image timing (bundle B) design

**Goal:** sänka mätbart både LLM-pausen (dörren stängd) och total tid per bild, utan att röra numeriken
i de tre första faserna.

**Architecture:** fas 0 lämnar en baslinje per steg; fas 1 flyttar host-arbetet ut ur interludet; fas 2
överlappar blockvikternas host->device med beräkningen; fas 3 håller så många DiT-block boende över
stegen som det yieldade minnet rymmer och räknar om admittarens beräknade need. Varje fas har sin egen
grind mot baslinjen och får inte försämra bundle A:s tre certer.

**Ordningen är en extern recensions ordning (se "Korsvisning"):** host först, sedan överlapp, sedan
resident budget -- med recensionens eget undantag: visar baslinjen att överföringarna klart dominerar
och finns certifierat minne över, körs fas 3 före finjustering av fas 2.

## Globala begränsningar

- Exakta vägar och tal i denna spec är kopierade från verktygsresultat i projektet, inte gissade.
- Kört i Debug av certerna, men prestandatal tas i Release med samma artifact (`out/Qwen-Image-2.1-FP8-v2.infernix`).
- `--disable-fast-paths` (den exakta routen) hålls bit-exakt; varje kännförändring blir en fast path som
  kvalificeras mot gällande oracle enligt `docs/maintainer/model-quality.md`.
- Bundle A:s certer (`infernix_image_need_ledger_real_test`, `..._image_interlude_prod_real_test`,
  `..._image_interlude_failures_real_test`) måste förbli gröna efter varje fas.
- Certerna kräver GPU:n för sig själv: svartfönster med `image-spike/blackout_bundle_a.ps1`.

## 1. Problemet, som koden ser ut idag

- Varje DiT-forward laddar **nia tensorer per block, per block, per steg** genom `upload_streamed`
  ([image_pipeline.cu:513-523](../../../src/models/qwen_image/image_pipeline.cu)), fria i slutet av sitt
  block, så det residenta toppar på ~ett block ([image_pipeline.h:3-15](../../../src/models/qwen_image/image_pipeline.h)).
  Med 32 block och ~40 steg blir det 11 520 strämmade objekt per bild, dubbla vid `true_cfg_scale > 1`
  (två forwards per steg, [image_pipeline.cu:570-573](../../../src/models/qwen_image/image_pipeline.cu)).
- Vikterna binds alla som `Residency::Streamed` ([image_weights.h:7-9](../../../src/models/qwen_image/image_weights.h)):
  varje objekt läses genom begränsad staging och `Uploaded`/`h2d` är **synkron `cudaMemcpy`**, vilket
  legacy default stream ordenar mot -- korrekt idag, men det utesluter överlappning.
- Euler-steget körs på **host**: `velocity D2H`, `latent D2H`, räkna om i en scalarslinga, `h2d` tillbaka
  ([image_pipeline.cu:538-598](../../../src/models/qwen_image/image_pipeline.cu)), med två
  `cudaStreamSynchronize` per steg utöver det.
- I fas 0 låg även text-tower-forward, VAE-decode, PNG-kodning, base64 och diskskriv inuti pausen.
  Fas 1 flyttade ut kodningen, base64:en och skrivningen (§8c); tower och VAE är kvar eftersom de
  behöver det yieldade minnet. Routens värd-sida är nu `encode_png_rgba8`
  ([image_generation_service.cpp:634](../../../src/serve/image_generation_service.cpp)) och
  `save_generated_image` ([:339](../../../src/serve/image_generation_service.cpp)) efter interludet,
  med `base64_encode_bytes` ([:311](../../../src/serve/image_generation_service.cpp)) i samma block.

## 2. Fas 0 -- baslinjen är en leverans

Ett körbart mätprogram (samma svartfönster som certerna, Release-build) skriver en tabell, per upplösning
i ledger-matrisen (512, 768, 768+ref, 768+CFG, 1024) och kall plus varm körning:

- **TVÅ huvudtal alltid:** LLM-paus (ms) och total tid per bild (ms), p50/p95 över upprepade körningar.
- Per block och steg: kopierade byte, H2D-tid och GB/s (API-anropet separat från GPU-tiden), kernel-tid,
  synk- och väntetid.
- Verklig överlapp:andel av H2D som döljs av beräkning, om copy engine eller minnesbandbredden blir
  flaskhals.
- Separat: tower, VAE, D2H, och varje CPU-steg (PNG, base64, disk-I/O).
- Minnesbudget: verkligt toppminne **inklusive mempoolens reserverade**, inte bara vikter och aktiveringar.
- Systemförhållanden: PCIe-generation och länkbredd, uppmätt hållbar överföringshastighet,
  `asyncEngineCount`, GPU-klockor, CPU-/NUMA-affinitet och GPU:s PCIe-topologi.

Baslinjetabellen skrivs in i denna spec (avsnitt 8) och programmet blir `bench/`-kategori med samma
matris som ledger-certen, så varje senare fas kan jämföras mot samma siffror.

**Grind:** fas 1–3 får inte börja utan tabellen, och ingen fas får påstå vinst utan en körd jämförelse.

## 3. Fas 1 -- host-arbetet ut ur pausen

**Körd 2026-10-23. Resultat, de mätbara talen och vad som inte kunde mätas finns i §8c.**

Gränsen läggs vid den sista nödvändiga D2H (den färdiga bilden i host-minne). Därefter frigörs leaset och
LLM:n återupptas medan PNG-kodning, base64 och diskskriv körs klart på host.

- Device-arbete som behöver minne (tower-forward, DiT, VAE) stannar kvar i interludet -- ett yield, en
  lease, samma felbild som idag.
- **Svarsavtalet får inte ändras:** ett misslyckat diskskriv eller kodningsfel måste fortfarande ge samma
  HTTP-fel och får inte annonsera en URL som aldrig skrevs. Därför: bildposten publiceras först efter att
  filen skrivits, och felet mappas på samma ställe som idag.
- Mätbar effekt: pausen minskad med hela host-delen; total tid kan stå sig.

## 4. Fas 2 -- överlappa blockvikternas överföring

- Host-cachen för blocken blir **pinned** (`cudaHostAlloc`), prövad mot `cudaHostAllocWriteCombined` för
  rent H2D-bruk (CPU-läsningar från WC-minne är långsamma, så valet mäts, inte antas). Pinned minne väljs
  selektivt: det är blocken som är varma, inte hela artifactet.
- Överföringen går på en egen **non-blocking copy-stream** med event-styrda beroenden; block `i+1`
  laddas medan block `i` räknar. Två device-buffertar per block (+1 block peak).
- Steget bort från synkron `cudaMemcpy` i `upload_streamed` för de här buffertarna; beräkningsordningen
  verifieras, och den osynkrona legacy default stream-fällan (som idag håller ihop blockframåt med
  `cudaMemcpy` på legacy stream) måste lösas: antingen blir allt arbete explicit stream-styrt, annars
  blir överlappningen noll oavsett buffertar.
- Poolbeteendet: `cudaFreeAsync`-retirering och mempoolens release threshold (bevist i prod-certen:
  "logisk free" återger inte minnet förrän enheten hunnit) mäts i varje fas, så att LLM:n verkligen får
  tillbaka kapaciteten efter jobbet.

## 5. Fas 3 -- resident blockbudget

**Avböjd 2026-10-23, motiverad under Korsvisningen.** Designen nedan står kvar som underlag för den dag
fas 3 återöppnas; den byggs inte i den här bundle:en, för fas 2:s överlapp åt upp den tidvinst som residens
skulle ha kunnat ge, medan dess kostnad betalas i den VRAM som avgör om en storlek överhuvudtaget admitteras.

- Budgeten kommer ur det faktiskt yieldade beloppet: `N = (tillgängliga bytes efter aktiveringar och
  arbetsytor) / image_block_bytes(dims)` ([image_device_need.h:39-42](../../../src/models/qwen_image/image_device_need.h)),
  valda block efter sparad kopieringstid per upptagen byte; resten streamas med prefetch ur fas 2.
- `image_device_need` och `kImageWeightResidentBytes` räknas om för den nya residenta mängden
  ([image_device_need.h:34](../../../src/models/qwen_image/image_device_need.h)) och **ledger-certen körs om på
  samma matrix**; garartin är att varje storlek som rymms idag fortfarande rymms, och 503 får bara flytta
  sig dit toppen verkliga stiger.
- Budgeten får aldrig ta minne som admission redan lovat en annan storlek: den beräknas före yield, inte
  under, annars kan ett jobb som admitteras inte genomföras.

## 6. Euler-steget på enheten (fas 3b, samma numerik)

Flytta `new_lat = lat + (sigma_next - sigma) * noise` till en kernel och låt latenten bo på enheten mellan
stegen, så de tre latent-överföringarna och de två extra synkroniseringarna per steg försvinner. Formeln och
avrundningen till BF16 är oförändrad ([image_pipeline.cu:586-598](../../../src/models/qwen_image/image_pipeline.cu));
är resultatet bit-exakt mot idag behövs ingen ny kvalifikation, annars körs den som fast path mot orakeln.

**Beslut 2026-10-23: fas 3b lämnas ogjord, villkoret är inte uppfyllt.** Ordningen i §9 och i
korsvisningen är att den får köras först om en timeline visar att de två synkroniseringarna per steg
faktiskt lägger igen. Ingen sådan timeline finns: bänken rapporterar loopens vägg, block-uppladdningar och
block0-kopiering, men ingen per-steg post för D2H/synk/väntan, och det enda måttet som finns pekar åt
andre hållet — efter fas 2 är en blocköverföring ~4 ms varm och loopen ≈ räknetid (§8b), så loopen är
kärnbindingsbunden och inte synkbindingsbunden. Att då flytta Euler-steget till enheten vore en
kännförändring som körs som fast path utan ett mått som säger att den vinner något, vilket §7 kräver
bevis för. Fas 3b återöppnas av en per-steg timeline (event-timers runt `velocity D2H`/`latent D2H`/
`h2d` och de två `cudaStreamSynchronize`, eller nsys på en 512-rad) som visar väntetid utöver kernelernas
egen tid; kostnaden för den mätningen är en timmes bänkarbete, inte en pipeline-ändring.

## 7. Numerik och routes

Fas 0–2 och 5 ändrar ingen numerik: samma kernel, samma reordning av arbete. Fas 3b och eventuell
sammanfogen block-kernel är kännförändringar: de körs som fast paths, den exakta routen hålls bit-exakt,
kvalificering mot naiv FP32/FP64-orakel på riktiga shapes plus prod-certens latent-/image-trösklar.

## 8. Baslinjen, körd 2026-10-22 (fas 0)

`bench/image/image_stage_bench.cpp` i Release, svartfönster 18:52 och 19:15, Engine med
`max_context=65536`, `expert_ram=8 GiB`, artifact `out/Qwen-Image-2.1-FP8-v2.infernix`. Logg:
`image-spike/stage_bench.out.log` och `...stage_bench.err.log`. ms, en körning per rad.

| rad | total | paus | device-job | loop | copy-span | uploaders | bytes |
|---|---|---|---|---|---|---|---|
| 512×512 kall | 29 899 | 29 578 | 22 708 | 19 536 | 19 100 | 1 280 | 266 350 MiB |
| 512×512 varm | 25 655 | 24 997 | 22 165 | 19 142 | 18 710 | 1 280 | 266 350 MiB |
| 768×768 kall | 35 932 | 35 931 | 33 010 | 28 934 | 28 197 | 1 280 | 266 350 MiB |
| 768×768 varm | 35 805 | 35 785 | 32 925 | 28 859 | 28 127 | 1 280 | 266 350 MiB |
| 768 + referens | 104 039 | 103 979 | 101 128 | 96 359 | 93 510 | 1 280 | 266 350 MiB |
| 768 + CFG | 72 736 | 72 428 | 62 713 | 58 579 | 57 078 | 2 560 | 532 701 MiB |
| 1024×1024 kall | 61 650 | 61 600 | 58 665 | 53 005 | 51 478 | 1 280 | 266 350 MiB |
| 1024×1024 varm | 60 905 | 60 870 | 57 934 | 52 190 | 50 697 | 1 280 | 266 350 MiB |

**Vad tabellen säger.** Varje bild flyttar 260 GiB blockvikter (520 GiB med CFG) över PCIe: 32 block à
uppmätt 208 MiB, om och om i 40 steg. Pausen är 99 % av totalen på varje rad, så fas 1 (host ut ur pausen)
är värd 0,3–3,2 s och inte mer.

**Vad den INTE säger, och varför fas 2 ändå behövs först.** `copy-span` mäts med värdklocka runt kopieringen,
men kopieringen är en synkron `cudaMemcpy` på legacy default stream — den väntar på att enheten är tom innan
den startar. Spanset är därför en övre gräns för kopieringen och ett mått på att loopens arbeta är
fullständigt serialiserat, inte ett bevis för att PCIe tar 97 % av tiden. Verklig kopiering av 208 MiB vid
mätt~14 GiB/s är ~15 ms per block, ~13 s av loopen på 512 men bara ~13 % av referensradens 96 s: på långa
sekvenser är det kärnorna som är långa, och väntetiden ser ut som kopiering. Fas 2 (egen copy-stream,
event) är det som skiljer åt; först efter det kan man säga vad residenta block verkligen tjänar.

**Beslutet fas 0 gav.** Det yieldade beloppet var 9 344–10 240 MiB och alla 32 block är 6,5 GiB: hela
blockuppsättningen kan bo kvar i det minnet som redan yieldas, på varje rad i matrisen. Fas 3 körs därför
-- inte som ett undantag från ordningen utan eftersom baslinjen visade att överföringarna är hela
trafiken -- med fas 2 som den mätbara grinden före och efter, och prefetch som kvarlevande väg för de
storlekar där aktiveringar äter budgeten.

## 8b. Fas 2, körd 2026-10-22 (blockvikterna överlappar beräkningen)

Implementation: blockens host-cache är pinned (`cudaHostAlloc`), de nio viktarna per block landar på en
egen non-blocking copy-stream med event-styrda beroenden och två device-slots (block i+1 laddas medan
block i räknar), och blockframåt körs nu på jobbets explicita image-stream i stället för legacy default
stream (den tidigare legacy-`cudaMemcpy`-fällan är löst). Kernels och numerik är orörda.

Samma bench, samma Engine och artifact, Release, svart fönster, en körning per rad. `loop` = loopen i
run_image_job, `copy` = en blocks H2D mätt med device-event (block 0, kall respektive varm). ms:

| rad | total | paus | device-job | loop | föreg. total | föreg. device | föreg. loop | block0-copy (varm) |
|---|---|---|---|---|---|---|---|---|
| 512×512 kall | 18 024 | 17 711 | 10 297 | 6 196 | 29 899 | 22 708 | 19 536 | 24→4 |
| 512×512 varm | 13 234 | 13 234 | 10 321 | 6 337 | 25 655 | 22 165 | 19 142 | 4 |
| 768×768 varm | 23 717 | 23 717 | 20 776 | 15 790 | 35 805 | 32 925 | 28 859 | 4 |
| 768 + referens | 92 413 | 92 366 | 89 538 | 83 757 | 104 039 | 101 128 | 96 359 | 4 |
| 768 + CFG | 42 453 | 42 376 | 36 638 | 31 624 | 72 736 | 62 713 | 58 579 | 4 |
| 1024×1024 varm | 49 437 | 49 401 | 46 472 | 39 865 | 60 905 | 57 934 | 52 190 | 4 |

**Överlappen inträffar, ärligt mätt.** `asyncEngineCount = 1` (en copy-engine) och `loop` föll till
räknetid: på 512 var loopen i baslinjen ~19 s som var kopieringsbunden (synkron `cudaMemcpy` på legacy
stream tvingade fram serialisering, ~15 ms/block); nu är en blocköverföring ~4 ms varm och ligger
parallellt med SM:et, så loopen är ~6 s och ~kärnbindingsbunden. Andelen H2D som döljs av beräkningen är
alltså i praktiken ~hela överföringen (loop ≈ räknetid; ~13 s av den tidigare loopen på 512 försvann).
På de kärntunga raderna (768 + referens, 83 s loop) var kopieringen redan en liten del, så vinsten där är
~13 %; vinsten är störst just där kopieringen dominerade (512, CFG: −67 % respektive −46 % loop).

**Negativa fynd, dokumenterade så ingen provar om.**
- *Write-combined är sämre.* `INFERNIX_IMAGE_HOST_CACHE_WC=1` mättes mot vanlig pinned cache på samma
  rader: 512 device 11 093 vs 10 082, 768 20 466 vs 20 135 (WC långsammare överallt, CPU-läsningarna
  från WC-minnet kostar). Valet mäldes alltså, inte antogs: **cache behålls, WC är av (env-flaggan finns
  kvar men är inte standard).**
- *Prefetch-djup ger ingen extra effekt.* `INFERNIX_IMAGE_OVERLAP=0` (ingen look-ahead, samma pinned
  cache och copy-stream) gav samma loop som överlappet (512 6 125 vs 6 103; 768 15 316 vs 15 227).
  Vinsten kommer från pinned + async copy-stream + explicit stream-styrt blockarbete, inte från
  look-ahead-djupet: även utan look-ahead överlappar intilliggande block var för sig över de två
  streamarna.

**Admission och need.** Två block är residenta under loopen, så `image_device_need` behöver
`blocks_resident = 2`. För att hålla den tjänade mängden oförändrad mot §8 sänktes
`kImageWeightResidentBytes` med exakt en blockstorlek (6183 -> 5972 MiB), vilket gör predicted-totalen
netto oförändrad: 1024 och 768 + CFG går alltså in som i baslinjen (de 503ade en kort stund i en
mellanrunning medan allowance ännu var 6183 med två residenta block).

## 8c. Fas 1, körd 2026-10-23 (host-arbetet utanför pausen)

Implementation (`1706a48`): device-jobbet returnerar `FinishedPicture` (RGBA8 + bredd/höjd) i stället
för färdig PNG, interludet tar slut där, och `encode_png_rgba8` → `save_generated_image` → bas64/URL
körs efter att `runner_->run` returnerat. Jobsvar och felbild är orörd: posten blir Succeeded först
när filen finns, och ett kodnings- eller skrivfel ger fortfarande 500 `image_generation_failed` genom
samma `job_started`-regel. Sista progress-enheten tas när bilden ligger på disk, inte när kodaren
returnerade.

Samma bench, samma Engine (`max_context=65536`, `expert_ram 8 GiB`) och artifact, Release, svart fönster
01:20–01:25, en körning per rad (`image-spike/stage_bench_fas1_full.out.log`); föregående kolumn är
fas 2:s körrad 23:17 (`stage_bench_A_full_overlap.out.log`). `utan paus` = `total - paus`, alltså det
som den här fasen flyttade ut. ms:

| rad | total | paus | device | utan paus | föreg. total | föreg. paus |
|---|---|---|---|---|---|---|
| 512×512 kall | 17 539 | 17 225 | 10 218 | 314 | 18 024 | 17 711 |
| 512×512 varm | 12 788 | 12 778 | 9 983 | 10 | 13 234 | 13 234 |
| 768×768 kall | 23 257 | 23 236 | 20 374 | 21 | 23 844 | 23 735 |
| 768×768 varm | 23 035 | 23 014 | 20 150 | 21 | 23 717 | 23 717 |
| 768 + referens | 89 621 | 89 561 | 86 712 | 60 | 92 413 | 92 366 |
| 768 + CFG | 41 088 | 41 067 | 35 525 | 21 | 42 453 | 42 376 |
| 1024×1024 | nekades | — | — | — | 49 437 | 49 401 |

**Vad som är mätt, och vad som inte är det.** Det strukturella resultatet syns direkt: pausen slutar
före kodningen, så 10–60 ms värdarbete (PNG-kodning + filskriv, utan base64 eftersom bänken körs med
`response_format=url`) ligger nu utanför fönstret LLM:n är pausad. Pausen blev alltså kortare med just
det beloppet, inte mer. §1:s uppskattning "fas 1 är värd 0,3–3,2 s" var `total - paus` i §8 — det vill
sig prompt-rendering och probe, inte bildkodningen — och den siffran håller inte som vinst.
Row-till-row-rörelsen mellan de två körrna är 0,3–2,8 s på HELA raden (total, paus och device rör sig
tillsammans med ~2–3 %), alltså 5–100× större än effekten som ska mätas: den kan inte tillskrivas den
här ändringen och ingen radvis "pausen föll med X ms"-poäng hävdas. Den 314 ms långa svansen på 512 kall
fanns i föregående kör också (18 024 − 17 711 = 313) och är förstone-kostnad före paus, inte kodning.

**1024-raden kunde inte mätas ikväll och varför det inte är fas 1.** Båda körningarna svarade
`the image job needs 7815 MiB; the model can yield 8384 MiB with 15 MiB already free and 256 MiB reserved
below the headroom` — admittenten nekar innan något yieldas, LLM:n är frisk, exakt som regeln ska.
Probet körs före interludet och kan inte påverkas av kod som springer efter det, och talen replikerades
identiskt i en andra körning med bara kallraderna (01:31), så nekas-tecken är inte den här ändringen.
Beloppet är ett gränsfall, inte en omöjlig storlek: 8 384 MiB plan + 15 MiB fritt = 8 399 MiB mot
7 815 + 512 marginal + 256 headroom = 8 583 MiB, alltså 184 MiB för kort. Samma storlek kördes end-to-end
i ledger-certen tjugo minuter tidigare (`one megapixel: 1024x1024 predicted 7812 MiB, measured peak
4007 MiB`, `TESTEXIT=0`), så det är bänk-processens gevning ikväll och inte produkten som satte gränsen.
Varför planen erbjöd 8 384 MiB mot ≥11,5 GiB i 23:17-körningen är **inte utrett**; maskinen hade 5,4 GiB
fritt av 63,7 GiB fysiskt minne under körningen och bänkens Engine är konfigurerad med `expert_ram 8 GiB`,
men det sambandet är en misstanke, inte ett mått. 1024-radens fas 1-effekt står alltså som omätt, och
dess `device`/`paus`-värden i tabellen är tomma i stället för uppskattade.

**Numerik och certer.** Certerna körda före leveransen (`RESULT ledger=0 prod=0 failures=0 healthy=True`,
svart fönster 01:11–01:17) och den certifierade routens numerik är identisk med före:
schedule `timesteps=0.000e+00 sigmas=0.000e+00`, `FINAL_LATENT rel_fro=4.401e-02`,
`FINAL_IMAGE rel_fro=5.097e-02`. Prod-certens ledger-peak läser 1 851 MiB mot 1 435 MiB i den äldre
loggen från 16:21; den skillnaden är fas 2:s copy-stream-staging (`ed7ae5d`), som ligger mellan loggarna.

## 9. Verifiering

1. Fas 0: baslinjetabellen körd och inklistrad här, två tal per upplösning.
2. Efter varje fas: samma program, samma matris; paus och total redovisade för och efter, inklusive
   sämsta fall (ingen fas får rapportera bara medelvärde).
3. A:s tre certer gröna i svartfönster efter varje fas (`RESULT ledger=0 prod=0 failures=0 healthy=True`).
4. Vid kännförändring: orakelkvalifikation enligt `docs/maintainer/op-development.md`.
5. Minneskravet: ledger-certen visar att `image_device_need().total()` aldrig hamnar under uppmätt peak.

**Fas 2 körd (2026-10-22).** Bundle A:s tre certer gröna i svart fönster: `RESULT ledger=0 prod=0
failures=0 healthy=True`. Prod-numeriken bit-identisk med baslinjen (`FINAL_LATENT 4.401e-02` tol 5e-1,
`FINAL_IMAGE 5.097e-02` tol 3e-1) -- fas 2 rörde ingen numerik. Ledger-certen per rad (predicted /
uppmätt peak / marginal, MiB), med `blocks_resident=2` och allowance 5972:

| rad | predicted | uppmätt peak | marginal | förra (allow 6183, 1 block) |
|---|---|---|---|---|
| 512 | 7 470 | 1 845 | 5 625 | 7 473 / 1 429 / 6 044 |
| 768 | 8 821 | 2 746 | 6 075 | 8 824 / 2 330 / 6 494 |
| 768 + referens | 8 835 | 2 760 | 6 075 | 8 838 / 2 343 / 6 495 |
| 768 + CFG | 8 825 | 2 751 | 6 074 | 8 828 / 2 334 / 6 494 |
| 1024 | 10 712 | 4 007 | 6 705 | 10 715 / 3 591 / 7 124 |

predicted-totalen är netto oförändrad (allowance sänkt med en blockstorlek väger upp blocks 1->2), så
antagna storlekar är desamma och marginalen är kvar 5,6--6,7 GiB på varje rad.

**Åtstramning körd (2026-10-22).** Need överskattade med ~4x. `kImageWeightResidentBytes` sänktes
5972 -> 3072 MiB (endast en konstant; ingen kernel, pipeline eller admittentlogik rörd, så
räkningstiden och antagningsregeln är oförändrade). Kalibreringen tar inte cert-certens uppmätta peak
som golv -- den matar nollifyllda trivialindata vars peak är lägre än en levande jobs -- utan den
VERKLIGA per-steg-peak ur stage-bänken (`stage_bench_A_full_overlap.err.log`), som bär riktig
prompt/negative/referens. Avdraget mellan verklig peak och inventariens uppräknade termer binder på
ref/CFG-raderna (~1 180 MiB, där certens triviala sekvens underskattar mest); 3072 MiB lämnar ~1,9 GiB
ovan det för 512 MiB golvmarginalen plus vad KV-poolen och LLM-restoren håller under en interlude.

Certernas tre grön: `RESULT ledger=0 prod=0 failures=0 healthy=True`, prod-numerik bit-identisk.
Per rad (MiB), predicted före/efter, certens uppmätta peak, verklig bench-peak, och den bindande
marginalen predicted minus VERKLIG peak:

| rad | predicted förr (5972) | predicted nu (3072) | cert-peak | verklig peak | marginal mot verklig | x verklig |
|---|---|---|---|---|---|---|
| 512 | 7 470 | 4 570 | 1 845 | 1 848 | 2 722 | 2,47 |
| 768 | 8 821 | 5 921 | 2 746 | 2 749 | 3 172 | 2,15 |
| 768 + referens | 8 835 | 5 935 | 2 760 | 3 615 | 2 320 | 1,64 |
| 768 + CFG | 8 825 | 5 925 | 2 751 | 3 615 | 2 310 | 1,64 |
| 1024 | 10 712 | 7 812 | 4 007 | 4 010 | 3 802 | 1,95 |

Garantierna: (1) predicted >= uppmätt peak på varje rad, både mot certens och mot den verkliga peaken,
med marginal > 512 MiB golv på varje rad (minst 2 310 på CFG) -- för storlekar utanför matrisen är
modellen parametrisk i `s_max`/VAE-sidan så predicted växer UPP strukturellt, allowance täcker resten.
(2) KV-tillväxt under interlude täckt: failures-certen (S6) körde ett scenario där KV-poolen växte till
576 pages (522 MiB) medan bilden begärde 24 576 MiB -- den mjuka KV-remapen föll tillbaka till 512
sidor istället för att OOM:a, och bilda jobbet blev klart (`failures=0`). (3) För-stort jobb nekas
fortfarande (admittentkoden orörd, `prod=0`), och ett jobb vid den nya gränsen BLIR KLART: 1024-raden
kördes end-to-end i ledger-certen till `live==0`, och vid antagningsgränsen gäller per konstruktion
`verklig peak <= predicted <= tillgängligt`, så ett admitterat jobb får plats -- marginalen 2,3--3,8 GiB
är själva garantin mot runtime-OOM.

Det finns alltså flera GiB tjänbar kapacitet som nu admitteras i stället för att 503:as i onödan.
Räkningstiden påverkades inte (constanten läses bara av serve-admittenten, inte av bänken), så
bench-matrisan behöver inte köras om för tid; dess fem storlekar kördes end-to-end via certerna.

**Garanti 0 verifierad (2026-10-23) -- predicted och measured mäter samma mängd.** Genomgång av de fyra
räkenskapsfelen: `Buf` är modulens ENA device-allocator ([image_device_util.h:89-115](../../../src/models/qwen_image/image_device_util.h))
och varje `Buf::alloc` adderar till `g_image_device_peak`, som är **max samtidiga** (inte en summa) --
fel (2) och (4) är alltså på rätt sida: `image_device_need` summerar kategorier som en övre gräns, ledgern
mäter max-samtidig, så `predicted >= measured` håller åt rätt håll. Varje bildvikt -- DiT-block, DiT-
globala (`img_in`/`proj_out`/...), text-tower-lager och VAE -- är en `artifact::ParameterReference` som
streamas på begäran ([image_weights.h:53-88](../../../src/models/qwen_image/image_weights.h)); inga
persistenta device-vikter finns utanför `Buf`, vilket `live==0` mellan jobb bevisar -- fel (1) gäller
inte. Fel (3) (allocated vs reserved) täcks av att marginalen ligger ovanför den sanna max-live-pekens
pool-reserverade (bänk: pool reserved ~1 664--2 048 vid 512 mot ledger live 1 349). **Faktor ~4 har en
räkenskapsförklaring:** allowance var gamla serialiserade routens hela-high-water-proxy (6 183), men fas
2:s överlapp + streaming sänkte den sanna max-live till ~1 845 -- så att sänka till verklig-max-live +
golv är FIXEN, inte en multiplier-fudge. Ledger-talet bekräftat täcka streamed-block-slots (`Buf slot[2]`
= 448 MiB pre-loop live, 864 under DiT-loopen), tower (text_tower allocerar ENbart via `Buf`), VAE-planen
(peak 1 845 nås vid VAE-decode) och latenten. Därför står `8dc9c63`: det är ingen OOM-generator.

**Känt, ej åtgärdat: 1024-radens negativa residual.** Där överstiger inventariets strukturella termer
(`dit_activations + latent_and_velocity + vae_peak + rope_and_tables` ≈ 4 318 MiB) redan radens mätta
peak (4 010 MiB) -- residualen mot allowance är negativ. Det är det tydligaste beviset för felkälla (b):
inventariet summerar buffertar som inte är samtidiga i stället för maximum samtidiga, och just där sitter
de sista ~2x konservatismen kvar. Det är ofarligt åt rätt håll (predicted förbliver en övre gräns) och att
rätta det vore en omritning av inventariet, inte en konstant -- det ligger utanför denna bundle.
Ledger-certen vaktar nu riktningen: varje rad måste klara både `predicted >= measured` OCH ett 512 MiB
reservationsgolv, så nästa sänkning av konstanten för långt är ett certfel.

**`need` bundet till `blocks_resident` (2026-10-23).** En enda symbol `kImageResidentBlocks = 2`
([image_device_need.h](../../../src/models/qwen_image/image_device_need.h)) binder now resident-block-
talet: serve-admittenten och ledger-certen sätter `blocks_resident = kImageResidentBlocks`, och
`CopyLane` har `static_assert(kImageResidentBlocks == 2)` (prefetchar betingat av exakt två slotar), så
ett tyst glid mellan den modellerade reserven och de verkliga slotarna är ett byggfel. Allowance 3072
kalibrerades vid detta tal. Admittentens **reservation** (`H = total − committed − future KV reserve −
safety margin`) är orörd: KV-explicit-kapaciteten och `kv_hold_pages` håller redan KV-tillväxten utanför
reserven, så marginalen behöver inte åter täcka den -- och som recensionen säger är det reserven, inte
talet, som utesluter runtime-OOM. Skuggläge (loggat need-förslag mot mätt peak per rad) är frivilligt och
ej byggt; säg till om det ska bli ett eget steg.

## Korsvisning

En extern recension (ChatGPT, tråd `chatgpt.com/c/6ada33c3-7b5c-83ed-9312-418f740d8b61`, 2026-10-22)
rekommenderade samma ordning (3 -> 1 -> 2 i dess numrering) och tre fallgropar som redan syns i vår kod:
mempoolens release threshold (prod-certen mätte det), att legacy default stream osynkront synkroniserar
bort all överlappning, och att asynkron H2D kräver pinned värdminne. Recensionen avråder också från
green context/MPS här: LLM-arbetet är redan pausat, så ingen av mekanismerna ger mer VRAM eller snabbare
PCIe.

### Andra korsvisningen, 2026-10-23 (efter fas 2:s mätresultat)

En ny recension (ChatGPT, tråd `chatgpt.com/c/6ada884d-b468-83ed-b70d-47f60e9f130d`) fick fas 2:s siffror och
de tre kvarvarande spåren, och rekommenderade: **rätta admittentens prediktion först, håll full residens
utanför kritiska vägen, undersök fas 3b mot en timeline.** Det är den ordning vi följer. Motiveringarna vi köper:

- Fas 2 flyttade butelneckan från överföring till räkning, så residens köper inte längre vägen. Att 260 GB
  per bild slutar överföras är inget mål i sig; vinsten måste mätas som wall time, svans eller
  kernel-duration, inte som byte-räknaren.
- Kostnaden är just den resurs som avgör om en bild betjänas alls: +211 MiB för att gå från 1 till 2
  residenta block, extrapolerat mot 31 × 211 MiB ≈ 6,4 GiB för full residens, mot KV-poolens utrymme.
  Recensionens princip: behandla residens som ett köp av latens med VRAM-huvudutrymme.
- Återöppningsvillkoret är en timeline som visar kvarstående kopieringsberoenden, kopieringsrelaterad
  kernel-slowdown eller mätbar svansvinst. Copy-engine-utilisering ensam är inget bevis — motorn kan vara
  upptagen hela loopen medan arbetet ändå är fullständigt dolt bakom beräkningen.
- Fas 3b är värt att pröva av en annan anledning än vikter: det tar bort *synkronisering* per steg, och
  synkroniseringar kan exponera gap som annars är dolda, även när kärnorna är räknetidsbundna. Villkoret
  är att en timeline först visar att de två synkarna per steg faktiskt lägger igen, och att numeriken hålls
  bit-identisk (eller inom avtalad tolerance) inklusive CFG- och referens-rad.

För `image_device_need` gav recensionen ett krav som väger tyngre än själva kalibreringen, och det gäller
innan konstanten rörs (**garanti 0**): predicted och measured måste mäta *samma* storhet. De fyra
räkenskapsfelen att utesluta är (1) persistenta vikter som redan finns i serverns baseline, (2) summation av
aktiveringar/workspace/latent som är samtidiga i stället för maximalt samtidiga, (3) *allocated* blandat med
allocator-*reserved*, (4) worst-case-summa av temp-buffertar där bara en delmängd är live. Ett fel på
faktor fyra har oftast en räkenskapsförklaring, och den ska fixas — inte kalibreras bort med en multiplier.
Ledger-talet måste vara bekräftat att täcka streamed-block-uppladdning, tower/VAE/text-encoder och latent;
gör det inte är "1845 mot 7470" äpplen och pomer, och att sänka `need` mot 1845 vore en OOM-generator.

Slutligen: admittenten ska vara en **reservation**, inte en gissning —
`H_admit = total − committed − future KV reserve − safety margin`, där KV-tillväxt under interluden inte får
äta reserven (KV-explicit-kapaciteten och `kv_hold_pages` är halva den mekanismen redan). Validering nära
full KV-pool, inte bara med LLM i vila; `need` bundet till den `blocks_resident`-konfiguration som
validerats så att talet inte kan glida tyst; utrullning i skuggläge (loggat förslag mot mätt peak per rad)
först, konservativ bana kvar för konfigurationer som inte validerats. Recensionen gav också den ärliga
gränsen: en modell kalibrerad mot max observerad peak plus marginal utesluter inte matematiskt en
runtime-OOM så länge KV-tillväxten kan vara obegränsad — garantin kommer från reserven, inte från talet.

### Tredje korsvisningen, 2026-10-23 (page-lock för blockcachen under expert-lagret)

En ny recension (ChatGPT, tråd `chatgpt.com/c/6adb0791-2ef8-83eb-a141-6221b77f3aa9`) fick mätningarna från
reproduktionen mot produktionsprofilen (`--expert-ram-mib 41500`): expert-lagret page-lockar 40,3 GiB
värdminne, `cudaHostAlloc` vägrar samtliga 205 cache-objekt, och DiT-loopen går från 6455 ms (pinned) till
19615 ms (pageable) medan bilden blir färdig med HTTP 200.

Svaret är **C + B + A i den preferensordningen**, och att D är fel här:

- **A, pageable fallback, är en tillgänglighetsgaranti och inte ett tillstånd att stanna i.** En hanterad
  vägrad pinning får aldrig bli en 500:an, eftersom beräkningen är numeriskt ekvivalent utan lock. Det är
  vad som landats i `PinnedBytes`, tillsammans med det klistriga CUDA-felet som annars dödar processen.
- **B, pinna så mycket budgeten faktiskt tillåter**, är allokationspolicyn men degnerar till A så länge
  expert-lagret håller nästan hela budgeten; och 205 separata allokationer bör bli ett fåtal slabbar med en
  gemensamt förvaltad budget i stället för ett beslut per objekt.
- **C, att släppa den pausade LLM:ens page-lock under interludet**, är den arkitektoniskt rätta åtgärden,
  eftersom interludet är just det fönster då expert-lagret varken används av enheten eller av VRAM. Kravet
  recensionen gav är implementationstekniskt: ett block som är låst via `cudaHostAlloc` kan bara släppas med
  `cudaFreeHost`, så expert-arenan måste hållas som vanlig värdallokering och registreras av/med
  `cudaHostRegister`/`cudaHostUnregister` runt interludet som en explicit tillståndsövergång (kviescera
  överföringar, avregistrera, bilda bildcachen, kör jobbet, släpp bildbudgeten, återregistrera, återuppta).
- **Varje återvinning måste försvaras med måttet, inte med mekanismen.** Skillnaden som står att tjäna är
  13 160 ms per loop och är därmed taket för vad av- och återregistrering får kosta; avgörandet görs på
  interludets totaltid och tid-till-första-token efter det, p50/p95/p99 över upprepade interluden, och på att
  återställningen aldrig kan lämna processen i ett trasigt tillstånd.
- **D, ren 500 vid vägrad pinning, är fel** eftersom användarens egen profil trots allt kan generera en
  korrekt bild; **E, sänkt expert-RAM**, löser inte att page-lock är en delad resurs utan budget.

Regeln recensionen påminde om som gäller den landade fallbacken: en `cudaMemcpyAsync` med pageable källa har
annan asynkron-semantik och annan livstid för källbufferten än med pinned minne. I den landade varianten är
det ofarligt eftersom cache-objekten fylls en gång per jobb och aldrig skrivs om innan deras händelse är
väntad; ett framtida `cudaHostRegister`-register måste behålla den egenskapen.

## Explicitly not in scope

- Mer än ett jobb åt gången, och att två bildjobb delar på minnet.
- Green context eller MPS (motiverat ovan).
- Full residens av blockvikterna, och att göra blockvikterna boende bortom certifierat minne (fas 3 avböjd
  ovan). Inte heller att *rycka i marginalen* för att få plats: åtkomligheten den här bundle:en vinner kommer
  ur att `image_device_need` slutar mäta fel, inte ur att 503-tröskeln sänks.
- Chatt-argumentet `image`, GC av genererade filer och variabel-dokumentationen: bundle C.
