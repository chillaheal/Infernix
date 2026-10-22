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
- Inuti pausen ligger också text-tower-forward, VAE-decode, PNG-kodning
  ([image_generation_service.cpp:215](../../../src/serve/image_generation_service.cpp)), base64
  ([:308](../../../src/serve/image_generation_service.cpp)) och diskskriv
  ([ :315-331](../../../src/serve/image_generation_service.cpp)).

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

**Öppet sparande -- åtstrama `image_device_need` mot ledgern.** Tabellen ovan visar att
`image_device_need().total()` är ~4x uppmätt peak på 512 (7 470 mot 1 845) och ~2,7x på 1024
(10 712 mot 4 007). Så länge need är så konservativ 503:ar admittenten tidigare än vad enheten själva
behöver -- det finns flera GiB tjänbar kapacitet som never används. Detta är en egen förbättring,
fristående från fas 2: sänk `kImageWeightResidentBytes` (den är död marginal, se kommentaren i
image_device_need.h) och/eller gör inventariens aktiveringstermer täta, och låt ledger-certen hålla
predicted >= uppmätt peak med en avsiktlig (inte 4x) marginal. Behöver inte göras i denna fas; den ska
göras med egen mätning och egen acceptans, inte som en sidoeffekt.

## Korsvisning

En extern recension (ChatGPT, tråd `chatgpt.com/c/6ada33c3-7b5c-83ed-9312-418f740d8b61`, 2026-10-22)
rekommenderade samma ordning (3 -> 1 -> 2 i dess numrering) och tre fallgropar som redan syns i vår kod:
mempoolens release threshold (prod-certen mätte det), att legacy default stream osynkront synkroniserar
bort all överlappning, och att asynkron H2D kräver pinned värdminne. Recensionen avråder också från
green context/MPS här: LLM-arbetet är redan pausat, så ingen av mekanismerna ger mer VRAM eller snabbare
PCIe.

## Explicitly not in scope

- Mer än ett jobb åt gången, och att två bildjobb delar på minnet.
- Green context eller MPS (motiverat ovan).
- Att göra blockvikterna boende bortom certifierat minne, eller att sänka 503-tröskeln för att få plats.
- Chatt-argumentet `image`, GC av genererade filer och variabel-dokumentationen: bundle C.
