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

## 8. Verifiering

1. Fas 0: baslinjetabellen körd och inklistrad här, två tal per upplösning.
2. Efter varje fas: samma program, samma matris; paus och total redovisade för och efter, inklusive
   sämsta fall (ingen fas får rapportera bara medelvärde).
3. A:s tre certer gröna i svartfönster efter varje fas (`RESULT ledger=0 prod=0 failures=0 healthy=True`).
4. Vid kännförändring: orakelkvalifikation enligt `docs/maintainer/op-development.md`.
5. Minneskravet: ledger-certen visar att `image_device_need().total()` aldrig hamnar under uppmätt peak.

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
