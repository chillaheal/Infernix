# Chatt-argumentet image, retention för genererade filer och variabel-dokumentationen (bundle C)

**Goal:** låta chattverktyget ta en bild som utgångspunkt, göra genererade filers livstid till en
dokumenterad policy (14 dagar som default), och samla image-pipelinnens variabler på ett ställe under
`docs/maintainer/`.

**Architecture:** tre fristående delar som bara delar de filer de rör. (1) Chatt-argumentet `image` blir
ett resolution-steg i chatt-lagret: det modellen skriver översätts till det format routen redan
accepterar (`OpenAIImagesRequest::image`, base64-referensbild). (2) En retention-sweep i serve-lagret
tar bort filer äldre än retention, efter mtime, med aktiva jobb skyddade. (3) En maintainer-doc binder
varje konstant och formatvariabel till sitt ställe i koden och sitt test.

## Globala begränsningar

- Exakta namn i denna spec är kopierade från verktygsresultat. Den som implementerar en task får inte
  gissa ett fältnamn: det tas ur kodens egen text först.
- OpenAI/Anthropic-beteende är ett externt avtal: berörda schema-test och `docs/serving.md` uppdateras
  tillsammans (AGENTS.md).
- Befintliga kontrakt i chatt-bilden får inte ändras: ett dåligt argument är modellens fel och blir ett
  error tool result, och en pipeline-failure (429/500) är också ett tool-svar
  ([chat_image_tool.h:64-73](../../../src/serve/chat_image_tool.h)).
- C rör inga kernels, ingen numerik och ingen device-väg.

## 1. Chatt-argumentet `image`

Idag: verktyget tar `prompt`, `size`, `steps`, `negative_prompt`, `true_cfg_scale` och `seed`
([chat_image_tool.cpp:16-21](../../../src/serve/chat_image_tool.cpp)), medan referensvägen redan finns i
routen som base64 ([openai_images.h:39-40](../../../src/serve/openai_images.h)) med tak 16 MiB
(`kMaximumImageInputBytes`, [openai_images.h:54](../../../src/serve/openai_images.h)).

**Schema:** ett frivilligt strängargument `image`, vars beskrivning nämnar de tre former verktyget löser
och inga andra.

**Upplösning, i denna ordning, i chatt-lagret (inte i routen):**

1. Ett jobbid som routen redan delar ut (formatet `img-…`) -> filen `images/<id>.png`.
2. En URL under denna servers eget prefix `/v1/images/` (`kImageArtifactRoutePrefix`) -> samma fil.
3. `attachment:N` -> den N:te bildmedien i *denna* chatt-request (den användaren laddade upp),
   nollindexerad. Fällets exakta plats i requesten tas ur kodens egen text vid implementationen, inte
   gissat.

**Avvisa uttryckligen,** som error tool result vars text namner de accepterade formerna (så modellen kan
pröva om): rå base64 eller data-URI i argumentet (modellen kan inte producera en 16 MiB-blob, och rå
base64 skulle bara dölja att bilden inte finns); främmande värdar (en URL som inte pekar på detta
processens eget prefix); okänt id; `attachment:N` utan att requesten har så mycket media; media som är
större än `kMaximumImageInputBytes`; och en fil som inte finns på disk (tagen av retention).

**Gränser som redan finns och lämnas orörda:** max fyra bilder per request
(`kMaximumChatImageGenerations`, [chat_image_tool.h:33](../../../src/serve/chat_image_tool.h)), och att
ett dåligt argument blir ett error result, inte ett HTTP-fel.

**Effekt på bildjobbet:** en upplöst referensbild går vidare till routen som base64 och blir
`ref_latents` exakt som en referensbild från HTTP-vägen gör; inget nytt i device-vägen. Behovet ökar med
referensen (A:s behovsberäkning tar redan höjd för det), så samma 503 kan komma tidigare -- det är
avsiktligt och ska stå i `docs/serving.md`.

## 2. Retention för genererade filer

Idag: filen skrivs till `current_path()/images/<id>.png`
([image_generation_service.cpp:315-331](../../../src/serve/image_generation_service.cpp)) och tas aldrig
bort. GET `/v1/images/<id>` läser direkt från disk
([http_server.cpp:607](../../../src/serve/http_server.cpp)), så filen är den verkliga varan, inte posten:
`ImageJobStore` har redan LRU-eviction och `prune_expired()` för posterna
([image_job_store.h:62-105](../../../src/serve/image_job_store.h)) och lämnar filen som en ensam efter sig.

**Policy (användarens val):** en fil får tas bort när den blir äldre än retention, default **14 dagar**.
Ingen antalsgräns och ingen byte-gräns -- bara ålder, eftersom URL:en är det enda användaren håller i.

**Mekanism:** en sweep i serve-lagret som

- körs en gång i starten och därefter med fast intervall (en timme),
- går igenom samma katalog som skrivaren (`current_path()/images`), med filernas mtime som ålder,
- tar bara bort filer vars post är terminal eller som ingen post har, så ett jobb som fortfarande skriver
  aldrig råkas ut, och
- loggar antal och bytes borttagna; att katalogen saknas är ett tyst no-op, inte ett fel.

**Konfiguration:** en serve-flagga med retention i dagar, default 14, 0 = av (ta aldrig bort). Flaggan
följer mönstret i `src/serve/serve_options.cpp` och namns där, inte här; `--help` och `docs/serving.md`
får samma text.

**Konsekvens som måste stå kvar i dokumentationen:** efter retention dör URL:en (404) även om en post
skulle finnas kvar; 14 dagar är alltså den längsta tid en bild är garanterat hämtbar, och den som vill ha
kvar den kopierar den.

## 3. `docs/maintainer/qwen-image-2-1-variables.md`

En enda sida som är auktoritet för image-pipelinnens variabler, med samma struktur för varje post:
**namn · vad det betyder · var det binds i koden · vilket test håller det**. Minst:

- format: BF16 för aktiveringar och uploads (`kImageElementBytes`), FP8 E4M3 row-scaled för de 17
  kvantiserade rollerna ([image_weights.h:11-14](../../../src/models/qwen_image/image_weights.h)),
  `Residency::Streamed` för alla vikter, och `image_block_bytes` som blockets residenta fotavtryck.
- gränser från routen: dimensionerna [32, 2048] med multipel 32, steps, CFG-intervallet
  (`kMinimumImageCFGScale`/`kMaximumImageCFGScale`), `kMaximumImagePromptBytes`,
  `kMaximumImageInputBytes`, `kMaximumChatImageGenerations` och retention (avsnitt 2).
- minne: `kImageWeightResidentBytes` och varför det är konservativt idag
  ([image_device_need.h:27-34](../../../src/models/qwen_image/image_device_need.h)), marginalen 512 MiB,
  och ledger-certen som sänker det.
- länkar till `docs/serving.md` (avtalet) och `docs/maintainer/model-quality.md` (kvalificering), inte en
  kopia av deras text.

Sidan dokumenterar bara det som finns; den får inte föreslå en blivande knob.

## 4. Verifiering

- **Schema och protokoll:** berörda OpenAI/Anthropic-schematest uppdaterade för `image`-argumentet, så
  att verktygets beskrivning (en del av exponerade `tools`) är testad.
- **Upplösning (host-test, befintlig fake-generator):** id, egen URL och `attachment:N` blir en
  referensbild till routen, och varje avvisad form ger ett error tool result med rätt kod och en text som
  namner de accepterade formerna.
- **Retention (host-test):** filer med gammal mtime tas, färska lämnas, ett aktivt jobb skyddas, 0 = av,
  och en URL inom retention hämtas fortfarande.
- **Route-end-to-end (svartfönster, befintlig prod-cert):** en bild med referens via chatt-argumentet ger
  en färdig URL och ett svar som bär den; A:s tre certer förblir gröna.
- **Dokumentation:** variabel-sidans länkar och flaggan i `--help` stämmas av mot sin egen kod.

## Explicitly not in scope

- Att låta modellen skicka egna base64-bilder, eller att hämta bilder från främmande värdar.
- Att göra LLM:n vision-capable som sådant; bara den bild som redan finns i requesten refereras.
- Molnlagring, kontokvoter, eller att behålla flera versioner av en bild.
- Tidsmätning, blockvikters placering och kernelarbete: bundle B.
