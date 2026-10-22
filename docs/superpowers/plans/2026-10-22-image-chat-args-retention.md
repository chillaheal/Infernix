# Chat image argument, retention och variabel-dokumentation (bundle C) implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans, task för task. Steg är
> ikryssningsbara. Tasks 1-3 (chatt-argumentet) och tasks 4-5 (retention) är oberoende av varandra och kan
> göras i valfri ordning; task 6-7 sist.

**Goal:** genomför `docs/superpowers/specs/2026-10-22-image-chat-args-retention-design.md`: chattverktyget får
`image`, genererade filer får en retention på 14 dagar som default, och image-pipelinnens variabler samlas i
`docs/maintainer/qwen-image-2-1-variables.md`.

**Architecture:** `image` löses upp i chatt-lagret till routens redan befintliga base64-referens; retention är
en sweep i serve-lagret som bara bryr sig om filålder; variabel-dokumentationen binder varje variabel till kod
och test.

## Global constraints

- Öppna avtal först: vidrört OpenAI/Anthropic-beteende ändras tillsammans med sina schema-test och
  `docs/serving.md`.
- Chatt-bildens befintliga kontrakt gäller: dåligt argument = error tool result, pipeline-failure = också ett
  tool-svar. Ingen HTTP-status läcker som chatt-svar.
- Inga gissade fältnamn: mediets plats i requesten, routens fält och flaggors namnstil tas ur kodens text.
- Bygg bara berörda test-mål; länka aldrig appen (levande serve håller exet). C rör ingen numerik.
- Host-test körds med `ctest --test-dir build -C Debug -R "<namn>"`; cert i svartfönster med
  `image-spike/blackout_bundle_a.ps1`.

---

### Task 1: schema och definition av `image`

**Files:** Modify `src/serve/chat_image_tool.cpp` (definitionen kring `:16-21`), `src/serve/chat_image_tool.h`,
berörda schema-test (hitta dem genom att söka efter test som redan hämtar `make_chat_image_tool_definition`).

- [ ] **Step 1:** Skriv det frivilliga strängargumentet `image` med en beskrivning som räknar de tre former
  som löses: eget jobbid (`img-…`), URL under `/v1/images/`, `attachment:N`. Ingen base64, inga främmande
  värdar.
- [ ] **Step 2:** Uppdatera schema-testen så att definitionen (namn, typer, beskrivning) är låst; kör dem.
- [ ] **Step 3:** Commit: `feat(serve): advertise the image argument on the chat tool`

### Task 2: upplösning av `image` i chatt-lagret

**Files:** Modify `src/serve/chat_image_tool.cpp` (`run_chat_image_tool_round` och dess validering),
`src/serve/image_generation_service.cpp` (`generated_image_path` `:315`) om fil-läsning behöver en helper.

- [ ] **Step 1:** Lägg resolutionen före samtalet med routen: id -> fil, egen URL -> fil, `attachment:N` ->
  mediets bytes i requesten (platsen i requesten tas ur kodens text, inte gissad).
- [ ] **Step 2:** Avvisa, med error tool result vars text namner de accepterade formerna: rå base64/data-URI,
  främmande värd, okänt id, `attachment:N` utan media, media över `kMaximumImageInputBytes`, saknad fil.
- [ ] **Step 3:** Skriva testen först (befintlig `FakeGenerator`-väg i `tests/test_chat_image_tool.cpp`):
  varje accepterad form blir en referensbild till routen; varje avvisad form rätt kod + text.
- [ ] **Step 4:** Kör `infernix_chat_image_tool_test`; grönt. Commit: `feat(serve): resolve the chat image argument to a reference image`

### Task 3: referensen vidare genom routen (kontroll, inte ny kod)

- [ ] **Step 1:** Följ länken `OpenAIImagesRequest::image` -> `encode_reference_image` -> `ref_latents` och
  bekräfta att en upplöst chatt-bild tar exakt samma väg som en HTTP-referens. Annars: det är här felet hör,
  inte i chatt-lagret.
- [ ] **Step 2:** Bekräfta att behovsberäkningen redan räknar med referens (A:s cert `768+ref` i matrisen) och
  att docs/serving.md säger att en referens kan ge 503 tidigare för samma storlek.
- [ ] **Step 3:** Om något av ovan inte håller: stoppa och rapportera, ändra inte i hemlighet.

### Task 4: retention-sweep

**Files:** Modify `src/serve/image_generation_service.{h,cpp}` (sweep), `src/serve/serve_options.cpp` (flagga,
`--help`), `apps/serve/main.cpp` om starten behöver en signal, `docs/serving.md`.

- [ ] **Step 1:** Lägg till flaggan i `serve_options.cpp` följandes filens eget mönster (dagar, default 14,
  0 = av) och `--help`-texten där den hör hemma.
- [ ] **Step 2:** Implementera sweept: en gång i starten, sedan timvis; `current_path()/images`, filernas
  mtime; ta bara filer vars post är terminal eller som saknar post; tyst no-op om katalogen saknas; logga
  antal och bytes.
- [ ] **Step 3:** Kör `infernix_serve_options*`/`infernix_openai_schema*`-målen som berörs av flaggan; gröna.
- [ ] **Step 4:** Commit: `feat(serve): retire generated images after a retention period`

### Task 5: retention-test

**Files:** Modify `tests/` (bild-tjästens test, hitta det genom att söka efter `save_generated_image`).

- [ ] **Step 1:** Test som förbereder filer med gammal respektive ny mtime: gamla tas, färska finns kvar.
- [ ] **Step 2:** Test av skyddet: en fil vars post fortfarande är aktiv tas inte; och 0 = av tar ingenting.
- [ ] **Step 3:** Test av att en URL inom retention fortfarande hämtas (samma väg som GET går).
- [ ] **Step 4:** Kör målet; grönt. Commit: `test(serve): cover the generated-image retention sweep`

### Task 6: end-to-end i svartfönster

- [ ] **Step 1:** Bygg berörda mål, stoppa serve i svartfönster, kör en riktig chatt-request vars modell ringer
  `generate_image` med `image` satt till en nyss genererad bild (id) och verifiera: ny URL i svaret, bilden
  går att hämta, A:s tre certer gröna (`RESULT ledger=0 prod=0 failures=0 healthy=True`).
- [ ] **Step 2:** Kör samma med `attachment:0` mot en request som bär en uppladdad bild.
- [ ] **Step 3:** Rapportera även misslyckanden: en modell som inte ringer verktyget är ett fynd, inte ett
  nej till featuren -- skriv vad som hände.

### Task 7: variabel-dokumentationen

- [ ] **Step 1:** Skriv `docs/maintainer/qwen-image-2-1-variables.md` enligt specens avsnitt 3: varje post med
  namn, betydelse, bindande ställe i kod, test som håller det. Bara det som finns idag.
- [ ] **Step 2:** Kontrollera varje länk och varje kodcitat mot filen det pekar på (exakt rad, inte ungefär).
- [ ] **Step 3:** Lägg in sidan i `docs/README.md`:ts kartan om den har en plats där; kör `git diff --check`.
- [ ] **Step 4:** Commit: `docs: document the image pipeline's variables`
