# INSTRUCTION.md — Första experimentet: DFlash2 på Qwen3.8-Flash-Next i Infernix

**Målgrupp:** en lokal kodmodell som arbetar direkt i den befintliga Infernix-checkouten.  
**Fas:** försiktig proof-of-concept.  
**Viktigt:** detta dokument ber dig inte träna en drafter ännu. Första målet är att förstå kompatibiliteten, återanvända så mycket DFlash2-kod som möjligt och få ett isolerat, diagnostiserbart test att köra utan att bryta dagens MTP- eller Qwen3.8-27B/DFlash2-stöd.

---

## 0. Sammanfattning av uppgiften

Vi har lokalt:

- Infernix-källkoden.
- En fungerande Qwen3.8-27B-artifact/configuration med DFlash2-drafter.
- En fungerande Qwen3.8-Flash-Next-artifact som idag använder MTP och som saknar en Flash-Next-specifik DFlash2-drafter.
- En GPU med 32 GB VRAM och en begränsad mängd system-RAM. Flash-Next-experter kan vara offloadade till RAM/SSD, så undvik att försöka hålla två target-modeller laddade samtidigt.

**Övergripande idé:** återanvänd DFlash2:s befintliga inferensimplementation, selector och dynamic-convolution-del från 27B. Anpassa endast de kontrakt som faktiskt är modellberoende. Börja med statisk kompatibilitetsdiagnostik och tensor-shape-test; gå sedan till ett opt-in runtime-smoketest. Ingen stor datasetgenerering eller träning i denna första uppgift.

**Förväntat resultat från denna första fas:**

1. En konkret kompatibilitetsrapport som listar varje känd skillnad och vad som måste göras åt den.
2. Test(er) som visar att Flash-Nexts valda hidden states kan fångas och levereras i rätt format.
3. En isolerad experimentell DFlash2-väg, eller ett dokumenterat stoppbeslut om en blockerande mismatch gör runtime-testet olämpligt just nu.
4. Om runtime-smoketestet är möjligt: en kort greedy-generation där targetens verifiering förblir korrekt. Acceptance rate och hastighet ska mätas, men förvänta dig inte bra resultat före modellspecifik träning.
5. Inga regressioner för befintlig `--spec mtp` på Flash-Next eller `--spec dflash2` på Qwen3.8-27B.

### Vad som INTE ska göras nu

- Träna inte modellen och generera inte miljontals tokens.
- Hämta inte nya modeller eller installera stora beroenden utan att först verifiera om de redan finns lokalt.
- Skriv inte över, konvertera eller modifiera originalets 27B DFlash2-vikter.
- Ändra inte betydelsen av existerande `--spec mtp` eller `--spec dflash2`.
- Marknadsför inte resultatet som en fungerande/snabb Flash-Next-DFlash2. Ett shape-test bevisar bara att tensorflödet fungerar; det bevisar inte att den otränade eller adapterade draftern är användbar.
- Dölj inte viktmismatchar med tyst `strict=False`, nollutfyllnad eller ignorerade fel. Varje avvikelse måste loggas och förklaras.

---

## 1. Arkitekturfakta att utgå från — men verifiera lokalt

Konfigurationsfiler och lokala artifact-versioner är sanningskällan för just den installerade modellen. Värdena nedan kommer från publicerade konfigurationer och är startpunkt, inte skäl att hoppa över lokal inspektion.

### 1.1 Qwen3.8-27B-DFlash2

Den publicerade `z-lab/Qwen3.8-27B-DFlash2/config.json` innehåller bland annat:

- `architectures`: `DFlash2DraftModel`
- drafter `hidden_size`: `5120`
- `vocab_size`: `248320`
- `num_hidden_layers`: `5`
- `num_target_layers`: `64`
- `dflash_config.block_size`: `8`
- `target_layer_ids`: `[5, 19, 33, 47, 61]`
- `conv_kernel_size`: `2`
- `conv_group_size`: `16`
- `selector_rank`: `256`
- `selector_top_k`: `16`
- `mask_token_id`: `248070`
- drafterns egna `head_dim`: `128`

Dessa är tränings-/checkpointvärden för en **Qwen3.8-27B-target**. Kopiera inte dess `target_layer_ids` direkt till Flash-Next: de sista indexen överstiger Flash-Nexts lagerantal. Ändra inte blockstorlek eller specialtoken-ID utan att följa implementationens kontrakt.

### 1.2 Qwen3.8-Flash-Next

Den publicerade HF-konfigurationen identifierar målmodellen som `Qwen4ExpForConditionalGeneration`, med textkonfiguration `qwen4_exp_text`. Viktiga värden:

- `hidden_size`: `2560`
- `num_hidden_layers`: `48`
- `vocab_size`: `248320`
- `num_experts`: `512`
- `num_experts_per_tok`: `10`
- `full_attention_interval`: `4`
- `layer_types`: tre `linear_attention`-lager följda av ett `full_attention`-lager, upprepat
- `head_dim`: `256`
- `linear_num_key_heads`: `16`
- `linear_num_value_heads`: `48`
- n-gram-embeddings och en inbyggd MTP-konfiguration
- överordnad arkitektur omfattar även vision; börja med **text-only** testfall

Flash-Next är alltså inte bara en Qwen3-modell med ett annat antal lager. Den har en annan modelltyp, annan attention/state-struktur och MoE-routing.

### 1.3 DFlash v1 vs DFlash2

DFlash v1 är redan en användbar arkitektonisk referens: en lätt block-diffusion-drafter tar in hidden features från flera target-lager och konditionerar alla drafterlager på target-informationen. DFlash2 behåller den grundläggande one-pass block-drafting-idén och lägger till bland annat:

- en path selector som poängsätter kandidat-tokenpar intill varandra och väljer en sammanhängande väg genom kandidatlistorna;
- två-tap dynamic depthwise convolution för att förbättra beroenden inom blocket och minska kvalitetsförlust mot blockets slut.

**Återanvänd DFlash2-kodvägen för detta experiment.** Återanvänd inte Qwen3.6-35B-A3B-DFlash-vikterna som om de vore DFlash2-vikter: det är DFlash v1. De kan hjälpa till att förstå mål-lager-tapping och MoE-stöd i DFlash-familjen, men deras tensorformer och inlärda representationer är inte ett substitut för en Flash-Next-drafter.

### 1.4 Den mest avgörande skillnaden

Samma `vocab_size` betyder inte att två drafter-checkpoints är kompatibla. Även om Flash-Next och Qwen3.8-27B har lika stort tokenutrymme skiljer sig deras interna dimensioner och target-representationer.

Speciellt:

- 27B-draftern förväntar sig en targetmodell med 64 lager och hidden size 5120.
- Flash-Next har 48 lager och hidden size 2560.
- 27B:s layer taps innehåller ID:n som är ogiltiga för Flash-Next.
- Flash-Next har linear-attention state och en hybrid arkitektur; target hidden-state capture får inte anta att alla lager har standard full-attention-beteende.
- DFlash-checkpointens mask-/specialtoken-ID ska verifieras mot faktisk tokenizer och den befintliga runtime-implementationen. Anta inte att `248070` från 27B utan vidare är rätt för Flash-Next.

Det här går att hantera i en experimentell adapter, men **shape-kompatibilitet är inte samma sak som kvalitetskompatibilitet**. Om vi använder 27B-vikter med en ny/random projection kan vi testa kopplingen, men acceptance rate kommer sannolikt inte säga något bra om vad en korrekt tränad Flash-Next-drafter skulle kunna prestera.

---

## 2. Arbetssätt och säkerhetsregler för kodändringar

Följ dessa regler under hela uppgiften:

1. Kör `git status` och notera aktuell branch/commit innan du gör något. Bevara alla befintliga, ocommittade ändringar. Byt inte branch eller återställ filer automatiskt.
2. Läs först dokumentation och befintlig implementation. Sök efter `DFlash2`, `dflash2`, `target_layer_ids`, `selector`, `conv_group_size`, `mask_token_id`, `--spec`, `MTP` och Flash-Next-modellregistreringen.
3. Följ projektets kodstil, teststruktur och namngivning. Gissa inte filnamn från det här dokumentet; lokalisera de verkliga filerna.
4. Dela upp ändringarna i små steg. Efter varje steg: bygg/kompilera det som rimligen påverkas och kör smala tester innan du går vidare.
5. Inga stora refaktoreringar. Ändra så få befintliga filer som möjligt och lägg experimentell logik separat där projektets struktur tillåter det.
6. Håll dagens fungerande väg oförändrad. Om någon ny flagga eller config behövs ska den vara explicit experimentell och avstängd som standard.
7. Spara en rapport över använda modeller/artifacts, konfigurationsvärden, viktmismatchar, testkommandon och resultat.
8. Om repo-läget eller artefakterna skiljer sig från det här dokumentets antaganden: stanna upp, rapportera skillnaden och anpassa planen i stället för att tvinga igenom antagandet.

---

## 3. Steg 0 — Säker inventering (ändra ingen runtime-kod)

### Gör

1. Bekräfta att du befinner dig i Infernix-repot.
2. Registrera:
   - branch och commit;
   - om working tree är rent;
   - aktuell byggmetod och fungerande testkommando;
   - modell-artifactens format (t.ex. Infernix/NInfer-artifact, safetensors eller annat);
   - var de lokala 27B target-/DFlash2-filerna och Flash-Next-filerna ligger, utan att flytta dem.
3. Läs Infernix README och Flash-Next-guiden. Bekräfta de faktiska CLI-flaggorna som används idag.
4. Hitta hur 27B-artifacten laddar DFlash2-vikterna och hur Flash-Nexts MTP-väg hämtar target hidden states/MTP-vikter.
5. Dokumentera vilka delar av modellens target-state som redan finns tillgängliga i Flash-Next-forward-pass: lagerutdata, MTP-state, embeddings, LM-head och/eller recurrent/linear-attention state.

### Viktigt att tänka på

- Räkna inte med att DFlash2-vikterna är en fristående Hugging Face-mapp. Infernix kan använda en konverterad artifact där drafter/proposal-vikter har packats ihop med targeten.
- Kopiera inte modeller till VRAM/RAM samtidigt för att “bara testa”. Börja med filmetadata och CPU-baserad inspektion.
- Om `git status` visar ändringar: håll dem orörda och arbeta runt dem.

### Klart när

En kort inventering finns, med verkliga sökvägar, modellformat, baseline-kommandon och filerna som hanterar DFlash2 och Flash-Next MTP. Ingen runtime-kod är ändrad.

---

## 4. Steg 1 — Skapa en statisk kompatibilitetsrapport

Det här ska vara det första konkreta kod-/verktygsarbetet. Lägg till ett litet diagnostikverktyg eller test enligt repo-konventionerna. Verktyget ska inte kräva att hela Flash-Next läses in på GPU.

### Krav på diagnostiken

Samla, så långt filformatet tillåter:

**Targetmodell Flash-Next**

- modell-/artifacttyp;
- lagerantal och layer types;
- target hidden size;
- vokabulärstorlek;
- embedding/LM-head dimensioner om de går att inspektera;
- tokenizer-/specialtokenmetadata, inklusive mask-tokenens faktiska tokensträng om den finns;
- om vald target-lagerutdata kan nås i befintlig Infernix-forward-väg.

**DFlash2 27B-checkpoint/artifact**

- draft hidden size;
- antal drafterlager;
- block size;
- target-lager-ID:n;
- selector-parametrar;
- convolution-parametrar;
- viktarnas tensorformer, inklusive embeddings och LM-head om de finns i artefakten;
- vokabulärstorlek och mask-/specialtoken-ID.

**DFlash v1 35B-A3B-referens om den finns lokalt**

- registrera endast arkitektur/config och tensorformer;
- markera den uttryckligen som DFlash v1, inte DFlash2;
- använd den som informationskälla, inte som vikter som ska kopieras in i DFlash2.

### Rapportformat

Spara gärna rapporten som en text- eller JSON-fil under ett experiment-/diagnostiknamn enligt projektets konventioner, exempelvis `dflash2_flashnext_compatibility_report.md` eller `.json`.

Varje kontrakt ska märkas:

- `MATCH`: verifierad kompatibel;
- `MISMATCH`: verifierad skillnad;
- `UNKNOWN`: kan inte avgöras från metadata, måste testas i runtime;
- `ADAPTER_CANDIDATE`: kan eventuellt hanteras med en uttrycklig adapter, men behöver bevisas.

Rapportera skillnaden mellan **obligatorisk formkompatibilitet** och **modellsemantisk kompatibilitet**. En linjär projektion kan lösa dimensioner, men den gör inte 27B-draftern tränad för Flash-Next.

### Viktigt att tänka på

- Använd det lokala artefaktformatets metadata-/indexfunktioner. Skanna inte igenom flera hundra GB med viktdata om header-/manifestinspektion räcker.
- Om artifacten är custom/binariserad, hitta befintligt verktyg för att läsa metadata. Implementera inte en gissad parser till produktionskoden.
- Skriv inga vikter.

### Klart när

Rapporten listar alla skillnader och varje `UNKNOWN` har ett konkret förslag på hur det kan verifieras.

---

## 5. Steg 2 — Kartlägg det befintliga DFlash2-kontraktet

Innan ny logik skapas, läs befintlig DFlash2-inferensväg från CLI till GPU-kernel(er).

Svara i en kort designnotering på följande frågor med exakta fil-/funktionsreferenser:

1. Var registreras `--spec dflash2` och var väljs algoritmen?
2. Hur laddas DFlash2-konfigurationen och vikterna? Kommer de från separata filer eller target-artifacten?
3. Vilken funktion samlar target-lager-output och i vilken ordning?
4. Förväntar feature-fusionen samma `hidden_size` hos target och drafter, eller finns det projection layers?
5. Hur läses block size och mask-token-ID? Är blockstorleken bestämd av checkpointen, CLI eller båda?
6. Var körs selector/pairwise scoring och dynamic convolution?
7. Hur väljs kandidattokens och hur lämnas de till verifieringsvägen?
8. Hur säkerställs korrekt greedy-/sampling-verifiering (inklusive fallback när token inte accepteras)?
9. Hur interagerar DFlash2 med `--lm-head-draft`, MTP och n-gram drafting? Kan flera vara aktiva samtidigt? Isolera dem i första experimentet.
10. Vilka modelltypspecifika antaganden finns för Qwen3.8-27B som kan gå sönder på `qwen4_exp`/Flash-Next?

### Viktigt att tänka på

**Anpassa befintlig implementation, inte bara configfilen.** Om dispatch eller hidden-state capture hårdkodar Qwen3-lagerlayout måste den delen hanteras explicit. Behåll DFlash2-selector och convolution oförändrade om inte deras gränssnitt verkligen kräver ändring.

### Klart när

Vi vet var adaptergränsen ska ligga, vilka befintliga komponenter som kan återanvändas direkt och vilka som är modellberoende. Ingen duplicerad “ny DFlash2”-implementation har skapats.

---

## 6. Steg 3 — Välj target hidden layers för ett rent formtest

DFlash2:s publicerade 27B-checkpoint använder target-layer-ID:n `[5, 19, 33, 47, 61]` för 64 lager. Dessa kan inte kopieras till Flash-Next med 48 lager.

För **ett första diagnostiskt försök endast** kan man undersöka en jämnt utspridd uppsättning på 48 lager som ungefär bevarar relativa djup, exempelvis `[4, 14, 25, 35, 46]`. Detta är endast en **testhypotes för att kontrollera capture och tensorflöde**, inte en rekommenderad tränad konfiguration. Den ska vara lätt att ändra och ska inte hårdkodas i flera filer.

### Krav

1. Kontrollera om layer-id:n i Infernix är nollbaserade eller ettbaserade och exakt vid vilken punkt lageroutput fångas.
2. Kontrollera att de valda ID:na verkligen motsvarar avsedda textlagers output i Flash-Next, inte vision-stack, MTP-lager, n-gram-embedding eller ett annat state.
3. Bekräfta shape, dtype, device och finita värden för varje tapp.
4. Logga varje taps ID, layer type, tensor shape och dtype i debugläge.
5. Lägg till ett enkelt test som avsiktligt väljer ett ogiltigt lagerindex och verifiera att implementationen returnerar ett tydligt fel, inte läser utanför arrayen.
6. Håll antalet taps och ordningen konsekventa mellan konfig, capture och fusion.

### Viktigt att tänka på

- Target tap-lager väljs normalt genom träning/validering, inte bara genom att fördela ID:n jämnt.
- Flash-Next blandar `linear_attention` och `full_attention`. Layer-output efter ett full-attention-block och output efter ett linear-attention-block har samma yttre hidden shape men kan ha olika användbarhet för drafter-träning.
- Återanvänd inte Flash-Nexts interna MTP hidden state som om det automatiskt vore en av DFlash2:s target taps. MTP kan användas för en separat jämförelse senare.

### Klart när

Vi kan fånga fem valda Flash-Next hidden-state tensors på ett kort textprompt-pass och har shape/dtype-assertions för dem. Ingen DFlash2-vikt behöver laddas ännu.

---

## 7. Steg 4 — Välj rätt sorts första smoke test

Dela detta i två tester. Blanda inte ihop dem.

### Test A: DFlash2-backbone isolerat

Ladda DFlash2-komponenterna i isolering, utan Flash-Next target. Använd små/syntetiska inputtensors med exakt de shapes som modellen förväntar sig. Verifiera att:

- drafterlagren kör;
- dynamic convolution kör med rätt block-/gruppdimensioner;
- selector tar emot förväntade kandidater och returnerar giltig path;
- logits/candidate-token-ID:n har förväntad shape;
- alla tensorer är finita och token-ID:n ligger inom target-vokabulären;
- block sizes på 8 (checkpointens värde) fungerar, och ogiltiga block sizes ger explicit fel.

Detta testar den befintliga DFlash2-koden, inte kompatibiliteten med Flash-Next.

### Test B: Flash-Next + adapter, bara efter Test A

Anslut de valda Flash-Next tapsen till DFlash2-vägen. Börja med att **undersöka den faktiska dimensioneringen i koden och checkpointen**. Välj adapter endast utifrån de verkliga tensorformerna.

Möjliga utfall:

1. **Direkt formkompatibilitet:** använd den befintliga vägen, men verifiera fortfarande modellsemantiken.
2. **Target hidden size 2560 måste mappas till drafterns 5120:** om den befintliga DFlash2-fusionen inte redan kan detta, gör en explicit, isolerad input projection/adapter endast för experimentet.
3. **Embeddings/LM-head är bundna till targetens hidden size:** om dimensionerna inte passar ska du inte dölja mismatchen. Ta reda på om DFlash2 förväntar sig targetens embedding/LM-head eller en kopia i draft-checkpointen. Dokumentera den blockerande dimensionen. Skapa vid behov en minimal experimentell adapter, men ändra inte originalcheckpointen.
4. **Saknat Flash-Next layer/state-stöd:** stoppa före token-generering och implementera först den minsta tydliga modellen för att läsa rätt layer outputs.

Om en projection initieras slumpmässigt eller med enkel deterministic init för shape-test ska koden/funktionen märkas `TEST_ONLY`. Dra inte slutsatsen att resultatet visar den verkliga potentialen hos en tränad drafter. Använd inte `pad`, `truncate` eller återanvändning av godtyckliga dimensioner utan att logga exakt vad som görs.

### Viktigt om vikter

- Behåll en oförändrad kopia av originalets 27B DFlash2-vikter.
- Ladda strikt där det är möjligt och lista alla oinlästa/oväntade nycklar.
- Om vissa parametrar måste nyskapas: logga dem separat, deras shape, initieringsmetod och att de är otränade.
- Återanvänd inte Qwen3.6-35B-A3B-DFlash-vikter i DFlash2-vägen. Det är DFlash v1 och annan checkpoint-konfiguration.

### Klart när

Antingen (a) den faktiska Flash-Next→DFlash2 tensorvägen klarar forward-smoketest med tydligt loggade former, eller (b) det finns en kort lista med blockerande mismatchar och rekommenderad nästa förändring. “Fungerar” betyder ännu inte bra tokenförslag.

---

## 8. Steg 5 — Lägg till en separat experimentell runtime-väg

Gör detta endast om Steg 0–4 är godkända.

### Designkrav

- Lägg till en separat experimentell mode/config med namn enligt Infernixs befintliga CLI-konvention. Namnge den så att användaren inte av misstag tror att vanlig Flash-Next DFlash2 är färdig.
- Rör inte beteendet för `--spec mtp` och den fungerande `--spec dflash2` för 27B.
- Låt experimentläge vara explicit opt-in och neka inkompatibla targettyper med ett tydligt meddelande.
- Ingen automatisk träning, datasetgenerering eller nedladdning får ske vid serverstart.
- Om checkpoint/config kräver `block_size=8`, använd detta först. Ändra inte till 4/16 bara för prestanda innan implementationen visat stöd och selector/conv-shapes är bekräftade.
- Börja text-only. Vision-input, multimodal promptsekvenser och lång kontext är utanför scope för första testet.
- Inaktivera n-gram drafting och andra draftmekanismer i första jämförelsen om de går att separera. Mät DFlash2-isolering först; kombinationer kan testas senare.
- Håll max context lågt och concurrency på 1 under testet för att minska VRAM/RAM-tryck.

### Diagnostik i loggen

I debugläge ska servern skriva en engångsheader med:

- targetmodelltyp/version/artifact;
- draftercheckpoint/artifact och config;
- experiment-mode och adapter-version;
- target-layer IDs och deras faktiska shapes;
- block size, selector top-k/rank och conv params;
- mask-token-ID och hur det verifierats;
- vilka draftervikter som laddats direkt, vilka som adapterats och vilka som är test-initierade.

Lägg inte in en lång dump per token i normal drift.

### Viktigt att tänka på

- Det är acceptabelt att implementationen ger dåliga draftförslag i detta test, så länge verifieringen är korrekt och mätningarna inte presenteras som ett prestandaresultat.
- Det är **inte** acceptabelt att fel draft tokens kan orsaka fel slutresultat. Speculative verification ska fortsatt ge exakt target-output enligt den befintliga algoritmens garanti.
- Om runtime-korrekthet inte kan garanteras från den befintliga verifieraren, implementera inte detta bakom en generell produktionsflagga. Begränsa till en unit/integration test-harness.

---

## 9. Steg 6 — Första mätning och korrekthetstest

Kör varje target separat; ladda inte 27B-target och Flash-Next samtidigt om det inte finns ett uttryckligt skäl.

### Baseline

1. Kör Flash-Next med befintlig fungerande MTP-konfiguration.
2. Använd greedy decoding (temperatur 0 om runtime stöder det), batch/concurrency 1 och ett kort textprompt.
3. Kör minst 3 fasta prompts: enkel kunskapsfråga, kort koduppgift och kort resonemangsuppgift. Undvik verktygskall/multimodal input i första omgången.
4. Spara exakt output-tokenföljd om möjligt, samt wall-clock decode-tid, TTFT, tok/s, VRAM/RAM och baseline-inställningar.

### Experimentell DFlash2 test

1. Samma target-artifact, tokenizer, promptmall, samplinginställningar och max-new-tokens.
2. DFlash2-testvägen ensam; ingen n-gram/MTP-kombination i första mätningen om inte kodens arkitektur gör den kombinationen obligatorisk.
3. Samma fasta prompts och outputlängd.
4. Logga accepted tokens per verification round och acceptance length om runtime stöder det. Logga separat draft latency, verification latency och total tid.
5. Kör samma prompt mer än en gång för att särskilja kallstart från steady state.

### Godkännandekriterier

**P0 — säkerhet/korrekthet:**

- Ingen crash, deadlock, CUDA illegal access, out-of-bounds layer/tensor eller NaN/Inf.
- Alla draft token-ID:n är inom tillåten vokabulär.
- Greedy slutresultat matchar targetens baseline token-för-token. Jämför det som faktiskt garanteras av runtime; var uppmärksam på skillnader i stop-token/templating.
- Existing Flash-Next MTP-test och existing 27B DFlash2-test fortsätter fungera.

**P1 — teknisk nytta:**

- Draftaren genererar kandidatblock och verifieraren accepterar/avvisar dem utan specialfall som döljer fel.
- Acceptance-mätetal fångas korrekt.

**P2 — prestanda, endast vägledande före träning:**

- Draft+verify-tiden jämförs mot samma target i baseline.
- Om acceptance är mycket låg eller latensen är hög dokumenteras det utan att försöka “tuna” bort problemet med godtyckliga konfigändringar.

### Viktigt att tänka på

En otränad/randomt adapterad drafter kan ge acceptance nära noll. Det betyder inte att DFlash2-arkitekturen är dålig; det betyder att den nuvarande draftervikten/representationen inte är anpassad till Flash-Next. Syftet här är att hitta integrationsfel och mäta kostnader, inte att dra slutsatser om slutlig acceleration.

---

## 10. Steg 7 — Regressionstester

Minst dessa tester ska köras efter ändring:

1. Befintlig Flash-Next MTP smoke test med exakt samma kommando och inställningar som före ändringen.
2. Befintlig Qwen3.8-27B DFlash2 smoke test.
3. DFlash2 synthetic-forward test.
4. Flash-Next taps shape/dtype/finite test.
5. Ogiltig config/layer ID ska ge tydligt fel och inte krascha processen.
6. Inkompatibel modeltype utan experimentläge ska fortfarande nekas (eller behålla dagens beteende).
7. Greedy exact-output-test på korta prompts om runtime-smoketestet implementerats.

Om någon befintlig baseline förändras, isolera och åtgärda regressionen innan du fortsätter.

---

## 11. Vad som ska lämnas in

När uppgiften är klar ska följande finnas:

1. **Kompatibilitetsrapport** med `MATCH/MISMATCH/UNKNOWN/ADAPTER_CANDIDATE` och tensorformer.
2. **Kort designnotering** som pekar ut de faktiska kodfilerna/funktionerna för DFlash2-vägen och Flash-Next hidden-state capture.
3. **Liten kodändring** som återanvänder befintlig DFlash2-kod. Ingen stor duplicerad implementation.
4. **Testresultat**: byggkommando, testkommandon, baseline och experimentresultat, korrekthetsjämförelse, acceptance, latency och memory.
5. **Kvarstående hinder** sorterade som:
   - blockerar form-/runtime-anslutning;
   - kräver tränade modellvikter;
   - prestandaoptimering efter att funktionaliteten bevisats.
6. **Ändringslista** över alla modifierade filer och varför de ändrades.
7. Kort rekommendation: fortsätta mot träningspipeline, göra en mindre kodjustering eller stoppa om det visar sig vara ett dåligt spår.

Uppdatera dokumentationen försiktigt och märk funktionen som experimentell. Skriv inte att Flash-Next “stöder DFlash2” utan kvalifikation om bara ett shape-smoketest passerar.

---

## 12. Vägen efter detta experiment (INTE del av första uppgiften)

Om runtime-tensorflödet fungerar och overheaden inte ser orimlig ut kan nästa uppgift bli träning. Först då ska man bygga en dataset-/teacher pipeline:

1. Frys samma Flash-Next target-artifact som ska användas vid inference. Om target kör NVFP4/FP8 i Infernix, avgör explicit om drafter ska tränas mot just den quantiserade targetens representationer/logits. Dokumentera valet.
2. Fånga target tokens, utvalda hidden states och de targets/logits som den valda loss-funktionen kräver.
3. Bestäm target-lager genom små ablationer i stället för att anta att jämn sampling är optimal.
4. Träna adapter och/eller DFlash2-drafter mot Flash-Next. Se till att selector loss och convolution-parametrar tränas om när det behövs.
5. Validera på prompts som inte användes i träning.
6. Mät acceptance length, latency och faktisk tok/s mot MTP på samma prompts.

DFlash2:s target-specifika vikter är vad som gör en användbar drafter; en ändrad config är inte en ersättning för modellträning. Håll denna framtida arbetsström utanför första PR/commit för att göra experimentet litet och lätt att bedöma.

---

## 13. Externa referenser att läsa (lokal repoimplementation har företräde)

- Infernix-repot och befintlig usage för Flash-Next/MTP och Qwen3.8-27B/DFlash2: https://github.com/Wallawalla47/Infernix
- DFlash repo och vilka checkpoints som anges som DFlash2 kontra DFlash v1: https://github.com/z-lab/dflash
- Qwen3.8-27B-DFlash2 config (block size, target layers, selector/conv): https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/blob/main/config.json
- Qwen3.6-35B-A3B-DFlash config (DFlash v1, MoE-referens): https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash/blob/main/config.json
- Qwen3.8-Flash-Next config (48 text layers, hidden size 2560, hybrid attention, MoE): https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json
- DFlash v1: grundidé om hidden-state taps och conditioning: https://z-lab.ai/projects/dflash/
- Inco AI:s DFlash2-översikt: selector och dynamic convolution: https://inco.ai/blog/dflash2/

---

## Slutlig instruktion till kodmodellen

Arbeta sekventiellt. **Börja med Steg 0 och Steg 1, skriv rapporten och inspektera resultatet innan du ändrar inferensimplementationen.** Gör inga stora ändringar bara för att få ett grönt build. Om en dimension eller vikt inte matchar ska du redovisa den, välja en explicit experimentell adapter eller stoppa med en blockeringsrapport. Målet är ett rent och mätbart första test, inte att låtsas att en otränad 27B-drafter redan fungerar för Flash-Next.
