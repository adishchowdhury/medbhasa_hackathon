---
title: "MedBhasha Round 2 — Technical Report"
subtitle: "Team JUSL Health Home · COMSYS Hackathon-8 · Annotate → NER → Cluster → Retrieve"
---

# 1. Summary

We built an end-to-end, **offline and deterministic** Bengali clinical NLP pipeline and packaged it as one runnable notebook (`MedBhasha_Pipeline.ipynb`):

1. **Data.** After the organisers' feedback (40 notes / 612 tokens in Round 1, target ≈ 10 000 tokens) we grew our own annotated corpus to **@@N_NOTES@@ notes / @@N_TOKENS@@ tokens / @@N_SPANS@@ entity spans** covering @@N_DOMAINS@@ clinical domains and all 10 entity types (rarest type: @@RAREST@@). The corpus passes the official `validate.py` with 0 errors and 0 warnings.
2. **NER.** A lexicon-augmented, section-aware linear-chain CRF. Cross-validated entity-level F1 on our own data: **@@F1_RANDOM@@** (random 5-fold) and **@@F1_DOMAIN@@** (whole clinical domains held out).
3. **Clustering.** Unsupervised clustering of the @@N_TEST@@ released notes from their predicted entities: **@@N_CLUSTERS@@ clusters** (@@CLUSTER_ALGO@@, chosen by internal metrics only), silhouette **@@SIL@@**, Davies–Bouldin **@@DB@@**.
4. **RAG.** Retrieval over our annotated notes (+ the released notes annotated by our own NER model) with grounded, extractive Bengali answers and `[note-id]` citations for the 30 test queries; a live-query function for the finale. No external API, no web retrieval, no pretrained generator.

**Three caveats we want the reader to see first.**

* All our training data are *synthetic and written by one author (an LLM, with the team)*; no clinician reviewed the annotations. Cross-validated scores are therefore an **upper bound** — the hidden test F1 will very likely be lower, especially if the organisers' gold uses other span conventions (§4.6).
* The RAG rule says "retrieve from your own annotated corpus only". The queries ask for *patients*, which only the released notes can supply. We therefore provide **two answer files** — released notes annotated by our own NER (default) and strictly team-corpus-only — and ask the organisers to confirm the intended reading (§6.5).
* We looked at a small number of released notes and the header statistics to learn the *format* of the notes (sections, shorthand headers such as `C/O`, `Dx`, `Rx`). We did not copy, annotate or train on any released text (verified: maximum character-5-gram Jaccard between any training note and any released note is **@@NOCOPY_MAX@@**).

# 2. Rule compliance at a glance

| Rule / requirement | How we comply |
|---|---|
| Approved sources only; source declaration with annotation method | 100 % synthetic team-authored notes; `round1_deliverables/SOURCE_DECLARATION.md` lists every batch and discloses *LLM-assisted manual drafting* (no rule-based tool, no clinician review). |
| Entity schema + BIO format, mandatory `validate.py` report | `data/round1/annotated_corpus.csv` (columns `note_id,token,tag`); report in `round1_deliverables/validation_report.txt`; prediction file validated too (`outputs/ner_predictions_validation_report.txt`). |
| EDA summary | `round1_deliverables/EDA_Summary.pdf` (+ §3.4 below). |
| NER trained on the team's **own** Round-1 annotations | CRF trained only on `annotated_corpus.csv`; **no pretrained model, no external data, no use of the released notes for training**. |
| Clustering unsupervised, features from NER output | Representation uses predicted entities only; model selection by silhouette / Davies–Bouldin only. |
| RAG: team's own corpus only, no external API / web | Everything local; knowledge base = our notes (+ released notes annotated by our NER); strict variant provided. |
| One runnable notebook; exact reproduction | `MedBhasha_Pipeline.ipynb`; fixed seed 42; every output hashed in `outputs/run_manifest.json`; `scripts/check_reproducibility.py` re-runs everything under a different `PYTHONHASHSEED` and compares hashes. |
| Cite tools / pretrained models | No pretrained models. Libraries: NumPy, pandas, SciPy, scikit-learn, CRFsuite via `sklearn-crfsuite`, `seqeval`, NetworkX, Matplotlib. The synthetic notes and span annotations were drafted by Claude (Anthropic) in conversation with the team; disclosed in the source declaration. |

# 3. Data

## 3.1 From Round 1 to Round 2

The Round-1 submission had 40 notes (612 tokens) and most entity types had < 30 mentions. The organisers recommended ≈ 10 000 tokens, prioritising rare entity types, and noted that few-shot LLM approaches work better on tiny corpora. We did the following:

* kept the 40 Round-1 notes (`n001–n040`) and **fixed 8 inconsistent tags** (`round1_fixes_log.tsv`), e.g. dosage forms such as `ইনহেলার` moved into the MEDICATION span, `ফলো-আপ` removed from TEMPORAL;
* wrote **@@N_NEW@@ new notes** (`n041–n232`) in 9 batches, deliberately mixing registers: long structured outpatient notes, admission/discharge summaries, shorthand notes (`C/O … O/E … Dx … Rx … Adv … F/U`), and very short informal first-person notes;
* repaired the metadata file: the Round-1 `note_metadata` covered only 20 notes with stale domain labels; the new `data/round1/note_metadata.csv` covers all @@N_NOTES@@ notes.

## 3.2 Annotation method (disclosure)

*LLM-assisted manual drafting.* The assistant wrote notes and entity spans as inline markup (`{DX|ডায়াবেটিস}`), a deterministic script aligned spans to tokens and produced BIO tags (it refuses spans that cut tokens, overlap, or contain punctuation). No clinician reviewed anything; there is **no inter-annotator agreement**. All decisions are documented in `docs/ANNOTATION_CONVENTIONS.md`. Source category: *synthetic clinical notes generated by the team* — no real patient data, no EHR, no web text.

## 3.3 Corpus statistics

@@TABLE_ENTITY_FREQ@@

Notes: @@N_NOTES@@ · tokens: @@N_TOKENS@@ (mean @@LEN_MEAN@@ per note, median @@LEN_MED@@, range @@LEN_MIN@@–@@LEN_MAX@@) · spans: @@N_SPANS@@ · domains: @@N_DOMAINS@@ (largest: @@TOP_DOMAINS@@).

![Entity frequency in our corpus](@@FIG@@/eda_entity_frequency.png){ width=70% }

## 3.4 EDA findings (what the numbers told us)

* `SYMPTOM`, `TEMPORAL`, `TEST` and `MEDICATION` dominate; `PROCEDURE` is rarest (@@PROC_N@@ spans) because most clinical acts in outpatient notes are tests or drugs. This is the schema's nature, not a labeling gap.
* Note length is bimodal by design (very short informal notes vs. ≈ 100-token structured notes); the released notes are all structured and ≈ @@TEST_LEN_MEAN@@ tokens long, so the structured part of our corpus matters most.
* Co-occurrence (figure `outputs/figures/eda_cooccurrence.png`): P(DOSAGE present | MEDICATION present) = @@P_DOS_MED@@, P(TEST_RESULT | TEST) = @@P_RES_TEST@@, P(TEMPORAL | SYMPTOM) = @@P_TMP_SX@@, P(NEGATION_UNCERTAINTY | DIAGNOSIS) = @@P_NEG_DX@@ — the footprint of prescription and examination templates (descriptive, not clinical findings). The EDA doubled as a self-check: it exposed a split-vs-merged inconsistency for "organ + symptom" phrases (`পেটে ব্যথা` as one SYMPTOM vs. `BODY_SITE`+`SYMPTOM`), which we resolved in favour of the organisers' worked example (BODY_SITE F1 rose by about 9 points under domain hold-out in our development runs after the fix).

# 3.5 Diversity and originality checks

* exact duplicates: none; maximum character-4-gram Jaccard between any two of our notes: **@@INTERNAL_MAXSIM@@** (near-duplicate threshold 0.6);
* similarity to the released test notes: max char-5-gram Jaccard **@@NOCOPY_MAX@@** (unit test `tests/test_basics.py::test_no_note_copied_from_release`);
* the style of batches 03–11 imitates the *genre* of the release (section headers, shorthand headers) — we disclose this openly; no sentence was copied.

# 4. Stage 1 — Named-entity recognition

## 4.1 Model

A linear-chain **Conditional Random Field** (CRFsuite via `sklearn-crfsuite`; L-BFGS, c1 = c2 = 0.05, 200 iterations, all transitions allowed). Why a CRF: with ≈ 12 k annotated tokens a neural encoder over-fits, and pretrained Bengali encoders (IndicBERT / MuRIL / BanglaBERT) cannot be downloaded in an offline, exactly-reproducible run. A CRF trains in seconds and is bit-for-bit deterministic. We did **not** use an LLM few-shot approach because it requires a hosted or very large local model, which violates the no-external-API / exact-reproduction constraints.

**Features per token:** word, 1–4-char prefixes and suffixes (Bengali inflection and drug morphology), orthographic shape (digit / `/` / `.` / `%` / Latin), ±2-token context window and bigrams, **gazetteer marks** and **note-section context**.

* **Gazetteer** (`src/medbhasha/lexicon.py`, @@LEX_TERMS@@ terms): bilingual drugs, diseases, symptoms, tests, body sites, procedures, negation and dosage words, written from general clinical knowledge — *not* mined from the released notes. Matching is spelling-normalised (hasanta removal, ড়/য় composition, case-ending stripping).
* **Section context** (`ner.py::section_context`): the nearest preceding header (`নির্ণয়:`, `ব্যবস্থাপত্র:`, `ভাইটাল সাইন:`, `C/O`, `Dx`, `Rx`, `Adv`, `F/U` …) is mapped to one of 11 groups (COMPLAINT, HISTORY, VITAL, EXAM, INVEST, DIAG, TREAT, ADVICE, FOLLOW …) and combined with word shape and gazetteer marks. Output is post-processed so punctuation is always `O` and BIO sequences are always valid.

## 4.2 Evaluation protocol

Entity-level micro precision / recall / F1 with `seqeval` (exact span **and** type match). Two cross-validation schemes on our corpus: **random 5-fold by note**, and **domain-held-out 5-fold** (all notes of a clinical specialty held out together — e.g. the model never sees a dermatology note when predicting on dermatology), the closer proxy for the hidden set with unseen diseases and drugs.

## 4.3 Results

@@TABLE_NER_CV@@

* The gazetteer is the single most useful component (**@@GAZ_GAIN@@ F1 points** under domain hold-out): it lets the model recognise drug and disease names it never saw in training.
* Section context helps less (@@SEC_GAIN@@ points) because many of our synthetic notes are short or unstructured; on the structured released notes we expect more.

Per-type F1, domain-held-out:

@@TABLE_NER_PER_TYPE@@

![Per-type F1](@@FIG@@/ner_per_type_f1.png){ width=70% }

**Learning curve** (fixed 40-note hold-out, growing training set) — the expansion mattered:

@@TABLE_LEARNING@@

## 4.4 Error analysis

* **BODY_SITE and SYMPTOM** are the weakest (they are also the most convention-sensitive: where does the site end and the symptom begin?). Many errors are boundary errors, which `seqeval` counts as full misses.
* **TEST_RESULT** is weaker than TEST: qualitative results ("সামান্য বেশি", "প্রসারিত") are open-ended and unseen in training.
* Unseen vocabulary in *shorthand* complaints (`C/O: রক্তে চর্বি বেশি ধরা পড়েছে`) is often left untagged: the CRF is conservative on words it has not seen.

## 4.5 Predictions for the released notes

@@TEST_NER_SUMMARY@@

Output formats (`outputs/`): `ner_predictions_bio.csv` (strict `note_id,token,tag`, validated by the official validator), `ner_predictions_spans.csv` (character offsets into `note_text`), `ner_predictions.jsonl`. The tokenizer is the same deterministic one used for training (whitespace split, edge punctuation split off, internal `- / .` kept, abbreviations such as `মি.গ্রা.` kept whole).

## 4.6 Why the hidden-test score will differ from our CV score

(i) Our data are synthetic and single-author; (ii) the hidden gold's conventions are unknown (units in `TEST_RESULT`? split of organ + symptom? history diagnoses?) — see "Known ambiguities" in `ANNOTATION_CONVENTIONS.md`; (iii) the released notes use richer vocabulary than ours. We expect a drop, mostly in BODY_SITE/SYMPTOM boundaries.

# 5. Stage 2 — Phenotype clustering

## 5.1 Representation (entities only)

For each of the @@N_TEST@@ released notes we use **only the entities predicted by the NER model** — not the note id, the file order, or any label:

* **View A:** DIAGNOSIS mentions → character 2–4-gram TF-IDF (robust to Bengali inflection and transliteration: `আর্টিকারিয়া` ≈ `আমবাত (আর্টিকারিয়া)`).
* **View B:** typed bag of DIAGNOSIS / SYMPTOM / MEDICATION / TEST / PROCEDURE / BODY_SITE phrases and their words → TF-IDF (entity co-occurrence). `DOSAGE`, `TEMPORAL`, `TEST_RESULT`, `NEGATION` are dropped (numbers and boiler-plate).
* Both views L2-normalised, weighted 1 : 1, concatenated, reduced to 64 dimensions (truncated SVD) and length-normalised.

## 5.2 Algorithm and model selection — internal metrics only

We compared k-means (k = 20…200), Ward agglomerative (k = 20…200) and Louvain communities on a cosine 10-NN graph (4 resolutions) and kept the configuration with the best **silhouette** (tie-break: lower **Davies–Bouldin**). No phenotype labels, no proxy labels were used for selection.

![Internal metrics](@@FIG@@/cluster_selection.png){ width=95% }

Result: **@@CLUSTER_ALGO@@ with @@N_CLUSTERS@@ clusters**, silhouette @@SIL@@, DB @@DB@@; cluster sizes @@CLUSTER_SIZES@@. The silhouette curve rises steeply up to ≈ 80–90 clusters and then plateaus or declines, and Louvain independently finds 59–76 communities (depending on resolution), i.e. the notes contain on the order of 60–90 distinct condition groups.

## 5.3 Sanity check against the notes' own diagnosis line (not a score)

@@PROXY_TEXT@@

This is **not** the hidden gold: the regex-extracted diagnosis strings are much finer-grained (@@PROXY_CLASSES@@ distinct strings for @@PROXY_N@@ notes) than a typical phenotype label, and our clusters use the same predicted DIAGNOSIS entities, so the check is partly circular. We report it only to show that the unsupervised structure is not arbitrary. NMI is tolerant of over-clustering, ARI is not; if the hidden labels are much coarser than ~80 groups ARI will be lower than NMI.

## 5.4 What the clusters look like

@@TABLE_CLUSTER_EXAMPLES@@

(Full table: `outputs/cluster_summary.csv`; assignment: `outputs/clusters.csv`.)

## 5.5 Frequency and relationship analysis on the NER output

* Frequency of predicted entities by type: `outputs/test_entity_frequency.csv`; most frequent entities per type: `test_top_entities.csv`.
* **Relationship analysis**: diagnosis → medication and diagnosis → symptom associations by within-note co-occurrence, lift and PMI (`test_diagnosis_medication_pairs.csv`, `test_diagnosis_symptom_pairs.csv`). Strongest diagnosis–drug pairs (highest lift among pairs co-occurring in ≥ 5 notes):

@@TABLE_DX_MED@@

![Entity-type co-occurrence on the released notes](@@FIG@@/test_cooccurrence.png){ width=70% }

# 6. Stage 3 — Retrieval-augmented question answering

## 6.1 Knowledge base

Every note becomes a document with (a) normalised word keys, (b) the set of clinical concepts it mentions, (c) age and sex parsed from the header, (d) its entity spans — gold spans for our notes, **predicted** spans for the released notes.

## 6.2 Retrieval

1. **Normalisation.** Spelling skeleton (drops hasanta, collapses ি/ী ু/ূ ণ/ন শ/ষ/স, `অ্যা`→`এ`) + light Bengali stemming, so `মেটফর্মিন` = `মেটফরমিন` and `ডায়াবেটিসের` = `ডায়াবেটিস`.
2. **Concept table** (`concepts.py`, @@N_CONCEPTS@@ concepts / @@N_VARIANTS@@ surface variants): bilingual synonyms, abbreviations (T2DM, NS1, PPI, SSRI, UTI, BPH …) and drug classes (SSRI → এসসিটালোপ্রাম, সার্ট্রালিন …). Multi-word concepts also match **order-free inside a window** (`রাতে বেশি চুলকানি` ≈ `রাতে চুলকানি`).
3. **Polarity-aware matching.** Concepts are detected per *clause*; a concept in a negated clause (`ডায়াবেটিস নেই`) or about a family member (`মায়ের ডায়াবেটিস`) does **not** count for the patient — unless the query itself asks for the negation (`অ্যান্টিবায়োটিকের প্রয়োজন নেই`).
4. **Constraints.** Child / elderly / sex from the note header, numeric thresholds (`HbA1c ৯-এর বেশি` is evaluated on the value in the note).
5. **Scoring.** concept coverage (0.55) + IDF-weighted coverage of remaining rare query terms (0.25) + BM25 (0.20, tie-breaker); constraints multiply the score by 0.3 when violated; a +0.10 bonus when the queried disease sits in a DIAGNOSIS span inside the note's diagnosis section (`নির্ণয়:` / `Dx:`) — the note is *about* it, not merely mentioning it in the history. All conditions satisfied → exact set; otherwise the closest partial matches are shown and **labelled** with the missing condition.

## 6.3 Answer generation — grounded by construction

No generative model is used. The answer is assembled from templates, entity strings and verbatim sentence fragments copied from the retrieved notes, each cited with `[note-id]`: a count and, depending on the query intent, an aggregated list (drugs / symptoms / findings / treatment with the number of notes), then per-patient lines (age, sex, diagnosis, drugs with dosage, evidence excerpt). If nothing satisfies the query, the system says so and refuses to guess.

## 6.4 Results on the 30 released queries

@@TABLE_RAG_QUERIES@@

All 30 answers: `outputs/rag_answers.md` (readable), `.csv`, `.jsonl`. Representative example (q19, numeric constraint):

```
@@RAG_EXAMPLE@@
```

**Generalisation test (queries not in the released set)** — the concept table was written with the public queries in view, so we checked unseen questions; unknown concepts fall back to BM25 + fuzzy stems:

@@TABLE_UNSEEN@@

## 6.5 The "own annotated corpus only" question

The rule says retrieval must draw only from the team's own annotated corpus. Our 232 notes alone can answer only part of the 30 queries (a handful of notes per condition). The queries, however, ask which *patients* have a condition, which presupposes the released patient notes. We therefore treat "annotated corpus" as *our notes + the released notes annotated by our own NER model* in the default run, and we additionally ship the strict variant (`rag_answers_team_corpus_only.*`, @@STRICT_COVERAGE@@ of the 30 queries return at least one note). **We will open an issue / ask the organisers which interpretation is intended**; switching is one argument (`ask --team-only`, or `release_df=None`).

## 6.6 Limitations of the RAG stage

* Extractive, template-based Bengali — useful and verifiable, but not fluent prose. A local LLM could polish wording but cannot be shipped in an offline, exactly-reproducible repo; if allowed, it should be constrained to rephrase only the retrieved text.
* Answers inherit NER errors for the released notes (e.g. a negated drug mentioned in advice can appear as "recorded"). Evidence excerpts are verbatim, so the reader can always verify.
* The concept table is hand-written; new abbreviations need new entries (fallback: BM25 + stems).
* No clinician evaluation yet; "correctness/usefulness" in the clinical advisor's rubric is untested by us.

# 7. Reproducibility

`python scripts/run_pipeline.py` (or *Restart & Run All* in the notebook) regenerates every file in `outputs/`. All randomness is seeded (CRF is deterministic; k-means, SVD, Louvain take `random_state=42`). Library versions are pinned in `requirements-lock.txt` and recorded with MD5 hashes of all outputs in `outputs/run_manifest.json`. `scripts/check_reproducibility.py` re-runs the pipeline with a different `PYTHONHASHSEED` and asserts that all hashes are unchanged. @@REPRO_RESULT@@

# 8. Limitations, risks and next steps

1. **Annotation quality** is the dominant risk: synthetic, single-author, no clinician review, no IAA. *Next step:* clinician spot-check of ≈ 50 notes; resolve the ambiguities listed in the conventions file.
2. **Domain shift** between our notes and the released notes (vocabulary, length). *Next step:* add notes in the observed registers (needs more synthetic authoring, not the released text).
3. **NER model class.** A CRF cannot use cross-sentence context or pretrained knowledge. *Next step (if offline weights are allowed):* IndicBERT/MuRIL token classifier initialised from public weights, ensembled with the CRF.
4. **Clustering granularity** is inferred from internal metrics; hidden label granularity is unknown.
5. **RAG** is extractive and concept-table dependent.

# 9. Questions for the organisers

1. Which interpretation of "own annotated corpus" is intended for RAG (§6.5)?
2. What exact format is expected for NER predictions (we provide BIO per token, spans with character offsets, and JSONL)?
3. Do TEST_RESULT spans include units; are organ + symptom phrases split?

# Appendix A — Repository map

| Path | Content |
|---|---|
| `MedBhasha_Pipeline.ipynb` | the single runnable notebook (executed, outputs embedded) |
| `src/medbhasha/` | library: `tokenizer`, `corpus`, `lexicon`, `ner`, `evaluation`, `cluster`, `concepts`, `rag`, `analysis`, `run_all`, `ask` |
| `data/round1/` | annotated corpus (BIO), raw text, metadata, annotation sources (span markup), the 40 submitted Round-1 notes |
| `data/round2/` | organiser-released test notes and queries (unchanged, with BOM removed) |
| `outputs/` | all generated results + `run_manifest.json` |
| `round1_deliverables/` | validation report, source declaration, EDA summary, Round-1 fix log |
| `docs/` | this report, `GUIDE_FROM_BASICS`, `ANNOTATION_CONVENTIONS` |
| `resources/` | organiser files used unchanged (`validate.py`, schema, rules, worked example) |
| `scripts/` | `build_corpus.py`, `run_pipeline.py`, `check_reproducibility.py`, `make_notebook.py`, `build_docs.py`, `make_zip.py`, `serve_app.py` |
| `tests/` | sanity tests |
