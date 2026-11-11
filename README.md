# MedBhasha — Bengali Clinical NLP (Team **JUSL Health Home**)

COMSYS Hackathon-8 · Jadavpur University · *Annotate → NER → Cluster → Retrieve*

One runnable notebook, **offline and deterministic**, that goes from our own annotated Bengali clinical corpus to entity extraction, unsupervised phenotype clustering and grounded question answering.

| Stage | Result (our own data / outputs) |
|---|---|
| **Corpus** (Round 1, expanded) | **232 notes · 12060 tokens · 2899 spans**, all 10 entity types, official `validate.py`: **PASS** (0 errors, 0 warnings) |
| **NER** (CRF + gazetteer + section features) | entity-level F1 **0.876** (random 5-fold) / **0.861** (domains held out) — *upper bounds: synthetic, single-author data* |
| **Clustering** (1691 released notes) | ward, **90 clusters**, silhouette 0.777, Davies–Bouldin 0.717 (selected by internal metrics only) |
| **RAG** (30 queries) | grounded extractive Bengali answers with `[note-id]` citations; no external API |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                      # exact versions: requirements-lock.txt
jupyter nbconvert --to notebook --execute --inplace MedBhasha_Pipeline.ipynb   # ≈ 6 min; or open the notebook and "Run All"
python scripts/run_pipeline.py                       # same thing, headless (--fast skips the ablation CV)
python scripts/ask.py "ডায়াবেটিস রোগীদের জন্য কোন কোন ওষুধ নথিভুক্ত করা আছে?"   # live query
python tests/test_basics.py                          # sanity tests
python scripts/check_reproducibility.py              # re-run under another PYTHONHASHSEED and compare hashes
```

## Where to look

| Deliverable | File |
|---|---|
| **Notebook** (executed, outputs embedded) | `MedBhasha_Pipeline.ipynb` |
| **NER predictions** | `outputs/ner_predictions_bio.csv` (`note_id,token,tag`), `ner_predictions_spans.csv` (char offsets), `ner_predictions.jsonl` |
| **Cluster + analytic results** | `outputs/clusters.csv`, `cluster_summary.csv`, `cluster_selection.csv`, `test_entity_frequency.csv`, `test_diagnosis_medication_pairs.csv`, `test_diagnosis_symptom_pairs.csv`, `figures/` |
| **RAG outputs** | `outputs/rag_answers.md` / `.csv` / `.jsonl` (default KB) and `rag_answers_team_corpus_only.*` (strict KB) |
| **Technical report** | `docs/TECHNICAL_REPORT.pdf` (`.md`, `.docx`) |
| **Explained from the basics** | `docs/GUIDE_FROM_BASICS.pdf` (`.md`, `.docx`) |
| Round-1 style deliverables | `round1_deliverables/` — `validation_report.txt`, `SOURCE_DECLARATION.md`, `EDA_Summary.pdf`, `round1_fixes_log.tsv` |
| Annotated corpus | `data/round1/annotated_corpus.csv` (+ `notes_raw.csv`, `note_metadata.csv`, `annotation_sources/`) |

## Repository layout

```
MedBhasha_Pipeline.ipynb      the single runnable notebook
src/medbhasha/                tokenizer · corpus · lexicon · ner · evaluation · cluster · concepts · rag · analysis · run_all · ask
data/round1/                  annotated corpus, raw text, metadata, annotation sources (span markup)
data/round2/                  organiser-released test notes + queries (unchanged)
outputs/                      every generated result + run_manifest.json (MD5 of all outputs)
round1_deliverables/          validation report, source declaration, EDA summary, fix log
docs/                         technical report, from-basics guide, annotation conventions
resources/                    organiser files used unchanged (validate.py, schema, rules, worked example)
scripts/                      build_corpus · run_pipeline · check_reproducibility · make_notebook · build_docs · make_zip · serve_app
tests/                        sanity tests
```

## Rules checklist

* Data: 100 % synthetic, team-authored (LLM-assisted, disclosed); no real patient data. **No clinician review** — stated everywhere.
* NER is trained **only** on our own annotations; the released notes are used as unlabeled input only.
* No pretrained model, no external API, no web retrieval; everything runs locally (CPU, ≈ 6 min).
* Reproducible: seed 42, pinned versions, hashed outputs.

## Decisions we need the organisers to confirm

1. **RAG corpus.** Rule: *retrieve only from the team's own annotated corpus*. Queries ask for patients that exist only in the released notes. Default answers search *our notes + the released notes annotated by our own NER*; strict team-only answers are also provided.
2. **NER output format.** We provide per-token BIO, character-offset spans and JSONL.
3. **Span conventions** (units inside `TEST_RESULT`; organ and symptom split) — see `docs/ANNOTATION_CONVENTIONS.md`.

## Honest limitations

Synthetic single-author data → our CV scores are optimistic; no clinician validation; extractive (template) answers; concept table hand-written. Details: `docs/TECHNICAL_REPORT.pdf` §8.

*Attribution: organiser resources in `resources/` and `data/round2/` are from the COMSYS Hackathon-8 MedBhasha release (`pabitracsers/COMSYS-Hackathon-8`). Corpus consent: the team agrees to a CC-BY-4.0 public release as stated in the competition rules.*
