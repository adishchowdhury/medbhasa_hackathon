#!/usr/bin/env python3
"""Generate MedBhasha_Pipeline.ipynb (un-executed). Execute it with:

    jupyter nbconvert --to notebook --execute --inplace MedBhasha_Pipeline.ipynb

The notebook is the single runnable deliverable: Round-1 data -> EDA -> NER -> clustering -> RAG.
"""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# MedBhasha — Bengali Clinical NLP: Annotate → NER → Cluster → Retrieve
**Team JUSL Health Home · COMSYS Hackathon-8 · Round 2 notebook**

One notebook, three stages, fully offline and deterministic (seed 42, no external API, no web retrieval, no model downloads):

| Stage | What this notebook does | Output files (`outputs/`) |
|---|---|---|
| 0 · Data | Loads our own annotated corpus (BIO), re-runs the official `validate.py`, EDA | `eda_*.csv`, `figures/eda_*.png` |
| 1 · NER | Lexicon-augmented, section-aware CRF; cross-validation + ablations; predictions for the released notes | `ner_predictions_bio.csv`, `ner_predictions_spans.csv`, `ner_predictions.jsonl`, `ner_cv_*.csv` |
| 2 · Clustering | Unsupervised phenotype clusters from NER output; frequency & relationship analysis | `clusters.csv`, `cluster_summary.csv`, `cluster_selection.csv`, `test_*pairs.csv` |
| 3 · RAG | Retrieval over our annotated notes + grounded extractive answers for the 30 queries and a live-query cell | `rag_answers.{csv,jsonl,md}`, `rag_answers_team_corpus_only.*` |

**Run:** `pip install -r requirements.txt` then *Kernel → Restart & Run All* (≈ 6 min on 4 CPU cores).
Read `docs/TECHNICAL_REPORT.md` for the reasoning.
""")

code(r"""
import os, sys, json, time, subprocess
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd
from IPython.display import display, Image, Markdown
pd.set_option("display.max_colwidth", 140, "display.width", 200)

from medbhasha import pipeline, corpus, analysis, cluster, run_all
from medbhasha.config import SEED, OUT, R1, R2, ENTITY_TYPES
from medbhasha.tokenizer import tokenize
print("repo root:", ROOT, "| seed:", SEED)
""")

md(r"""
## 0 · Data: our annotated corpus (Round 1, expanded)

The organisers asked us (after Round 1) to grow the corpus to ≈ 10 000 annotated tokens and to prioritise rare entity types.
We kept the 40 Round-1 notes (with 8 documented consistency fixes, `round1_deliverables/round1_fixes_log.tsv`) and added 192 new
team-authored synthetic notes (source and annotation method: `round1_deliverables/SOURCE_DECLARATION.md`).
Annotation sources are human-readable span markup under `data/round1/annotation_sources/`; `scripts/build_corpus.py` aligns them to tokens and writes the BIO CSV.
""")
code(r"""
notes, meta, raw = pipeline.load_train()
n_tok = sum(len(t) for t, _ in notes.values())
print(f"notes: {len(notes)}   tokens: {n_tok}   domains: {meta.domain.nunique()}")

# official validator (mandatory part of the submission)
r = subprocess.run([sys.executable, "resources/validate.py", "--input", str(R1 / "annotated_corpus.csv"),
                    "--scheme", "BIO", "--report", "round1_deliverables/validation_report.txt"],
                   capture_output=True, text=True)
print(r.stdout[-420:])
""")
code(r"""
def show(nid, notes=notes):
    toks, tags = notes[nid]
    out, i = [], 0
    for s, e, ty in corpus.spans_from_bio(tags):
        out += toks[i:s]
        out.append("[" + " ".join(toks[s:e]) + "]{" + ty + "}")
        i = e
    return " ".join(out + toks[i:])

for nid in ("n005", "n052", "n224"):
    print(nid, "→", show(nid), "\n")
""")
code(r"""
eda = run_all.stage_eda_round1()
print({k: eda[k] for k in ("notes", "tokens", "spans", "domains")}, eda["length_stats"])
display(eda["freq"])
""")
code(r"""
display(Image(str(OUT / "figures/eda_entity_frequency.png"), width=520))
display(Image(str(OUT / "figures/eda_cooccurrence.png"), width=560))
""")

md(r"""
## 1 · NER — entity extraction

**Model.** Linear-chain CRF (`sklearn-crfsuite`, L-BFGS, c1 = c2 = 0.05) over rich features: word / prefix / suffix / shape, ±2 context, a hand-written
bilingual medical **gazetteer** (`src/medbhasha/lexicon.py`, general medical knowledge — *not* mined from the test notes) and **note-section context**
(the header group — `নির্ণয়:` / `Dx:`, `ব্যবস্থাপত্র:` / `Rx:`, `ভাইটাল সাইন:` … — is a very strong cue for the entity type).
Why not a transformer? With ≈ 12 k annotated tokens a fine-tuned encoder over-fits, and pretrained Bengali encoders cannot be downloaded in an offline, exactly-reproducible run.

**Evaluation (on our own data).** Entity-level micro P/R/F1 with `seqeval`. Two protocols: random 5-fold by note, and the harder *domain-held-out* 5-fold
(whole clinical specialties unseen at training time — a better proxy for unseen diseases/drugs in the hidden test set). Both are optimistic relative to the hidden test set
because the corpus is synthetic and written by one author (see *Limitations*).
""")
code(r"""
t0 = time.time()
ner_eval = run_all.stage_ner_eval(fast=False)
print(f"{time.time()-t0:.0f}s")
display(ner_eval["cv"])
""")
code(r"""
display(ner_eval["per_type"].round(3))
display(ner_eval["learning_curve"])
display(Image(str(OUT / "figures/ner_per_type_f1.png"), width=520))
""")
md("### Train on the full corpus and predict the released (unlabeled) notes")
code(r"""
pred = run_all.stage_ner_predict()
tdf, qdf = pipeline.load_test()
print(len(tdf), "released notes ->", len(pred["span_rows"]), "predicted entity spans;",
      f"{len(pred['span_rows'])/len(tdf):.1f} per note")
display(pred["freq"])

def show_pred(nid):
    rows = [r for r in pred["tok_rows"] if r[0] == nid]
    toks, tags = [r[1] for r in rows], [r[2] for r in rows]
    out, i = [], 0
    for s, e, ty in corpus.spans_from_bio(tags):
        out += toks[i:s]; out.append("[" + " ".join(toks[s:e]) + "]{" + ty + "}"); i = e
    return " ".join(out + toks[i:])
print(show_pred("MB-0301"))
""")
code(r"""
# stage_ner_predict() already ran the official validator on the prediction file
print(open(OUT / "ner_predictions_validation_report.txt", encoding="utf-8").read())
""")

md(r"""
## 2 · Phenotype clustering (unsupervised)

Each note is represented **only by its predicted entities** (never by note id, file order or any label): (A) DIAGNOSIS mentions as character 2–4-gram TF-IDF
(robust to Bengali inflection/transliteration), (B) a typed bag of DIAGNOSIS / SYMPTOM / MEDICATION / TEST / PROCEDURE / BODY_SITE phrases (entity co-occurrence).
Both blocks are weighted equally, reduced with truncated SVD (64-d) and clustered. We compare k-means, Ward agglomerative and Louvain-on-a-kNN-graph and choose
**the algorithm and number of clusters by internal metrics only** (silhouette, Davies–Bouldin). No phenotype labels were used for selection.
""")
code(r"""
clu = run_all.stage_cluster(pred["span_rows"])
print("chosen:", clu["config"], "| internal scores:", {k: round(v, 3) for k, v in clu["scores"].items()})
print("sanity check vs. the diagnosis line printed in each note (NOT the hidden gold):", clu["proxy"])
display(Image(str(OUT / "figures/cluster_selection.png"), width=700))
""")
code(r"""
s = clu["summary"]
print("clusters:", len(s), "| sizes: min", s["size"].min(), "median", int(s["size"].median()), "max", s["size"].max())
display(s[["cluster", "size", "top_diagnoses", "top_medications"]].head(12))
""")
md("### Frequency and relationship analysis on the NER output")
code(r"""
top = pd.read_csv(OUT / 'test_top_entities.csv')
display(top.groupby('type').head(5).reset_index(drop=True))
print("Top diagnosis → medication associations (co-occurrence over notes, lift, PMI):")
display(pred["dx_med"].head(12))
print("Top diagnosis → symptom associations:")
display(pred["dx_sym"].head(10))
display(Image(str(OUT / "figures/test_cooccurrence.png"), width=560))
""")

md(r"""
## 3 · Retrieval-augmented question answering (no external API)

**Knowledge base.** Our annotated notes (gold spans) **+** the released notes annotated by *our own* NER model (predicted spans).
The queries ask for *patients* ("রোগীদের দেখান"), which only the released notes can answer; to respect the rule "retrieve from your own annotated corpus only"
we also write a strict **team-corpus-only** answer file (`rag_answers_team_corpus_only.*`). **Please pick the interpretation the organisers confirm** (see report §6.4).

**Retriever.** Spelling-skeleton + light Bengali stemming → BM25, a bilingual concept table (abbreviations, drug classes), negation/family-history-aware concept detection,
age/sex and numeric constraints (e.g. *HbA1c ৯-এর বেশি*). Notes satisfying **all** conditions rank first; otherwise the closest partial matches are shown and labelled.

**Generator.** Extractive and grounded: every sentence/entity in an answer is copied from a retrieved note and cited `[note-id]`; if nothing matches, the system says so instead of guessing.
""")
code(r"""
rag = run_all.stage_rag(pred["span_rows"])
res = rag["rag_answers"]["results"]
display(res[["query_id", "query_text", "detected_concepts", "n_matched_notes", "n_partial_notes"]])
""")
code(r"""
for qid in ("q01", "q10", "q19", "q22"):
    row = res[res.query_id == qid].iloc[0]
    display(Markdown(f"**{qid}: {row.query_text}**\n\n```\n{row.answer}\n```"))
""")
md("### Live query (finale) — type any Bengali question. The example uses the synonym *অ্যাজমা* = *হাঁপানি* and the *children* constraint.")
code(r"""
from medbhasha.ask import ask
R = rag["rag_answers"]["retriever"]
out = ask(R, "অ্যাজমা আছে এমন শিশুদের কী ওষুধ দেওয়া হয়েছে?")
print(out["answer"]); print("\nretrieved:", out["retrieved"])
""")
code(r"""
# strict "team corpus only" variant for the same query (kept for rule compliance)
R_team = rag["rag_answers_team_corpus_only"]["retriever"]
print(ask(R_team, "অ্যাজমা আছে এমন শিশুদের কী ওষুধ দেওয়া হয়েছে?")["answer"])
""")

md(r"""
## 4 · Reproducibility
Every output file is hashed in `outputs/run_manifest.json` (with library versions). Running the notebook (or `python scripts/run_pipeline.py`) twice, even with different
`PYTHONHASHSEED`, yields identical hashes — verified by `python scripts/check_reproducibility.py`.
""")
code(r"""
man = run_all.write_manifest()
print({k: man[k] for k in ("python", "seed")}); print(man["versions"])
for k, v in list(man["output_md5"].items())[:40]:
    print(f"{v}  {k}")
""")

md(r"""
## 5 · Limitations (honest summary)
* **Synthetic, single-author training data** → our cross-validated F1 (≈ 0.87) is an *upper bound*; the hidden gold may follow different span conventions (e.g. whether units belong to `TEST_RESULT`).
* No clinician has reviewed the annotations; no inter-annotator agreement exists.
* The RAG "generator" is extractive (templates + verbatim excerpts); fluent abstractive answers would need a local LLM, which cannot be shipped for an offline exact-reproducibility run.
* The concept table (`concepts.py`) was written with the public query set in view; unseen concepts fall back to BM25 + fuzzy stems (tested on unseen queries in the report).
* Clustering granularity (≈ 80–90 clusters) is chosen by internal metrics; the number of hidden phenotype classes is unknown.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = ROOT / "MedBhasha_Pipeline.ipynb"
nbf.write(nb, out)
print("wrote", out)
