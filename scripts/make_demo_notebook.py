#!/usr/bin/env python3
"""Generate MedBhasha_Demo.ipynb — a presentation-ready notebook with
beginner-friendly explanations at every step, designed to be run live
in front of judges.

Execute:
    python scripts/make_demo_notebook.py
    jupyter nbconvert --to notebook --execute --inplace MedBhasha_Demo.ipynb
"""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TITLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
md(r"""
# MedBhasha — Bengali Clinical NLP Pipeline
### Team JUSL Health Home · COMSYS Hackathon-8

---

**What this notebook does (run it top to bottom, ~6 min):**

| Stage | What happens | Key idea |
|:------|:-------------|:---------|
| **0 — Setup** | Load libraries, set seed for reproducibility | Everything is deterministic |
| **1 — Data** | Load our 232 annotated clinical notes | Show what the raw data looks like |
| **2 — Tokenization** | Break Bengali text into words | Handle Bengali-specific punctuation |
| **3 — NER** | Train a CRF to find medical entities | The core ML model |
| **4 — Evaluation** | Cross-validate, ablation study | Prove the model works |
| **5 — Prediction** | Run NER on the 30 unseen test notes | The deliverable |
| **6 — Clustering** | Group patients by disease profile | Unsupervised, entities-only |
| **7 — RAG** | Answer 30 Bengali medical queries | Offline, extractive, cited |
| **8 — Live Demo** | Ask any question interactively | The wow factor |
| **9 — Reproducibility** | Verify MD5 hashes match | Bit-for-bit identical runs |

**Constraints:** Fully offline (no internet), deterministic (seed 42), no external AI/LLM.
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 0 — SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
md(r"""
---
## Stage 0 — Setup

We import libraries and set seed = 42 so every run produces the **exact same output**.

> **Why seed 42?** Every random operation (fold shuffles, k-means init, SVD) uses this seed.
> Run the pipeline twice → identical MD5 hashes for every output file.
""")

code(r"""
import os, sys, json, time, subprocess
from pathlib import Path

# Find the project root
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd
from IPython.display import display, Image, Markdown, HTML
pd.set_option("display.max_colwidth", 140, "display.width", 200)

# Our modules
from medbhasha import pipeline, corpus, analysis, cluster, run_all
from medbhasha.config import SEED, OUT, R1, R2, ENTITY_TYPES
from medbhasha.tokenizer import tokenize
from medbhasha.lexicon import match as gaz_match, GAZ
from medbhasha.ner import section_context, featurize, NERModel

print(f"Project root: {ROOT}")
print(f"Seed: {SEED}")
print(f"Entity types: {ENTITY_TYPES}")
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 1 — DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
md(r"""
---
## Stage 1 — Our Annotated Corpus

We wrote **232 Bengali clinical notes** and annotated every word with BIO tags.

**What is BIO?**
- **B-TYPE** = this word **begins** an entity of that type
- **I-TYPE** = this word **continues** the entity
- **O** = this word is **not** part of any entity

**Our 10 entity types:** DIAGNOSIS, SYMPTOM, MEDICATION, DOSAGE, TEST, TEST_RESULT, BODY_SITE, PROCEDURE, TEMPORAL, NEGATION_UNCERTAINTY
""")

code(r"""
# Load the annotated corpus
notes, meta, raw = pipeline.load_train()
n_tok = sum(len(t) for t, _ in notes.values())
print(f"Total notes: {len(notes)}")
print(f"Total tokens: {n_tok}")
print(f"Clinical domains: {meta.domain.nunique()}")
print(f"Domains: {sorted(meta.domain.unique())}")
""")

md(r"""
### Let's look at a real annotated note

The function below shows a note with its entities highlighted in `[brackets]{TYPE}` format.
This is exactly how the NER model should learn to tag text.
""")

code(r"""
def show_note(nid, notes=notes):
    toks, tags = notes[nid]
    out, i = [], 0
    for s, e, ty in corpus.spans_from_bio(tags):
        out += toks[i:s]
        out.append(f"**[{' '.join(toks[s:e])}]{{{ty}}}**")
        i = e
    return " ".join(out + toks[i:])

# Show 3 example notes from different domains
for nid in ("n005", "n052", "n224"):
    domain = meta.loc[meta.note_id == nid, "domain"].values[0] if nid in meta.note_id.values else "unknown"
    display(Markdown(f"**{nid}** ({domain}):\n\n{show_note(nid)}"))
    print()
""")

md(r"""
### Corpus statistics — entity type distribution

How many entities of each type are in our corpus? This matters because rare types
(like PROCEDURE) are harder to learn.
""")

code(r"""
# Run EDA and show entity frequency
eda = run_all.stage_eda_round1()
print(f"Corpus: {eda['notes']} notes, {eda['tokens']} tokens, {eda['spans']} entity spans")
display(eda["freq"])
""")

code(r"""
# Show the entity frequency chart
display(Image(str(OUT / "figures/eda_entity_frequency.png"), width=520))
""")

code(r"""
# Validate with the official validator (mandatory)
r = subprocess.run([sys.executable, "resources/validate.py", "--input", str(R1 / "annotated_corpus.csv"),
                    "--scheme", "BIO", "--report", "round1_deliverables/validation_report.txt"],
                   capture_output=True, text=True)
print("Official validator output (last lines):")
print(r.stdout[-300:])
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 2 — TOKENIZATION DEMO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
md(r"""
---
## Stage 2 — Tokenization (How the computer reads text)

**Problem:** Computers can't "read" Bengali. The first step is breaking text into tokens (words).

**Why is this hard?**
- Bengali danda `।` must be split from words
- Compounds like `সর্দি-কাশি` (cold-cough) must stay together
- Blood pressure `১১০/৭০` must stay together
- Abbreviations `মি.গ্রা.` (mg) must stay together

Let's see our tokenizer in action:
""")

code(r"""
# Demo: tokenize a clinical sentence
examples = [
    "রোগীর গত দুই দিন ধরে বুকে ব্যথা ও শ্বাসকষ্ট আছে।",
    "BP ১১০/৭০, SpO2 ৯৫%, তাপমাত্রা ১০১°F।",
    "ব্যবস্থাপত্র: অ্যামোক্সিসিলিন ৫০০ মি.গ্রা. দিনে তিনবার।",
    "সর্দি-কাশি ও (ডায়াবেটিস) আছে।",
]

for text in examples:
    tokens = tokenize(text)
    print(f"Input:  {text}")
    print(f"Tokens: {tokens}")
    print(f"Count:  {len(tokens)} tokens")
    print()
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 2B — FEATURES DEMO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
md(r"""
---
## Stage 2b — Feature Engineering (How the CRF "sees" each token)

The CRF model can't read text directly. We describe each token with **~30+ features**.
These are the signals the model uses to decide "is this a drug name? a symptom? nothing?"

**Key features:**
1. **Word affixes** — first/last 1-4 characters (drug names often end in -মাইসিন, -প্রিল)
2. **Shape** — what kind of characters (all Bengali? digits? mixed?)
3. **Gazetteer** — is this word in our 762-term medical dictionary?
4. **Section context** — is this under "নির্ণয়:" (diagnosis) or "ব্যবস্থাপত্র:" (prescription)?
5. **Neighbours** — what are the ±2 surrounding words?

Let's see the actual features for a sample sentence:
""")

code(r"""
# Demo: show features for each token in a sentence
sample_tokens = tokenize("ব্যবস্থাপত্র: মেটফরমিন ৫০০ মি.গ্রা. দিনে দুইবার।")
sample_tags = ["O"] * len(sample_tokens)  # dummy tags

# Build section context
sec_ctx = section_context(sample_tokens)

# Featurize and show
features = featurize(sample_tokens, sec_ctx)
print(f"Sentence: {' '.join(sample_tokens)}\n")
for i, (tok, feat) in enumerate(zip(sample_tokens, features)):
    print(f"Token {i}: '{tok}'")
    # Show the most interesting features
    interesting = {k: v for k, v in feat.items() if not k.startswith(("prev2", "next2", "bigram"))}
    for k, v in sorted(interesting.items()):
        print(f"  {k:20s} = {v}")
    print()
""")

md(r"""
### The Medical Dictionary (Gazetteer) — 762 terms

We hand-wrote a dictionary of medical terms in Bengali and English.
When a token matches, the CRF gets a very strong hint about its entity type.
""")

code(r"""
# Demo: gazetteer matching
test_words = ["মেটফরমিন", "ডায়াবেটিস", "জ্বর", "HbA1c", "রোগী", "ইসিজি-তে", "বুকে"]
print(f"Gazetteer size: {sum(len(v) for v in GAZ.values())} terms across {len(GAZ)} entity types\n")
for w in test_words:
    result = gaz_match([w])
    match_str = result[0] if result[0] else "no match"
    print(f"  '{w}' → {match_str}")
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 3 — NER TRAINING + EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
md(r"""
---
## Stage 3 — NER Model Training & Evaluation

**Algorithm:** Linear-chain CRF (Conditional Random Field)

**Why CRF?**
- Works well on small data (~12k tokens) — transformers/BERT need millions
- No internet needed — no pretrained model to download
- Deterministic — same output every run
- Interpretable — we can see which features matter

**How it works (simplified):**
1. For each token, compute features (we just saw this)
2. The CRF learns weights: "if suffix=মাইসিন AND section=TREAT → probably MEDICATION"
3. It also learns transitions: "after B-MEDICATION, the next label is usually I-MEDICATION or B-DOSAGE"
4. At prediction time, the **Viterbi algorithm** finds the globally best label sequence

**Hyperparameters:** L-BFGS optimiser, L1 reg = L2 reg = 0.05, max 200 iterations.
""")

md(r"""
### Cross-Validation — Testing the model fairly

We use **5-fold cross-validation** (two protocols):

| Protocol | How | What it tests |
|----------|-----|---------------|
| **Random 5-fold** | Notes shuffled randomly into 5 groups | General accuracy |
| **Domain-held-out** | Entire clinical specialties held out (e.g., all Cardiology notes in one fold) | Can the model handle unseen diseases? |

Domain-held-out is harder and more realistic — the hidden test set may have diseases our training data doesn't cover.
""")

code(r"""
t0 = time.time()
ner_eval = run_all.stage_ner_eval(fast=False)
elapsed = time.time() - t0
print(f"Cross-validation completed in {elapsed:.0f} seconds\n")

print("=== Cross-Validation Results ===")
display(ner_eval["cv"])
""")

code(r"""
print("=== Per-Entity-Type F1 Scores ===")
display(ner_eval["per_type"].round(3))
""")

code(r"""
# F1 by entity type — visual
display(Image(str(OUT / "figures/ner_per_type_f1.png"), width=520))
""")

md(r"""
### Ablation Study — Which features matter most?

We remove one feature at a time and measure the F1 drop.
**Bigger drop = more important feature.**
""")

code(r"""
# Show learning curve (how performance changes with more training data)
display(ner_eval["learning_curve"])
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 4 — PREDICT TEST NOTES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
md(r"""
---
## Stage 4 — Predict Entities on the 30 Test Notes

Now we train on ALL 232 notes and run predictions on the 30 unseen notes from the organisers.
This is the actual deliverable — the entities the judges will evaluate.
""")

code(r"""
# Train on full corpus, predict test notes
pred = run_all.stage_ner_predict()
tdf, qdf = pipeline.load_test()
print(f"Test notes: {len(tdf)}")
print(f"Predicted entity spans: {len(pred['span_rows'])}")
print(f"Average spans per note: {len(pred['span_rows'])/len(tdf):.1f}")
print()
display(pred["freq"])
""")

md(r"""
### Let's see the model's predictions on a real test note

The model has never seen these notes during training. Let's check how it does:
""")

code(r"""
def show_pred(nid):
    rows = [r for r in pred["tok_rows"] if r[0] == nid]
    toks, tags = [r[1] for r in rows], [r[2] for r in rows]
    out, i = [], 0
    for s, e, ty in corpus.spans_from_bio(tags):
        out += toks[i:s]
        out.append(f"**[{' '.join(toks[s:e])}]{{{ty}}}**")
        i = e
    return " ".join(out + toks[i:])

# Show first 3 test notes with predictions
test_ids = sorted(set(r[0] for r in pred["tok_rows"]))[:3]
for nid in test_ids:
    display(Markdown(f"**{nid}:**\n\n{show_pred(nid)}"))
    print()
""")

code(r"""
# Official validator on predictions
print(open(OUT / "ner_predictions_validation_report.txt", encoding="utf-8").read())
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 5 — CLUSTERING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
md(r"""
---
## Stage 5 — Phenotype Clustering (Unsupervised)

**Goal:** Group the 30 test patients by disease profile — WITHOUT any labels.

**How it works (step by step):**

1. **Extract entities** from each note using our NER model
2. **Build two feature views** (using ONLY predicted entities, never raw text):
   - View A: Character 2-4-gram TF-IDF of DIAGNOSIS mentions
   - View B: Typed entity bag TF-IDF (MEDICATION_মেটফরমিন, SYMPTOM_ব্যথা, etc.)
3. **Combine + reduce** with SVD to 64 dimensions
4. **Try 3 algorithms** (K-means, Ward hierarchical, Louvain community detection) with many cluster counts
5. **Pick the best** by silhouette score + Davies-Bouldin index (**no labels used!**)

**Why character n-grams?** Bengali has many spelling variants: ডায়াবেটিস, ডায়াবেটিক, ডায়াবেটিস মেলিটাস.
Character chunks naturally overlap between variants, so they cluster together.
""")

code(r"""
clu = run_all.stage_cluster(pred["span_rows"])
print(f"Chosen algorithm + k: {clu['config']}")
print(f"Internal scores: {', '.join(f'{k}={v:.3f}' for k, v in clu['scores'].items())}")
print(f"Sanity check vs diagnosis headers (NOT used for selection): {clu['proxy']}")
""")

code(r"""
# Cluster selection chart — shows silhouette across all algorithm/k combinations
display(Image(str(OUT / "figures/cluster_selection.png"), width=700))
""")

code(r"""
# What's in each cluster?
s = clu["summary"]
print(f"Number of clusters: {len(s)}")
print(f"Cluster sizes: min={s['size'].min()}, median={int(s['size'].median())}, max={s['size'].max()}")
print()
display(s[["cluster", "size", "top_diagnoses", "top_medications"]].head(12))
""")

md(r"""
### Entity Frequency & Co-occurrence Analysis
""")

code(r"""
top = pd.read_csv(OUT / 'test_top_entities.csv')
print("Top entities by type:")
display(top.groupby('type').head(5).reset_index(drop=True))
""")

code(r"""
print("Top diagnosis → medication associations:")
display(pred["dx_med"].head(12))
""")

code(r"""
print("Top diagnosis → symptom associations:")
display(pred["dx_sym"].head(10))
display(Image(str(OUT / "figures/test_cooccurrence.png"), width=560))
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 6 — RAG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
md(r"""
---
## Stage 6 — Retrieval-Augmented Question Answering (RAG)

**Goal:** Answer 30 Bengali medical queries with cited evidence from the notes.

**How it works:**

1. **Parse the query** → extract medical concepts, demographic/numeric constraints, intent
2. **Match concepts** using a hand-written table of 73 concepts with 411 spelling variants
3. **Handle negation** at clause level ("has diabetes **but** no heart disease")
4. **Score each note** = 0.55 × concept_coverage + 0.25 × term_coverage + 0.20 × BM25
5. **Apply hard constraints** (age, sex, numeric thresholds) as ×0.3 penalties
6. **Generate extractive answer** — every word copied from real notes, with [note-id] citations

**Key features:**
- Spelling skeleton normalisation (collapses Bengali transliteration variants)
- Clause-level negation detection
- Primary diagnosis bonus (+0.10) to prefer actual patients over family history mentions
- Two knowledge bases: default (team + test notes) and strict team-corpus-only

**Zero hallucination by design:** Every claim is copied from a real note and cited. If nothing matches, it says "no results."
""")

code(r"""
# Run RAG on all 30 queries
rag = run_all.stage_rag(pred["span_rows"])
res = rag["rag_answers"]["results"]

print(f"Queries answered: {len(res)}")
print(f"Queries with matched notes: {(res.n_matched_notes > 0).sum()}")
print(f"Queries with zero results: {(res.n_matched_notes == 0).sum()}")
print()
display(res[["query_id", "query_text", "detected_concepts", "n_matched_notes", "n_partial_notes"]])
""")

md(r"""
### Sample answers — let's see what the system produces
""")

code(r"""
for qid in ("q01", "q10", "q19", "q22"):
    row = res[res.query_id == qid].iloc[0]
    display(Markdown(f"---\n### {qid}: {row.query_text}\n\n**Detected concepts:** {row.detected_concepts}\n\n**Matched notes:** {row.n_matched_notes}\n\n```\n{row.answer}\n```"))
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 7 — LIVE DEMO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
md(r"""
---
## Stage 7 — Live Interactive Demo

**Ask any Bengali medical question and get a grounded answer with citations.**

Change the query string below and re-run the cell to try different questions!

Example queries you can try:
- `"ডায়াবেটিস রোগীদের কী ওষুধ দেওয়া হয়েছে?"` (What medicines for diabetes patients?)
- `"অ্যাজমা আছে এমন শিশুদের কী ওষুধ দেওয়া হয়েছে?"` (Medicines for children with asthma?)
- `"উচ্চ রক্তচাপ আছে এমন বৃদ্ধ রোগীদের কী ওষুধ দেওয়া হয়?"` (Medicines for elderly hypertension patients?)
- `"HbA1c ৯-এর বেশি এমন রোগীদের কী চিকিৎসা?"` (Treatment for patients with HbA1c > 9?)
""")

code(r"""
from medbhasha.ask import ask

# The main retriever (team corpus + test notes with predicted entities)
R = rag["rag_answers"]["retriever"]

# === CHANGE THIS QUERY TO ASK ANYTHING ===
query = "অ্যাজমা আছে এমন শিশুদের কী ওষুধ দেওয়া হয়েছে?"

result = ask(R, query)
print(f"Query: {query}")
print(f"Retrieved notes: {result['retrieved']}")
print(f"\n{'='*60}")
print(result["answer"])
""")

md(r"""
### Team corpus only variant (strict rule interpretation)
""")

code(r"""
R_team = rag["rag_answers_team_corpus_only"]["retriever"]
result_strict = ask(R_team, "অ্যাজমা আছে এমন শিশুদের কী ওষুধ দেওয়া হয়েছে?")
print("STRICT (team corpus only):")
print(result_strict["answer"])
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 8 — REPRODUCIBILITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
md(r"""
---
## Stage 8 — Reproducibility Verification

Every output file is hashed with MD5. Running this pipeline twice (even with different `PYTHONHASHSEED`)
produces **identical** hashes — verified by `scripts/check_reproducibility.py`.
""")

code(r"""
man = run_all.write_manifest()
print(f"Python: {man['python']}, Seed: {man['seed']}")
print(f"Library versions: {man['versions']}")
print(f"\nOutput file MD5 hashes ({len(man['output_md5'])} files):")
for k, v in list(man["output_md5"].items())[:30]:
    print(f"  {v}  {k}")
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LIMITATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
md(r"""
---
## Honest Limitations

1. **Synthetic single-author corpus** → ~0.87 F1 is an upper bound; real doctor handwriting/style varies more
2. **No inter-annotator agreement** — one person annotated, no second reviewer
3. **Extractive RAG only** — answers are factual but not fluent; a local LLM would help but violates offline constraint
4. **Concept table was written with query set in view** — unseen concepts fall back to BM25 + fuzzy stems
5. **Clustering granularity** chosen by internal metrics; true number of phenotype classes is unknown

---

**Team JUSL Health Home** · 232 notes · 12,060 tokens · 10 entity types · 73 concepts · 762 gazetteer terms · seed 42
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WRITE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}
}
out = ROOT / "MedBhasha_Demo.ipynb"
nbf.write(nb, out)
print("wrote", out)
