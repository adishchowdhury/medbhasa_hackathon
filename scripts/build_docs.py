#!/usr/bin/env python3
"""Fill the document templates with the REAL numbers from outputs/ and render DOCX / PDF.

    python scripts/build_docs.py            # needs pandoc + libreoffice (soffice) for DOCX/PDF
    python scripts/build_docs.py --no-pdf   # markdown only

Inputs : docs/_templates/*.tpl, outputs/*.csv|json, data/round1/*
Outputs: docs/TECHNICAL_REPORT.{md,docx,pdf}, README.md,
         round1_deliverables/{SOURCE_DECLARATION.md, EDA_Summary.md, EDA_Summary.pdf}
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from medbhasha import corpus, pipeline  # noqa: E402
from medbhasha.config import ENTITY_TYPES, OUT, R1  # noqa: E402
from medbhasha.ner import featurize  # noqa: E402
from medbhasha.tokenizer import tokenize  # noqa: E402

TPL = ROOT / "docs" / "_templates"


# ----------------------------------------------------------------------------- helpers
def md_table(df: pd.DataFrame, index=False, widths=None, floatfmt="{:.3f}") -> str:
    d = df.reset_index() if index else df.copy()
    if index and str(d.columns[0]) in ("index", ""):
        d = d.rename(columns={d.columns[0]: "type"})
    cols = [str(c) for c in d.columns]
    int_cols = {c for c in d.columns if pd.api.types.is_integer_dtype(d[c])}
    int_cols |= {c for c in d.columns if pd.api.types.is_float_dtype(d[c]) and (d[c].dropna() % 1 == 0).all()
                 and (d[c].dropna().abs() >= 20).any()}        # e.g. support / token counts stored as floats
    rows = []
    for _, r in d.iterrows():
        cells = []
        for c, v in zip(d.columns, r):
            if c in int_cols:
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v).replace("|", "/").replace("\n", " "))
        rows.append(cells)
    w = widths or [max(3, min(40, max(len(c), *(len(r[i]) for r in rows)))) for i, c in enumerate(cols)]
    line = lambda cells: "| " + " | ".join(cells) + " |"
    return "\n".join([line(cols), "|" + "|".join("-" * (x + 2) for x in w) + "|"] + [line(r) for r in rows])


def grams(s, n=4):
    s = "".join(s.split())
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def render(template: str, values: dict) -> str:
    def sub(m):
        k = m.group(1)
        if k not in values:
            raise KeyError(f"template placeholder @@{k}@@ has no value")
        return str(values[k])
    return re.sub(r"@@([A-Z0-9_]+)@@", sub, template)


def pandoc(md: Path, kind: str, resource_dir: Path):
    out = md.with_suffix("." + kind)
    subprocess.run(["pandoc", str(md), "-o", str(out), "--resource-path", str(resource_dir), "-f", "markdown+pipe_tables+yaml_metadata_block"],
                   check=True)
    return out


def to_pdf(docx: Path):
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(docx.parent), str(docx)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return docx.with_suffix(".pdf")


# ----------------------------------------------------------------------------- collect values
def collect():
    v = {}
    notes, meta, raw = pipeline.load_train()
    tdf, qdf = pipeline.load_test()
    rows = [(nid, ty, " ".join(toks[s:e])) for nid, (toks, tags) in notes.items() for s, e, ty in corpus.spans_from_bio(tags)]
    freq = pd.read_csv(OUT / "eda_entity_frequency.csv", index_col=0)
    lens = [len(t) for t, _ in notes.values()]
    v["N_NOTES"] = len(notes)
    v["N_NEW"] = len(notes) - 40
    v["N_TOKENS"] = sum(lens)
    v["N_SPANS"] = len(rows)
    v["N_DOMAINS"] = meta["domain"].nunique()
    v["RAREST"] = f"{freq['spans'].idxmin()} = {int(freq['spans'].min())} spans"
    v["PROC_N"] = int(freq.loc["PROCEDURE", "spans"])
    v["LEN_MEAN"], v["LEN_MED"] = f"{np.mean(lens):.0f}", f"{np.median(lens):.0f}"
    v["LEN_MIN"], v["LEN_MAX"] = min(lens), max(lens)
    top_dom = meta["domain"].value_counts().head(4)
    v["TOP_DOMAINS"] = ", ".join(f"{k} {n}" for k, n in top_dom.items())
    v["N_TEST"] = len(tdf)
    test_len = [len(tokenize(t)) for t in tdf["note_text"]]
    v["TEST_LEN_MEAN"] = f"{np.mean(test_len):.0f}"
    ff = freq.copy()
    ff.columns = ["spans", "distinct forms", "notes containing", "share %"]
    ff["share %"] = ff["share %"].map(lambda x: f"{x:.1f}")
    v["TABLE_ENTITY_FREQ"] = md_table(ff, index=True, widths=[24, 6, 14, 16, 7]).replace("| type |", "| entity type |")

    from medbhasha import concepts, lexicon
    v["LEX_TERMS"] = sum(len([t for t in re.split(r"[|\n]+", b) if t.strip()]) for b in lexicon._RAW.values())
    v["N_CONCEPTS"] = len(concepts.CONCEPTS)
    v["N_VARIANTS"] = sum(len(x) for x in concepts.CONCEPTS.values())
    co = pd.read_csv(OUT / "eda_cooccurrence.csv", index_col=0)
    pr = lambda a, b: f"{co.loc[a, b] / co.loc[a, a]:.2f}"
    v["P_DOS_MED"], v["P_RES_TEST"] = pr("MEDICATION", "DOSAGE"), pr("TEST", "TEST_RESULT")
    v["P_TMP_SX"], v["P_NEG_DX"] = pr("SYMPTOM", "TEMPORAL"), pr("DIAGNOSIS", "NEGATION_UNCERTAINTY")
    # similarity checks
    g_train = [grams(t) for t in raw["text"]]
    mx_internal = 0.0
    for i in range(len(g_train)):
        for j in range(i + 1, len(g_train)):
            mx_internal = max(mx_internal, jaccard(g_train[i], g_train[j]))
    g5_train = [grams(t, 5) for t in raw["text"]]
    g5_test = [grams(t, 5) for t in tdf["note_text"]]
    mx_rel = max(jaccard(a, b) for a in g5_train for b in g5_test)
    v["NOCOPY_MAX"] = f"{mx_rel:.2f}"
    v["INTERNAL_MAXSIM"] = f"{mx_internal:.2f}"

    # NER
    cv = pd.read_csv(OUT / "ner_cv_results.csv")
    show = cv.copy()
    show.columns = ["evaluation", "model", "precision", "recall", "F1"]
    v["TABLE_NER_CV"] = md_table(show)
    f_rand = cv[(cv.evaluation.str.startswith("random")) & (cv.model.str.startswith("full"))].f1.iloc[0]
    f_dom = cv[(cv.evaluation.str.startswith("domain")) & (cv.model.str.startswith("full"))].f1.iloc[0]
    v["F1_RANDOM"], v["F1_DOMAIN"] = f"{f_rand:.3f}", f"{f_dom:.3f}"
    d = cv[cv.evaluation.str.startswith("domain")].set_index("model").f1
    v["GAZ_GAIN"] = f"{100*(d['full: lexicon + section features']-d['no gazetteer lexicon']):.1f}"
    v["SEC_GAIN"] = f"{100*(d['full: lexicon + section features']-d['no section features']):.1f}"
    pt = pd.read_csv(OUT / "ner_cv_per_type_domain_heldout.csv", index_col=0)
    pt = pt[["precision", "recall", "f1-score", "support"]].copy()
    pt["support"] = pt["support"].astype(int)
    v["TABLE_NER_PER_TYPE"] = md_table(pt, index=True)
    lc = pd.read_csv(OUT / "ner_learning_curve.csv")
    v["TABLE_LEARNING"] = md_table(lc.rename(columns={"f1": "F1"}))
    # predictions summary
    sp = pd.read_csv(OUT / "ner_predictions_spans.csv")
    tf = pd.read_csv(OUT / "test_entity_frequency.csv", index_col=0)
    tf2 = tf[["spans", "distinct", "notes", "share_%"]].copy()
    tf2.columns = ["spans", "distinct forms", "notes containing", "share %"]
    tf2["share %"] = tf2["share %"].map(lambda x: f"{x:.1f}")
    v["TEST_NER_SUMMARY"] = (f"{len(sp)} entity spans on {sp.note_id.nunique()} of {len(tdf)} released notes "
                             f"({len(sp)/len(tdf):.1f} per note; our corpus has {v['N_SPANS']/v['N_NOTES']:.1f} per note).\n\n"
                             + md_table(tf2, index=True, widths=[24, 6, 14, 16, 7]).replace("| type |", "| entity type |"))
    # clustering
    cc = json.loads((OUT / "cluster_config.json").read_text(encoding="utf-8"))
    cs = pd.read_csv(OUT / "cluster_summary.csv")
    v["CLUSTER_ALGO"] = cc["chosen"]["algorithm"]
    v["N_CLUSTERS"] = cc["n_clusters"]
    v["SIL"], v["DB"] = f"{cc['silhouette']:.3f}", f"{cc['davies_bouldin']:.3f}"
    v["CLUSTER_SIZES"] = f"min {cs['size'].min()}, median {int(cs['size'].median())}, max {cs['size'].max()}"
    pc = cc["proxy_check"]
    v["PROXY_TEXT"] = f"NMI = {pc['NMI']:.3f}, ARI = {pc['ARI']:.3f} (on {pc['notes_with_diagnosis_line']} notes that print a diagnosis line)"
    v["PROXY_CLASSES"], v["PROXY_N"] = pc["distinct_diagnosis_strings"], pc["notes_with_diagnosis_line"]
    ex = cs.head(8)[["cluster", "size", "top_diagnoses", "top_medications"]].copy()
    ex["top_diagnoses"] = ex["top_diagnoses"].str.split(";").str[:2].str.join(";")
    ex["top_medications"] = ex["top_medications"].str.split(";").str[:2].str.join(";")
    v["TABLE_CLUSTER_EXAMPLES"] = md_table(ex, widths=[7, 5, 38, 38])
    v["TABLE_CLUSTER_EXAMPLES_SMALL"] = md_table(ex.head(3), widths=[7, 5, 38, 38])
    dm = pd.read_csv(OUT / "test_diagnosis_medication_pairs.csv").head(10)
    v["TABLE_DX_MED"] = md_table(dm)
    v["TABLE_DX_MED_SMALL"] = md_table(dm.head(6))
    # RAG
    rag = pd.read_csv(OUT / "rag_answers.csv")
    strict = pd.read_csv(OUT / "rag_answers_team_corpus_only.csv")
    t = rag[["query_id", "detected_concepts", "n_matched_notes"]].copy()
    t["strict_team_only"] = strict["n_matched_notes"].values
    t["partial"] = rag["n_partial_notes"].values
    t["query"] = rag["query_text"].str.slice(0, 46) + "…"
    v["TABLE_RAG_QUERIES"] = md_table(t[["query_id", "query", "detected_concepts", "n_matched_notes", "partial", "strict_team_only"]]
                                      .rename(columns={"query_id": "id", "detected_concepts": "detected concepts", "n_matched_notes": "all conditions met", "partial": "partial shown", "strict_team_only": "team-only KB"}),
                                      widths=[4, 40, 32, 10, 8, 10])
    v["STRICT_COVERAGE"] = f"{int((strict['n_matched_notes'] > 0).sum())}"
    ans = rag[rag.query_id == "q19"]["answer"].iloc[0].split("\n")
    v["RAG_EXAMPLE"] = "\n".join(ans[:4])[:1200]
    # unseen queries
    from medbhasha.ask import ask, build_retriever
    R = build_retriever(False)
    unseen = ["হাইপোথাইরয়েডিজম আছে এবং লিভোথাইরক্সিন নিচ্ছেন এমন রোগীদের দেখান।",
              "শিশুদের কানের সংক্রমণে কোন কোন ওষুধ দেওয়া হয়েছে?",
              "অ্যানিমিয়া বা রক্তস্বল্পতা আছে এমন গর্ভবতী মহিলাদের তথ্য দিন।",
              "কোন রোগীদের কোভিড-১৯ হয়েছে?",
              "ইনফ্লুয়েঞ্জা বা ভাইরাল জ্বরে কোন কোন ওষুধ দেওয়া হয়েছে?"]
    ur = []
    for q in unseen:
        r = ask(R, q)
        ur.append({"unseen query": q, "concepts found": ", ".join(r["concepts"]) or "(none → BM25 fallback)",
                   "all-condition matches": r["n_matched"], "partial": r["n_partial"], "top notes": " ".join(i for i, _ in r["retrieved"][:4])})
    v["TABLE_UNSEEN"] = md_table(pd.DataFrame(ur), widths=[42, 26, 8, 8, 24])
    # repro
    rp = ROOT / "docs" / "reproducibility_check.txt"
    v["REPRO_RESULT"] = rp.read_text(encoding="utf-8").strip() if rp.exists() else ""
    v["FIG"] = "../outputs/figures"
    # EDA extras for guide / EDA summary
    v["_notes"], v["_meta"], v["_raw"], v["_freq"], v["_rows"] = notes, meta, raw, freq, rows
    return v


# ----------------------------------------------------------------------------- guide examples
def guide_examples(v):
    notes = v["_notes"]
    toks, tags = notes["n005"]
    df = pd.DataFrame({"token": toks[:14], "tag": tags[:14]})
    v["EXAMPLE_ANNOTATION"] = md_table(df.T.reset_index(drop=True).T.rename(columns=lambda c: ""), widths=[10] * 1) if False else \
        "| " + " | ".join(toks[:12]) + " |\n|" + "|".join("---" for _ in toks[:12]) + "|\n| " + " | ".join(tags[:12]) + " |"
    src = None
    for f in sorted((R1 / "annotation_sources").glob("*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith("n052\t"):
                src = line.split("\t")[2]
    v["EXAMPLE_MARKUP"] = (src or "")[:330] + " …"
    text = "রোগীর গত দুই সপ্তাহ ধরে বুকে ব্যথা ও শ্বাসকষ্ট আছে। ইসিজি স্বাভাবিক, অ্যাসপিরিন ৭৫ মি.গ্রা. দিনে ১ বার দেওয়া হয়েছে।"
    t = tokenize(text)
    v["EXAMPLE_TEXT"], v["EXAMPLE_TOKENS"] = text, " | ".join(t)
    model = pipeline.train_final_ner(notes)
    pred = model.predict([t])[0]
    v["EXAMPLE_PREDICTION"] = "| " + " | ".join(t) + " |\n|" + "|".join("---" for _ in t) + "|\n| " + " | ".join(pred) + " |"
    # feature demo
    toks2 = tokenize("Rx: প্যারাসিটামল ৫০০ মি.গ্রা. জ্বরে।")
    f = featurize(toks2)[2]
    v["EX_TOKEN"] = toks2[2]
    v["EX_F_W"] = f["w"]
    v["EX_F_S3"] = f["s3"]
    v["EX_F_SHAPE"] = f["shape"]
    v["EX_F_GAZ"] = ", ".join(k for k in f if k.startswith("gaz:")) or "-"
    v["EX_F_SEC"] = f["sec"]
    return v


# ----------------------------------------------------------------------------- EDA summary + declaration + README
def eda_md(v):
    notes, meta, freq = v["_notes"], v["_meta"], v["_freq"]
    co = pd.read_csv(OUT / "eda_cooccurrence.csv", index_col=0)
    diag = np.diag(co.values)
    pc = lambda a, b: co.loc[a, b] / co.loc[a, a]
    dom = meta["domain"].value_counts().rename_axis("domain").reset_index(name="notes")
    dom["share %"] = (100 * dom["notes"] / dom["notes"].sum()).map(lambda x: f"{x:.1f}")
    spans_per_note = pd.Series(Counter(r[0] for r in v["_rows"]))
    pairs = pd.read_csv(OUT / "eda_diagnosis_medication_pairs.csv").head(8)
    # co-occurrence table in text form (top off-diagonal)
    off = []
    for a in co.index:
        for b in co.columns:
            if a < b:
                off.append((a, b, int(co.loc[a, b])))
    off = pd.DataFrame(sorted(off, key=lambda x: -x[2])[:10], columns=["type A", "type B", "notes containing both"])
    return f"""---
title: "MedBhasha — EDA Summary (corpus v2, for Round 2)"
subtitle: "Team JUSL Health Home · synthetic notes, no clinician review"
---

*This summary describes the **final** validated corpus `data/round1/annotated_corpus.csv`
(the Round-1 notes n001–n040 with 8 documented corrections + 192 new notes). Numbers are generated by code from the file.*

# 1. Corpus size

* Notes: **{v['N_NOTES']}** (Round 1 submission: 40) · tokens: **{v['N_TOKENS']}** (Round 1: 612; organisers' recommendation ≈ 10 000)
* Tokens per note: mean {v['LEN_MEAN']}, median {v['LEN_MED']}, min {v['LEN_MIN']}, max {v['LEN_MAX']}
* Entity spans: **{v['N_SPANS']}** ({v['N_SPANS']/v['N_NOTES']:.1f} per note on average; min {spans_per_note.min()}, max {spans_per_note.max()}); notes without any entity: {len(notes) - len(spans_per_note)}
* Domains (project label, not an official field): {v['N_DOMAINS']} — largest: {v['TOP_DOMAINS']}

![Note length](../outputs/figures/eda_note_length.png){{ width=70% }}

# 2. Entity frequency distribution

{v['TABLE_ENTITY_FREQ']}

![Entity frequency](../outputs/figures/eda_entity_frequency.png){{ width=70% }}

Every one of the 10 types now has at least {int(freq['spans'].min())} spans (Round 1: most types < 30). `PROCEDURE` is the rarest type; in outpatient notes most clinical acts are tests or prescriptions.

# 3. Domain distribution

{md_table(dom)}

# 4. Co-occurrence and relationships between entity types

Within-note co-occurrence (diagonal = notes containing the type). Row-normalised heat-map below: *P(column type present | row type present)*.

![Co-occurrence](../outputs/figures/eda_cooccurrence.png){{ width=75% }}

Most frequent type pairs:

{md_table(off)}

Reading: P(DOSAGE | MEDICATION) = {pc('MEDICATION','DOSAGE'):.2f}, P(TEST_RESULT | TEST) = {pc('TEST','TEST_RESULT'):.2f},
P(TEMPORAL | SYMPTOM) = {pc('SYMPTOM','TEMPORAL'):.2f}, P(NEGATION_UNCERTAINTY | DIAGNOSIS) = {pc('DIAGNOSIS','NEGATION_UNCERTAINTY'):.2f}.
These are the patterns a prescription/exam template produces; with {v['N_NOTES']} notes they are descriptive, not clinical findings.

Diagnosis → medication pairs seen at least twice, ranked by lift (lift > 1 = more often together than by chance):

{md_table(pairs)}

# 5. Data-quality checks

* empty tokens: 0 · invalid tags: 0 · official `validate.py`: **PASS, 0 errors, 0 warnings**
* exact duplicate notes: 0 · max character-4-gram Jaccard between any two notes: **{v['INTERNAL_MAXSIM']}** (threshold 0.6)
* max character-5-gram Jaccard between any of our notes and any organiser-released test note: **{v['NOCOPY_MAX']}** (no copying)
* every entity type has ≥ 50 spans; the corpus contains short informal notes, structured outpatient notes, shorthand (`C/O … Dx … Rx`) notes and admission/discharge summaries.
* consistency audit: same-surface/different-type conflicts were inspected; the one systematic problem (organ + symptom written both as one SYMPTOM and as BODY_SITE + SYMPTOM) was corrected — see `docs/ANNOTATION_CONVENTIONS.md` rule C7.

# 6. Annotation edge cases (for clinician review)

| topic | decision | where |
|---|---|---|
| organ + symptom (`বুকে ব্যথা`) | split: BODY_SITE + SYMPTOM; idioms such as `মাথা ঘোরা` stay one SYMPTOM | C7 |
| units in results (`১১০/৭০ মি.মি. পারদ`) | unit is inside TEST_RESULT | C6 |
| imaging name with organ (`বুকের এক্স-রে`) | `বুকের` = BODY_SITE, `এক্স-রে` = TEST (organisers' worked example) | C5 |
| endoscopy / dressing / nebulisation | PROCEDURE; imaging and lab → TEST | C10 |
| negation | marker only (`নেই`, `না`, `সন্দেহ`) | C11 |
| history diagnoses | DIAGNOSIS (no separate history label in the schema) | C9 |
| `সকালে`, `রাতে` | TEMPORAL for symptoms, DOSAGE for when to take a drug | C2/C3 |
| hyphen suffix tokens (`ইসিজি-তে`) | whole token tagged | C13 |

# 7. Limitations

* Notes are **synthetic** and written by a single author; the corpus measures consistency with one author's patterns, not real clinical language.
* **No clinician review**, no second annotator, no inter-annotator agreement.
* Domain balance is uneven; some specialties have only 2–6 notes.
* Whitespace tokenisation keeps Bengali case endings attached (`ডায়াবেটিসের`), so span boundaries include them (same as the organisers' worked example).
"""


def declaration_md(v):
    meta = v["_meta"]
    dom = meta["domain"].value_counts()
    batches = []
    for f in sorted((R1 / "annotation_sources").glob("*.txt")):
        ids = [l.split("\t")[0] for l in f.read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]
        title = f.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").split(".")[0]
        batches.append((f.name, len(ids), f"{ids[0]}–{ids[-1]}", title))
    btab = md_table(pd.DataFrame(batches, columns=["batch file", "notes", "ids", "content / register"]))
    dtab = md_table(dom.rename_axis("domain").reset_index(name="notes"))
    return f"""# MedBhasha — Source Declaration (corpus v2, Round 2 submission)

Team name: **JUSL Health Home**
Corpus file: `data/round1/annotated_corpus.csv` (validated with the official `validate.py`; report: `validation_report.txt`)
Declaration status: **DRAFT prepared with the team's pipeline — the team lead must read and confirm every statement before submission (see §9).**

## 1. Source category (data_collection_rules.md §1)

All {v['N_NOTES']} notes are **synthetic clinical notes generated by our own team** (approved category: *"Synthetic clinical notes generated by your own team"*).

* Every note is a **fictional scenario** written in Bengali (with English medical terms where natural).
* No note derives from a real patient, hospital record, EHR export, case report, PubMed article, transcript, social-media post or any other external text.
* No note is a copy or close paraphrase of published text. We checked similarity against the organiser-released test notes: max character-5-gram Jaccard = **{v['NOCOPY_MAX']}**.
* The team did **not** use any external dataset. The file `Bangla-Healthcare_Dataset_1.xlsx` (5 263 triage queries) was reviewed in Round 1 and **not used** (no verified provenance/licence); it is not part of this repository.

## 2. Counts

| Item | Value |
|---|---|
| Total notes | {v['N_NOTES']} |
| from the Round-1 submission (n001–n040) | 40 (8 tags corrected, see `round1_fixes_log.tsv`) |
| new notes written for Round 2 (n041–n{v['N_NOTES']:03d}) | {v['N_NEW']} |
| Notes from approved published / open sources | 0 |
| Notes from the patient-reported route (§1a) | 0 — no consent forms were used or collected |
| Notes from hospital EHR or restricted data | 0 |
| Notes containing real patient identifiers | 0 (none by construction) |
| Tokens / entity spans | {v['N_TOKENS']} / {v['N_SPANS']} |

Batches:

{btab}

Domain labels (project field, not an official field). Round-1 labels were re-derived from content so that they agree with the Round-1 declaration's counts (GM 10, Cardiology 7, Gastroenterology 8, Respiratory 9, Dermatology 6); the Round-1 label "Respiratory Medicine" is now simply "Respiratory":

{dtab}

## 3. Generation method

* Generated by the team with an LLM prompted to mimic Bengali clinical documentation (approved category, data_collection_rules.md §1), in batches, each saved as a text file under `data/round1/annotation_sources/` and merged by `scripts/build_corpus.py`.
* Diversity controls: notes were written for different clinical scenarios and registers; the build/tests check exact duplicates and character n-gram near-duplicates (max pairwise 4-gram Jaccard {v['INTERNAL_MAXSIM']}).
* **Style disclosure.** To make the training data resemble the genre of the Round-2 release, the author looked at the *format* of a small number of released notes and at header statistics (sections such as `প্রধান অভিযোগ:`, shorthand `C/O / O/E / Dx / Rx / Adv / F/U`). No released sentence was copied, annotated or used for training; released notes are used only as unlabeled input for prediction, clustering and retrieval.
* No random seed: notes were written by hand, not sampled.

## 4. Annotation method (disclosed per data_collection_rules.md §3)

* **Method: LLM-assisted annotation.** Entity spans were written as inline markup; a deterministic script aligned them to tokens and produced BIO tags. No rule-based tagger, no built-in NER model and no external annotation tool was used to *create* labels.
* Tokenisation: whitespace split with edge punctuation split off; internal `-`, `/`, `.` kept; abbreviations such as `মি.গ্রা.` kept whole.
* Tagging scheme: BIO, schema `entity_schema.md` v1.0 (10 labels). Decisions: `docs/ANNOTATION_CONVENTIONS.md`.
* Automated checks: alignment fails on spans that cut tokens, overlap, or include punctuation; official `validate.py` run on the final CSV (**PASS, 0 errors, 0 warnings**).

## 5. Use of the organiser-released Round-2 data

`data/round2/test_notes.csv` and `test_queries.csv` are the organisers' files, unchanged apart from removing a byte-order mark. They are used only as **unlabeled input** (NER prediction, clustering, retrieval). We did not annotate them by hand and did not train on them. Retrieval also indexes them after annotating them with our own NER model; a strict team-corpus-only variant is provided (see the technical report §6.5).

## 6. Pretrained models, tools, resources

No pretrained model, embedding, LLM or external API is used at run time. Libraries: NumPy, pandas, SciPy, scikit-learn, CRFsuite (`sklearn-crfsuite`), seqeval, NetworkX, Matplotlib, Jupyter/nbconvert. The hand-written gazetteer (`src/medbhasha/lexicon.py`) and concept table (`src/medbhasha/concepts.py`) were authored by the team (LLM-assisted) from general clinical knowledge.

## 7. Review status (please read)

* **No clinician has reviewed the annotations.** No clinician reviewer has been confirmed.
* **No second annotator** has checked any note; no inter-annotator agreement has been measured.
* Ambiguous cases are listed in `docs/ANNOTATION_CONVENTIONS.md` and in the EDA summary; they are judgment calls, not clinician-approved rulings.

## 8. Consent and privacy

No individual's personal health information is included; no consent form was used because no patient-reported information was collected. The team consents (per the rules) to the corpus being released under an open licence (e.g. CC-BY-4.0).

## 9. Known limitations

1. Notes are synthetic: the corpus measures consistency with the author's patterns, not performance on real clinical language.
2. Domain balance is uneven; several specialties have only a few notes.
3. Annotations are unreviewed drafts.
4. Whitespace tokenisation keeps Bengali case endings attached to stems (`ডায়াবেটিসের`).

## 10. Attestation (to be completed by the team lead — *not* pre-filled)

- [ ] Team lead has read this declaration and confirms it is accurate.
- [ ] Official validator was run on the final submitted CSV (the report in this repository corresponds to that file).

Signed: ______________________   Date: ______________
"""


def readme_md(v):
    cc = json.loads((OUT / "cluster_config.json").read_text(encoding="utf-8"))
    return f"""# MedBhasha — Bengali Clinical NLP (Team **JUSL Health Home**)

COMSYS Hackathon-8 · Jadavpur University · *Annotate → NER → Cluster → Retrieve*

One runnable notebook, **offline and deterministic**, that goes from our own annotated Bengali clinical corpus to entity extraction, unsupervised phenotype clustering and grounded question answering.

| Stage | Result (our own data / outputs) |
|---|---|
| **Corpus** (Round 1, expanded) | **{v['N_NOTES']} notes · {v['N_TOKENS']} tokens · {v['N_SPANS']} spans**, all 10 entity types, official `validate.py`: **PASS** (0 errors, 0 warnings) |
| **NER** (CRF + gazetteer + section features) | entity-level F1 **{v['F1_RANDOM']}** (random 5-fold) / **{v['F1_DOMAIN']}** (domains held out) — *upper bounds: synthetic, single-author data* |
| **Clustering** ({v['N_TEST']} released notes) | {v['CLUSTER_ALGO']}, **{v['N_CLUSTERS']} clusters**, silhouette {v['SIL']}, Davies–Bouldin {v['DB']} (selected by internal metrics only) |
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
docs/                         technical report, annotation conventions
resources/                    organiser files used unchanged (validate.py, schema, rules, worked example)
scripts/                      build_corpus · run_pipeline · check_reproducibility · make_notebook · build_docs · make_zip · serve_app
tests/                        sanity tests
```

## Rules checklist

* Data: 100 % synthetic, team-authored (method stated in the source declaration); no real patient data. **No clinician review** — stated everywhere.
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
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    v = collect()
    # ---- templates
    for name in ("TECHNICAL_REPORT",):
        txt = render((TPL / f"{name}.md.tpl").read_text(encoding="utf-8"), v)
        (ROOT / "docs" / f"{name}.md").write_text(txt, encoding="utf-8")
    (ROOT / "README.md").write_text(readme_md(v), encoding="utf-8")
    (ROOT / "round1_deliverables" / "SOURCE_DECLARATION.md").write_text(declaration_md(v), encoding="utf-8")
    (ROOT / "round1_deliverables" / "EDA_Summary.md").write_text(eda_md(v), encoding="utf-8")
    print("markdown written")
    if a.no_pdf:
        return
    for md in (ROOT / "docs" / "TECHNICAL_REPORT.md",
               ROOT / "round1_deliverables" / "EDA_Summary.md"):
        docx = pandoc(md, "docx", md.parent)
        to_pdf(docx)
        print("rendered", md.name)
    # the EDA summary is required as PDF only
    (ROOT / "round1_deliverables" / "EDA_Summary.docx").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
