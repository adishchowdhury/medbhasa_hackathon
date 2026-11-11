"""End-to-end orchestration. The notebook and ``scripts/run_pipeline.py`` call these stages.

Every stage is deterministic (fixed seeds, no network, no wall-clock dependence), so running the
pipeline twice gives byte-identical output files (checked by ``scripts/check_reproducibility.py``).
"""
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from . import analysis, cluster, corpus, pipeline
from .config import ENTITY_TYPES, OUT, R1, SEED
from .evaluation import cross_validate, entity_scores, make_folds, per_type_report
from .ner import NERModel
from .rag import Retriever
from .tokenizer import tokenize

FIG = OUT / "figures"


def _ensure():
    for d in (OUT, FIG):
        d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------------ Round 1 EDA
def stage_eda_round1():
    _ensure()
    notes, meta, raw = pipeline.load_train()
    rows = analysis.spans_from_notes(notes)
    freq = analysis.entity_frequency(rows)
    freq.to_csv(OUT / "eda_entity_frequency.csv", encoding="utf-8")
    analysis.top_entities(rows, 8).to_csv(OUT / "eda_top_entities.csv", index=False, encoding="utf-8")
    lengths = [len(t) for t, _ in notes.values()]
    stats = analysis.note_length_stats(lengths)
    co = analysis.cooccurrence(rows)
    co.to_csv(OUT / "eda_cooccurrence.csv", encoding="utf-8")
    assoc = analysis.association_pairs(rows, "DIAGNOSIS", "MEDICATION", min_count=2, top=20)
    assoc.to_csv(OUT / "eda_diagnosis_medication_pairs.csv", index=False, encoding="utf-8")
    analysis.plot_entity_frequency(freq, "Round-1 corpus: entity frequency", FIG / "eda_entity_frequency.png")
    analysis.plot_length_hist(lengths, "Round-1 corpus: note length", FIG / "eda_note_length.png")
    analysis.plot_cooccurrence(co, "Round-1 corpus: entity-type co-occurrence", FIG / "eda_cooccurrence.png")
    dom = meta["domain"].value_counts()
    dom.to_csv(OUT / "eda_domain_counts.csv", encoding="utf-8")
    n_tokens = sum(lengths)
    return {"notes": len(notes), "tokens": n_tokens, "length_stats": stats, "spans": len(rows),
            "domains": int(dom.size), "freq": freq, "cooccurrence": co, "pairs": assoc, "domain_counts": dom}


# ------------------------------------------------------------------------------------ NER evaluation
def stage_ner_eval(fast=False):
    """Cross-validated entity-level P/R/F1 (seqeval) on OUR corpus + ablations + learning curve."""
    _ensure()
    notes, meta, raw = pipeline.load_train()
    dom = dict(zip(meta["note_id"], meta["domain"]))
    results = []
    configs = [("full: lexicon + section features", dict()),
               ("no gazetteer lexicon", dict(use_lexicon=False)),
               ("no section features", dict(use_section=False)),
               ("neither (plain CRF)", dict(use_lexicon=False, use_section=False))]
    if fast:
        configs = configs[:1]
    report_full = None
    for scheme, groups in (("random 5-fold (by note)", None), ("domain-held-out 5-fold", dom)):
        for name, kw in configs:
            g, p, _ = cross_validate(notes, lambda kw=kw: NERModel(**kw), 5, groups)
            s = entity_scores(g, p)
            results.append({"evaluation": scheme, "model": name, **{k: round(v, 4) for k, v in s.items()}})
            if name.startswith("full") and groups is not None:
                report_full = per_type_report(g, p)
    res = pd.DataFrame(results)
    res.to_csv(OUT / "ner_cv_results.csv", index=False, encoding="utf-8")
    rep = pd.DataFrame(report_full).T.loc[ENTITY_TYPES + ["micro avg", "macro avg"]]
    rep.round(4).to_csv(OUT / "ner_cv_per_type_domain_heldout.csv", encoding="utf-8")
    analysis.plot_ner_per_type(report_full, "Per-type F1, domain-held-out CV", FIG / "ner_per_type_f1.png")

    # learning curve: fixed 40-note hold-out, growing training set
    ids = sorted(notes)
    rng = np.random.RandomState(SEED)
    perm = list(rng.permutation(ids))
    hold, pool = perm[:40], perm[40:]
    lc = []
    for n in ([20, 40, 80, 120, 160, len(pool)] if not fast else [40, len(pool)]):
        tr = pool[:n]
        m = NERModel().fit([notes[i][0] for i in tr], [notes[i][1] for i in tr])
        pr = m.predict([notes[i][0] for i in hold])
        s = entity_scores([notes[i][1] for i in hold], pr)
        n_tok = sum(len(notes[i][0]) for i in tr)
        lc.append({"train_notes": n, "train_tokens": n_tok, **{k: round(v, 4) for k, v in s.items()}})
    lc = pd.DataFrame(lc)
    lc.to_csv(OUT / "ner_learning_curve.csv", index=False, encoding="utf-8")
    return {"cv": res, "per_type": rep, "learning_curve": lc}


# ------------------------------------------------------------------------------------ NER predictions
def stage_ner_predict():
    _ensure()
    notes, meta, raw = pipeline.load_train()
    tdf, qdf = pipeline.load_test()
    model = pipeline.train_final_ner(notes)
    tok_rows, span_rows = pipeline.predict_notes(model, tdf)
    pipeline.write_ner_outputs(tok_rows, span_rows)
    # the prediction file must satisfy the official format validator too
    import subprocess
    from .config import ROOT
    v = subprocess.run([sys.executable, str(ROOT / "resources" / "validate.py"), "--input", str(OUT / "ner_predictions_bio.csv"),
                        "--scheme", "BIO", "--report", str(OUT / "ner_predictions_validation_report.txt")],
                       capture_output=True, text=True)
    if v.returncode != 0:
        raise RuntimeError("NER predictions failed validate.py:\n" + v.stdout[-500:])
    # analysis of the NER output
    srows = [(r[0], r[1], r[2]) for r in span_rows]
    freq = analysis.entity_frequency(srows)
    freq.to_csv(OUT / "test_entity_frequency.csv", encoding="utf-8")
    analysis.top_entities(srows, 15).to_csv(OUT / "test_top_entities.csv", index=False, encoding="utf-8")
    co = analysis.cooccurrence(srows)
    co.to_csv(OUT / "test_cooccurrence.csv", encoding="utf-8")
    dm = analysis.association_pairs(srows, "DIAGNOSIS", "MEDICATION", min_count=5, top=40)
    dm.to_csv(OUT / "test_diagnosis_medication_pairs.csv", index=False, encoding="utf-8")
    ds = analysis.association_pairs(srows, "DIAGNOSIS", "SYMPTOM", min_count=5, top=40)
    ds.to_csv(OUT / "test_diagnosis_symptom_pairs.csv", index=False, encoding="utf-8")
    analysis.plot_entity_frequency(freq, "Released notes (NER output): entity frequency", FIG / "test_entity_frequency.png")
    analysis.plot_cooccurrence(co, "Released notes (NER output): entity-type co-occurrence", FIG / "test_cooccurrence.png")
    n_tok = pd.Series([len(tokenize(t)) for t in tdf["note_text"]])
    analysis.plot_length_hist(n_tok, "Released notes: length", FIG / "test_note_length.png")
    return {"model": model, "tok_rows": tok_rows, "span_rows": span_rows, "freq": freq, "co": co,
            "dx_med": dm, "dx_sym": ds, "length_stats": analysis.note_length_stats(list(n_tok))}


# ------------------------------------------------------------------------------------ clustering
def stage_cluster(span_rows):
    _ensure()
    tdf, _ = pipeline.load_test()
    ids = list(tdf["note_id"])
    ents = pipeline.entities_by_note(span_rows)
    Z, _ = cluster.build_features(ents, ids, w_diag=1.0, w_bag=1.0, svd_dim=64, seed=SEED)
    labels, cfg, rows = cluster.select_and_cluster(Z)
    sel = pd.DataFrame(rows)
    sel.to_csv(OUT / "cluster_selection.csv", index=False, encoding="utf-8")
    # renumber clusters by size for readability (deterministic)
    order = pd.Series(labels).value_counts()
    remap = {old: new for new, old in enumerate(order.index)}
    labels = np.array([remap[x] for x in labels])
    pd.DataFrame({"note_id": ids, "cluster": labels}).to_csv(OUT / "clusters.csv", index=False)
    summ = pd.DataFrame(cluster.describe_clusters(labels, ids, ents))
    summ.to_csv(OUT / "cluster_summary.csv", index=False, encoding="utf-8")
    sc = cluster.internal_scores(Z, labels)
    analysis.plot_cluster_selection(rows, cfg, FIG / "cluster_selection.png")
    analysis.plot_cluster_map(Z, labels, FIG / "cluster_map.png")
    # sanity (NOT a score): agreement with the note's own 'diagnosis line', extracted by regex
    proxy = _proxy_diagnosis_labels(tdf)
    chk = _proxy_check(ids, labels, proxy)
    with open(OUT / "cluster_config.json", "w", encoding="utf-8") as f:
        json.dump({"chosen": cfg, "n_clusters": int(len(set(labels))), **sc, "proxy_check": chk,
                   "seed": SEED}, f, ensure_ascii=False, indent=2)
    return {"labels": labels, "ids": ids, "config": cfg, "scores": sc, "selection": sel, "summary": summ,
            "proxy": chk, "Z": Z}


def _proxy_diagnosis_labels(tdf) -> Dict[str, str]:
    import re
    pat = re.compile(r"(?:রোগ নির্ণয় করা হয়েছে|রোগ নির্ণয়|প্রাথমিক নির্ণয়|চূড়ান্ত নির্ণয়|নির্ণয়|ডায়াগনোসিস|Dx)\s*:\s*([^।]+)।")
    out = {}
    for nid, t in zip(tdf["note_id"], tdf["note_text"]):
        m = pat.search(t)
        if m:
            out[nid] = re.sub(r"\([^)]*\)", "", m.group(1)).strip()
    return out


def _proxy_check(ids, labels, proxy):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    idx = [i for i, n in enumerate(ids) if n in proxy]
    y = [proxy[ids[i]] for i in idx]
    l = [labels[i] for i in idx]
    return {"note": "sanity check against the diagnosis line printed in the note (regex), NOT the hidden gold",
            "notes_with_diagnosis_line": len(idx), "distinct_diagnosis_strings": len(set(y)),
            "NMI": round(float(normalized_mutual_info_score(y, l)), 4),
            "ARI": round(float(adjusted_rand_score(y, l)), 4)}


# ------------------------------------------------------------------------------------ RAG
def stage_rag(span_rows):
    _ensure()
    notes, meta, raw = pipeline.load_train()
    tdf, qdf = pipeline.load_test()
    outputs = {}
    for name, release in (("rag_answers", tdf), ("rag_answers_team_corpus_only", None)):
        docs = pipeline.build_rag_docs(notes, raw, release, span_rows)
        R = Retriever(docs)
        res = pipeline.run_queries(R, qdf)
        df = pd.DataFrame(res)
        df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(OUT / f"{name}.md", "w", encoding="utf-8") as f:
            f.write(f"# Answers ({'team corpus + released notes annotated by our NER' if release is not None else 'team annotated corpus ONLY'})\n\n")
            for r in res:
                f.write(f"## {r['query_id']}: {r['query_text']}\n\n")
                f.write(f"*Detected concepts:* {r['detected_concepts'] or '-'} · *notes satisfying all conditions:* {r['n_matched_notes']} · *partial matches shown:* {r['n_partial_notes']}\n\n")
                f.write(r["answer"].replace("\n", "  \n") + "\n\n")
        outputs[name] = {"retriever": R, "results": df}
    return outputs


# ------------------------------------------------------------------------------------ manifest
def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def write_manifest(extra: dict = None):
    files = sorted(p for p in OUT.glob("*") if p.is_file() and p.name != "run_manifest.json")
    from importlib import metadata
    vers = {}
    for pkg in ("numpy", "pandas", "scikit-learn", "scipy", "sklearn-crfsuite", "python-crfsuite", "seqeval", "networkx", "matplotlib"):
        try:
            vers[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            vers[pkg] = "n/a"
    man = {"python": sys.version.split()[0], "platform": platform.platform(), "seed": SEED, "versions": vers,
           "output_md5": {p.name: md5(p) for p in files}}
    if extra:
        man.update(extra)
    with open(OUT / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
    return man


def run_everything(fast=False, verbose=True):
    t0 = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    eda = stage_eda_round1(); log(f"[eda] {eda['notes']} notes, {eda['tokens']} tokens  ({time.time()-t0:.0f}s)")
    ner_eval = stage_ner_eval(fast=fast); log(f"[ner-eval] done ({time.time()-t0:.0f}s)")
    pred = stage_ner_predict(); log(f"[ner-predict] {len(pred['span_rows'])} spans ({time.time()-t0:.0f}s)")
    clu = stage_cluster(pred["span_rows"]); log(f"[cluster] {clu['config']} ({time.time()-t0:.0f}s)")
    rag = stage_rag(pred["span_rows"]); log(f"[rag] done ({time.time()-t0:.0f}s)")
    man = write_manifest()
    return {"eda": eda, "ner_eval": ner_eval, "pred": pred, "cluster": clu, "rag": rag, "manifest": man}
