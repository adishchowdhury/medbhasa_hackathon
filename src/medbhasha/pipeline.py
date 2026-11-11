"""Glue code shared by the notebook and the command-line runner."""
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from . import corpus
from .config import ENTITY_TYPES, OUT, R1, R2
from .ner import NERModel
from .tokenizer import tokenize_with_offsets


def load_train():
    """Training corpus: id -> (tokens, tags); plus metadata (domain) and raw text."""
    notes = corpus.read_bio_csv(R1 / "annotated_corpus.csv")
    meta = pd.read_csv(R1 / "note_metadata.csv", encoding="utf-8")
    raw = pd.read_csv(R1 / "notes_raw.csv", encoding="utf-8")
    return notes, meta, raw


def load_test():
    notes = pd.read_csv(R2 / "test_notes.csv", encoding="utf-8")
    queries = pd.read_csv(R2 / "test_queries.csv", encoding="utf-8")
    return notes, queries


def train_final_ner(notes: Dict[str, Tuple[List[str], List[str]]]) -> NERModel:
    ids = sorted(notes)
    return NERModel().fit([notes[i][0] for i in ids], [notes[i][1] for i in ids])


def predict_notes(model: NERModel, df: pd.DataFrame, id_col="note_id", text_col="note_text"):
    """Run NER on a dataframe of notes.

    Returns (token_rows, span_rows):
      token_rows: [(note_id, token, tag, char_start, char_end)]
      span_rows : [(note_id, entity_type, text, char_start, char_end, tok_start, tok_end)]
    """
    tok_rows, span_rows = [], []
    all_toks = [tokenize_with_offsets(t) for t in df[text_col]]
    preds = model.predict([[x[0] for x in toks] for toks in all_toks])
    for nid, text, toks, tags in zip(df[id_col], df[text_col], all_toks, preds):
        for (tok, s, e), g in zip(toks, tags):
            tok_rows.append((nid, tok, g, s, e))
        for s, e, ty in corpus.spans_from_bio(tags):
            cs, ce = toks[s][1], toks[e - 1][2]
            span_rows.append((nid, ty, text[cs:ce], cs, ce, s, e))
    return tok_rows, span_rows


def write_ner_outputs(tok_rows, span_rows, out_dir: Path = OUT):
    out_dir.mkdir(parents=True, exist_ok=True)
    # 1) strict 3-column BIO file (same layout as the Round-1 corpus; validate.py-compatible)
    with open(out_dir / "ner_predictions_bio.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["note_id", "token", "tag"])
        for nid, tok, tag, _, _ in tok_rows:
            w.writerow([nid, tok, tag])
    # 2) span file with character offsets into note_text
    with open(out_dir / "ner_predictions_spans.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["note_id", "entity_type", "text", "char_start", "char_end", "token_start", "token_end"])
        w.writerows(span_rows)
    # 3) JSON lines, one note per line
    by_note: Dict[str, list] = {}
    for nid, ty, text, cs, ce, ts, te in span_rows:
        by_note.setdefault(nid, []).append({"type": ty, "text": text, "start": cs, "end": ce})
    ids = list(dict.fromkeys(r[0] for r in tok_rows))
    with open(out_dir / "ner_predictions.jsonl", "w", encoding="utf-8") as f:
        for nid in ids:
            f.write(json.dumps({"note_id": nid, "entities": by_note.get(nid, [])}, ensure_ascii=False) + "\n")


def entities_by_note(span_rows) -> Dict[str, List[Tuple[str, str]]]:
    d: Dict[str, List[Tuple[str, str]]] = {}
    for nid, ty, text, *_ in span_rows:
        d.setdefault(nid, []).append((ty, text))
    return d


# ----------------------------------------------------------------------------------------- RAG
def build_rag_docs(train_notes, train_raw: pd.DataFrame, release_df: pd.DataFrame = None, span_rows=None):
    """Knowledge base for retrieval.

    * team notes  : the hand-annotated corpus (gold spans).
    * release notes (optional): the organiser-released unlabeled notes, annotated by OUR NER model
      (``span_rows``).  Pass ``release_df=None`` for the strict "team corpus only" interpretation.
    """
    from .rag import make_doc
    docs = []
    raw = dict(zip(train_raw["note_id"], train_raw["text"]))
    for nid in sorted(train_notes):
        toks, tags = train_notes[nid]
        spans = [(ty, " ".join(toks[s:e]), s, e) for s, e, ty in corpus.spans_from_bio(tags)]
        docs.append(make_doc(nid, raw[nid], "team", spans))
    if release_df is not None:
        by = {}
        for nid, ty, text, cs, ce, ts, te in span_rows:
            by.setdefault(nid, []).append((ty, text, ts, te))
        for nid, text in zip(release_df["note_id"], release_df["note_text"]):
            docs.append(make_doc(nid, text, "release", by.get(nid, [])))
    return docs


def run_queries(retriever, queries: pd.DataFrame, max_hits=10):
    from .rag import generate_answer
    results = []
    for qid, q in zip(queries["query_id"], queries["query_text"]):
        pq, selected, top = retriever.search(q, max_hits=max_hits)
        ans = generate_answer(pq, selected, top)
        exact = [h for h in selected if h.concept_cov == 1.0 or not pq.concepts]
        results.append({
            "query_id": qid, "query_text": q, "answer": ans,
            "detected_concepts": ", ".join(pq.concepts),
            "n_matched_notes": len(exact),
            "n_partial_notes": len(selected) - len(exact),
            "retrieved_note_ids": " ".join(h.doc.note_id for h in (selected or top)[:max_hits]),
            "retrieved_scores": " ".join(f"{h.score:.3f}" for h in (selected or top)[:max_hits]),
            "knowledge_base": "+".join(sorted({d.source for d in retriever.docs})),
        })
    return results
