"""Interactive / live querying.

    python -m medbhasha.ask "হার্ট ফেইলিওর আছে এমন রোগীদের বর্ণনা দিন।"
    python -m medbhasha.ask --team-only "ডায়াবেটিস রোগীদের জন্য কোন কোন ওষুধ নথিভুক্ত করা আছে?"

Requires ``outputs/ner_predictions_spans.csv`` (created by the pipeline) when the released notes are
part of the knowledge base.  Everything runs locally; no network access is used.
"""
import argparse
import sys

import pandas as pd

from . import pipeline
from .config import OUT
from .rag import Retriever, generate_answer


def build_retriever(team_only: bool = False) -> Retriever:
    notes, meta, raw = pipeline.load_train()
    if team_only:
        return Retriever(pipeline.build_rag_docs(notes, raw))
    tdf, _ = pipeline.load_test()
    spans = pd.read_csv(OUT / "ner_predictions_spans.csv", encoding="utf-8")
    rows = [tuple(r) for r in spans.itertuples(index=False)]
    return Retriever(pipeline.build_rag_docs(notes, raw, tdf, rows))


def ask(retriever: Retriever, query: str, max_hits: int = 10) -> dict:
    pq, selected, top = retriever.search(query, max_hits=max_hits)
    exact = [h for h in selected if h.concept_cov == 1.0 or not pq.concepts]
    return {"query": query, "concepts": pq.concepts, "n_matched": len(exact), "n_partial": len(selected) - len(exact),
            "retrieved": [(h.doc.note_id, round(h.score, 3)) for h in (selected or top)[:max_hits]],
            "answer": generate_answer(pq, selected, top)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("--team-only", action="store_true", help="retrieve only from the team's own annotated corpus")
    a = ap.parse_args(argv)
    r = ask(build_retriever(a.team_only), " ".join(a.query))
    print(r["answer"])
    print("\n[retrieved]", " ".join(f"{i}:{s}" for i, s in r["retrieved"]), file=sys.stderr)


if __name__ == "__main__":
    main()
