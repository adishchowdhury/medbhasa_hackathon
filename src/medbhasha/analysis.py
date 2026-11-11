"""Frequency, co-occurrence and relationship analysis + plotting helpers.

Used twice: for the Round-1 annotated corpus (EDA) and for the NER output on the released notes
(Round-2 "frequency analysis and relationship analysis using the NER output").
All plot labels are in English because matplotlib cannot shape Bengali conjuncts; Bengali strings
are kept in the CSV tables.
"""
import math
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import ENTITY_TYPES  # noqa: E402
from .surface import SurfaceMap, key_of  # noqa: E402

PALETTE = ["#1b6ca8", "#e07b39", "#3a9d5d", "#b0413e", "#7a5bb5", "#8c6d31", "#d6609a", "#5d6d7e", "#c9a227", "#2a9d9f"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def spans_from_notes(notes) -> List[Tuple[str, str, str]]:
    """Gold spans (note_id, type, text) from ``{id: (tokens, tags)}``."""
    from .corpus import spans_from_bio
    rows = []
    for nid, (toks, tags) in notes.items():
        for s, e, ty in spans_from_bio(tags):
            rows.append((nid, ty, " ".join(toks[s:e])))
    return rows


def entity_frequency(span_rows) -> pd.DataFrame:
    df = pd.DataFrame(span_rows, columns=["note_id", "type", "text"])
    out = df.groupby("type").agg(spans=("text", "size"), distinct=("text", "nunique"), notes=("note_id", "nunique"))
    out = out.reindex(ENTITY_TYPES).fillna(0).astype(int)
    out["share_%"] = (100 * out["spans"] / out["spans"].sum()).round(1)
    return out.sort_values("spans", ascending=False)


def top_entities(span_rows, per_type=10) -> pd.DataFrame:
    c = defaultdict(Counter)
    sm = defaultdict(SurfaceMap)
    for nid, ty, text in span_rows:
        k = sm[ty].add(text)
        if k:
            c[ty][k] += 1
    rows = []
    for ty in ENTITY_TYPES:
        for k, n in sorted(c[ty].items(), key=lambda kv: (-kv[1], kv[0]))[:per_type]:
            rows.append((ty, sm[ty].show(k), n))
    return pd.DataFrame(rows, columns=["type", "entity", "count"])


def cooccurrence(span_rows) -> pd.DataFrame:
    """Within-note co-occurrence counts of entity TYPES (diagonal = notes containing the type)."""
    by_note = defaultdict(set)
    for nid, ty, _ in span_rows:
        by_note[nid].add(ty)
    M = pd.DataFrame(0, index=ENTITY_TYPES, columns=ENTITY_TYPES, dtype=int)
    for types in by_note.values():
        for a in types:
            for b in types:
                M.loc[a, b] += 1
    return M


def association_pairs(span_rows, type_a="DIAGNOSIS", type_b="MEDICATION", min_count=3, top=25) -> pd.DataFrame:
    """Relationship analysis: which (type_a entity, type_b entity) pairs co-occur more than chance.

    Reports count, lift and PMI over notes (entities normalised by light stemming).
    """
    notes = defaultdict(lambda: (set(), set()))
    sm_a, sm_b = SurfaceMap(), SurfaceMap()
    for nid, ty, text in span_rows:
        if ty == type_a:
            k = sm_a.add(text)
            if k:
                notes[nid][0].add(k)
        elif ty == type_b:
            k = sm_b.add(text)
            if k:
                notes[nid][1].add(k)
    N = len(notes)
    ca, cb, cab = Counter(), Counter(), Counter()
    for nid in sorted(notes):                      # sorted iteration: output must not depend on hash order
        a, b = notes[nid]
        for x in sorted(a):
            ca[x] += 1
        for y in sorted(b):
            cb[y] += 1
        for x in sorted(a):
            for y in sorted(b):
                cab[(x, y)] += 1
    rows = []
    for (x, y), n in cab.items():
        if n < min_count:
            continue
        lift = (n * N) / (ca[x] * cb[y])
        rows.append((sm_a.show(x), sm_b.show(y), n, ca[x], cb[y], round(lift, 2), round(math.log2(lift), 2)))
    df = pd.DataFrame(rows, columns=[type_a.lower(), type_b.lower(), "co_notes", "n_a", "n_b", "lift", "pmi"])
    # fully deterministic order: ties broken by the entity strings themselves
    return (df.sort_values(["lift", "co_notes", type_a.lower(), type_b.lower()], ascending=[False, False, True, True],
                           kind="mergesort").head(top).reset_index(drop=True))


def note_length_stats(lengths: List[int]) -> Dict[str, float]:
    a = np.array(lengths)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "min": int(a.min()),
            "max": int(a.max()), "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}


# --------------------------------------------------------------------------------------- plots
def plot_entity_frequency(freq: pd.DataFrame, title: str, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    f = freq.sort_values("spans")
    ax.barh(f.index, f["spans"], color=PALETTE[0])
    for i, v in enumerate(f["spans"]):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    ax.set_xlabel("entity spans")
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_length_hist(lengths, title, path, xlabel="tokens per note"):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.hist(lengths, bins=25, color=PALETTE[1], edgecolor="white")
    ax.axvline(np.mean(lengths), color="black", lw=1, ls="--")
    ax.text(np.mean(lengths), ax.get_ylim()[1] * 0.92, f" mean {np.mean(lengths):.0f}", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("notes")
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cooccurrence(M: pd.DataFrame, title: str, path):
    diag = np.diag(M.values).astype(float)
    norm = M.values / np.maximum(diag[:, None], 1)       # P(col present | row present)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(M)))
    ax.set_yticks(range(len(M)))
    short = [t.replace("NEGATION_UNCERTAINTY", "NEG/UNC").replace("TEST_RESULT", "TEST_RES") for t in M.columns]
    ax.set_xticklabels(short, rotation=60, ha="right")
    ax.set_yticklabels(short)
    for i in range(len(M)):
        for j in range(len(M)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_title(title, loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, label="P(column type | row type) over notes")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cluster_selection(rows, chosen, path):
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for algo, col in (("kmeans", PALETTE[0]), ("ward", PALETTE[1])):
        d = df[df.algorithm == algo]
        axes[0].plot(d["n_clusters"], d["silhouette"], marker="o", ms=3, color=col, label=algo)
        axes[1].plot(d["n_clusters"], d["davies_bouldin"], marker="o", ms=3, color=col, label=algo)
    d = df[df.algorithm == "louvain"]
    axes[0].scatter(d["n_clusters"], d["silhouette"], marker="^", color=PALETTE[2], label="louvain")
    axes[1].scatter(d["n_clusters"], d["davies_bouldin"], marker="^", color=PALETTE[2], label="louvain")
    axes[0].set_ylabel("silhouette (higher = better)")
    axes[1].set_ylabel("Davies-Bouldin (lower = better)")
    for a in axes:
        a.set_xlabel("number of clusters")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Choosing the clustering from internal metrics only", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cluster_map(Z, labels, path):
    from sklearn.decomposition import PCA
    P = PCA(n_components=2, random_state=42).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(P[:, 0], P[:, 1], c=labels, cmap="nipy_spectral", s=7, alpha=0.75)
    ax.set_title(f"Notes in entity space (PCA of SVD features), {len(set(labels))} clusters", loc="left", fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ner_per_type(report: Dict[str, Dict[str, float]], title, path):
    types = [t for t in ENTITY_TYPES if t in report]
    f1 = [report[t]["f1-score"] for t in types]
    sup = [int(report[t]["support"]) for t in types]
    order = np.argsort(f1)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh([types[i] for i in order], [f1[i] for i in order], color=PALETTE[2])
    for k, i in enumerate(order):
        ax.text(f1[i], k, f" {f1[i]:.2f} (n={sup[i]})", va="center", fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("entity-level F1")
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
