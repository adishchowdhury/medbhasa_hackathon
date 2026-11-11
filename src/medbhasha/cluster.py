"""Unsupervised phenotype clustering of notes from NER output.

Representation (per note, built ONLY from predicted entities - never from the raw label, the
note id or the position of the note in the file):

  A. DIAGNOSIS mentions -> character 2-4-gram TF-IDF.  Character n-grams make
     ``আর্টিকারিয়া`` / ``আমবাত (আর্টিকারিয়া)`` / ``তীব্র আর্টিকারিয়া`` land close together
     even though the surface strings differ (Bengali is highly inflected / transliterated).
  B. A typed bag of entities (DIAGNOSIS, SYMPTOM, MEDICATION, TEST, PROCEDURE, BODY_SITE) ->
     word-level TF-IDF.  This is the entity co-occurrence signal; DOSAGE / TEMPORAL /
     TEST_RESULT / NEGATION are dropped because they carry numbers and boiler-plate, not
     phenotype.

The two blocks are L2-normalised, weighted, concatenated, reduced with truncated SVD and
clustered. Algorithms compared: k-means (spherical, via L2-normalised SVD space),
agglomerative (Ward linkage on the unit sphere) and Louvain on a k-NN similarity graph.
Selection of the final configuration uses ONLY internal metrics (silhouette, Davies-Bouldin).
"""
import re
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize

from .config import SEED
from .lexicon import norm, stem

PHENOTYPE_TYPES = ("DIAGNOSIS", "SYMPTOM", "MEDICATION", "TEST", "PROCEDURE", "BODY_SITE")
_PAREN = re.compile(r"[()\[\]{},;:।]")


def _clean(text: str) -> str:
    return " ".join(stem(w) for w in _PAREN.sub(" ", text).split())


def note_documents(entities: Dict[str, List[Tuple[str, str]]], ids: List[str]):
    """Two text views per note: diagnosis string(s) and a typed entity bag."""
    diag_docs, bag_docs = [], []
    for nid in ids:
        ents = entities.get(nid, [])
        diag_docs.append(" | ".join(_clean(t) for ty, t in ents if ty == "DIAGNOSIS"))
        bag = []
        for ty, t in ents:
            if ty not in PHENOTYPE_TYPES:
                continue
            c = _clean(t)
            if not c:
                continue
            bag.append(f"{ty}_{c.replace(' ', '_')}")      # whole phrase
            for w in c.split():                              # constituent words
                if len(w) > 2:
                    bag.append(f"{ty}_w_{w}")
        bag_docs.append(" ".join(bag))
    return diag_docs, bag_docs


def build_features(entities, ids, w_diag=1.0, w_bag=1.0, svd_dim=64, seed=SEED):
    diag_docs, bag_docs = note_documents(entities, ids)
    v1 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2)
    X1 = v1.fit_transform(diag_docs)
    v2 = TfidfVectorizer(analyzer=str.split, sublinear_tf=True, min_df=2, lowercase=False)
    X2 = v2.fit_transform(bag_docs)
    X = sp.hstack([w_diag * normalize(X1), w_bag * normalize(X2)]).tocsr()
    svd = TruncatedSVD(n_components=min(svd_dim, X.shape[1] - 1), random_state=seed, algorithm="randomized", n_iter=7)
    Z = normalize(svd.fit_transform(X))
    return Z, (v1, v2, svd)


def cluster_kmeans(Z, k, seed=SEED):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return km.fit_predict(Z)


def cluster_agglomerative(Z, k):
    # Z is L2-normalised, so Euclidean Ward linkage is equivalent to a cosine-based merge criterion
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)


def cluster_louvain(Z, n_neighbors=10, resolution=1.0, seed=SEED):
    """Louvain communities on a symmetric cosine k-NN graph (graph-based representation)."""
    import networkx as nx
    S = Z @ Z.T
    n = Z.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        idx = np.argpartition(-S[i], n_neighbors + 1)[: n_neighbors + 1]
        for j in idx:
            if j != i and S[i, j] > 0:
                G.add_edge(i, int(j), weight=float(S[i, j]))
    comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    labels = np.zeros(n, dtype=int)
    for c, members in enumerate(sorted(comms, key=lambda s: (-len(s), min(s)))):
        for m in members:
            labels[m] = c
    return labels


def internal_scores(Z, labels):
    if len(set(labels)) < 2:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan")}
    return {
        "silhouette": float(silhouette_score(Z, labels, metric="cosine")),
        "davies_bouldin": float(davies_bouldin_score(Z, labels)),
    }


def describe_clusters(labels, ids, entities, top=5):
    """Human-readable summary of each cluster: size, top diagnoses / drugs / symptoms (surface forms)."""
    from collections import Counter, defaultdict
    from .surface import SurfaceMap
    members = defaultdict(list)
    for nid, c in zip(ids, labels):
        members[int(c)].append(nid)
    sms = {t: SurfaceMap() for t in ("DIAGNOSIS", "MEDICATION", "SYMPTOM", "TEST")}
    rows = []
    for c in sorted(members, key=lambda c: (-len(members[c]), c)):
        cnt = {t: Counter() for t in sms}
        for nid in members[c]:
            seen = set()
            for ty, t in entities.get(nid, []):
                if ty in cnt:
                    key = sms[ty].add(t)
                    if key and (ty, key) not in seen:
                        cnt[ty][key] += 1
                        seen.add((ty, key))
        fmt = lambda ty: "; ".join(f"{sms[ty].show(k)} ({v})" for k, v in
                                    sorted(cnt[ty].items(), key=lambda kv: (-kv[1], kv[0]))[:top])
        rows.append({
            "cluster": c, "size": len(members[c]),
            "top_diagnoses": fmt("DIAGNOSIS"), "top_medications": fmt("MEDICATION"),
            "top_symptoms": fmt("SYMPTOM"), "top_tests": fmt("TEST"),
            "example_notes": " ".join(members[c][:5]),
        })
    return rows


def select_and_cluster(Z, k_grid=range(20, 201, 10), louvain_resolutions=(0.5, 1.0, 2.0, 4.0), seed=SEED):
    """Grid over algorithm x k (or Louvain resolution); pick best silhouette (ties: lower DB).

    Returns (best_labels, best_config_dict, results_table_rows).
    Only INTERNAL metrics are used - no labels of any kind.
    """
    rows, best = [], None
    for algo, fn in (("kmeans", lambda k: cluster_kmeans(Z, k, seed)), ("ward", lambda k: cluster_agglomerative(Z, k))):
        for k in k_grid:
            lab = fn(k)
            s = internal_scores(Z, lab)
            rows.append({"algorithm": algo, "param": k, "n_clusters": len(set(lab)), **s})
            key = (s["silhouette"], -s["davies_bouldin"])
            if best is None or key > best[0]:
                best = (key, lab, {"algorithm": algo, "k": k})
    for res in louvain_resolutions:
        lab = cluster_louvain(Z, resolution=res, seed=seed)
        s = internal_scores(Z, lab)
        rows.append({"algorithm": "louvain", "param": res, "n_clusters": len(set(lab)), **s})
        key = (s["silhouette"], -s["davies_bouldin"])
        if best is None or key > best[0]:
            best = (key, lab, {"algorithm": "louvain", "resolution": res})
    return best[1], best[2], rows
