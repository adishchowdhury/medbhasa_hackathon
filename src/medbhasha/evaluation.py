"""NER evaluation helpers (entity-level, seqeval-style) and cross-validation."""
import random
from typing import Callable, Dict, List

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from .config import SEED


def entity_scores(gold: List[List[str]], pred: List[List[str]]) -> Dict[str, float]:
    """Entity-level micro P/R/F1 (conlleval-style, as in seqeval's default mode)."""
    return {
        "precision": precision_score(gold, pred),
        "recall": recall_score(gold, pred),
        "f1": f1_score(gold, pred),
    }


def per_type_report(gold, pred) -> Dict[str, Dict[str, float]]:
    return classification_report(gold, pred, output_dict=True, digits=4)


def make_folds(ids: List[str], k=5, groups: Dict[str, str] = None, seed=SEED):
    """Deterministic k folds. With ``groups`` whole groups (e.g. clinical domains) are held out."""
    rng = random.Random(seed)
    if groups is None:
        ids = sorted(ids)
        rng.shuffle(ids)
        return [ids[i::k] for i in range(k)]
    by_group: Dict[str, List[str]] = {}
    for i in sorted(ids):
        by_group.setdefault(groups[i], []).append(i)
    names = sorted(by_group, key=lambda g: (-len(by_group[g]), g))
    folds = [[] for _ in range(k)]
    for g in names:  # greedy balance: biggest group -> currently smallest fold
        j = min(range(k), key=lambda x: (len(folds[x]), x))
        folds[j].extend(by_group[g])
    return folds


def cross_validate(notes, make_model: Callable, k=5, groups=None, seed=SEED):
    """Pooled out-of-fold predictions. ``notes``: id -> (tokens, tags)."""
    ids = sorted(notes)
    folds = make_folds(ids, k, groups, seed)
    gold_all, pred_all, order = [], [], []
    for held in folds:
        held_set = set(held)
        tr = [i for i in ids if i not in held_set]
        m = make_model().fit([notes[i][0] for i in tr], [notes[i][1] for i in tr])
        pred = m.predict([notes[i][0] for i in sorted(held)])
        for i, p in zip(sorted(held), pred):
            gold_all.append(notes[i][1])
            pred_all.append(p)
            order.append(i)
    return gold_all, pred_all, order
