"""Lexicon-augmented, section-aware linear-chain CRF for Bengali clinical NER.

Why a CRF?  With ~10k annotated tokens a neural encoder cannot be trained from scratch, and
no pretrained Bengali encoder can be downloaded in the offline/reproducible setting required
by the competition.  A CRF with careful features is fast, deterministic (identical output on
every run) and strong on small data.  Features:

* word identity / lower-cased / normalised form, prefixes + suffixes (Bengali is highly
  inflected, so 1-4 character suffixes carry case endings and drug/disease morphology),
* orthographic shape (Bengali / Latin / digits / punctuation, '/', '.', '%'),
* hand-written medical gazetteer marks (B/I per entity type, longest match),
* +-2 token context window,
* note-section context: notes in this genre have headers such as ``নির্ণয়:`` or
  ``ব্যবস্থাপত্র:``; the header group (diagnosis, treatment, vitals, ...) is a very strong cue
  for the entity type.
"""
import re
from typing import Dict, List, Sequence, Tuple

import sklearn_crfsuite

from . import lexicon
from .config import SEED
from .corpus import spans_from_bio

_BN_DIGIT = re.compile(r"[০-৯]")
_DIGIT = re.compile(r"[0-9০-৯]")
_BOUNDARY = {"।", "?", "!"}

# header keywords -> group (first match wins)
_HEADER_GROUPS: List[Tuple[Tuple[str, ...], str]] = [
    (("নির্ণয়", "ডায়াগনোসিস"), "DIAG"),
    (("চিকিৎসা", "ব্যবস্থাপত্র", "ওষুধ", "চশমা"), "TREAT"),
    (("পরামর্শ", "উপদেশ"), "ADVICE"),
    (("ফলোআপ", "সাক্ষাৎ", "রিভিউ"), "FOLLOW"),
    (("ভাইটাল",), "VITAL"),
    (("প্রস্তাবিত", "তদন্ত", "ফলাফল"), "INVEST"),
    (("পরীক্ষা", "দেখা যায়", "অবস্থা"), "EXAM"),
    (("অভিযোগ", "কারণ"), "COMPLAINT"),
    (("ইতিহাস",), "HISTORY"),
    (("সময়কাল",), "DURATION"),
    (("তথ্য", "বিবরণ"), "DEMOG"),
]
_HEADER_KEYS = tuple(k for keys, _ in _HEADER_GROUPS for k in keys)
# Shorthand headers (exact match): C/O, H/O, O/E, Ix, Dx, Rx, Adv, F/U
_SHORT_HEADERS = {"C/O": "COMPLAINT", "H/O": "HISTORY", "O/E": "EXAM", "IX": "INVEST",
                  "DX": "DIAG", "RX": "TREAT", "ADV": "ADVICE", "F/U": "FOLLOW", "HX": "HISTORY"}


def header_group(phrase: str) -> str:
    if phrase.strip().upper() in _SHORT_HEADERS:
        return _SHORT_HEADERS[phrase.strip().upper()]
    for keys, g in _HEADER_GROUPS:
        if any(k in phrase for k in keys):
            return g
    return "OTHER"


def section_context(tokens: Sequence[str]):
    """Per token: (header phrase, header group, tokens since header, sentence index in section)."""
    out = []
    header, group, since, sent = "START", "START", 0, 0
    for i, t in enumerate(tokens):
        out.append((header, group, min(since, 12), min(sent, 3)))
        since += 1
        if t in _BOUNDARY:
            sent += 1
        if t == ":" and i > 0:
            # collect up to 4 tokens back to the previous sentence boundary
            j = i - 1
            cand = []
            while j >= 0 and len(cand) < 4 and tokens[j] not in _BOUNDARY and tokens[j] not in (";", ":"):
                cand.insert(0, tokens[j])
                j -= 1
            phrase = " ".join(cand)
            if cand and (phrase.upper() in _SHORT_HEADERS or any(k in phrase for k in _HEADER_KEYS)):
                header, group = phrase, header_group(phrase)
                since, sent = 0, 0
    return out


def _shape(t: str) -> str:
    s = []
    for ch in t:
        if "\u0980" <= ch <= "\u09ff":
            s.append("9" if ch in "০১২৩৪৫৬৭৮৯" else "b")
        elif ch.isascii() and ch.isalpha():
            s.append("L" if ch.isupper() else "l")
        elif ch.isdigit():
            s.append("9")
        else:
            s.append(ch)
    # collapse repeats
    return re.sub(r"(.)\1+", r"\1\1", "".join(s))


def featurize(tokens: Sequence[str], use_lexicon=True, use_section=True) -> List[Dict[str, object]]:
    n = len(tokens)
    low = [lexicon.norm(t) for t in tokens]
    shp = [_shape(t) for t in tokens]
    gaz = lexicon.match(tokens) if use_lexicon else [[] for _ in tokens]
    sec = section_context(tokens) if use_section else [("NA", "NA", 0, 0)] * n
    feats = []
    for i, t in enumerate(tokens):
        w = low[i]
        f: Dict[str, object] = {
            "bias": 1.0, "w": w, "shape": shp[i],
            "p1": w[:1], "p2": w[:2], "p3": w[:3], "p4": w[:4],
            "s1": w[-1:], "s2": w[-2:], "s3": w[-3:], "s4": w[-4:],
            "isdigit": bool(_DIGIT.search(t)), "allpunct": all(not c.isalnum() for c in t),
            "slash": "/" in t, "dot": "." in t, "pct": "%" in t, "hyph": "-" in t,
            "latin": bool(re.search(r"[A-Za-z]", t)), "len": min(len(t), 10),
        }
        for ty, bi in gaz[i]:
            f[f"gaz:{ty}:{bi}"] = 1.0
        f["gaz_n"] = len(gaz[i])
        if use_section:
            hdr, grp, since, sent = sec[i]
            f["sec"] = grp
            f["sec_hdr"] = hdr
            f["sec_since"] = since
            f["sec_sent"] = sent
            f["sec+shape"] = f"{grp}|{shp[i]}"
            f["sec+w"] = f"{grp}|{w}"
            f["sec+s3"] = f"{grp}|{w[-3:]}"
            for ty, bi in gaz[i]:
                f[f"sec+gaz:{grp}:{ty}:{bi}"] = 1.0
        for d in (-2, -1, 1, 2):
            j = i + d
            if 0 <= j < n:
                f[f"{d}:w"] = low[j]
                f[f"{d}:shape"] = shp[j]
                f[f"{d}:s3"] = low[j][-3:]
                for ty, bi in gaz[j]:
                    f[f"{d}:gaz:{ty}:{bi}"] = 1.0
                f[f"{d}:digit"] = bool(_DIGIT.search(tokens[j]))
            else:
                f[f"{d}:pad"] = 1.0
        if i > 0:
            f["bg-1"] = f"{low[i-1]}|{w}"
        if i + 1 < n:
            f["bg+1"] = f"{w}|{low[i+1]}"
        f["BOS"] = i == 0
        f["EOS"] = i == n - 1
        feats.append(f)
    return feats


def repair_bio(tags: List[str]) -> List[str]:
    """Turn an orphan I-X (no preceding B-X/I-X) into B-X so output is always valid BIO."""
    out, prev = [], "O"
    for g in tags:
        if g.startswith("I-") and prev[2:] != g[2:]:
            g = "B-" + g[2:]
        out.append(g)
        prev = g
    return out


class NERModel:
    def __init__(self, use_lexicon=True, use_section=True, c1=0.05, c2=0.05, max_iter=200):
        self.use_lexicon, self.use_section = use_lexicon, use_section
        self.c1, self.c2, self.max_iter = c1, c2, max_iter
        self.crf = None

    def _X(self, seqs):
        return [featurize(s, self.use_lexicon, self.use_section) for s in seqs]

    def fit(self, token_seqs: List[List[str]], tag_seqs: List[List[str]]):
        self.crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs", c1=self.c1, c2=self.c2, max_iterations=self.max_iter,
            all_possible_transitions=True,
        )
        self.crf.fit(self._X(token_seqs), tag_seqs)
        return self

    def predict(self, token_seqs: List[List[str]]) -> List[List[str]]:
        out = []
        for toks, p in zip(token_seqs, self.crf.predict(self._X(token_seqs))):
            # schema rule: punctuation tokens are always O (training data never has them inside spans)
            p = ["O" if all(not c.isalnum() for c in t) and t not in ("%",) else g for t, g in zip(toks, p)]
            out.append(repair_bio(p))
        return out

    def predict_marginals(self, tokens: List[str]):
        return self.crf.predict_marginals_single(featurize(tokens, self.use_lexicon, self.use_section))


def tags_to_spans(tokens_offsets, tags):
    """Character spans from token offsets + BIO tags: [(start, end, type, surface)]."""
    out = []
    for s, e, ty in spans_from_bio(list(tags)):
        out.append((tokens_offsets[s][1], tokens_offsets[e - 1][2], ty))
    return out
