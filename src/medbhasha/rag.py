"""Retrieval-augmented question answering over the team's own annotated notes.

NO external API, NO web retrieval, NO pretrained generator: everything is computed locally and
deterministically.  Pipeline:

  1. Index   - every note -> normalised word keys (spelling skeleton + light stemming), the set of
               clinical *concepts* it mentions (``concepts.py``), age / sex, and its NER spans.
  2. Query   - the Bengali query is parsed into concept constraints, free content terms,
               demographic constraints (child / elderly / sex) and numeric constraints
               (e.g. "HbA1c ৯-এর বেশি").
  3. Retrieve- score = concept coverage + IDF-weighted term coverage + BM25 (tie-breaker); hard
               constraints (age / sex / numeric) gate the score.  Conjunctive queries therefore rank
               notes that satisfy ALL conditions first.
  4. Answer  - *extractive, grounded* generation: the answer is assembled from sentences and entity
               strings copied from the retrieved notes, every claim carries a [note-id] citation, and
               the system says so when no note satisfies the query instead of inventing content.
"""
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .concepts import CONCEPTS
from .lexicon import norm
from .tokenizer import PUNCT, tokenize

_BN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
KEY_LEN = 8

# ----------------------------------------------------------------------------- normalisation
_CASE_SUFFIXES = sorted(["গুলোর", "গুলো", "দের", "েরা", "ের", "তে", "েতে", "কে", "য়ে", "টির", "টি", "টা", "র", "ে", "এ", "য়", "য়ের"],
                        key=len, reverse=True)
_VOWEL_SIGNS = "ািীুূেো"


def skeleton(t: str) -> str:
    """Spelling skeleton: collapses the usual Bengali transliteration variants."""
    t = unicodedata.normalize("NFC", t).replace("‍", "").replace("‌", "")
    t = t.replace("অ্যা", "এ").replace("র‍্য", "র্য")
    t = t.replace("ড়", "ড়").replace("য়", "য়")
    t = t.replace("্", "")                       # drop hasanta: মেটফর্মিন == মেটফরমিন
    t = (t.replace("ী", "ি").replace("ূ", "ু").replace("ণ", "ন").replace("ষ", "স").replace("শ", "স")
          .replace("ঈ", "ই").replace("ঊ", "উ").replace("ঁ", ""))
    return t


def _strip_suffix(t: str) -> str:
    for _ in range(2):
        for s in _CASE_SUFFIXES:
            if t.endswith(s) and len(t) - len(s) >= 2:
                t = t[: -len(s)]
                break
        else:
            break
    if len(t) > 3 and t[-1] in _VOWEL_SIGNS:
        t = t[:-1]
    return t


def key(tok: str) -> str:
    t = tok.translate(_BN).lower()
    t = skeleton(norm(t))
    t = _strip_suffix(t)
    return t[:KEY_LEN]


def keys_of(text: str) -> List[str]:
    """Retrieval keys of a text; hyphenated compounds are split (সর্দি-কাশি -> সর্দি, কাশি)."""
    out = []
    for tok in tokenize(text):
        if all(c in PUNCT or not c.isalnum() for c in tok):
            continue
        parts = [p for p in re.split(r"[-/]", tok) if p] if re.search(r"[^\d.\-/]", tok) else [tok]
        for p in parts:
            out.append(key(p))
    return out


STOP_TEXT = """
কোন কোনো কী কি কত কারা কাদের কাকে কেউ কিছু এমন যাদের যারা যে যা তা তাদের তাঁদের তারা এই ওই সেই এবং ও আর বা কিংবা অথবা
আছে আছেন আছে? আছেন? নেই হয়েছে হয় হচ্ছে হচ্ছেন করা করেছে করেছেন করুন দিন দেখান খুঁজুন বলুন জানান দেওয়া দেয়া হয়েছে
নথিভুক্ত উল্লেখ বর্ণনা বিস্তারিত তথ্য এনে রোগী রোগীদের রোগীর রোগীকে রোগীরা জন্য থেকে মধ্যে সাথে সঙ্গে সহ নিয়ে নিচ্ছেন নিচ্ছে চলছে চলছেন
পাওয়া যায় যাচ্ছে পেয়েছেন পরামর্শ তাদের তাকে কাদের কেমন কীভাবে কিভাবে কোথায় এর তে র কে বেশি কম এড়িয়ে চলার
শিশু শিশুদের বাচ্চা বাচ্চাদের বয়স্ক বৃদ্ধ পুরুষ মহিলা নারী গর্ভবতী
আক্রান্ত ক্ষেত্রে উপসর্গ লক্ষণ ক্লিনিক্যাল ফাইন্ডিংস ওষুধ ঔষধ চিকিৎসা করা হয়েছে বলা বলা হয়েছে শুরু পেয়েছেন নিচ্ছেন আসা এনে বর্তমানে
"""
STOP = {key(w) for w in STOP_TEXT.split()} | {""}
# words that carry meaning in a query but are not content for matching ("এড়িয়ে চলা" etc. are advice verbs)

# ----------------------------------------------------------------------------- concept index
def _build_concept_index():
    idx: Dict[Tuple[str, ...], set] = defaultdict(set)
    maxlen = 1
    for name, variants in CONCEPTS.items():
        for v in variants:
            ks = tuple(k for k in keys_of(v) if k)
            if ks:
                idx[ks].add(name)
                maxlen = max(maxlen, len(ks))
    return idx, maxlen


CONCEPT_INDEX, CONCEPT_MAXLEN = _build_concept_index()


def _build_window_variants():
    out = []
    for name, variants in CONCEPTS.items():
        for v in variants:
            ks = tuple(dict.fromkeys(k for k in keys_of(v) if k))
            if 2 <= len(ks) <= 3:
                out.append((ks, name))
    return out


WINDOW_VARIANTS = _build_window_variants()


def _window_match(ks: List[str], found: set):
    """Order-free match of short multi-word variants inside a window (e.g. 'রাতে চুলকানি বেশি')."""
    kset = set(ks)
    hits, covered = set(), set()
    for vk, name in WINDOW_VARIANTS:
        if name in found or name in hits or not all(k in kset for k in vk):
            continue
        pos = [[i for i, k in enumerate(ks) if k == x] for x in vk]
        # smallest window containing all keys
        best = min(max(c) - min(c) for c in _combos(pos))
        if best <= len(vk) + 1:
            hits.add(name)
    return hits


def _combos(pos):
    from itertools import product
    for c in product(*[p[:4] for p in pos]):
        yield c


def find_concepts(ks: List[str]):
    """Longest-match concept detection (+ order-free window match). Returns (set, covered_positions)."""
    found, covered = _find_contiguous(ks)
    if len(ks) >= 2:
        found |= _window_match(ks, found)
    return found, covered


def _find_contiguous(ks: List[str]):
    found, covered = set(), set()
    i = 0
    while i < len(ks):
        hit = None
        for L in range(min(CONCEPT_MAXLEN, len(ks) - i), 0, -1):
            c = CONCEPT_INDEX.get(tuple(ks[i:i + L]))
            if c:
                hit = (L, c)
                break
        if hit:
            L, c = hit
            found |= c
            covered.update(range(i, i + L))
            i += L
        else:
            i += 1
    return found, covered


# ----------------------------------------------------------------------------- documents
@dataclass
class Doc:
    note_id: str
    text: str
    source: str                                   # 'team' (gold-annotated) | 'release' (NER-annotated)
    spans: List[Tuple[str, str, int, int]] = field(default_factory=list)   # (type, text, tok_start, tok_end)
    keys: List[str] = field(default_factory=list)
    keyset: set = field(default_factory=set)
    concepts: set = field(default_factory=set)       # positively mentioned (patient, not negated)
    neg_concepts: set = field(default_factory=set)   # mentioned only in negated clauses ('... নেই')
    dx_concepts: set = field(default_factory=set)   # concepts found inside DIAGNOSIS spans
    neg_keys: set = field(default_factory=set)      # word keys that occur only in negated clauses
    age: Optional[float] = None
    sex: Optional[str] = None


_AGE_RE = [
    (re.compile(r"(\d+)\s*Y\s*/\s*([MF])"), "yx"),
    (re.compile(r"(\d+)\s*বছর\s*বয়সী"), "y"),
    (re.compile(r"(\d+)\s*মাস\s*বয়সী"), "m"),
    (re.compile(r"(\d+)\s*দিন\s*বয়সী"), "d"),
]


def parse_demographics(text: str):
    head = text[:200].translate(_BN)
    age, sex = None, None
    for rx, kind in _AGE_RE:
        m = rx.search(head)
        if m:
            n = float(m.group(1))
            age = {"yx": n, "y": n, "m": n / 12, "d": n / 365}[kind]
            if kind == "yx":
                sex = "M" if m.group(2) == "M" else "F"
            break
    if sex is None:
        if re.search(r"পুরুষ|ছেলে|বাবা", head):
            sex = "M"
        elif re.search(r"মহিলা|মেয়ে|গর্ভবতী|মা জানান", head):
            sex = "F"
    return age, sex


_NEG_CUE = re.compile(r"(?:^|\s)(?:নেই|নাই|নয়|না|অস্বীকার|ছিল না|হয়নি|হননি|যায়নি|আসেনি|করেনি|করেননি|করাননি|খাননি|খায়নি|নেননি|পাননি|দেননি)(?:\s|$)")
_FAM_CUE = re.compile(r"পরিবারে|বাবার|মায়ের|বাবা|ভাইয়ের|বোনের|দাদার|পিতার|মাতার|বংশগত|আত্মীয়")
_CLAUSE_SPLIT = re.compile(r"[;,।]|\sএবং\s|\sকিন্তু\s|\sতবে\s")


def concepts_by_polarity(text: str):
    """Concepts mentioned positively / negated / about family members (clause-level)."""
    pos, neg, fam = set(), set(), set()
    for clause in _CLAUSE_SPLIT.split(text):
        cs, _ = find_concepts(keys_of(clause))
        if not cs:
            continue
        if _NEG_CUE.search(" " + clause + " "):
            neg |= cs
        elif _FAM_CUE.search(clause):
            fam |= cs
        else:
            pos |= cs
    return pos, neg - pos, fam - pos


def negated_keys(text: str) -> set:
    neg, pos = set(), set()
    for clause in _CLAUSE_SPLIT.split(text):
        ks = set(keys_of(clause))
        (neg if _NEG_CUE.search(" " + clause + " ") else pos).update(ks)
    return neg - pos


def make_doc(note_id, text, source, spans) -> Doc:
    ks = keys_of(text)
    cons, neg, _fam = concepts_by_polarity(text)
    age, sex = parse_demographics(text)
    # disease concepts inside DIAGNOSIS spans that sit in the note's diagnosis section (Dx / নির্ণয়).
    # Short notes without any section header: every DIAGNOSIS span counts.
    from .ner import section_context
    sec = section_context(tokenize(text))
    structured = any(g != "START" for _, g, _, _ in sec)
    dxc = set()
    for ty, t, ts, _te in spans:
        if ty != "DIAGNOSIS":
            continue
        if structured and not (ts < len(sec) and sec[ts][1] == "DIAG"):
            continue
        dxc |= find_concepts(keys_of(t))[0]
    return Doc(note_id, text, source, spans, ks, set(ks), cons, neg, dxc, negated_keys(text), age, sex)


# ----------------------------------------------------------------------------- query parsing
@dataclass
class ParsedQuery:
    text: str
    concepts: List[str]
    free_terms: List[str]
    req_terms: List[str] = field(default_factory=list)   # rare free terms: must be matched
    child: bool = False
    elderly: bool = False
    sex: Optional[str] = None
    numeric: List[Tuple[str, str, float]] = field(default_factory=list)  # (concept_or_term, op, value)
    intent: str = "list"
    wants_negation: bool = False
    antibiotic_only: bool = False


_NUM_RE = re.compile(r"([A-Za-z][A-Za-z0-9]*|[^\s\d.]+)\s*([0-9]+(?:\.[0-9]+)?)\s*%?\s*[-–]?\s*(?:এর|র)?\s*(বেশি|উপরে|ঊর্ধ্বে|কম|নিচে)")


def parse_query(q: str, idf: Dict[str, float], df: Dict[str, int], n_docs: int) -> ParsedQuery:
    qn = q.translate(_BN)
    ks = keys_of(q)
    concepts, covered = find_concepts(ks)
    free = []
    for i, k in enumerate(ks):
        if i in covered or k in STOP or k in free or re.fullmatch(r"[0-9.%]+", k):
            continue
        if df.get(k, 0) == 0 or df.get(k, 0) / n_docs > 0.5:
            continue
        free.append(k)
    pq = ParsedQuery(q, sorted(concepts), free)
    pq.req_terms = [k for k in free if df.get(k, 0) / n_docs < 0.08]
    pq.child = bool(re.search(r"শিশু|বাচ্চা|ছেলেমেয়ে", q))
    pq.elderly = bool(re.search(r"বয়স্ক|বৃদ্ধ", q))
    if re.search(r"পুরুষ", q):
        pq.sex = "M"
    elif re.search(r"মহিলা|নারী", q):
        pq.sex = "F"
    for m in _NUM_RE.finditer(qn):
        term, val, op = m.group(1), float(m.group(2)), m.group(3)
        pq.numeric.append((term, ">" if op in ("বেশি", "উপরে", "ঊর্ধ্বে") else "<", val))
    if re.search(r"ওষুধ|ঔষধ|মেডিসিন", q) and re.search(r"কোন কোন|কী কী|নথিভুক্ত|কোন", q):
        pq.intent = "medication"
    elif re.search(r"কোন কোন অ্যান্টিবায়োটিক|অ্যান্টিবায়োটিক .*বলুন", q):
        pq.intent, pq.antibiotic_only = "medication", True
    elif re.search(r"উপসর্গ|লক্ষণ", q):
        pq.intent = "symptom"
    elif re.search(r"ফাইন্ডিংস|ক্লিনিক্যাল", q):
        pq.intent = "findings"
    elif re.search(r"বর্ণনা|বিস্তারিত", q):
        pq.intent = "describe"
    elif re.search(r"কী চিকিৎসা|চিকিৎসা দেওয়া", q):
        pq.intent = "treatment"
    pq.wants_negation = bool(re.search(r"প্রয়োজন নেই|দরকার নেই|এড়িয়ে চল|না নিতে|নিষেধ", q))
    return pq


# ----------------------------------------------------------------------------- retriever
@dataclass
class Hit:
    doc: Doc
    score: float
    concept_cov: float
    term_cov: float
    bm25: float
    matched: List[str]
    satisfied: bool = True
    primary: bool = True        # the queried disease concept(s) appear in the note's diagnosis section, not only in history
    constraint_notes: List[str] = field(default_factory=list)


class Retriever:
    def __init__(self, docs: List[Doc], k1=1.5, b=0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.df: Dict[str, int] = Counter()
        for d in docs:
            for k in d.keyset:
                self.df[k] += 1
        self.avgdl = sum(len(d.keys) for d in docs) / max(1, self.N)
        self.idf = {k: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for k, n in self.df.items()}
        self.tf = [Counter(d.keys) for d in docs]
        self.dxc = set().union(*[d.dx_concepts for d in docs]) if docs else set()

    # -- BM25 over free terms (+ concept keys so concepts also contribute lexically)
    def _bm25(self, i: int, terms: List[str]) -> float:
        d = self.docs[i]
        dl = len(d.keys)
        s = 0.0
        for t in terms:
            f = self.tf[i].get(t, 0)
            if f:
                s += self.idf.get(t, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def _numeric_ok(self, d: Doc, pq: ParsedQuery):
        notes = []
        for term, op, val in pq.numeric:
            tk = [k for k in keys_of(term) if k]
            names = set(find_concepts(tk)[0])
            variants = []
            for n in names:
                variants += CONCEPTS[n]
            if not variants:
                variants = [term]
            text = d.text.translate(_BN)
            ok = False
            for v in variants:
                vv = v.translate(_BN)
                for m in re.finditer(re.escape(vv), text, flags=re.I):
                    tail = text[m.end(): m.end() + 40]
                    mm = re.search(r"([0-9]+(?:\.[0-9]+)?)", tail)
                    if mm:
                        x = float(mm.group(1))
                        if (op == ">" and x > val) or (op == "<" and x < val):
                            ok = True
                            notes.append(f"{vv} {x:g}")
                            break
                if ok:
                    break
            if not ok:
                return False, notes
        return True, notes

    def search(self, query: str, max_hits: int = 10, min_cov: float = 0.5):
        pq = parse_query(query, self.idf, self.df, self.N)
        # concept keys are looked up through the concept table, so they need no free-term weight
        terms = list(pq.free_terms)
        concept_terms = []
        for c in pq.concepts:
            for v in CONCEPTS[c]:
                concept_terms += [k for k in keys_of(v) if k in self.idf]
        bm_terms = terms + sorted(set(concept_terms))
        denom_terms = sum(self.idf.get(t, 0) for t in terms) or 1.0
        bm = [self._bm25(i, bm_terms) for i in range(self.N)]
        bmax = max(bm) or 1.0
        hits: List[Hit] = []
        for i, d in enumerate(self.docs):
            have = d.concepts | d.neg_concepts if pq.wants_negation else d.concepts
            matched = [c for c in pq.concepts if c in have]
            ccov = (len(matched) / len(pq.concepts)) if pq.concepts else 0.0
            tcov = (sum(self.idf.get(t, 0) for t in terms if t in d.keyset and (pq.wants_negation or t not in d.neg_keys)) / denom_terms) if terms else 0.0
            if pq.concepts and terms:
                s = 0.55 * ccov + 0.25 * tcov + 0.20 * bm[i] / bmax
            elif pq.concepts:
                s = 0.80 * ccov + 0.20 * bm[i] / bmax
            else:
                s = 0.70 * tcov + 0.30 * bm[i] / bmax
            hit = Hit(d, s, ccov, tcov, bm[i], matched)
            dx_q = [c for c in pq.concepts if c in self.dxc]
            hit.primary = all(c in d.dx_concepts for c in dx_q) if dx_q else True
            if dx_q and hit.primary:
                hit.score += 0.10
            # ---- hard-ish constraints
            if pq.child and not ((d.age is not None and d.age < 13) or "শিশু" in d.text):
                hit.score *= 0.3; hit.satisfied = False; hit.constraint_notes.append("not-child")
            if pq.elderly and not (d.age is not None and d.age >= 60):
                hit.score *= 0.3; hit.satisfied = False; hit.constraint_notes.append("not-elderly")
            if pq.sex and d.sex and d.sex != pq.sex:
                hit.score *= 0.3; hit.satisfied = False; hit.constraint_notes.append("sex-mismatch")
            if pq.numeric:
                ok, nn = self._numeric_ok(d, pq)
                if not ok:
                    hit.score *= 0.3; hit.satisfied = False; hit.constraint_notes.append("numeric-fail")
                else:
                    hit.constraint_notes += nn
            hits.append(hit)
        hits.sort(key=lambda h: (-h.score, h.doc.note_id))
        top = hits[0] if hits else None
        # select the answer set
        if top is None:
            return pq, [], []
        if pq.concepts:
            req = pq.req_terms
            def req_ok(h):
                if not req:
                    return True
                return sum(1 for t in req if t in h.doc.keyset and (pq.wants_negation or t not in h.doc.neg_keys)) / len(req) >= 0.5
            full = [h for h in hits if h.concept_cov == 1.0 and h.satisfied and req_ok(h)]
            partial = [h for h in hits if 0.5 <= h.concept_cov < 1.0 and h.satisfied][:8]
            # few exact matches -> also surface the closest partial matches (clearly labelled later)
            selected = full if len(full) >= 3 else full + partial
        else:
            selected = [h for h in hits if h.score >= 0.6 * top.score and h.term_cov > 0][:max_hits]
        return pq, selected[:200], hits[:max_hits]


# ----------------------------------------------------------------------------- answer generation
_TYPE_BN = {"DIAGNOSIS": "রোগ নির্ণয়", "SYMPTOM": "উপসর্গ", "MEDICATION": "ওষুধ", "TEST": "পরীক্ষা",
            "TEST_RESULT": "ফলাফল", "PROCEDURE": "পদ্ধতি", "BODY_SITE": "অঙ্গ"}


def bn(x) -> str:
    return str(x).translate(str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯"))


def _seg_split(text: str) -> List[str]:
    segs = []
    for s in re.split(r"[।]", text):
        for p in re.split(r"[;]", s):
            p = p.strip()
            if p:
                segs.append(p)
    return segs


def _clip(s: str, n=150) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _numeric_segment_ok(seg: str, pq: "ParsedQuery") -> bool:
    """True if this text fragment itself carries the queried numeric value (e.g. 'HbA1c ৯.৪%')."""
    text = seg.translate(_BN)
    for term, op, val in pq.numeric:
        names = find_concepts([k for k in keys_of(term) if k])[0]
        variants = [v for n in names for v in CONCEPTS[n]] or [term]
        for v in variants:
            for m in re.finditer(re.escape(v.translate(_BN)), text, flags=re.I):
                mm = re.search(r"([0-9]+(?:\.[0-9]+)?)", text[m.end(): m.end() + 40])
                if mm and ((op == ">" and float(mm.group(1)) > val) or (op == "<" and float(mm.group(1)) < val)):
                    return True
    return False


def evidence_sentences(d: Doc, pq: ParsedQuery, limit=2) -> List[str]:
    """Fragments of the note that mention the matched concepts / free terms (verbatim excerpts)."""
    scored = []
    want_concepts = set(pq.concepts)
    cands = []
    for seg in _seg_split(d.text):
        cands += [x.strip() for x in seg.split(",") if x.strip()] if len(seg) > 90 else [seg]
    for seg in cands:
        ks = keys_of(seg)
        c, _ = find_concepts(ks)
        sc = len(c & want_concepts) * 2 + sum(1 for k in set(ks) if k in pq.free_terms)
        if pq.wants_negation and re.search(r"নেই|না\b|নয়|এড়ি", seg):
            sc += 1
        if pq.numeric and _numeric_segment_ok(seg, pq):
            sc += 5                      # the fragment that actually carries the queried value
        if sc:
            scored.append((-sc, len(seg), seg))
    scored.sort()
    out = []
    for _, _, seg in scored:
        seg = _clip(seg)
        if seg not in out:
            out.append(seg)
        if len(out) >= limit:
            break
    return out


def _canon_med(s: str) -> str:
    return " ".join(key(w) for w in re.split(r"\s+", s) if w)


def _entities(d: Doc, ty: str) -> List[str]:
    seen, out = set(), []
    for t, text, _, _ in d.spans:
        if t == ty:
            k = _canon_med(text)
            if k and k not in seen:
                seen.add(k)
                out.append(text)
    return out


def _meds_with_dose(d: Doc) -> List[str]:
    out, seen = [], set()
    sp = d.spans
    for i, (t, text, s, e) in enumerate(sp):
        if t != "MEDICATION":
            continue
        k = _canon_med(text)
        if k in seen:
            continue
        seen.add(k)
        dose = ""
        if i + 1 < len(sp) and sp[i + 1][0] == "DOSAGE" and sp[i + 1][2] - e <= 1:
            dose = " " + sp[i + 1][1]
        out.append(text + dose)
    return out


def _fmt_patient(d: Doc) -> str:
    age = f"{bn(int(d.age))} বছর" if d.age is not None and d.age >= 1 else (f"{bn(int(round(d.age * 12)))} মাস" if d.age else "")
    sex = {"M": "পুরুষ", "F": "মহিলা"}.get(d.sex or "", "")
    return ", ".join(x for x in (age, sex) if x)


def _label(c: str) -> str:
    return CONCEPTS[c][0] if c in CONCEPTS else c


def _patient_line(h: Hit, pq: ParsedQuery) -> str:
    d = h.doc
    parts = []
    demo = _fmt_patient(d)
    if demo:
        parts.append(demo)
    dx = _entities(d, "DIAGNOSIS")
    if dx:
        parts.append("রোগ নির্ণয়: " + ", ".join(dx[:3]))
    if pq.intent in ("describe", "symptom", "findings"):
        sy = _entities(d, "SYMPTOM")
        if sy:
            parts.append("উপসর্গ/ফাইন্ডিংস: " + ", ".join(sy[:5]))
    if pq.intent in ("medication", "describe", "treatment", "list"):
        ms = _meds_with_dose(d)
        if ms:
            parts.append("ওষুধ: " + ", ".join(ms[:4]))
    ev = evidence_sentences(d, pq)
    if ev:
        parts.append("নোট থেকে: «" + " | ".join(ev) + "»")
    if h.concept_cov < 1.0 and pq.concepts:
        missing = [_label(c) for c in pq.concepts if c not in h.matched]
        parts.append("মেলেনি: " + ", ".join(missing))
    return f"- [{d.note_id}] " + " — ".join(parts)


def generate_answer(pq: ParsedQuery, selected: List[Hit], top_hits: List[Hit], max_patients=8) -> str:
    """Extractive, grounded answer in Bengali. Nothing is generated that is not copied from a note."""
    exact = [h for h in selected if h.concept_cov == 1.0 or not pq.concepts]
    partial = [h for h in selected if h not in exact]
    if not selected:
        best = top_hits[0] if top_hits else None
        hint = ""
        if best is not None and best.matched:
            hint = f" (সবচেয়ে কাছের নোট: [{best.doc.note_id}], আংশিক মিল: {', '.join(_label(c) for c in best.matched)})"
        return ("নথিভুক্ত নোটে এই প্রশ্নের শর্ত পূরণ করে এমন কোনো রোগী পাওয়া যায়নি, তাই অনুমানভিত্তিক উত্তর দেওয়া হলো না" + hint + "।")
    primary = [h for h in exact if h.primary]
    agg = primary if len(primary) >= 3 else exact
    lines = []
    if exact:
        head = f"নথিতে {bn(len(exact))} জন রোগীর নোট পাওয়া গেছে যেখানে প্রশ্নের সব শর্ত মিলেছে।"
        if 0 < len(primary) < len(exact) and len(primary) >= 3:
            head += f" এর মধ্যে {bn(len(primary))}টিতে রোগটি নোটের নির্ণয় (Dx) অংশে আছে; সারসংক্ষেপ সেগুলোর ভিত্তিতে তৈরি।"
    else:
        head = f"প্রশ্নের সব শর্ত একসাথে মেলে এমন নোট নেই; আংশিক মিল আছে এমন {bn(len(partial))}টি নোট নিচে দেওয়া হলো (কোন শর্ত মেলেনি তা উল্লেখ করা আছে)।"
    lines.append(head)
    docs = [h.doc for h in (agg if exact else partial)]
    src_cite = lambda ds: " ".join(f"[{d.note_id}]" for d in ds[:6]) + (" …" if len(ds) > 6 else "")

    def tally(types, restrict=None):
        cnt, who, rep = Counter(), defaultdict(list), {}
        for d in docs:
            for ty in types:
                for m in _entities(d, ty):
                    if restrict and not restrict(m):
                        continue
                    k = _canon_med(m)
                    cnt[k] += 1
                    who[k].append(d)
                    rep.setdefault(k, m)
        return cnt, who, rep

    if pq.intent == "medication":
        restrict = (lambda m: bool(find_concepts(keys_of(m))[0] & {"antibiotic", "amoxicillin"})) if pq.antibiotic_only else None
        cnt, who, rep = tally(("MEDICATION",), restrict)
        if cnt:
            lines.append("নথিভুক্ত ওষুধ (কতগুলো নোটে): " + "; ".join(
                f"{rep[k]} ({bn(c)}টি নোট, {src_cite(who[k])})" for k, c in cnt.most_common(10)) + "।")
    elif pq.intent == "symptom":
        cnt, who, rep = tally(("SYMPTOM",))
        if cnt:
            lines.append("উল্লেখিত উপসর্গ: " + "; ".join(f"{rep[k]} ({bn(c)}টি নোট)" for k, c in cnt.most_common(10)) + "।")
    elif pq.intent == "findings":
        cnt, who, rep = tally(("SYMPTOM", "TEST_RESULT"))
        if cnt:
            lines.append("পাওয়া ক্লিনিক্যাল ফাইন্ডিংস/ফলাফল: " + "; ".join(f"{rep[k]} ({bn(c)}টি নোট)" for k, c in cnt.most_common(12)) + "।")
    elif pq.intent == "treatment":
        cnt, who, rep = tally(("MEDICATION", "PROCEDURE"))
        if cnt:
            lines.append("দেওয়া চিকিৎসা: " + "; ".join(f"{rep[k]} ({bn(c)}টি নোট)" for k, c in cnt.most_common(10)) + "।")

    lines.append("রোগীভিত্তিক বিবরণ (উৎস নোট সহ):")
    shown = sorted(exact, key=lambda h: (not h.primary, -h.score, h.doc.note_id))[:max_patients]
    for h in shown:
        lines.append(_patient_line(h, pq))
    if len(exact) > max_patients:
        rest = sorted(exact, key=lambda h: (not h.primary, -h.score, h.doc.note_id))[max_patients:max_patients + 12]
        lines.append(f"- … আরও {bn(len(exact) - max_patients)}টি মিলে যাওয়া নোট: " + " ".join(f"[{h.doc.note_id}]" for h in rest))
    if partial and exact:
        lines.append("আংশিক মিল (সব শর্ত নয়):")
    for h in (partial if exact else sorted(partial, key=lambda h: (-h.concept_cov, -h.score))[:max_patients]):
        lines.append(_patient_line(h, pq))
    return "\n".join(lines)
