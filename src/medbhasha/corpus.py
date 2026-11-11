"""Build the annotated corpus (BIO CSV) from hand-annotated span markup.

Annotation source format (data/round1/annotation_sources/*.txt), one note per
line, three TAB-separated fields::

    note_id <TAB> domain <TAB> text with {CODE|annotated span} markup

Example::

    n041	Cardiology	রোগীর {TMP|গত দুই দিন ধরে} {BODY|বুকে} {SX|ব্যথা} আছে।

The builder strips the markup, tokenizes the plain text with
``medbhasha.tokenizer`` and projects every span onto tokens as B-/I- tags.
Every span MUST start and end on a token boundary and spans cannot nest;
violations raise ``ValueError`` so annotation mistakes cannot slip through.
"""
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

from .config import ENTITY_TYPES, SHORT_CODES, R1
from .tokenizer import PUNCT, tokenize_with_offsets

_SPAN = re.compile(r"\{([A-Z]+)\|([^{}]+)\}")


def parse_markup(marked: str) -> Tuple[str, List[Tuple[int, int, str]]]:
    """Return ``(plain_text, [(start, end, ENTITY_TYPE), ...])``."""
    plain, spans, pos = [], [], 0
    out_len = 0
    for m in _SPAN.finditer(marked):
        code, body = m.group(1), m.group(2)
        if code not in SHORT_CODES:
            raise ValueError(f"unknown code {code!r} in: {marked[:80]}")
        pre = marked[pos:m.start()]
        plain.append(pre)
        out_len += len(pre)
        spans.append((out_len, out_len + len(body), SHORT_CODES[code]))
        plain.append(body)
        out_len += len(body)
        pos = m.end()
    plain.append(marked[pos:])
    text = "".join(plain)
    if "{" in text or "}" in text or "|" in text.replace("।", ""):
        # stray braces / pipes mean a malformed span
        if "{" in text or "}" in text:
            raise ValueError(f"malformed markup: {marked[:120]}")
    return text, spans


def _hyphen_cut(text, s, e, a, b):
    """A span may stop right before / start right after a hyphen that glues a suffix
    to the token (``{TEST|ইসিজি}-তে`` -> the whole token ``ইসিজি-তে`` is tagged)."""
    ok = True
    if b > e:
        ok &= text[e:e + 1] == "-"
    if a < s:
        ok &= text[s - 1:s] == "-"
    return ok


def to_bio(text: str, spans: List[Tuple[int, int, str]]):
    toks = tokenize_with_offsets(text)
    tags = ["O"] * len(toks)
    for s, e, ty in spans:
        idx = [i for i, (_, a, b) in enumerate(toks) if a < e and b > s]
        if not idx:
            raise ValueError(f"span {text[s:e]!r} covers no token")
        for i in idx:
            _, a, b = toks[i]
            if (a < s or b > e) and not _hyphen_cut(text, s, e, a, b):
                raise ValueError(
                    f"span {text[s:e]!r} cuts through token {toks[i][0]!r} in: {text[:80]}")
            if tags[i] != "O":
                raise ValueError(f"overlapping spans at token {toks[i][0]!r}")
        if any(toks[i][0] in PUNCT for i in idx):
            raise ValueError(f"punctuation inside span {text[s:e]!r} (schema: punctuation must be O)")
        tags[idx[0]] = "B-" + ty
        for i in idx[1:]:
            tags[i] = "I-" + ty
    return [t for t, _, _ in toks], tags


def spans_from_bio(tags: List[str]) -> List[Tuple[int, int, str]]:
    """Token-index spans ``(start, end_exclusive, type)`` from BIO tags."""
    spans, cur = [], None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-") or tag == "O" or (tag.startswith("I-") and (cur is None or cur[2] != tag[2:])):
            if cur is not None:
                spans.append((cur[0], i, cur[2]))
                cur = None
            if tag.startswith("B-") or tag.startswith("I-"):
                cur = (i, None, tag[2:])
    return spans


def detokenize(tokens: List[str]) -> str:
    """Rebuild readable text from tokens (used for Round-1 notes whose raw text was not kept)."""
    out = ""
    for i, t in enumerate(tokens):
        if i == 0:
            out = t
        elif t in ("।", ",", ";", ":", "?", "!", ")", "”") or out.endswith(("(", "“")):
            out += t
        else:
            out += " " + t
    return out


def read_sources(src_dir: Path = R1 / "annotation_sources"):
    """Yield (note_id, domain, plain_text, tokens, tags) from every *.txt batch file."""
    for f in sorted(Path(src_dir).glob("*.txt")):
        for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{f.name}:{ln}: expected 3 tab-separated fields")
            nid, domain, marked = parts
            text, spans = parse_markup(marked)
            try:
                tokens, tags = to_bio(text, spans)
            except ValueError as e:
                raise ValueError(f"{f.name}:{ln} ({nid}): {e}") from None
            yield nid, domain, text, tokens, tags


def read_bio_csv(path: Path) -> Dict[str, Tuple[List[str], List[str]]]:
    notes: Dict[str, Tuple[List[str], List[str]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            toks, tags = notes.setdefault(r["note_id"], ([], []))
            toks.append(r["token"])
            tags.append(r["tag"])
    return notes


def write_bio_csv(path: Path, notes: List[Tuple[str, List[str], List[str]]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["note_id", "token", "tag"])
        for nid, toks, tags in notes:
            for t, g in zip(toks, tags):
                w.writerow([nid, t, g])
