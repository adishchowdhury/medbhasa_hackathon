"""Deterministic tokenizer used for training AND for the hidden test notes.

Rules (identical to the Round 1 submission):
  * split on whitespace;
  * leading/trailing punctuation is split off as separate tokens;
  * internal hyphens, slashes and dots stay inside a token (``১১০/৭০``,
    ``সর্দি-কাশি``, ``৮.২``);
  * an abbreviation that ends in a dot such as ``মি.গ্রা.`` keeps its final dot.
"""
import re
from typing import List, Tuple

PUNCT = set("।,;:!?()[]{}\"'“”‘’—–|​")
# Abbreviation with internal dots and a final dot: মি.গ্রা.  মি.মি.  Dr.
_ABBR = re.compile(r"^[ঀ-৿A-Za-z]+(?:\.[ঀ-৿A-Za-z]+)+\.$")


def _split_edges(chunk: str, offset: int) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    start, end = 0, len(chunk)
    lead: List[Tuple[str, int, int]] = []
    trail: List[Tuple[str, int, int]] = []
    while start < end and (chunk[start] in PUNCT):
        lead.append((chunk[start], offset + start, offset + start + 1))
        start += 1
    while end > start and (chunk[end - 1] in PUNCT):
        trail.append((chunk[end - 1], offset + end - 1, offset + end))
        end -= 1
    # a trailing '.' is punctuation unless the word is an abbreviation
    core = chunk[start:end]
    while core.endswith(".") and not _ABBR.match(core) and len(core) > 1:
        trail.append((".", offset + end - 1, offset + end))
        end -= 1
        core = chunk[start:end]
        # punctuation may be revealed again behind the dot
        while end > start and chunk[end - 1] in PUNCT:
            trail.append((chunk[end - 1], offset + end - 1, offset + end))
            end -= 1
        core = chunk[start:end]
    if core:
        out.append((core, offset + start, offset + end))
    return lead + out + trail[::-1]


def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Return ``[(token, char_start, char_end), ...]``."""
    tokens: List[Tuple[str, int, int]] = []
    for m in re.finditer(r"\S+", text):
        tokens.extend(_split_edges(m.group(0), m.start()))
    return tokens


def tokenize(text: str) -> List[str]:
    return [t for t, _, _ in tokenize_with_offsets(text)]


def sentence_split(tokens: List[str]) -> List[Tuple[int, int]]:
    """Sentence spans over token indices (split on the Bengali danda / '?' / '!')."""
    spans, s = [], 0
    for i, t in enumerate(tokens):
        if t in ("।", "?", "!"):
            spans.append((s, i + 1))
            s = i + 1
    if s < len(tokens):
        spans.append((s, len(tokens)))
    return spans
