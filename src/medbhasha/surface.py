"""Group entity strings by a stemmed key but DISPLAY the most frequent original surface form
(so reports show 'তীব্র আর্টিকারিয়া', not the stem 'তীব্ আর্টিকারিয়া')."""
import re
from collections import Counter, defaultdict

from .lexicon import stem

_PUNCT = re.compile(r"[()\[\],;:।]")


def key_of(text: str) -> str:
    return " ".join(stem(w) for w in _PUNCT.sub(" ", text).split())


def clean(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text).split())


class SurfaceMap:
    def __init__(self):
        self._c = defaultdict(Counter)

    def add(self, text: str) -> str:
        k = key_of(text)
        if k:
            self._c[k][clean(text)] += 1
        return k

    def show(self, k: str) -> str:
        items = self._c.get(k)
        if not items:
            return k
        return sorted(items.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
