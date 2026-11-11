"""Fast sanity tests:  python -m pytest -q   (or: python tests/test_basics.py)"""
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medbhasha import corpus, rag  # noqa: E402
from medbhasha.config import ENTITY_TYPES, R1  # noqa: E402
from medbhasha.tokenizer import tokenize  # noqa: E402


def test_tokenizer_keeps_abbreviations_and_splits_punctuation():
    t = tokenize("মেটফরমিন ৫০০ মি.গ্রা. দিনে ২ বার। রক্তচাপ ১১০/৭০; সর্দি-কাশি।")
    assert "মি.গ্রা." in t and "১১০/৭০" in t and "সর্দি-কাশি" in t
    assert t[-1] == "।" and ";" in t


def test_markup_to_bio_roundtrip():
    text, spans = corpus.parse_markup("রোগীর {TMP|গত তিন দিন ধরে} {SX|জ্বর} {NEG|নেই}।")
    toks, tags = corpus.to_bio(text, spans)
    assert tags == ["O", "B-TEMPORAL", "I-TEMPORAL", "I-TEMPORAL", "I-TEMPORAL", "B-SYMPTOM", "B-NEGATION_UNCERTAINTY", "O"]
    assert [(s, e, ty) for s, e, ty in corpus.spans_from_bio(tags)][0] == (1, 5, "TEMPORAL")


def test_corpus_file_is_valid_bio_and_large_enough():
    rows = list(csv.DictReader(open(R1 / "annotated_corpus.csv", encoding="utf-8")))
    assert list(rows[0]) == ["note_id", "token", "tag"]
    assert len(rows) >= 10000                      # organisers' recommended size
    valid = {"O"} | {p + "-" + t for p in "BI" for t in ENTITY_TYPES}
    assert {r["tag"] for r in rows} <= valid
    counts = {t: sum(1 for r in rows if r["tag"] == "B-" + t) for t in ENTITY_TYPES}
    assert min(counts.values()) >= 50, counts      # no rare entity type left under-represented


def test_official_validator_passes():
    r = subprocess.run([sys.executable, str(ROOT / "resources" / "validate.py"), "--input", str(R1 / "annotated_corpus.csv"),
                        "--report", "/tmp/_val.txt"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_no_note_copied_from_release():
    """Training notes must not be near-duplicates of the organiser-released test notes (char 5-gram Jaccard)."""
    import pandas as pd
    def grams(s):
        s = "".join(s.split())
        return {s[i:i + 5] for i in range(len(s) - 4)}
    train = pd.read_csv(R1 / "notes_raw.csv", encoding="utf-8")
    test = pd.read_csv(ROOT / "data/round2/test_notes.csv", encoding="utf-8")
    tg = [grams(t) for t in test["note_text"]]
    worst = 0.0
    for t in train["text"]:
        g = grams(t)
        for h in tg:
            worst = max(worst, len(g & h) / max(1, len(g | h)))
    assert worst < 0.5, worst


def test_retrieval_handles_negation_and_spelling_variants():
    assert rag.key("মেটফর্মিন") == rag.key("মেটফরমিন")
    pos, neg, fam = rag.concepts_by_polarity("পূর্ব ইতিহাস: ডায়াবেটিস ও উচ্চ রক্তচাপ নেই; মায়ের ডায়াবেটিস ছিল।")
    assert "diabetes" not in pos and "diabetes" in neg | fam


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
