"""Central configuration: paths, schema, random seeds.

Everything that affects reproducibility lives here.
"""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
R1 = DATA / "round1"
R2 = DATA / "round2"
OUT = ROOT / "outputs"
RESOURCES = ROOT / "resources"

ENTITY_TYPES = [
    "DIAGNOSIS", "SYMPTOM", "MEDICATION", "DOSAGE", "TEST",
    "TEST_RESULT", "BODY_SITE", "PROCEDURE", "TEMPORAL", "NEGATION_UNCERTAINTY",
]

# Short codes used only in the hand-authored annotation sources
# (data/round1/annotation_sources/*.txt), e.g.  {SX|জ্বর}
SHORT_CODES = {
    "DX": "DIAGNOSIS", "SX": "SYMPTOM", "MED": "MEDICATION", "DOS": "DOSAGE",
    "TEST": "TEST", "RES": "TEST_RESULT", "BODY": "BODY_SITE",
    "PROC": "PROCEDURE", "TMP": "TEMPORAL", "NEG": "NEGATION_UNCERTAINTY",
}
