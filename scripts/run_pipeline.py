#!/usr/bin/env python3
"""Headless run of the whole pipeline (same code the notebook calls).

    python scripts/run_pipeline.py          # full: EDA + NER CV/ablations + predictions + clustering + RAG
    python scripts/run_pipeline.py --fast   # skip the ablation CV (about 1 minute)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from medbhasha.run_all import run_everything  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip NER ablation cross-validation")
    a = ap.parse_args()
    run_everything(fast=a.fast)
