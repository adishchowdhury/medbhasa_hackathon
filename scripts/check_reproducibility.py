#!/usr/bin/env python3
"""Re-run the whole pipeline under a different PYTHONHASHSEED and compare every output file
with the hashes recorded in outputs/run_manifest.json (from the previous run).

    python scripts/check_reproducibility.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
before = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))["output_md5"]
env = dict(os.environ, PYTHONHASHSEED="12345")
subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py")], check=True, env=env)
after = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))["output_md5"]
bad = [k for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)]
print("files compared:", len(after))
if bad:
    print("DIFFERENT:", bad)
    sys.exit(1)
print("REPRODUCIBLE: all output files are byte-identical.")
