#!/usr/bin/env python3
"""Create the submission archive  dist/JUSL_Health_Home_MedBhasha.zip  from the repository tree.

Everything except .git, dist/, caches and checkpoints is included (the zip lives in dist/ so the
archive never contains itself).
"""
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "JUSL_Health_Home_MedBhasha"
SKIP_DIRS = {".git", "dist", "__pycache__", ".ipynb_checkpoints", ".venv", "venv", ".pytest_cache"}
dist = ROOT / "dist"
dist.mkdir(exist_ok=True)
out = dist / f"{NAME}.zip"
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = sorted(d for d in dn if d not in SKIP_DIRS)
        for f in sorted(fn):
            p = Path(dp) / f
            if p.suffix == ".pyc":
                continue
            info = zipfile.ZipInfo.from_file(p, f"{NAME}/{p.relative_to(ROOT).as_posix()}")
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
            n += 1
print(f"{out}  ({out.stat().st_size/1e6:.1f} MB, {n} files)")
