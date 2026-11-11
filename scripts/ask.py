#!/usr/bin/env python3
"""Live query from the command line (works from any directory, no installation needed).

    python scripts/ask.py "হার্ট ফেইলিওর আছে এমন রোগীদের বর্ণনা দিন।"
    python scripts/ask.py --team-only "ডায়াবেটিস রোগীদের জন্য কোন কোন ওষুধ নথিভুক্ত করা আছে?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from medbhasha.ask import main  # noqa: E402

if __name__ == "__main__":
    main()
