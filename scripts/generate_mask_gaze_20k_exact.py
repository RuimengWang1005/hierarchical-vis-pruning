#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generate_mask_gaze_20k_exact import main


if __name__ == "__main__":
    raise SystemExit(main())

