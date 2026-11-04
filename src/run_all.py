"""Run the whole pipeline from the raw CSV.

Usage:  python -m src.run_all --stages audit e1 e2 [e3] [--quick]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["audit", "e1", "e2"], choices=["audit", "e1", "e2", "e3"])
    ap.add_argument("--quick", action="store_true", help="2 trials, 3-fold CV; writes to results/_quick")
    args = ap.parse_args()
    if args.quick:
        os.environ["CVD_QUICK"] = "1"

    from src.config import path  # imported after CVD_QUICK is set
    from src.data import audit
    from src import run_e1, run_e2, run_e3

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    (path("results") / "environment.txt").write_text(
        subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    if "audit" in args.stages:
        log(f"Audit: {audit.run_audit()}")
    if "e1" in args.stages:
        run_e1.main(log)
    if "e2" in args.stages:
        run_e2.main(log)
    if "e3" in args.stages:
        run_e3.main(log)


if __name__ == "__main__":
    main()
