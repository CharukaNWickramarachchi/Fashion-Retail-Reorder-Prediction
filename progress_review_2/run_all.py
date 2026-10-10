"""Entry point; all paths resolve relative to this file, not the shell directory."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Rebuild the Progress Review 2 modelling project.")
    parser.add_argument("--stage", choices=["all", "prepare", "train", "forecast", "inventory", "verify"], default="all")
    args = parser.parse_args()
    try:
        from src.data import prepare
        from src.train import train
        from src.forecast import make_forecast
        from src.inventory import build_inventory
        from verify import verify
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}. Run: python -m pip install -r requirements.txt")
    if args.stage in ["all", "prepare"]:
        prepare(ROOT)
    if args.stage in ["all", "train"]:
        train(ROOT)
    if args.stage in ["all", "forecast"]:
        make_forecast(ROOT)
    if args.stage in ["all", "inventory"]:
        build_inventory(ROOT)
    if args.stage in ["all", "verify"]:
        verify()
    print("Done. Open results/STOCK_REVIEW.html to view saved results.")


if __name__ == "__main__":
    main()
