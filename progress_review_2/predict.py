"""Inspect one saved historical forecast and optionally calculate a stock scenario."""
import argparse
import json
from pathlib import Path
import pandas as pd
from src.forecast import reorder_quantity

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Historical forecast demonstration, not a current 2026 forecast.")
    parser.add_argument("--sku", help="SKU from results/next_week_forecast.csv; omit for the highest predicted SKU")
    parser.add_argument("--on-hand", type=float)
    parser.add_argument("--on-order", type=float, default=0)
    parser.add_argument("--backorders", type=float, default=0)
    parser.add_argument("--lead-days", type=float, default=7)
    parser.add_argument("--review-days", type=float, default=7)
    parser.add_argument("--safety-units", type=float, default=0)
    args = parser.parse_args()
    data = pd.read_csv(ROOT / "results/next_week_forecast.csv")
    rows = data[data.SKU == args.sku] if args.sku else data.nlargest(1, "predicted_weekly_shipped_units")
    if rows.empty:
        parser.error("SKU is not in the historical cohort. Choose one from results/next_week_forecast.csv.")
    row = rows.iloc[0].to_dict()
    if args.on_hand is not None:
        try:
            row["scenario_reorder_quantity"] = reorder_quantity(row["predicted_weekly_shipped_units"], args.on_hand,
                args.on_order, args.backorders, args.lead_days, args.review_days, args.safety_units)
        except ValueError as exc:
            parser.error(str(exc))
        row["scenario_inputs"] = {k: v for k, v in vars(args).items() if k != "sku"}
        row["scenario_only"] = True
        row["assumption"] = "Forecast daily rate stays constant over lead time plus review period; costs and service level are not optimised."
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
