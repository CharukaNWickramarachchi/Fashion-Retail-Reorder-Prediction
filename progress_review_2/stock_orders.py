"""Batch scenario calculator: rejects unknown stock; does not send orders."""
import argparse
from pathlib import Path
import pandas as pd
from src.data import ROOT
from src.forecast import reorder_quantity

REQUIRED = ["SKU", "forecast_week", "weekly_usage", "on_hand", "on_order", "backorders", "lead_days", "review_days", "safety_units", "stock_basis"]

def calculate(frame):
    missing = set(REQUIRED)-set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    d = frame.copy()
    if d[REQUIRED].isna().any().any():
        raise ValueError("Unknown inputs found. Supply all required fields; unknown stock is not zero.")
    if d.SKU.astype(str).str.strip().eq("").any() or d.stock_basis.astype(str).str.strip().eq("").any():
        raise ValueError("SKU and stock_basis must not be empty.")
    if d.duplicated(["SKU", "forecast_week"]).any():
        raise ValueError("Duplicate SKU and forecast week.")
    pd.to_datetime(d.forecast_week, format="%Y-%m-%d", errors="raise")
    for c in REQUIRED[2:-1]:
        d[c] = pd.to_numeric(d[c], errors="raise")
    if d.review_days.le(0).any():
        raise ValueError("Review days must be greater than zero.")
    for c in ["on_hand", "on_order", "backorders", "lead_days", "review_days", "safety_units"]:
        if not d[c].mod(1).eq(0).all():
            raise ValueError(f"{c} must contain whole numbers.")
    quantities = []
    for r in d.itertuples():
        quantities.append(reorder_quantity(r.weekly_usage, r.on_hand, r.on_order, r.backorders,
                                          r.lead_days, r.review_days, r.safety_units))
    d["inventory_position"] = d.on_hand+d.on_order-d.backorders
    d["target_units"] = d.weekly_usage*(d.lead_days+d.review_days)/7+d.safety_units
    d["suggested_order_units"] = quantities
    d["action"] = ["Order suggested at review" if q else "No order at this review" for q in quantities]
    return d

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT/"inputs/stock_scenarios.csv")
    p.add_argument("--output", type=Path, default=ROOT/"results/custom_stock_orders.csv")
    a = p.parse_args()
    try:
        result = calculate(pd.read_csv(a.input))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(a.output, index=False)
    except (ValueError, OSError) as exc:
        p.exit(1, f"Input error: {exc}\n")
    print(f"Saved {len(result)} scenario calculations to {a.output}. These are not purchase orders.")
