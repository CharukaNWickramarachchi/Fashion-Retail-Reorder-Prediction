"""Historical forecast export and explicit, non-optimised reorder scenarios."""
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import ROOT, FEATURES, add_features
from .models import predict_nonnegative


def reorder_quantity(weekly_sales, on_hand, on_order=0, backorders=0,
                     lead_time_days=7, review_period_days=7, safety_stock_units=0):
    values = [weekly_sales, on_hand, on_order, backorders, lead_time_days, review_period_days, safety_stock_units]
    if any(v is None or not np.isfinite(v) or v < 0 for v in values):
        raise ValueError("All scenario inputs must be known, finite, nonnegative numbers. Unknown stock is not zero.")
    # Simple periodic-review order-up-to scenario. Constant daily rate over the
    # protection period is an assumption; a one-week model does not validate it.
    cover_days = lead_time_days + review_period_days
    target = weekly_sales * cover_days / 7 + safety_stock_units
    inventory_position = on_hand + on_order - backorders
    return int(math.ceil(max(0.0, target - inventory_position)))


def make_forecast(root=ROOT):
    root = Path(root)
    weekly = pd.read_csv(root / "data/processed/weekly_sales.csv", parse_dates=["week_start", "first_observed_date"])
    last = weekly.week_start.max()
    next_week = last + pd.Timedelta(days=7)
    future = weekly[weekly.week_start == last].copy()
    future["week_start"] = next_week
    future["units"] = np.nan
    features = add_features(pd.concat([weekly, future], ignore_index=True))
    future = features[features.week_start == next_week].copy()
    saved = joblib.load(root / "models/forecast_model.joblib")
    future["predicted_weekly_shipped_units"] = predict_nonnegative(saved["model"], future[FEATURES])
    result = future[["SKU", "category", "size", "week_start", "history_available_through", "predicted_weekly_shipped_units"]].copy()
    result["model"] = saved["name"]
    result["forecast_type"] = "historical_demonstration_not_current_forecast"
    result.to_csv(root / "results/next_week_forecast.csv", index=False)
    future[["SKU", "week_start"] + FEATURES].to_csv(root / "data/processed/forecast_features.csv", index=False)
    # Assumed values, not the undated source snapshot.
    examples = result.nlargest(6, "predicted_weekly_shipped_units").copy()
    examples["assumed_on_hand"] = [0, 5, 10, 20, 50, 100][:len(examples)]
    examples["assumed_on_order"] = 0
    examples["assumed_backorders"] = 0
    examples["lead_time_days"] = 7
    examples["review_period_days"] = 7
    examples["safety_stock_units"] = 2
    examples["scenario_only"] = True
    examples["recommended_units"] = [reorder_quantity(r.predicted_weekly_shipped_units, r.assumed_on_hand,
        lead_time_days=r.lead_time_days, review_period_days=r.review_period_days, safety_stock_units=r.safety_stock_units) for r in examples.itertuples()]
    examples.to_csv(root / "results/reorder_scenarios.csv", index=False)
    print(f"Historical forecast exported for week starting {next_week.date()}; illustrative reorder scenarios saved.")
    return result
