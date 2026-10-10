"""Reproducible preparation with a fixed historical cohort and complete weeks."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NUMERIC_FEATURES = ["lag_1", "lag_2", "lag_3", "lag_4", "mean_4", "std_4", "zero_weeks_4", "change_1"]
CATEGORICAL_FEATURES = ["category", "size"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def config(root=ROOT):
    return json.loads((Path(root) / "config.json").read_text())


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def add_features(panel):
    """All numeric predictors depend strictly on earlier complete calendar weeks."""
    out = panel.sort_values(["SKU", "week_start"]).reset_index(drop=True).copy()
    group = out.groupby("SKU", sort=False)["units"]
    for k in range(1, 5):
        out[f"lag_{k}"] = group.shift(k)
    lag = out[[f"lag_{k}" for k in range(1, 5)]]
    out["mean_4"] = lag.mean(axis=1)
    out["std_4"] = lag.std(axis=1, ddof=0)
    out["zero_weeks_4"] = (lag == 0).sum(axis=1)
    out["change_1"] = out["lag_1"] - out["lag_2"]
    out["history_available_through"] = out["week_start"] - pd.Timedelta(days=1)
    return out


def clean_stock(raw):
    """Keep undated stock separate from modelling; never impute unknown stock to zero."""
    stock = raw.copy()
    stock.columns = stock.columns.str.strip()
    stock = stock.dropna(subset=["SKU Code"])
    stock["SKU"] = stock["SKU Code"].astype(str).str.strip()
    stock = stock[~stock["SKU"].isin(["", "#REF!"])]
    stock["Stock"] = pd.to_numeric(stock["Stock"], errors="coerce")
    stock.loc[stock.Stock < 0, "Stock"] = np.nan
    # Duplicated SKUs have unclear warehouse/date semantics. Expose ambiguity;
    # do not choose an arbitrary max and present it as a known current balance.
    def unique_stock(s):
        values = s.dropna().unique()
        return float(values[0]) if len(values) == 1 else np.nan
    result = stock.groupby("SKU", as_index=False).agg(
        undated_stock_snapshot=("Stock", unique_stock),
        distinct_stock_values=("Stock", "nunique"),
        source_rows=("Stock", "size"),
        display_color=("Color", "first"),
    )
    result["usable_as_current_stock"] = False
    return result


def prepare(root=ROOT):
    root = Path(root)
    cfg = config(root)
    raw_path = root / "data/raw/Amazon Sale Report.csv"
    stock_path = root / "data/raw/Sale Report.csv"
    raw = pd.read_csv(raw_path, low_memory=False)
    raw.columns = raw.columns.str.strip()
    raw_rows = len(raw)
    # Exact record duplicates only; preserve legitimate multi-item orders.
    raw = raw.drop(columns=["index", "Unnamed: 22"], errors="ignore")
    duplicate_count = int(raw.duplicated().sum())
    raw = raw.drop_duplicates().copy()
    raw["date"] = pd.to_datetime(raw["Date"], format="%m-%d-%y", errors="coerce")
    raw["SKU"] = raw["SKU"].astype("string").str.strip()
    valid_key = raw["date"].notna() & raw.SKU.notna() & raw.SKU.ne("")
    invalid_keys = int((~valid_key).sum())
    raw = raw.loc[valid_key].copy()
    raw["Qty"] = pd.to_numeric(raw.Qty, errors="coerce")
    raw["Amount"] = pd.to_numeric(raw.Amount, errors="coerce")
    valid_qty = raw.Qty.notna() & np.isfinite(raw.Qty) & raw.Qty.ge(0) & raw.Qty.mod(1).eq(0)
    invalid_qty = int((~valid_qty).sum())
    raw = raw.loc[valid_qty].copy()
    raw["Qty"] = raw.Qty.astype(int)
    raw["category"] = raw.Category.fillna("UNKNOWN").astype(str).str.strip().str.upper()
    raw["size"] = raw.Size.fillna("UNKNOWN").astype(str).str.strip().str.upper().replace({"XXXL": "3XL", "XXL": "2XL"})
    raw["Status"] = raw.Status.fillna("UNKNOWN").str.strip()
    start = pd.Timestamp(cfg["first_week"])
    end = pd.Timestamp(cfg["last_complete_week"]) + pd.Timedelta(days=6)
    if raw.date.min() > start or raw.date.max() < end:
        raise ValueError("Raw data does not cover the configured full-week window.")
    window = raw[raw.date.between(start, end)].copy()
    missing_days = pd.date_range(start, end).difference(window.date.unique())
    if len(missing_days):
        raise ValueError(f"Dataset-wide missing dates: {missing_days.tolist()}")
    cutoff = pd.Timestamp(cfg["cohort_cutoff"])
    initial = window[window.date < cutoff].sort_values("date", kind="stable")
    # Cohort and attributes fixed using only the initial four-week history.
    cohort = initial.drop_duplicates("SKU")[["SKU", "category", "size", "date"]].rename(columns={"date": "first_observed_date"})
    cohort = cohort.sort_values("SKU").reset_index(drop=True)
    shipped = window[window.Status.isin(cfg["accepted_statuses"])].copy()
    shipped["week_start"] = shipped.date - pd.to_timedelta(shipped.date.dt.dayofweek, unit="D")
    aggregate = shipped.groupby(["SKU", "week_start"]).Qty.sum()
    weeks = pd.date_range(start, pd.Timestamp(cfg["last_complete_week"]), freq="7D")
    index = pd.MultiIndex.from_product([cohort.SKU, weeks], names=["SKU", "week_start"])
    weekly = aggregate.reindex(index, fill_value=0).rename("units").reset_index()
    weekly = weekly.merge(cohort, on="SKU", how="left", validate="many_to_one")
    panel = add_features(weekly)
    panel = panel[panel.week_start >= cutoff].dropna(subset=["lag_4"]).copy()
    panel["split"] = "train"
    panel.loc[panel.week_start.isin(pd.to_datetime(cfg["validation_weeks"])), "split"] = "validation"
    panel.loc[panel.week_start.isin(pd.to_datetime(cfg["test_weeks"])), "split"] = "test"
    if panel[FEATURES].isna().any().any():
        raise ValueError("Unexpected missing model feature.")
    if panel.duplicated(["SKU", "week_start"]).any():
        raise ValueError("Duplicate SKU-week row.")
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(processed / "cohort.csv", index=False)
    weekly.to_csv(processed / "weekly_sales.csv", index=False)
    panel.to_csv(processed / "model_panel.csv", index=False)
    clean_stock(pd.read_csv(stock_path)).to_csv(processed / "stock_snapshot_reference.csv", index=False)
    status = window.groupby("Status").agg(rows=("Qty", "size"), units=("Qty", "sum")).reset_index()
    status["included_in_target"] = status.Status.isin(cfg["accepted_statuses"])
    status.to_csv(results / "status_policy_audit.csv", index=False)
    audit = {
        "raw_transaction_rows": raw_rows, "exact_duplicate_rows_removed": duplicate_count,
        "invalid_key_rows_removed": invalid_keys, "invalid_quantity_rows_removed": invalid_qty,
        "raw_date_range": [str(raw.date.min().date()), str(raw.date.max().date())],
        "full_week_window": [str(start.date()), str(end.date())],
        "rows_outside_full_week_window": int((~raw.date.between(start, end)).sum()),
        "target_definition": "Recorded units on orders with status Shipped or Shipped - Delivered to Buyer, aggregated by order date.",
        "cohort_SKUs": len(cohort), "complete_weeks": len(weeks),
        "model_rows": len(panel), "rows_by_split": panel.split.value_counts().to_dict(),
        "new_SKUs_after_cohort_cutoff_excluded": int(window.loc[window.date >= cutoff, "SKU"].nunique() - window.loc[(window.date >= cutoff) & window.SKU.isin(cohort.SKU), "SKU"].nunique()),
        "heldout_units_for_excluded_new_SKUs": int(shipped.loc[(shipped.date >= pd.Timestamp(cfg['test_weeks'][0])) & ~shipped.SKU.isin(cohort.SKU), "Qty"].sum()),
        "zero_target_fraction": float(panel.units.eq(0).mean()),
        "missing_amount_rows_in_accepted_window": int(shipped.Amount.isna().sum()),
        "features": FEATURES,
        "source_sha256": {raw_path.name: hashlib.sha256(raw_path.read_bytes()).hexdigest(), stock_path.name: hashlib.sha256(stock_path.read_bytes()).hexdigest()},
        "assumptions_and_limits": [
            "No transaction in a week means zero recorded shipped units; availability and lost demand are unknown.",
            "Before first observation, zero denotes no recorded sales, not proof that a product was listed.",
            "Fixed cohort contains SKUs seen in the initial four weeks, including cancelled-only items; new products are excluded.",
            "Final order statuses are retrospective snapshot labels; event-time status history is unavailable.",
            "Stock is undated and never used as a historical model predictor or a validated current balance.",
            "Category and size are first observed pre-cutoff attributes; subsequent attribute changes are not modelled.",
            "Only twelve complete weeks and two final test weeks are available; no annual seasonality or deployment claims.",
            "Data appears to concern Amazon India; Sri Lankan retailer applicability has not been validated.",
            "Original data publisher and reuse licence must be verified before external redistribution."
        ]
    }
    save_json(results / "data_audit.json", audit)
    print(f"Prepared {len(cohort):,} SKUs; {len(panel):,} modelling rows; full weeks {start.date()} to {end.date()}.")
    return panel
