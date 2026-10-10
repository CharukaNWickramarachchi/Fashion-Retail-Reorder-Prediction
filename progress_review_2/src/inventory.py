"""Explicit weekly inventory replay. No source stock snapshot is used."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from .data import ROOT
from .forecast import reorder_quantity


def simulate(demand, estimates, initial_stock, lead_weeks=1, safety_units=2):
    """Receive -> review/order -> fulfil; lost-demand proxy, integer stock units."""
    if isinstance(lead_weeks, bool) or not isinstance(lead_weeks, int) or lead_weeks < 1:
        raise ValueError("Weekly simulator requires an integer lead time of at least one week.")
    if len(demand) != len(estimates) or not len(demand):
        raise ValueError("Demand and estimate series must have equal nonzero length.")
    values = [initial_stock, safety_units, *demand, *estimates]
    if any(v is None or not np.isfinite(v) or v < 0 for v in values):
        raise ValueError("All simulation values must be finite and nonnegative.")
    if initial_stock != int(initial_stock) or any(x != int(x) for x in demand):
        raise ValueError("Physical stock and observed demand proxy must be whole units.")
    hand, pipeline, rows = int(initial_stock), {}, []
    for t, (wanted, estimate) in enumerate(zip(demand, estimates)):
        opening = hand
        receipt = pipeline.pop(t, 0)
        hand += receipt
        on_order_before = sum(pipeline.values())
        target = float(estimate) * (lead_weeks + 1) + safety_units
        quantity = reorder_quantity(float(estimate), hand, on_order_before,
            lead_time_days=lead_weeks*7, review_period_days=7, safety_stock_units=safety_units)
        pipeline[t + lead_weeks] = pipeline.get(t + lead_weeks, 0) + quantity
        fulfilled = min(hand, int(wanted))
        hand -= fulfilled
        rows.append(dict(week_index=t, opening_stock=opening, received_units=receipt,
            available_before_demand=opening+receipt, estimated_weekly_usage=float(estimate),
            target_stock=target, on_order_before=on_order_before,
            inventory_position_before_order=opening+receipt+on_order_before,
            ordered_units=quantity, order_due_week_index=t+lead_weeks,
            demand_proxy=int(wanted), fulfilled_units=fulfilled,
            unmet_proxy_units=int(wanted)-fulfilled, ending_stock=hand,
            outstanding_units=sum(pipeline.values())))
    return rows


def choose_subset(panel, count):
    train = panel[panel.split == "train"]
    ranked = train.groupby("SKU", as_index=False).agg(train_units=("units", "sum"), train_mean=("units", "mean"))
    return ranked.sort_values(["train_units", "SKU"], ascending=[False, True], kind="stable").head(count).reset_index(drop=True)


def build_inventory(root=ROOT):
    root = Path(root)
    cfg = json.loads((root / "inventory_config.json").read_text())
    if cfg["review_weeks"] != 1:
        raise ValueError("This simulator supports weekly reviews only.")
    panel = pd.read_csv(root / "data/processed/model_panel.csv")
    subset = choose_subset(panel, cfg["subset_size"])
    subset.to_csv(root / "results/inventory_subset.csv", index=False)
    selected = json.loads((root / "results/selection.json").read_text())["selected_model"]
    predictions = pd.read_csv(root / "results/test_predictions.csv")
    rows = []
    for scenario in cfg["scenarios"]:
        for item in subset.itertuples():
            initial = math.ceil(item.train_mean * scenario["initial_cover_weeks"])
            for model in ["Mean4Weeks", selected]:
                series = predictions[(predictions.SKU == item.SKU) & (predictions.model == model)].sort_values("week_start")
                if len(series) != 2:
                    raise ValueError("Expected exactly two held-out weeks per selected SKU and method.")
                ledger = simulate(series.actual.tolist(), series.predicted.tolist(), initial, cfg["lead_weeks"], cfg["safety_units"])
                for step, week in zip(ledger, series.week_start):
                    rows.append(dict(scenario=scenario["name"], model=model, SKU=item.SKU,
                                     week_start=week, initial_stock=initial, **step))
    ledger = pd.DataFrame(rows)
    ledger.to_csv(root / "results/inventory_ledger.csv", index=False)
    metrics = []
    for (scenario, model), g in ledger.groupby(["scenario", "model"], sort=False):
        last = g.sort_values("week_start").groupby("SKU", sort=False).tail(1)
        demand = int(g.demand_proxy.sum())
        metrics.append(dict(scenario=scenario, model=model, sku_count=g.SKU.nunique(),
            weeks=g.week_start.nunique(), demand_proxy_units=demand,
            fulfilled_units=int(g.fulfilled_units.sum()), unmet_proxy_units=int(g.unmet_proxy_units.sum()),
            simulated_fill_rate_pct=100*g.fulfilled_units.sum()/demand if demand else None,
            mean_end_stock_per_sku_week=float(g.ending_stock.mean()),
            order_events=int(g.ordered_units.gt(0).sum()), ordered_units=int(g.ordered_units.sum()),
            final_stock_units=int(last.ending_stock.sum()), final_in_transit_units=int(last.outstanding_units.sum())))
    summary = pd.DataFrame(metrics)
    summary.to_csv(root / "results/inventory_comparison.csv", index=False)
    config_out = {**cfg, "selected_model":selected,
        "weeks": sorted(ledger.week_start.unique().tolist()),
        "subset_train_last_week": panel.loc[panel.split == "train", "week_start"].max(),
        "scope": "Exploratory simulated comparison, not observed inventory performance"}
    (root / "results/inventory_assumptions.json").write_text(json.dumps(config_out, indent=2), encoding="utf-8")
    forecast = pd.read_csv(root / "results/next_week_forecast.csv")
    demo = subset.merge(forecast, on="SKU", validate="one_to_one")
    demo["assumed_on_hand"] = np.ceil(demo.train_mean).astype(int)
    demo["assumed_on_order"] = 0
    demo["assumed_backorders"] = 0
    demo["lead_days"] = cfg["lead_weeks"] * 7
    demo["review_days"] = 7
    demo["safety_units"] = cfg["safety_units"]
    demo["target_units"] = demo.predicted_weekly_shipped_units * (cfg["lead_weeks"] + 1) + cfg["safety_units"]
    demo["suggested_order_units"] = [reorder_quantity(r.predicted_weekly_shipped_units, r.assumed_on_hand,
        lead_time_days=r.lead_days, review_period_days=r.review_days, safety_stock_units=r.safety_units) for r in demo.itertuples()]
    demo["action"] = np.where(demo.suggested_order_units > 0, "Order suggested at review", "No order at this review")
    demo["stock_basis"] = "ASSUMED training mean; not observed current stock"
    demo.to_csv(root / "results/stock_decisions.csv", index=False)
    from .inventory_report import write_inventory_report
    write_inventory_report(root, summary, demo, selected)
    print("Stock decisions, inventory simulation, and STOCK_REVIEW.html saved.")
    return summary
