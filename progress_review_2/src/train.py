"""Fit once per evaluation stage; keep the final test out of model selection."""
import html
import json
import platform
from pathlib import Path
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import ROOT, FEATURES, config, save_json
from .models import make_candidates, predict_nonnegative


def score(y, pred):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    total = y.sum()
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "WAPE_pct": float(100 * np.abs(y - pred).sum() / total) if total > 0 else None,
        "R2": float(r2_score(y, pred)) if len(y) > 1 and np.var(y) > 0 else None,
        "mean_error_pred_minus_actual": float(np.mean(pred - y)),
        "rows": len(y), "actual_total": float(total), "predicted_total": float(pred.sum())
    }


def select_model(validation_metrics):
    return validation_metrics.sort_values(["MAE", "RMSE", "model"], kind="stable").iloc[0]["model"]


def plot_results(results, root, test_predictions, selected, importance):
    root = Path(root)
    figdir = root / "results/figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    test = results[results.stage == "test"].sort_values("MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    colors = ["#147d92" if x == selected else "#94a3b8" for x in test.model]
    ax.barh(test.model, test.MAE, color=colors)
    ax.set_xlabel("MAE (recorded shipped units per SKU-week; lower is better)")
    ax.set_title("Final two-week test | Highlighted model selected on validation")
    for i, value in enumerate(test.MAE):
        ax.text(value + .01, i, f"{value:.3f}", va="center")
    ax.set_xlim(0, test.MAE.max() * 1.2)
    fig.tight_layout(); fig.savefig(figdir / "test_model_comparison.png", dpi=160); plt.close(fig)
    chosen = test_predictions[test_predictions.model == selected]
    totals = chosen.groupby("week_start")[["actual", "predicted"]].sum()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    totals.plot.bar(ax=ax, color=["#172b4d", "#147d92"], rot=0)
    ax.set_ylabel("Recorded shipped units across the fixed SKU cohort")
    ax.set_xlabel("Forecast week start")
    ax.set_title(f"Observed and predicted totals | {selected}")
    ax.legend(["Observed", "Predicted"])
    fig.tight_layout(); fig.savefig(figdir / "test_weekly_totals.png", dpi=160); plt.close(fig)
    imp = importance.sort_values("mean_MAE_increase")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(imp.feature, imp.mean_MAE_increase, xerr=imp.std_MAE_increase, color="#147d92")
    ax.axvline(0, color="#64748b", linewidth=.8)
    ax.set_xlabel("Increase in validation MAE after shuffling a feature")
    ax.set_title("RandomForestAttributes | Validation permutation importance")
    fig.tight_layout(); fig.savefig(figdir / "validation_feature_importance.png", dpi=160); plt.close(fig)


def write_results_html(root, metrics, selected):
    root = Path(root)
    audit = json.loads((root / "results/data_audit.json").read_text())
    test = metrics[metrics.stage == "test"].copy()
    selected_row = test[test.model == selected].iloc[0]
    columns = ["model", "MAE", "RMSE", "WAPE_pct", "R2"]
    rows = metrics[["stage"] + columns].to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0)
    limits = "".join(f"<li>{html.escape(x)}</li>" for x in audit["assumptions_and_limits"])
    output = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fashion retail model results</title><style>
body{{font:16px/1.6 system-ui,sans-serif;background:#f4f7fa;color:#172b4d;margin:0}}main{{max-width:1120px;margin:auto;padding:36px 24px}}
h1{{font-size:32px;line-height:1.2}}h2{{margin-top:38px}}.label{{color:#147d92;font-weight:700}}img{{width:100%;max-width:1000px;background:white;border-radius:8px;margin:14px 0}}
table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #d7e0ea}}.scroll{{overflow:auto}}a{{color:#126879}}.metrics{{display:flex;gap:32px;flex-wrap:wrap}}.metrics b{{display:block;font-size:28px}}code{{background:#e6edf3;padding:2px 5px}}
</style><main><p class="label">PROGRESS REVIEW 2 / MODEL BUILDING AND TESTING</p>
<p><a href="STOCK_REVIEW.html">Open stock replenishment review</a></p><h1>Forecasting component: evaluation</h1><p>Historical experiment using twelve complete weeks, a fixed cohort of {audit['cohort_SKUs']:,} SKUs, and one-week-ahead predictions. All quantities are recorded shipped units, not unconstrained customer demand.</p>
<p>Selected on validation: <strong>{html.escape(selected)}</strong>. Test weeks start 13 and 20 June 2022. Models are frozen before both test weeks; each forecast uses observations available through the preceding Sunday.</p>
<div class="metrics"><div>Test MAE<b>{selected_row.MAE:.3f}</b></div><div>Test RMSE<b>{selected_row.RMSE:.3f}</b></div><div>Test WAPE<b>{selected_row.WAPE_pct:.1f}%</b></div></div>
<h2>Measured model comparison</h2><p>Validation weeks: 30 May and 6 June. Model selection uses validation MAE, then RMSE for ties. Test scores are reported without changing the selection. WAPE is an error measure, not accuracy.</p><div class="scroll">{rows}</div>
<img src="figures/test_model_comparison.png" alt="MAE comparison across predeclared models on the final test">
<img src="figures/test_weekly_totals.png" alt="Actual and predicted total shipped units in the two test weeks">
<h2>Feature check</h2><p>Permutation importance is computed on validation data for RandomForestAttributes. It measures predictive reliance, not causation. Negative values can occur.</p>
<img src="figures/validation_feature_importance.png" alt="Validation feature permutation importance">
<h2>Scope and limitations</h2><ul>{limits}</ul>
<h2>Files</h2><ul><li><a href="model_comparison.csv">Model scores</a></li><li><a href="test_predictions.csv">Test predictions</a></li><li><a href="test_metrics_by_week.csv">Per-week test metrics</a></li><li><a href="test_metrics_by_category.csv">Per-category test metrics</a></li><li><a href="next_week_forecast.csv">Historical next-week forecast demonstration</a></li><li><a href="reorder_scenarios.csv">Illustrative reorder scenarios with assumed stock</a></li><li><a href="verification.json">Automated verification results</a></li><li><a href="run_summary.json">Run metadata and model selection</a></li></ul>
<p>Stock snapshot values are not model inputs. Reorder scenarios are assumptions, not evaluated optimal inventory policies. No measured profit improvement is claimed.</p></main></html>"""
    (root / "results/OPEN_RESULTS.html").write_text(output, encoding="utf-8")


def train(root=ROOT):
    root = Path(root)
    cfg = config(root)
    data = pd.read_csv(root / "data/processed/model_panel.csv")
    train_df = data[data.split == "train"]
    val_df = data[data.split == "validation"]
    test_df = data[data.split == "test"]
    assert train_df.week_start.max() < val_df.week_start.min() <= val_df.week_start.max() < test_df.week_start.min()
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    results = root / "results"
    metric_rows, val_predictions, val_fitted = [], [], {}
    candidates = make_candidates(cfg)
    with threadpool_limits(limits=2):
        for name, model in candidates.items():
            print(f"Validation: {name}", flush=True)
            model.fit(train_df[FEATURES], train_df.units)
            pred = predict_nonnegative(model, val_df[FEATURES])
            metric_rows.append({"stage": "validation", "model": name, **score(val_df.units, pred)})
            val_predictions.append(pd.DataFrame({"SKU": val_df.SKU, "week_start": val_df.week_start, "model": name, "actual": val_df.units, "predicted": pred}))
            val_fitted[name] = model
        selected = select_model(pd.DataFrame(metric_rows))
        selection = {"selected_model": selected, "criterion": "validation MAE, then RMSE, then model name", "test_used_for_selection": False, "config": cfg}
        # Persist decision before any test-model fitting or evaluation.
        save_json(results / "selection.json", selection)
        pd.concat(val_predictions).to_csv(results / "validation_predictions.csv", index=False)
        # Interpretation on validation only, independent of test results.
        importance = permutation_importance(val_fitted["RandomForestAttributes"], val_df[FEATURES], val_df.units,
            scoring="neg_mean_absolute_error", n_repeats=3, random_state=cfg["seed"], n_jobs=1)
        importance_df = pd.DataFrame({"feature": FEATURES, "mean_MAE_increase": importance.importances_mean, "std_MAE_increase": importance.importances_std})
        importance_df.to_csv(results / "validation_feature_importance.csv", index=False)
        development = pd.concat([train_df, val_df], ignore_index=True)
        test_predictions = []
        for name, template in candidates.items():
            print(f"Final test: {name}", flush=True)
            model = clone(template).fit(development[FEATURES], development.units)
            pred = predict_nonnegative(model, test_df[FEATURES])
            metric_rows.append({"stage": "test", "model": name, **score(test_df.units, pred)})
            test_predictions.append(pd.DataFrame({"SKU": test_df.SKU, "week_start": test_df.week_start, "category": test_df.category, "model": name, "actual": test_df.units, "predicted": pred}))
            joblib.dump({"model": model, "name": name, "features": FEATURES, "training_target_last_week": development.week_start.max(), "purpose": "Frozen final-test model"}, models_dir / f"evaluation_{name}.joblib", compress=3)
        # Refit only the validation-selected algorithm for a separate historical
        # next-week demonstration. This artifact is NEVER used to score the test.
        final_model = clone(candidates[selected]).fit(data[FEATURES], data.units)
        joblib.dump({"model": final_model, "name": selected, "features": FEATURES, "training_target_last_week": data.week_start.max(), "purpose": "Historical next-week forecast demonstration; refit includes test period"}, models_dir / "forecast_model.joblib", compress=3)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(results / "model_comparison.csv", index=False)
    predictions = pd.concat(test_predictions, ignore_index=True)
    predictions.to_csv(results / "test_predictions.csv", index=False)
    for group_col, filename in [("week_start", "test_metrics_by_week.csv"), ("category", "test_metrics_by_category.csv")]:
        grouped = [{"model": model, group_col: key, **score(g.actual, g.predicted)} for (model, key), g in predictions.groupby(["model", group_col])]
        pd.DataFrame(grouped).to_csv(results / filename, index=False)
    chosen = predictions[predictions.model == selected]
    chosen.assign(absolute_error=(chosen.actual - chosen.predicted).abs()).sort_values("absolute_error", ascending=False).head(100).to_csv(results / "largest_test_errors.csv", index=False)
    summary = {
        "selected_model": selected, "selection_metric": "validation MAE", "test_used_for_selection": False,
        "protocol": cfg["forecast_protocol"],
        "train_weeks": sorted(train_df.week_start.unique().tolist()),
        "validation_weeks": sorted(val_df.week_start.unique().tolist()),
        "test_weeks": sorted(test_df.week_start.unique().tolist()),
        "selected_test_metrics": score(chosen.actual, chosen.predicted),
        "features": FEATURES, "seed": cfg["seed"],
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "joblib", "threadpoolctl"]},
        "technical_references": ["https://scikit-learn.org/stable/common_pitfalls.html", "https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data"],
        "limitations": "Two test weeks; fixed pre-existing cohort; retrospective final statuses; no lost demand, stock history, annual seasonality, or business-impact validation."
    }
    save_json(results / "run_summary.json", summary)
    plot_results(metrics, root, predictions, selected, importance_df)
    write_results_html(root, metrics, selected)
    print(f"Selected on validation: {selected}; test MAE = {summary['selected_test_metrics']['MAE']:.4f}")
    return summary
