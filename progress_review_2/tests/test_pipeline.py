"""Meaningful data, temporal leakage, saved-model and inventory checks."""
import json
from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from src.data import ROOT, FEATURES, NUMERIC_FEATURES, add_features, clean_stock
from src.forecast import reorder_quantity
from src.models import predict_nonnegative
from src.train import select_model, score


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = pd.read_csv(ROOT / "data/processed/model_panel.csv", parse_dates=["week_start", "history_available_through"])
        cls.weekly = pd.read_csv(ROOT / "data/processed/weekly_sales.csv", parse_dates=["week_start"])

    def test_unique_complete_week_grid(self):
        d = self.weekly.sort_values(["SKU", "week_start"])
        self.assertFalse(d.duplicated(["SKU", "week_start"]).any())
        gaps = d.groupby("SKU").week_start.diff().dropna()
        self.assertTrue(gaps.eq(pd.Timedelta(days=7)).all())
        self.assertTrue(d.groupby("SKU").size().eq(12).all())
        self.assertTrue(d.units.ge(0).all())

    def test_zero_sales_weeks_are_preserved(self):
        self.assertGreater(int(self.weekly.units.eq(0).sum()), 0)
        zero_then_next = self.weekly.sort_values(["SKU", "week_start"])
        built = add_features(zero_then_next)
        expected = zero_then_next.groupby("SKU").units.shift(1)
        np.testing.assert_allclose(built.lag_1.to_numpy(), expected.to_numpy(), equal_nan=True)

    def test_chronological_disjoint_splits(self):
        d = self.panel
        self.assertLess(d[d.split == "train"].week_start.max(), d[d.split == "validation"].week_start.min())
        self.assertLess(d[d.split == "validation"].week_start.max(), d[d.split == "test"].week_start.min())
        self.assertTrue((d.history_available_through < d.week_start).all())

    def test_future_sales_mutation_does_not_change_earlier_features(self):
        origin = pd.Timestamp("2022-06-13")
        old = add_features(self.weekly)
        mutated = self.weekly.copy()
        mutated.loc[mutated.week_start >= origin, "units"] += 99999
        new = add_features(mutated)
        mask = old.week_start <= origin
        assert_frame_equal(old.loc[mask, FEATURES].reset_index(drop=True), new.loc[mask, FEATURES].reset_index(drop=True))

    def test_forbidden_predictors_are_absent(self):
        forbidden = {"units", "beginning_stock", "ending_stock", "stock_cover_proxy", "top_ship_state", "revenue", "item_class", "final_window_ending_stock"}
        self.assertFalse(forbidden.intersection(FEATURES))
        self.assertTrue(np.isfinite(self.panel[NUMERIC_FEATURES].to_numpy()).all())

    def test_attributes_are_known_before_modelling(self):
        cohort = pd.read_csv(ROOT / "data/processed/cohort.csv", parse_dates=["first_observed_date"])
        self.assertTrue((cohort.first_observed_date < self.panel.week_start.min()).all())
        self.assertEqual(set(cohort.SKU), set(self.panel.SKU))

    def test_validation_only_selection(self):
        m = pd.read_csv(ROOT / "results/model_comparison.csv")
        choice = json.loads((ROOT / "results/selection.json").read_text())
        self.assertEqual(select_model(m[m.stage == "validation"]), choice["selected_model"])
        self.assertFalse(choice["test_used_for_selection"])
        mutated = m.copy()
        mutated.loc[mutated.stage == "test", "MAE"] = -9999
        self.assertEqual(select_model(mutated[mutated.stage == "validation"]), choice["selected_model"])

    def test_saved_models_reproduce_all_test_predictions(self):
        pred = pd.read_csv(ROOT / "results/test_predictions.csv")
        test = self.panel[self.panel.split == "test"].copy()
        for name, rows in pred.groupby("model"):
            artifact = joblib.load(ROOT / f"models/evaluation_{name}.joblib")
            self.assertLess(pd.Timestamp(artifact["training_target_last_week"]), test.week_start.min())
            actual = predict_nonnegative(artifact["model"], test[FEATURES])
            np.testing.assert_allclose(actual, rows.predicted.to_numpy(), rtol=1e-9, atol=1e-9)
            self.assertTrue((actual >= 0).all())

    def test_metrics_recompute_from_predictions(self):
        pred = pd.read_csv(ROOT / "results/test_predictions.csv")
        m = pd.read_csv(ROOT / "results/model_comparison.csv")
        for name, rows in pred.groupby("model"):
            values = score(rows.actual, rows.predicted)
            saved = m[(m.stage == "test") & (m.model == name)].iloc[0]
            for key in ["MAE", "RMSE", "WAPE_pct", "R2"]:
                self.assertAlmostEqual(values[key], saved[key], places=9)

    def test_forecast_uses_separate_refitted_model(self):
        artifact = joblib.load(ROOT / "models/forecast_model.joblib")
        inputs = pd.read_csv(ROOT / "data/processed/forecast_features.csv")
        output = pd.read_csv(ROOT / "results/next_week_forecast.csv")
        self.assertLess(pd.Timestamp(artifact["training_target_last_week"]), pd.Timestamp(inputs.week_start.min()))
        self.assertTrue(output.forecast_type.eq("historical_demonstration_not_current_forecast").all())
        np.testing.assert_allclose(predict_nonnegative(artifact["model"], inputs[FEATURES]), output.predicted_weekly_shipped_units, rtol=1e-9, atol=1e-9)

    def test_unknown_categories_do_not_crash(self):
        artifact = joblib.load(ROOT / "models/evaluation_RandomForestAttributes.joblib")
        x = self.panel[FEATURES].head(3).copy()
        x["category"] = "NEVER_SEEN_CATEGORY"
        x["size"] = "NEVER_SEEN_SIZE"
        self.assertTrue(np.isfinite(predict_nonnegative(artifact["model"], x)).all())

    def test_unrelated_stock_columns_cannot_change_predictions(self):
        artifact = joblib.load(ROOT / "models/evaluation_RandomForestAttributes.joblib")
        x = self.panel[FEATURES].head(5).copy()
        a = predict_nonnegative(artifact["model"], x)
        x["beginning_stock"] = 999999
        x["ending_stock"] = -12345
        np.testing.assert_allclose(a, predict_nonnegative(artifact["model"], x))

    def test_stock_conflict_and_unknown_stay_unknown(self):
        raw = pd.DataFrame({"SKU Code": ["A", "A", "B", "C"], "Stock": [10, 20, np.nan, 0], "Color": ["RED"] * 4})
        clean = clean_stock(raw).set_index("SKU")
        self.assertTrue(pd.isna(clean.loc["A", "undated_stock_snapshot"]))
        self.assertTrue(pd.isna(clean.loc["B", "undated_stock_snapshot"]))
        self.assertEqual(clean.loc["C", "undated_stock_snapshot"], 0)

    def test_reorder_inventory_position_and_rounding(self):
        # 10/week * 14/7 + 2 safety - (5 on hand + 3 on order - 1 backlog) = 15
        self.assertEqual(reorder_quantity(10, 5, 3, 1, 7, 7, 2), 15)
        self.assertEqual(reorder_quantity(10, 100), 0)
        self.assertEqual(reorder_quantity(.1, 0, lead_time_days=0, review_period_days=7), 1)
        for bad in [None, np.nan, np.inf, -1]:
            with self.assertRaises(ValueError):
                reorder_quantity(10, bad)

    def test_reorder_samples_are_explicit_scenarios(self):
        d = pd.read_csv(ROOT / "results/reorder_scenarios.csv")
        self.assertTrue(d.scenario_only.all())
        for r in d.itertuples():
            self.assertEqual(r.recommended_units, reorder_quantity(r.predicted_weekly_shipped_units, r.assumed_on_hand,
                r.assumed_on_order, r.assumed_backorders, r.lead_time_days, r.review_period_days, r.safety_stock_units))

    def test_raw_files_match_source_manifest(self):
        import hashlib
        manifest = json.loads((ROOT / "data/source_manifest.json").read_text())
        for name, entry in manifest.items():
            self.assertEqual(hashlib.sha256((ROOT / "data/raw" / name).read_bytes()).hexdigest(), entry["sha256"])

    def test_weekly_target_matches_status_filtered_raw_records(self):
        cfg = json.loads((ROOT / "config.json").read_text())
        raw = pd.read_csv(ROOT / "data/raw/Amazon Sale Report.csv", low_memory=False)
        raw.columns = raw.columns.str.strip()
        raw = raw.drop(columns=["index", "Unnamed: 22"], errors="ignore").drop_duplicates()
        raw["date"] = pd.to_datetime(raw.Date, format="%m-%d-%y", errors="coerce")
        raw["SKU"] = raw.SKU.astype("string").str.strip()
        raw["Qty"] = pd.to_numeric(raw.Qty, errors="coerce")
        last_day = self.weekly.week_start.max() + pd.Timedelta(days=6)
        rows = raw[raw.date.between(self.weekly.week_start.min(), last_day)
            & raw.Status.str.strip().isin(cfg["accepted_statuses"])
            & raw.SKU.isin(self.weekly.SKU.unique())
            & raw.Qty.ge(0) & raw.Qty.mod(1).eq(0)].copy()
        rows["week_start"] = rows.date - pd.to_timedelta(rows.date.dt.dayofweek, unit="D")
        expected = rows.groupby(["SKU", "week_start"]).Qty.sum()
        actual = self.weekly.set_index(["SKU", "week_start"]).units
        np.testing.assert_array_equal(actual.to_numpy(), expected.reindex(actual.index, fill_value=0).to_numpy())

    def test_results_page_local_links_exist(self):
        from html.parser import HTMLParser
        class LinkCollector(HTMLParser):
            def __init__(self):
                super().__init__()
                self.paths = []
            def handle_starttag(self, tag, attrs):
                for key, value in attrs:
                    if key in ["src", "href"] and value and not value.startswith(("https:", "http:", "#")):
                        self.paths.append(value)
        links = LinkCollector()
        links.feed((ROOT / "results/OPEN_RESULTS.html").read_text())
        self.assertGreater(len(links.paths), 0)
        for path in links.paths:
            # verification.json is written at the end of this test run.
            if path != "verification.json":
                self.assertTrue((ROOT / "results" / path).is_file(), path)


if __name__ == "__main__":
    unittest.main()
