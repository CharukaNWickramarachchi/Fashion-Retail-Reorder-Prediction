"""Inventory conservation, timing, fair comparison and rejected bad inputs."""
import unittest
import json
import numpy as np
import pandas as pd
from src.data import ROOT
from src.inventory import simulate, choose_subset
from src.forecast import reorder_quantity
from stock_orders import calculate

class InventoryTests(unittest.TestCase):
    def test_hand_worked_receipt_timing(self):
        rows = simulate([8,8], [10,10], 5, lead_weeks=1, safety_units=2)
        self.assertEqual([r['ordered_units'] for r in rows], [17,5])
        self.assertEqual([r['received_units'] for r in rows], [0,17])
        self.assertEqual([r['unmet_proxy_units'] for r in rows], [3,0])
        self.assertEqual(rows[-1]['ending_stock'], 9)
        self.assertEqual(rows[-1]['outstanding_units'], 5)

    def test_long_lead_keeps_orders_in_position(self):
        rows = simulate([5,5,5], [10,10,10], 10, lead_weeks=2, safety_units=0)
        self.assertEqual([r['received_units'] for r in rows], [0,0,20])
        self.assertEqual([r['ordered_units'] for r in rows], [20,5,5])

    def test_zero_demand_and_sufficient_stock(self):
        r = simulate([0,0], [0,0], 50, safety_units=0)
        self.assertTrue(all(x['ordered_units']==0 and x['ending_stock']==50 for x in r))

    def test_bad_inputs_rejected(self):
        for bad in [-1, np.nan, np.inf, None]:
            with self.assertRaises(ValueError): simulate([1], [1], bad)
        for lead in [0, -1, 0.5, True]:
            with self.assertRaises(ValueError): simulate([1], [1], 1, lead_weeks=lead)
        with self.assertRaises(ValueError): simulate([1], [1,2], 1)
        with self.assertRaises(ValueError): simulate([0.5], [1], 1)

    def test_saved_ledger_conservation_and_timing(self):
        d = pd.read_csv(ROOT/'results/inventory_ledger.csv')
        np.testing.assert_array_equal(d.opening_stock+d.received_units-d.fulfilled_units, d.ending_stock)
        np.testing.assert_array_equal(d.fulfilled_units+d.unmet_proxy_units, d.demand_proxy)
        self.assertTrue(d[['ending_stock','unmet_proxy_units','outstanding_units']].ge(0).all().all())
        for _, g in d.groupby(['scenario','model','SKU']):
            g = g.sort_values('week_index')
            self.assertEqual(g.iloc[1].received_units, g.iloc[0].ordered_units)
            self.assertEqual(g.iloc[1].opening_stock, g.iloc[0].ending_stock)
            self.assertEqual(g.iloc[-1].ending_stock+g.fulfilled_units.sum(),g.iloc[0].initial_stock+g.received_units.sum())
            self.assertEqual(g.ordered_units.sum(),g.received_units.sum()+g.iloc[-1].outstanding_units)

    def test_comparison_uses_identical_inputs(self):
        d = pd.read_csv(ROOT/'results/inventory_ledger.csv')
        for c in ['demand_proxy','initial_stock']:
            self.assertTrue(d.groupby(['scenario','SKU','week_start'])[c].nunique().eq(1).all())
        self.assertTrue(d.groupby(['scenario','SKU','week_start']).size().eq(2).all())

    def test_sku_selection_ignores_validation_and_test_targets(self):
        panel = pd.read_csv(ROOT/'data/processed/model_panel.csv')
        before = choose_subset(panel,100)
        panel.loc[panel.split!='train','units'] += 999999
        pd.testing.assert_frame_equal(before,choose_subset(panel,100))
        pd.testing.assert_frame_equal(before,pd.read_csv(ROOT/'results/inventory_subset.csv'))

    def test_summary_recomputed(self):
        ledger = pd.read_csv(ROOT/'results/inventory_ledger.csv')
        summary = pd.read_csv(ROOT/'results/inventory_comparison.csv')
        for row in summary.itertuples():
            g=ledger[(ledger.scenario==row.scenario)&(ledger.model==row.model)]
            self.assertEqual(row.unmet_proxy_units,g.unmet_proxy_units.sum())
            self.assertAlmostEqual(row.simulated_fill_rate_pct,100*g.fulfilled_units.sum()/g.demand_proxy.sum())
            self.assertAlmostEqual(row.mean_end_stock_per_sku_week,g.ending_stock.mean())
            self.assertEqual(row.ordered_units,g.ordered_units.sum())

    def test_stock_decisions_recompute(self):
        for r in pd.read_csv(ROOT/'results/stock_decisions.csv').itertuples():
            self.assertIn('ASSUMED',r.stock_basis)
            self.assertEqual(r.suggested_order_units,reorder_quantity(r.predicted_weekly_shipped_units,r.assumed_on_hand,
                r.assumed_on_order,r.assumed_backorders,r.lead_days,r.review_days,r.safety_units))

    def test_batch_inputs_known_and_valid(self):
        d=pd.read_csv(ROOT/'inputs/stock_scenarios.csv')
        self.assertEqual(calculate(d).suggested_order_units.tolist(),[15,0,1])
        for bad in [np.nan, -1, np.inf]:
            damaged = d.copy()
            damaged["on_hand"] = damaged["on_hand"].astype(float)
            damaged.loc[0, "on_hand"] = bad
            with self.assertRaises(ValueError): calculate(damaged)
        with self.assertRaises(ValueError): calculate(pd.concat([d,d]))

    def test_stock_report_assets_and_guide_exist(self):
        from html.parser import HTMLParser
        class Links(HTMLParser):
            def __init__(self): super().__init__(); self.links=[]
            def handle_starttag(self, tag, attrs):
                self.links += [v for k,v in attrs if k in ['href','src'] and not v.startswith(('http','#'))]
        p=Links(); p.feed((ROOT/'results/STOCK_REVIEW.html').read_text())
        for path in p.links:
            if path!='verification.json': self.assertTrue((ROOT/'results'/path).is_file(),path)
