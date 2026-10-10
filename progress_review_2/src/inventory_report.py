"""Self-contained local HTML stock demonstration with transparent assumptions."""
import json
import html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSS = """body{margin:0;background:#f3f6fa;color:#152a43;font:16px/1.65 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:32px 24px}h1{font-size:36px;line-height:1.15}h2{margin-top:34px}a{color:#096a76}section,.card{background:white;border:1px solid #dce4ed;border-radius:12px;padding:22px;margin:18px 0}.tag{font-weight:700;color:#096a76}.notice{background:#fff4d9;padding:16px;border-left:4px solid #b97912}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;border-bottom:1px solid #dce4ed;text-align:left}.scroll{overflow:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}label{display:block;font-size:14px}input,select,button{font:inherit;padding:8px;box-sizing:border-box;width:100%;border:1px solid #9cabbc;border-radius:5px}button{background:#096a76;color:white;cursor:pointer;margin-top:16px}output{display:block;font-size:20px;font-weight:700;margin-top:16px}img{max-width:100%}nav{display:flex;gap:22px;flex-wrap:wrap}code{background:#eef2f5;padding:3px} @media print{body{background:white}main{padding:0}section{break-inside:avoid}button{display:none}}"""


def write_inventory_report(root, summary, demo, selected):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, metric, label in zip(axes, ["unmet_proxy_units", "mean_end_stock_per_sku_week"],
                                ["Unmet proxy units (lower is better)", "Mean end-of-week units per SKU"]):
        pivot = summary.pivot(index="scenario", columns="model", values=metric)
        pivot.plot.bar(ax=ax, color=["#96a6bb", "#087d89"], rot=15)
        ax.set_xlabel("Assumed starting-stock scenario")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Exploratory inventory simulation | 100 training-selected SKUs | 2 weeks")
    fig.tight_layout()
    fig.savefig(root/"results/figures/inventory_comparison.png", dpi=150)
    plt.close(fig)
    cards = []
    for scenario, g in summary.groupby("scenario", sort=False):
        baseline = g[g.model == "Mean4Weeks"].iloc[0]
        model = g[g.model == selected].iloc[0]
        cards.append(f"<li><b>{html.escape(scenario)}</b>: {selected} unmet proxy = {model.unmet_proxy_units:,}; four-week average = {baseline.unmet_proxy_units:,}. Mean ending stock per SKU-week: {model.mean_end_stock_per_sku_week:.2f} versus {baseline.mean_end_stock_per_sku_week:.2f} units.</li>")
    tablecols = ["scenario", "model", "unmet_proxy_units", "simulated_fill_rate_pct", "mean_end_stock_per_sku_week", "order_events", "ordered_units", "final_in_transit_units"]
    stockcols = ["SKU", "category", "predicted_weekly_shipped_units", "assumed_on_hand", "suggested_order_units", "action"]
    payload = demo[["SKU", "predicted_weekly_shipped_units", "assumed_on_hand"]].to_json(orient="records").replace("<", "\\u003c")
    template = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stock Replenishment | Group Review</title><style>__CSS__</style><main>
<p class="tag">GROUP 05 · PROGRESS REVIEW 2 · RESEARCH PROTOTYPE</p><h1>Stock replenishment decisions,<br>with evidence your group can explain.</h1>
<p>Decide how many units to order at a weekly stock review. Historical usage estimates feed a transparent stock calculation. A controlled simulation compares two ways of estimating usage.</p>
<nav><a href="../docs/GROUP_GUIDE.html">Read the group guide first</a><a href="OPEN_RESULTS.html">Forecast model evidence</a><a href="verification.json">Verification</a></nav>
<div class="notice"><b>Historical demonstration, not live shop advice.</b> Sales records are from 2022. Stock balances and delivery settings below are assumed. The source stock snapshot is undated and is not used as current stock. No Sri Lankan pilot or measured savings are claimed.</div>
<section><h2>1. What did the stock simulation show?</h2><p>100 SKUs selected using training-period volume only. Two held-out weeks starting 13 and 20 June 2022. Compare the validation-selected __MODEL__ with a four-week average under the same weekly ordering rule.</p><ul>__FINDINGS__</ul>
<p>Read shortages and stock held together. Fewer ordered units alone does not prove a better policy. Orders still in transit at the end are reported, not discarded.</p>
<img src="figures/inventory_comparison.png" alt="Simulated unmet proxy units and mean week-end stock for both methods under three starting-stock scenarios">
<div class="scroll">__SUMMARY__</div><p><a href="inventory_comparison.csv">Download comparison CSV</a> · <a href="inventory_ledger.csv">Every stock movement</a> · <a href="inventory_assumptions.json">All assumptions</a> · <a href="inventory_subset.csv">SKU selection</a></p></section>
<section><h2>2. Try a stock decision</h2><p>Select a product to load its historical forecast for 27 June 2022, or adjust the usage estimate. All stock inputs are illustrative and editable. Changes here stay in your browser and do not change the saved experiment.</p>
<label>Product<select id="sku"></select></label><div class="grid">
<label>Expected units per week<input id="usage" type="number" min="0" step="any"></label>
<label>Available units<input id="hand" type="number" min="0" step="1"></label>
<label>Incoming units<input id="incoming" type="number" min="0" step="1" value="0"></label>
<label>Backordered units<input id="backlog" type="number" min="0" step="1" value="0"></label>
<label>Delivery time (days)<input id="lead" type="number" min="0" step="1" value="7"></label>
<label>Review interval (days)<input id="review" type="number" min="1" step="1" value="7"></label>
<label>Safety stock (units)<input id="safety" type="number" min="0" step="1" value="2"></label></div>
<button id="calculate">Calculate suggested order</button><output id="answer" aria-live="polite"></output><p id="math"></p>
<p>Target = weekly usage × (delivery days + review days) ÷ 7 + safety stock.<br>Inventory position = available + incoming − backorders.<br>Suggested quantity = target − inventory position, rounded up; minimum zero.</p><p>Assumes incoming orders arrive in the relevant period, a constant usage rate, no order-size constraints and owner approval. It is a periodic weekly review, not a separately fitted reorder-point model.</p></section>
<section><h2>3. Product-level stock recommendations</h2><p>First 20 of the 100 training-selected products. Assumed available stock = rounded-up training weekly average; no incoming orders or backorders; seven-day delivery and seven-day review; safety stock two units.</p><div class="scroll">__STOCK__</div><p><a href="stock_decisions.csv">All 100 product recommendations</a> · <a href="next_week_forecast.csv">Full-cohort historical usage estimates</a></p></section>
<section><h2>4. What this experiment cannot establish</h2><ul><li>Historical shipped units are a demand proxy. Unrecorded lost demand is unknown.</li><li>The same external historical series feeds both approaches even if the simulation runs out of stock. This is a controlled replay, not a complete closed-loop shop simulation.</li><li>Only two evaluation weeks. Initial stock heavily affects outcomes; three scenarios are shown without choosing a winning setup.</li><li>The inventory experiment was added after forecast scores were available. It is exploratory, not a fresh independent holdout.</li><li>Week-end stock averages are not daily average stock. No cost, profitability, waste or sustainability benefits were measured.</li><li>A local pilot needs dated stock balances, receipts, returns, supplier lead times and stockout records. New products and annual seasonality need more data.</li></ul></section>
</main><script id="products" type="application/json">__PRODUCTS__</script><script>
const products=JSON.parse(document.getElementById('products').textContent);
const $=id=>document.getElementById(id);
products.forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent=p.SKU;$('sku').appendChild(o)});
function calc(){const ids=['usage','hand','incoming','backlog','lead','review','safety'];const raw=ids.map(id=>$(id).value);const v=raw.map(Number);if(raw.some(x=>x.trim()==='')||v.some(x=>!Number.isFinite(x)||x<0)||v[5]<=0||v.slice(1).some(x=>!Number.isInteger(x))){$('answer').textContent='Enter known, nonnegative values; stock counts and days must be whole numbers, and review days must exceed zero.';$('math').textContent='';return;}const [usage,hand,incoming,backlog,lead,review,safety]=v;const target=usage*(lead+review)/7+safety;const pos=hand+incoming-backlog;const q=Math.ceil(Math.max(0,target-pos));$('answer').textContent=q>0?'Suggested order: '+q+' units':'No order suggested at this review';$('math').textContent='Target '+target.toFixed(2)+' units − inventory position '+pos+' units → '+q+' units. Scenario only.';}
function select(){const p=products[Number($('sku').value)];$('usage').value=p.predicted_weekly_shipped_units;$('hand').value=p.assumed_on_hand;calc();}
$('sku').addEventListener('change',select);$('calculate').addEventListener('click',calc);select();
</script></html>'''
    for key, value in {"__CSS__":CSS, "__MODEL__":html.escape(selected), "__FINDINGS__":"".join(cards),
                       "__SUMMARY__":summary[tablecols].to_html(index=False, border=0, float_format=lambda v:f"{v:.2f}"),
                       "__STOCK__":demo[stockcols].head(20).to_html(index=False, border=0, float_format=lambda v:f"{v:.2f}"),
                       "__PRODUCTS__":payload}.items():
        template = template.replace(key, value)
    (root/"results/STOCK_REVIEW.html").write_text(template, encoding="utf-8")
