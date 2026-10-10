# Stock Replenishment Decision Support for Clothing Retail

A complete Progress Review 2 research prototype: raw-data preparation, chronological model evaluation, transparent replenishment calculations, a controlled inventory simulation, saved results, and a local interactive demonstration.

Start with **docs/GROUP_GUIDE.html**. Then open **results/STOCK_REVIEW.html** in a browser. Read **docs/WINDOWS_STEPS.md** for every VS Code/PowerShell step.

## Research question

How do a simple recent-average usage estimate and a machine-learning estimate affect simulated shortages and stock held when used with the same replenishment rule?

The model estimates recorded weekly shipped units. The stock layer uses an explicit periodic-review order-up-to calculation. It does not learn an optimal reorder quantity, date or safety stock from data. Available stock and incoming orders affect the final decision.

## Run

Python 3.12, pinned dependencies in requirements.txt:

```bash
python -m pip install -r requirements.txt
python run_all.py
python verify.py
python stock_orders.py
```

Windows users should use the .venv interpreter commands in START_HERE.txt. A full run executes prepare, train, forecast, inventory and verify. Individual stages are supported through `--stage`. Core operation needs no Jupyter, Streamlit, external API or database.

## Evidence

- `results/OPEN_RESULTS.html`: measured forecast comparison, charts and limits.
- `results/STOCK_REVIEW.html`: stock comparison, editable scenario calculator and decisions.
- `results/inventory_comparison.csv`: all three initial-stock scenarios, both approaches.
- `results/inventory_ledger.csv`: receipts, orders, fulfilled and unmet proxy units, ending balances.
- `results/stock_decisions.csv`: 100 historical recommendations with assumed stock.
- `results/verification.json`: automated verification status.
- `results/package_verification.json`: clean extraction/build check for this delivered snapshot.
- `docs/GROUP_GUIDE.md`: plain-language explanation, findings, demonstration and panel questions.

## Experiment design

Twelve complete weeks; fixed 5,615-SKU cohort observed before modelling. Features use past sales only plus pre-cutoff category/size. Training: four target weeks; validation: two weeks; final test: two weeks. Ridge is selected on validation MAE. A separate refitted model produces the historical 27 June 2022 demonstration; it is not used for held-out scoring.

Inventory replay uses the top 100 training-volume SKUs and compares Mean4Weeks with the validation-selected model. Both receive identical starting conditions, external historical proxy demand, seven-day supplier lead time, weekly review and safety buffer. Three starting-stock assumptions are shown. These settings were not optimised. The inventory study was added after forecast evaluation and is explicitly exploratory.

The recorded result shows a trade-off: in the one-week starting-stock scenario Ridge holds less average ending stock but leaves more proxy demand unmet than the simple average. Do not describe this as universally improved inventory performance.

## Scope and limitations

Source transactions appear to concern India, not a Sri Lankan pilot. Historical order statuses are retrospective. Zero recorded sales do not establish no demand. Source stock is undated and excluded as a historical model feature or current balance. Simulated shortages are not observed lost sales. Two test weeks cannot establish stable annual or commercial performance. The simulator uses the same external historical demand/feature stream for both policies rather than modelling feedback from simulated stockouts. No costs, waste reduction or profit changes were measured.

A local HTML interface is included for review and scenario calculations. It is not a production stock database or purchasing system. Changes made in its calculator do not edit experiment results. No purchase orders are sent.

Original input CSVs are preserved with source hashes. Their upstream publisher and reuse licence remain unverified. Resolve this before public redistribution. No claim of original data collection is made.

Saved joblib artifacts require the pinned versions; only load trusted files. Automated execution was checked on Linux with Python 3.12. Windows instructions are provided but native Windows execution must be confirmed by the group.

## References

- https://scikit-learn.org/stable/common_pitfalls.html
- https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data
- https://ocw.mit.edu/courses/15-763j-manufacturing-system-and-supply-chain-design-spring-2005/e1800430fdccaf618b22bd551fc00fac_summary.pdf
- https://sdgs.un.org/goals/goal12
