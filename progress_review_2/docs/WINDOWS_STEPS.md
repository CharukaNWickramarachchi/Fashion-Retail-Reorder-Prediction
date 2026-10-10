# Run in Windows VS Code, from beginning to end

## First: view what is already finished

Extract the entire ZIP into a NEW folder, for example Documents\Stock_Review_2. Keep your earlier copy separately. Open the inner Fashion_Retail_Progress_Review_2 folder: you must see run_all.py, requirements.txt, results and docs together.

Double-click docs\GROUP_GUIDE.html and read it. Double-click results\STOCK_REVIEW.html to view the stock demonstration. These saved pages do not need Python, internet, a server or a login. Keep the folder structure intact so charts and links work.

## Set up Python and VS Code

Use 64-bit Python 3.12. In VS Code choose File > Open Folder and select the folder containing run_all.py. Open Terminal > New Terminal (PowerShell). Run each line separately:

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_all.py
```

The first installation needs internet. The pipeline may take several minutes. Wait for the final completion message. Do not close the terminal while it trains. The complete command prepares data, compares models, exports the historical forecast, builds stock outputs and runs verification.

Activation is optional. If you prefer it, run `.\.venv\Scripts\Activate.ps1`, then use `python` instead of `.\.venv\Scripts\python.exe`. If PowerShell blocks activation, use the direct commands above; you do not need to change execution policy. Select .venv as your Python interpreter in VS Code if using the Run button; terminal commands above are less ambiguous.

## Open outputs after run_all.py

```powershell
Start-Process .\results\STOCK_REVIEW.html
Start-Process .\docs\GROUP_GUIDE.html
Start-Process .\results\OPEN_RESULTS.html
```

Stock page: assumptions, simulation comparison, editable stock calculator and recommendations. Model page: model scores and charts. Guide: full explanation and team demonstration script.

## Run the checks without training again

```powershell
.\.venv\Scripts\python.exe verify.py
```

The completed package has 29 checks. All should end with OK. Results are also saved in results\test_log.txt and results\verification.json. If an error appears, keep the full terminal message; do not replace it with old output.

## Try the batch stock calculator

Open inputs\stock_scenarios.csv in VS Code or Excel. The included rows are assumed examples. Every required field must be known. Blank stock is rejected. Keep forecast_week matched to the usage estimate's period and label stock_basis honestly. Do not attach a 2022 forecast to today's real inventory.

```powershell
.\.venv\Scripts\python.exe stock_orders.py
Start-Process .\results\custom_stock_orders.csv
```

The sample outputs are 15, 0 and 1 units. To use a separate scenario file:

```powershell
.\.venv\Scripts\python.exe stock_orders.py --input .\inputs\my_scenarios.csv --output .\results\my_orders.csv
```

This calculates suggestions only. It does not contact suppliers or submit purchase orders.

## Existing single-product command

```powershell
.\.venv\Scripts\python.exe predict.py --on-hand 10 --on-order 5 --lead-days 7 --review-days 7 --safety-units 2
```

This uses the saved historical demonstration model. It is not a live forecast. For a visible group demonstration, the stock page is easier to follow.

## Rebuild only the simulation

After all model outputs exist, you may run:

```powershell
.\.venv\Scripts\python.exe run_all.py --stage inventory
.\.venv\Scripts\python.exe verify.py
```

The delivered simulation uses the documented settings. If you change inventory_config.json, disclose the change and rerun; do not select only favourable results. Some verification checks intentionally enforce the delivered two-week experiment.

## Troubleshooting

- `py` not recognised: install Python 3.12 or use its full executable path to create .venv.
- `No module named ...`: run installation using the exact .venv interpreter shown above.
- `can't open file run_all.py`: you opened the wrong directory. Open the folder where that file exists.
- Missing image or link: extract everything; do not open a file directly inside the ZIP.
- Browser shows old results: refresh after a run, or close and reopen the page.
- Permission error: extract into a normal writable folder under Documents, not Program Files.
- Current Python is a different version: recreate .venv with Python 3.12 before installing the pinned dependencies.

No API keys or database are needed. Recorded automated execution was on Linux/Python 3.12, not native Windows. The group should complete the Windows run before the review.

## GitHub later

The repository structure and .gitignore are included. Do not upload .venv, temporary files or credentials. The source dataset's upstream licence has not been verified; establish redistribution permission before putting raw data or derived data publicly on GitHub. Do not invent a licence for third-party data. Code can be reviewed locally now. Nothing has been published by this package.
