# Fashion Retail Reorder Prediction

## Project Title
AI-Powered Trend Forecasting and Inventory Optimization for Fashion Retail

## Team
Charuka Wickramarachchi
Dilmith Amarawickrama
Imasha Edirisinghe
Linuash Sabeshkumar

## Project Description
This project aims to develop an AI-powered decision support system for fashion retailers by forecasting product demand and optimizing inventory levels using machine learning techniques.

## Dataset
- Amazon_Sale_Report.csv
- Sale_Report.csv

## Project Structure

```
Fashion-Retail-Reorder-Prediction
│
├── data
│   ├── raw
│   └── processed
├── notebooks
├── scripts
├── images
├── presentation
├── reports
├── README.md
├── requirements.txt
└── .gitignore
```

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Matplotlib
- Jupyter Notebook


## Progress Review 2: Stock Replenishment Prototype

The original EDA work remains in its existing folders. The modelling and
stock-replenishment work is available in [progress_review_2](progress_review_2/).

### Start here

- [One-page overview](progress_review_2/docs/ONE_PAGE_OVERVIEW.html)
- [Project explanation and findings](progress_review_2/docs/GROUP_GUIDE.md)
- [Windows and VS Code instructions](progress_review_2/docs/WINDOWS_STEPS.md)

After cloning, open the HTML files locally in a browser. GitHub's normal file
viewer does not run the interactive stock calculator.

### Rebuild the prototype

From the repository root, run in Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force .\progress_review_2\data\raw
Copy-Item ".\data\raw\Amazon Sale Report.csv" ".\progress_review_2\data\raw\"
Copy-Item ".\data\raw\Sale Report.csv" ".\progress_review_2\data\raw\"
Set-Location .\progress_review_2
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_all.py
Start-Process .\results\STOCK_REVIEW.html
```

The pipeline rebuilds processed data, fitted models, evaluation results and
inventory simulations, then runs verification.

### Interpretation

The model estimates historical weekly shipped units. A separate inventory
calculation combines estimated usage with stock and delivery inputs to suggest
replenishment quantities.

Inventory results are exploratory simulations using assumed stock conditions
and two evaluation weeks. They show a trade-off between shortages and stock
held, not a universal machine-learning advantage. Real Sri Lankan shop
performance, profit improvements and waste reductions have not been measured.

The corrected Progress Review 2 methodology supersedes earlier claims based
on stock reconstructed using same-period or future sales.
