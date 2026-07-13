"""
Data Cleaning & Merging Pipeline
Fashion Retail Reorder Prediction Project
---------------------------------------------------
Inputs : Amazon_Sale_Report.csv   (transaction-level sales data)
         Sale_Report.csv          (SKU-level current stock snapshot)
Output : merged_clean_dataset.csv (cleaned, merged, analysis-ready dataset)

Run with: python clean_and_merge.py
"""

import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# 0. LOAD RAW DATA
# ------------------------------------------------------------------
amz = pd.read_csv("Amazon_Sale_Report.csv", low_memory=False)
stock = pd.read_csv("Sale_Report.csv", low_memory=False)

print(f"Raw Amazon Sale Report:  {amz.shape[0]:,} rows, {amz.shape[1]} cols")
print(f"Raw Sale Report (stock): {stock.shape[0]:,} rows, {stock.shape[1]} cols")

# ------------------------------------------------------------------
# 1. CLEAN THE STOCK FILE (Sale_Report.csv)
# ------------------------------------------------------------------

# 1a. Drop fully-blank trailing rows (no SKU Code at all)
stock = stock.dropna(subset=["SKU Code"]).copy()

# 1b. Drop broken Excel references (#REF! errors)
stock = stock[stock["SKU Code"] != "#REF!"].copy()

# 1c. Standardise category text (casing + whitespace)
stock["Category"] = stock["Category"].str.strip().str.upper()

# 1d. Resolve genuine duplicate SKU Codes.
#     Policy: keep the MAX stock value per SKU Code (conservative choice —
#     documents the assumption clearly; alternative policies, e.g. mean,
#     could be justified instead, just be explicit about which you pick).
stock_clean = (
    stock.groupby("SKU Code", as_index=False)
    .agg({
        "Design No.": "first",
        "Stock": "max",
        "Category": "first",
        "Size": "first",
        "Color": "first",
    })
)

# 1e. Handle any remaining missing Stock values (impute with 0 — treat as
#     "not currently stocked" rather than dropping, since dropping would
#     lose the SKU entirely from the merge)
stock_clean["Stock"] = stock_clean["Stock"].fillna(0)

print(f"\nCleaned stock file: {stock_clean.shape[0]:,} unique SKUs")
print(f"Rows dropped in cleaning: {stock.shape[0] - stock_clean.shape[0]:,} (duplicates collapsed)")

# ------------------------------------------------------------------
# 2. CLEAN THE AMAZON SALE REPORT (transactions)
# ------------------------------------------------------------------

# 2a. Drop the unnamed junk column and unused ASIN/promotion noise if not needed
amz_clean = amz.drop(columns=["Unnamed: 22"], errors="ignore")

# 2b. Parse Date properly
amz_clean["Date"] = pd.to_datetime(amz_clean["Date"], format="%m-%d-%y", errors="coerce")

# 2c. Create a Month period column — this is your monthly aggregation key
amz_clean["Month"] = amz_clean["Date"].dt.to_period("M").astype(str)

# 2d. Standardise category text to match stock file casing
amz_clean["Category"] = amz_clean["Category"].str.strip().str.upper()

# 2e. Remove cancelled orders for TRUE demand analysis (optional — keep both
#     versions so you can compare "gross demand" vs "fulfilled sales")
amz_clean["is_cancelled"] = amz_clean["Status"].str.contains("Cancelled", case=False, na=False)

# 2f. Drop rows with missing SKU (can't be merged or aggregated meaningfully)
amz_clean = amz_clean.dropna(subset=["SKU"])

# 2g. Basic sanity filter: negative or zero Amount on non-cancelled orders
#     are worth flagging, not silently dropping (log them for your report)
suspicious = amz_clean[(amz_clean["Amount"] <= 0) & (~amz_clean["is_cancelled"])]
print(f"\nFlag: {len(suspicious):,} non-cancelled rows have Amount <= 0 (review before final use)")

print(f"\nCleaned Amazon transactions: {amz_clean.shape[0]:,} rows")
print(f"Date range: {amz_clean['Date'].min().date()} to {amz_clean['Date'].max().date()}")

# ------------------------------------------------------------------
# 3. MERGE ON SKU
# ------------------------------------------------------------------
merged = amz_clean.merge(
    stock_clean,
    left_on="SKU",
    right_on="SKU Code",
    how="left",
    suffixes=("", "_stock"),
)

match_rate = merged["Stock"].notna().mean() * 100
print(f"\nMerge match rate: {match_rate:.2f}% of transaction rows received a Stock value")

# ------------------------------------------------------------------
# 4. BUILD THE MONTHLY SKU-LEVEL ANALYTICAL TABLE
#    (this is the level your reorder model will most likely operate at)
# ------------------------------------------------------------------
monthly = (
    merged.groupby(["SKU", "Category", "Month"], as_index=False)
    .agg(
        units_sold=("Qty", "sum"),
        gross_sales_amount=("Amount", "sum"),
        num_orders=("Order ID", "nunique"),
        cancelled_orders=("is_cancelled", "sum"),
        current_stock_snapshot=("Stock", "first"),   # same value repeated each month — see note below
        ship_state_mode=("ship-state", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
    )
)

print(f"\nMonthly SKU-level table: {monthly.shape[0]:,} rows "
      f"({monthly['SKU'].nunique():,} SKUs x up to {monthly['Month'].nunique()} months)")

# ------------------------------------------------------------------
# 5. SAVE OUTPUTS
# ------------------------------------------------------------------
merged.to_csv("merged_clean_dataset.csv", index=False)
monthly.to_csv("monthly_sku_level_dataset.csv", index=False)

print("\nSaved: merged_clean_dataset.csv (transaction-level, merged with stock)")
print("Saved: monthly_sku_level_dataset.csv (monthly SKU-level aggregation)")

# ------------------------------------------------------------------
# IMPORTANT CAVEAT TO STATE IN YOUR REPORT:
# 'current_stock_snapshot' is a SINGLE point-in-time value per SKU,
# repeated across all months for that SKU. It is NOT a true monthly
# stock time series. Treat it as "most recently known stock level"
# rather than "stock at the end of each specific month."
# ------------------------------------------------------------------
