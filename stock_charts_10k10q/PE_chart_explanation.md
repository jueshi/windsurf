# PE Chart Overview

## Purpose

Visualize the relationship between valuation (P/E), price action, and the earnings power (annualized EPS) of a ticker over any yfinance-supported period.

## Data Pipeline

1. **Historical Prices**: `yf.Ticker(ticker).history(period)` pulls up to the requested window of OHLCV data.
2. **EPS Retrieval**:
   - Quarterly and annual statements are scanned for Diluted/Basic EPS rows.
   - Quarterly EPS is converted to trailing-twelve-month (TTM) values via a 4-quarter rolling sum.
   - Annual EPS entries remain as reported.
3. **Index Normalization**: All timestamps are converted to timezone-naive `DatetimeIndex` objects to keep comparisons consistent.
4. **Daily Alignment**:
   - Each EPS timestamp is expanded forward so every trading day inherits the most recent TTM EPS.
   - `ffill().bfill()` ensures the entire price history inside the requested window has EPS coverage.
5. **P/E Calculation**: Daily PE is simply `Close / EPS`. Rows missing either value are dropped.

## Plot Layout

```text
┌──────────────────────────────────────────────┐
│  Pane 1: PE (left axis) + Close Price (right)│
└──────────────────────────────────────────────┘
┌──────────────────────────────────────────────┐
│  Pane 2: Annualized EPS (TTM)                │
└──────────────────────────────────────────────┘
```

- **Pane 1**
  - Blue line: trailing PE ratio.
  - Orange line: closing price plotted on a twin y-axis for direct valuation/price comparison.
- **Pane 2**
  - Green line: annualized EPS (TTM) trend with a floating annotation for the latest value.

## Usage

```python
if __name__ == "__main__":
    ticker = "intc"
    plot_pe_ratio_over_time(ticker, period="10y")
```

- Any yfinance-recognized ticker works (e.g., "AAPL", "AMZN", "ALAB").
- Period can be `"1y"`, `"5y"`, `"10y"`, `"max"`, etc.

## Output

- Interactive Matplotlib figure showing both panes with shared x-axis.
- Console summary of PE distribution (`count`, `mean`, quartiles, min, max`) for quick stats.

## Notes

- Requires `yfinance`, `pandas`, `numpy`, and `matplotlib`.
- Annualized EPS combines both quarterly and annual statements; fallback uses the latest `trailingEps` if no financials are available.
