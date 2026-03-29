# -*- coding: utf-8 -*-
"""Relative strength analysis for stocks vs SPY benchmark.

Provides functions to:
- Calculate relative strength rankings
- Track rank changes over time
- Build HTML reports with charts
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Ensure project directory is on the path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from simple_data_loader import load_data, update_data
from ticker_lists import mega_tickers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

BENCHMARK = "SPY"
DEFAULT_BIG_CAPS = mega_tickers[:20]  # Top 20 from mega_tickers


def load_stock_data(tickers):
    """Load price data for a list of tickers.

    Returns dict: {ticker: DataFrame with Close prices}
    """
    data = {}
    for ticker in tickers:
        try:
            df = load_data(ticker)
            if df is not None and not df.empty and "Close" in df.columns:
                data[ticker] = df[["Close"]].copy()
        except Exception as e:
            logging.warning(f"Failed to load {ticker}: {e}")

    return data


def calculate_relative_strength(stock_data, benchmark_data, lookback_days=252):
    """Calculate relative strength (% outperformance vs benchmark) for each stock.

    Returns dict: {ticker: {period: relative_return}}
    """
    periods = {
        "1w": 5,
        "2w": 10,
        "1m": 21,
        "3m": 63,
    }

    bench_close = benchmark_data["Close"]
    results = {}

    for ticker, stock_df in stock_data.items():
        stock_close = stock_df["Close"]

        # Align indices
        common_index = stock_close.index.intersection(bench_close.index)
        if len(common_index) < max(periods.values()):
            logging.warning(f"Insufficient data for {ticker}")
            continue

        stock_aligned = stock_close[common_index].tail(lookback_days)
        bench_aligned = bench_close[common_index].tail(lookback_days)

        results[ticker] = {}
        for period_name, period_days in periods.items():
            if len(stock_aligned) >= period_days:
                stock_ret = ((stock_aligned.iloc[-1] / stock_aligned.iloc[-period_days]) - 1) * 100
                bench_ret = ((bench_aligned.iloc[-1] / bench_aligned.iloc[-period_days]) - 1) * 100
                results[ticker][period_name] = stock_ret - bench_ret
            else:
                results[ticker][period_name] = np.nan

    return results


def build_relative_strength_table(stock_data, benchmark_data):
    """Build a ranking table of stocks by relative strength.

    Returns DataFrame with rankings and performance metrics.
    """
    rs_values = calculate_relative_strength(stock_data, benchmark_data)

    rows = []
    for ticker, metrics in rs_values.items():
        # Calculate composite score (weighted average)
        weights = {"1w": 1, "2w": 1, "1m": 2, "3m": 3}
        composites = [metrics.get(p, np.nan) for p in weights.keys()]
        if all(np.isnan(c) for c in composites):
            continue

        composite = np.nanmean([c * w for c, w in zip(composites, weights.values())])

        rows.append({
            "Ticker": ticker,
            "1W": metrics.get("1w", np.nan),
            "2W": metrics.get("2w", np.nan),
            "1M": metrics.get("1m", np.nan),
            "3M": metrics.get("3m", np.nan),
            "Composite": composite,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("Composite", ascending=False, na_position="last").reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)

    return df[["Rank", "Ticker", "1W", "2W", "1M", "3M", "Composite"]]


def build_rolling_ranks(stock_data, benchmark_data, window=21, lookback_days=252):
    """Build time-series of relative strength ranks over time.

    Returns DataFrame with DatetimeIndex and one column per ticker (value = rank).
    """
    bench_close = benchmark_data["Close"]

    # Align all data
    all_tickers = list(stock_data.keys())
    all_data = {ticker: stock_data[ticker]["Close"] for ticker in all_tickers}

    # Create DataFrame with all prices
    price_df = pd.DataFrame(all_data)
    price_df = price_df.dropna(how="all")

    # Limit to lookback period
    if len(price_df) > lookback_days:
        price_df = price_df.iloc[-lookback_days:]

    # Align benchmark
    bench_aligned = bench_close.reindex(price_df.index)

    # Calculate rolling relative returns
    rel_returns = pd.DataFrame(index=price_df.index)
    for ticker in all_tickers:
        if ticker in price_df.columns:
            stock_ret = price_df[ticker].pct_change(window)
            bench_ret = bench_aligned.pct_change(window)
            rel_returns[ticker] = stock_ret - bench_ret

    # Rank each day (1 = best)
    ranks = rel_returns.rank(axis=1, ascending=False, method="min")
    return ranks.dropna(how="all")


if __name__ == "__main__":
    # Test: load data and calculate rankings
    tickers = ["AAPL", "MSFT", "NVDA", "SPY"]
    print(f"Loading data for {len(tickers)} stocks...")

    all_data = load_stock_data(tickers + [BENCHMARK])
    stock_data = {t: all_data[t] for t in tickers if t in all_data}
    benchmark_data = all_data.get(BENCHMARK)

    if stock_data and benchmark_data is not None:
        print("\nRelative Strength Rankings:")
        rs_table = build_relative_strength_table(stock_data, benchmark_data)
        print(rs_table)

        print("\nRank changes last week:")
        rolling = build_rolling_ranks(stock_data, benchmark_data)
        if len(rolling) >= 5:
            recent = rolling.iloc[-5:]
            for ticker in rolling.columns:
                start_rank = int(recent[ticker].iloc[0])
                end_rank = int(recent[ticker].iloc[-1])
                print(f"  {ticker}: {start_rank} -> {end_rank} ({start_rank - end_rank:+d})")
