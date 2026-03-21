# -*- coding: utf-8 -*-
"""Sector rotation detection and visualization.

Analyzes relative strength, momentum rankings, and rotation patterns
across sector ETFs to identify where capital is flowing.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

# Sector ETF -> human-readable name mapping
SECTOR_ETF_MAP = {
    "XLK": "Technology",
    "XLV": "Healthcare",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discr.",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication",
    "VNQ": "REITs",
}

# Core sector ETFs (the 11 SPDR sectors)
CORE_SECTOR_ETFS = [
    "XLK", "XLV", "XLF", "XLE", "XLI",
    "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC",
]

BENCHMARK = "SPY"


def load_sector_data(manager, etfs=None, benchmark=BENCHMARK):
    """Load price data for sector ETFs and benchmark.

    Returns dict of {ticker: DataFrame} with at least a 'Close' column.
    """
    etfs = etfs or CORE_SECTOR_ETFS
    all_tickers = list(etfs) + [benchmark]
    data = {}
    for ticker in all_tickers:
        try:
            df = manager.load_data(ticker)
            if df is not None and not df.empty:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df.index = df.index.tz_localize(None) if df.index.tz else df.index
                data[ticker] = df
        except Exception as e:
            logging.warning(f"sector_rotation: could not load {ticker}: {e}")
    return data


def compute_relative_strength(sector_data, benchmark_data, window=20):
    """Compute rolling relative strength ratio: sector_close / benchmark_close, normalized."""
    aligned = pd.DataFrame({
        "sector": sector_data["Close"],
        "bench": benchmark_data["Close"],
    }).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    ratio = aligned["sector"] / aligned["bench"]
    # Normalize to 100 at start of window
    rs = (ratio / ratio.iloc[0]) * 100
    return rs


def compute_momentum(close_series, periods=(5, 10, 21, 63)):
    """Return dict of rate-of-change (%) for given lookback periods (trading days)."""
    result = {}
    for p in periods:
        if len(close_series) > p:
            roc = ((close_series.iloc[-1] / close_series.iloc[-p]) - 1) * 100
            result[p] = roc
        else:
            result[p] = np.nan
    return result


def build_rotation_table(data, benchmark=BENCHMARK, periods=(5, 10, 21, 63)):
    """Build a DataFrame ranking sectors by momentum across timeframes.

    Returns DataFrame with columns for each period's ROC and a composite rank.
    """
    if benchmark not in data:
        raise ValueError(f"Benchmark {benchmark} not found in data")

    bench_close = data[benchmark]["Close"]
    rows = []

    for etf in CORE_SECTOR_ETFS:
        if etf not in data:
            continue
        close = data[etf]["Close"]
        mom = compute_momentum(close, periods)
        bench_mom = compute_momentum(bench_close, periods)
        row = {"ETF": etf, "Sector": SECTOR_ETF_MAP.get(etf, etf)}
        for p in periods:
            row[f"{p}d_roc"] = mom.get(p, np.nan)
            row[f"{p}d_vs_spy"] = mom.get(p, np.nan) - bench_mom.get(p, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Composite score: weighted average of vs_spy across timeframes
    weights = {5: 0.15, 10: 0.25, 21: 0.35, 63: 0.25}
    df["composite"] = sum(
        df[f"{p}d_vs_spy"].fillna(0) * w for p, w in weights.items() if f"{p}d_vs_spy" in df.columns
    )
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def build_rolling_ranks(data, benchmark=BENCHMARK, window=21, lookback_days=252):
    """Build time-series of sector momentum ranks over the last `lookback_days` trading days.

    Returns DataFrame with DatetimeIndex and one column per sector ETF (value = rank 1..N).
    """
    if benchmark not in data:
        return pd.DataFrame()

    bench_close = data[benchmark]["Close"]
    sector_closes = {}
    for etf in CORE_SECTOR_ETFS:
        if etf in data:
            sector_closes[etf] = data[etf]["Close"]

    if not sector_closes:
        return pd.DataFrame()

    # Align all series to common dates
    all_close = pd.DataFrame(sector_closes)
    all_close = all_close.dropna(how="all")

    # Limit to lookback period
    if len(all_close) > lookback_days:
        all_close = all_close.iloc[-lookback_days:]

    # Align benchmark
    bench_aligned = bench_close.reindex(all_close.index)

    # Rolling relative return vs benchmark
    rel_returns = pd.DataFrame(index=all_close.index)
    for etf in all_close.columns:
        sector_ret = all_close[etf].pct_change(window)
        bench_ret = bench_aligned.pct_change(window)
        rel_returns[etf] = sector_ret - bench_ret

    # Rank each day (1 = best)
    ranks = rel_returns.rank(axis=1, ascending=False, method="min")
    return ranks.dropna(how="all")


def plot_sector_heatmap(rotation_table, save_path=None):
    """Create a heatmap showing sector momentum across timeframes.

    Returns matplotlib Figure.
    """
    original_backend = matplotlib.get_backend()
    matplotlib.use("Agg")

    periods = [5, 10, 21, 63]
    period_labels = ["1W", "2W", "1M", "3M"]

    # Build heatmap data
    sectors = rotation_table["Sector"].tolist()
    vs_spy_cols = [f"{p}d_vs_spy" for p in periods]
    heatmap_data = rotation_table[vs_spy_cols].values

    fig, ax = plt.subplots(figsize=(10, max(6, len(sectors) * 0.5 + 1)))

    # Custom diverging colormap: red -> white -> green
    cmap = LinearSegmentedColormap.from_list("rg", ["#d32f2f", "#ffffff", "#2e7d32"])

    # Determine symmetric color range
    vmax = max(abs(np.nanmin(heatmap_data)), abs(np.nanmax(heatmap_data)), 1)
    im = ax.imshow(heatmap_data, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(period_labels)))
    ax.set_xticklabels(period_labels, fontsize=11)
    ax.set_yticks(range(len(sectors)))
    etf_labels = [f"{rotation_table['ETF'].iloc[i]}  {s}" for i, s in enumerate(sectors)]
    ax.set_yticklabels(etf_labels, fontsize=10)

    # Annotate cells with values
    for i in range(len(sectors)):
        for j in range(len(periods)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    ax.set_title("Sector Relative Performance vs SPY", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Lookback Period", fontsize=11)
    fig.colorbar(im, ax=ax, label="% vs SPY", shrink=0.8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    matplotlib.use(original_backend)
    return fig


def plot_rotation_ranks(ranks_df, save_path=None, visible_etfs=None):
    """Plot rolling sector rank changes over time as a line chart.

    Args:
        ranks_df: DataFrame with one column per ETF, values are ranks.
        save_path: Optional path to save the figure.
        visible_etfs: Optional list of ETF tickers to show. None = show all.

    Returns matplotlib Figure.
    """
    original_backend = matplotlib.get_backend()
    matplotlib.use("Agg")

    # Filter to visible ETFs if specified
    all_etfs = list(ranks_df.columns)
    if visible_etfs is not None:
        plot_etfs = [e for e in all_etfs if e in visible_etfs]
    else:
        plot_etfs = all_etfs

    fig, ax = plt.subplots(figsize=(12, 7))

    # Use consistent colors: each ETF always gets the same color regardless of filtering
    color_map = {etf: plt.cm.tab20(i / max(len(all_etfs), 1)) for i, etf in enumerate(all_etfs)}

    for etf in plot_etfs:
        label = f"{etf} ({SECTOR_ETF_MAP.get(etf, etf)})"
        ax.plot(ranks_df.index, ranks_df[etf], label=label, color=color_map[etf], linewidth=1.5, alpha=0.85)

    n_sectors = len(all_etfs)

    ax.set_ylabel("Rank (1 = strongest)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_title("Sector Momentum Rank Over Time (21-day rolling vs SPY)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()  # Rank 1 at top
    ax.set_yticks(range(1, n_sectors + 1))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    matplotlib.use(original_backend)
    return fig


def plot_rrg_scatter(data, benchmark=BENCHMARK, window=21, save_path=None):
    """Plot a Relative Rotation Graph (RRG) scatter.

    X-axis = RS-Ratio (relative strength vs benchmark, normalized)
    Y-axis = RS-Momentum (rate of change of RS-Ratio)

    Each sector is plotted as a point with a trail showing direction.
    Returns matplotlib Figure.
    """
    original_backend = matplotlib.get_backend()
    matplotlib.use("Agg")

    if benchmark not in data:
        return None

    bench_close = data[benchmark]["Close"]
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab20(np.linspace(0, 1, len(CORE_SECTOR_ETFS)))
    trail_len = 4  # number of historical points to show

    for i, etf in enumerate(CORE_SECTOR_ETFS):
        if etf not in data:
            continue
        sector_close = data[etf]["Close"]

        # Align
        aligned = pd.DataFrame({"s": sector_close, "b": bench_close}).dropna()
        if len(aligned) < window * 3:
            continue

        # RS-Ratio: rolling relative strength normalized to 100
        ratio = aligned["s"] / aligned["b"]
        rs_ratio = (ratio / ratio.rolling(window * 5, min_periods=window).mean()) * 100

        # RS-Momentum: rate of change of RS-Ratio
        rs_momentum = rs_ratio.pct_change(window) * 100 + 100

        # Get last few points for trail
        rs_r = rs_ratio.dropna().iloc[-(trail_len + 1):]
        rs_m = rs_momentum.reindex(rs_r.index).dropna()

        if len(rs_m) < 2:
            continue

        common_idx = rs_r.index.intersection(rs_m.index)
        rs_r = rs_r[common_idx]
        rs_m = rs_m[common_idx]

        # Plot trail (fading)
        for j in range(len(common_idx) - 1):
            alpha = 0.2 + 0.6 * (j / max(len(common_idx) - 1, 1))
            ax.plot(
                [rs_r.iloc[j], rs_r.iloc[j + 1]],
                [rs_m.iloc[j], rs_m.iloc[j + 1]],
                color=colors[i], alpha=alpha, linewidth=1.5,
            )

        # Plot current position
        label = f"{etf} ({SECTOR_ETF_MAP.get(etf, etf)})"
        ax.scatter(rs_r.iloc[-1], rs_m.iloc[-1], color=colors[i], s=80, zorder=5, label=label)
        ax.annotate(etf, (rs_r.iloc[-1], rs_m.iloc[-1]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8, fontweight="bold")

    # Quadrant lines and labels
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=100, color="gray", linestyle="--", alpha=0.5)

    # Quadrant labels
    props = dict(fontsize=10, alpha=0.3, fontweight="bold", ha="center")
    ax.text(101.5, 101.5, "LEADING", color="green", **props)
    ax.text(98.5, 101.5, "IMPROVING", color="blue", **props)
    ax.text(98.5, 98.5, "LAGGING", color="red", **props)
    ax.text(101.5, 98.5, "WEAKENING", color="orange", **props)

    ax.set_xlabel("RS-Ratio (Relative Strength)", fontsize=11)
    ax.set_ylabel("RS-Momentum", fontsize=11)
    ax.set_title("Relative Rotation Graph (RRG) vs SPY", fontsize=13, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    matplotlib.use(original_backend)
    return fig


def get_missing_sector_tickers(manager, etfs=None, benchmark=BENCHMARK):
    """Return list of sector ETF tickers that don't have local data."""
    import os
    etfs = etfs or CORE_SECTOR_ETFS
    all_tickers = list(etfs) + [benchmark]
    missing = []
    for ticker in all_tickers:
        data_path = manager._get_data_path(ticker)
        if not os.path.exists(data_path):
            missing.append(ticker)
    return missing
