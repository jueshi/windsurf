import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


COLOR_UP = "#2ca02c"
COLOR_DOWN = "#d62728"
NEUTRAL_COLOR = "#7f7f7f"
DOT_SIZE = 140


def _to_naive_datetime_index(index):
    """Ensure index is timezone-naive DatetimeIndex for safe comparisons."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx


def _format_share_delta(value):
    """Return a compact, signed string for share deltas."""
    try:
        magnitude = abs(float(value))
    except (TypeError, ValueError):
        return "0"

    if magnitude >= 1_000_000_000:
        formatted = f"{magnitude / 1_000_000_000:.1f}B"
    elif magnitude >= 1_000_000:
        formatted = f"{magnitude / 1_000_000:.1f}M"
    elif magnitude >= 1_000:
        formatted = f"{magnitude / 1_000:.0f}K"
    else:
        formatted = f"{magnitude:.0f}"

    sign = "+" if value >= 0 else "-"
    return f"{sign}{formatted}"


def _fetch_institutional_frame(stock):
    """Return the institutional holders DataFrame, trying both attribute and method access."""
    holders = getattr(stock, "institutional_holders", None)
    if holders is None or holders.empty:
        get_fn = getattr(stock, "get_institutional_holders", None)
        if callable(get_fn):
            holders = get_fn()
    if isinstance(holders, pd.DataFrame) and not holders.empty:
        return holders.copy()
    return None


def _build_institutional_timeseries(stock):
    """Aggregate institutional shares by reported date and compute share deltas."""
    df = _fetch_institutional_frame(stock)
    if df is None:
        return None

    if "Date Reported" not in df.columns or "Shares" not in df.columns:
        print("Institutional data missing 'Date Reported' or 'Shares' columns.")
        return None

    timeline = df[["Date Reported", "Shares"]].copy()
    timeline["Date Reported"] = pd.to_datetime(timeline["Date Reported"], errors="coerce")
    timeline["Shares"] = pd.to_numeric(timeline["Shares"], errors="coerce")
    timeline = timeline.dropna(subset=["Date Reported", "Shares"])

    if timeline.empty:
        print("Institutional ownership dataset is empty after cleaning.")
        return None

    grouped = (
        timeline.groupby("Date Reported")["Shares"]
        .sum()
        .sort_index()
        .to_frame(name="shares")
    )
    grouped["delta"] = grouped["shares"].diff()
    grouped["delta"] = grouped["delta"].fillna(grouped["shares"])
    grouped.index = _to_naive_datetime_index(grouped.index)
    return grouped


def _attach_price(grouped, hist):
    """Attach the nearest closing price to each institutional reporting date."""
    close_series = hist["Close"].copy()
    close_series.index = _to_naive_datetime_index(close_series.index)
    grouped = grouped.copy()
    grouped["price"] = close_series.reindex(grouped.index, method="nearest")
    grouped = grouped.dropna(subset=["price"])
    return grouped


def _plot_institutional_chart(ticker, hist, timeline):
    """Render the price line with institutional ownership deltas as dots."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(hist.index, hist["Close"], color="#1f77b4", linewidth=1.8, label="Close Price")
    ax.set_title(f"{ticker.upper()} Institutional Ownership vs. Price")
    ax.set_ylabel("Close Price ($)")
    ax.grid(True, linestyle="--", alpha=0.35)

    colors = timeline["delta"].apply(
        lambda val: COLOR_UP if val > 0 else (COLOR_DOWN if val < 0 else NEUTRAL_COLOR)
    )
    scatter = ax.scatter(
        timeline.index,
        timeline["price"],
        c=colors,
        s=DOT_SIZE,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )

    for idx, row in timeline.iterrows():
        delta = row["delta"]
        if pd.isna(delta) or np.isclose(delta, 0.0):
            continue
        text = _format_share_delta(delta)
        color = COLOR_UP if delta > 0 else COLOR_DOWN
        offset = 12 if delta > 0 else -14
        va = "bottom" if delta > 0 else "top"
        ax.annotate(
            text,
            xy=(idx, row["price"]),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color="#1f77b4", lw=1.8, label="Close Price"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            label="Shares Increased",
            markerfacecolor=COLOR_UP,
            markeredgecolor="white",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            label="Shares Reduced",
            markerfacecolor=COLOR_DOWN,
            markeredgecolor="white",
            markersize=10,
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left")
    plt.tight_layout()
    plt.show()

    latest_row = timeline.iloc[-1]
    latest_total = latest_row["shares"]
    print(f"Latest reported institutional shares: {latest_total:,.0f}")


def plot_institutional_ownership(ticker, period="5y"):
    """Download price + institutional data and render the ownership scatter chart."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    if hist.empty:
        print("Unable to download historical price data.")
        return
    hist.index = _to_naive_datetime_index(hist.index)

    timeline = _build_institutional_timeseries(stock)
    if timeline is None or timeline.empty:
        print("Unable to build institutional ownership timeline for this ticker.")
        return

    timeline = _attach_price(timeline, hist)
    if timeline.empty:
        print("No overlapping dates between price history and institutional filings.")
        return

    _plot_institutional_chart(ticker, hist, timeline)


if __name__ == "__main__":
    ticker = "crdo"  # Replace with ticker of interest, e.g., "ALAB" or "AMD"
    plot_institutional_ownership(ticker, period="10y")
