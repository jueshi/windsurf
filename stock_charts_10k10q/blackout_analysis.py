"""Blackout-period price performance analysis.

Insider trading blackout periods are derived from quarterly earnings
announcement dates: blackout ends at the earnings date and starts a
configurable number of calendar days before it. For each past quarter we
measure price performance N calendar days after blackout end (rolling
forward to the next trading day), then compound those returns over
chronological quarters — one compounded curve per horizon N.

All computation is pure and GUI-free so it can be unit-tested with
synthetic data (see test_blackout_analysis.py). Only get_earnings_dates()
and analyze_ticker() touch the network.
"""

import io
import json
import logging
import os
import time
from bisect import bisect_left, bisect_right
from datetime import date, datetime, timedelta
from random import uniform
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

# Calendar-day horizons after blackout end
HORIZONS = [1, 3, 7, 14, 21, 28, 35, 42]

# Blackout starts this many calendar days before the earnings date
DEFAULT_BLACKOUT_START_DAYS = 30

# Earnings dates are cached on disk and reused within this many days
EARNINGS_CACHE_DIR = 'earnings_cache'
EARNINGS_CACHE_MAX_AGE_DAYS = 1


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _earnings_cache_path(ticker: str) -> str:
    os.makedirs(EARNINGS_CACHE_DIR, exist_ok=True)
    return os.path.join(EARNINGS_CACHE_DIR, f'{ticker.upper()}_earnings.json')


def _load_cached_earnings(ticker: str) -> Optional[List[Dict]]:
    """Load earnings dates from disk cache if fresh enough."""
    path = _earnings_cache_path(ticker)
    try:
        if not os.path.exists(path):
            return None
        age_days = (time.time() - os.path.getmtime(path)) / 86400.0
        if age_days > EARNINGS_CACHE_MAX_AGE_DAYS:
            return None
        with open(path, 'r', encoding='utf-8') as f:
            rows = json.load(f)
        out = []
        for r in rows:
            out.append({
                'date': date.fromisoformat(r['date']),
                'eps_estimate': r.get('eps_estimate'),
                'reported_eps': r.get('reported_eps'),
                'surprise_pct': r.get('surprise_pct'),
            })
        if out:
            logging.info(f"Earnings cache hit for {ticker} ({len(out)} quarters)")
            return out
    except Exception as e:
        logging.debug(f"Earnings cache read failed for {ticker}: {e}")
    return None


def _save_cached_earnings(ticker: str, earnings: List[Dict]):
    """Persist earnings dates to the disk cache."""
    try:
        rows = [{
            'date': e['date'].isoformat(),
            'eps_estimate': None if e.get('eps_estimate') is None or pd.isna(e.get('eps_estimate')) else float(e['eps_estimate']),
            'reported_eps': None if e.get('reported_eps') is None or pd.isna(e.get('reported_eps')) else float(e['reported_eps']),
            'surprise_pct': None if e.get('surprise_pct') is None or pd.isna(e.get('surprise_pct')) else float(e['surprise_pct']),
        } for e in earnings]
        with open(_earnings_cache_path(ticker), 'w', encoding='utf-8') as f:
            json.dump(rows, f)
    except Exception as e:
        logging.debug(f"Earnings cache write failed for {ticker}: {e}")


def get_earnings_dates(ticker: str, max_retries: int = 3,
                       use_cache: bool = True) -> List[Dict]:
    """Fetch historical quarterly earnings announcement dates for a ticker.

    Uses yfinance historical earnings dates (exact announcement timestamps,
    including EPS estimate/reported/surprise), with a disk cache (reused for
    up to EARNINGS_CACHE_MAX_AGE_DAYS) so repeat analyses don't re-hit the
    network. Tries a larger limit first (yfinance defaults to only ~12
    quarters) and falls back to the default.

    Returns:
        Chronologically sorted list of dicts with keys:
        date (datetime.date), eps_estimate, reported_eps, surprise_pct

    Raises:
        RuntimeError: if all retries fail or no dates are available.
    """
    if use_cache:
        cached = _load_cached_earnings(ticker)
        if cached is not None:
            return cached
    last_err = None
    for attempt in range(max_retries):
        try:
            ticker_obj = yf.Ticker(ticker)
            df = None
            # Newer yfinance supports a limit; try to pull deeper history
            try:
                df = ticker_obj.get_earnings_dates(limit=100)
            except Exception:
                df = None
            if df is None or len(df) == 0:
                df = ticker_obj.earnings_dates
            if df is None or len(df) == 0:
                raise ValueError("no historical earnings dates returned")

            df = df.sort_index()
            out = []
            for ts, row in df.iterrows():
                d = pd.to_datetime(ts)
                if isinstance(d, pd.Timestamp):
                    d = d.date()
                out.append({
                    'date': d,
                    'eps_estimate': row.get('EPS Estimate'),
                    'reported_eps': row.get('Reported EPS'),
                    'surprise_pct': row.get('Surprise(%)'),
                })
            _save_cached_earnings(ticker, out)
            return out
        except Exception as e:
            last_err = e
            wait = (2 ** attempt) + uniform(0, 1)
            logging.warning(f"get_earnings_dates attempt {attempt + 1} failed for "
                            f"{ticker}: {e}. Waiting {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch earnings dates for {ticker}: {last_err}")


# ---------------------------------------------------------------------------
# Blackout windows and per-quarter returns (pure functions)
# ---------------------------------------------------------------------------

def _extract_series(price_history: pd.DataFrame):
    """Return (dates, closes): sorted trading days and parallel closes.

    data_manager.load_data() may return dates as a 'Date' column or as the
    (named) DatetimeIndex — accept both.
    """
    if 'Date' in price_history.columns:
        dates = pd.to_datetime(price_history['Date'], utc=True).dt.date.tolist()
    else:
        dates = [d.date() if isinstance(d, pd.Timestamp) else d
                 for d in pd.to_datetime(price_history.index, utc=True)]
    closes = pd.to_numeric(price_history['Close'], errors='coerce').tolist()
    order = sorted(range(len(dates)), key=lambda i: dates[i])
    return [dates[i] for i in order], [closes[i] for i in order]


def get_blackout_periods(earnings_dates: List[Dict],
                         blackout_start_days: int = DEFAULT_BLACKOUT_START_DAYS) -> List[Dict]:
    """Derive blackout windows from earnings dates.

    Args:
        earnings_dates: output of get_earnings_dates() (chronological).
        blackout_start_days: calendar days between blackout start and earnings.

    Returns:
        Chronological list of dicts with keys:
        label, blackout_start (date), blackout_end (date = earnings date)
    """
    periods = []
    for item in sorted(earnings_dates, key=lambda x: x['date']):
        end = item['date']
        start = end - timedelta(days=blackout_start_days)
        label = f"{end.year} Q{(end.month - 1) // 3 + 1}"
        periods.append({'label': label, 'blackout_start': start, 'blackout_end': end})
    return periods


def compute_quarter_gains(price_history: pd.DataFrame,
                          blackout_periods: List[Dict],
                          horizons: List[int] = None,
                          as_of: Optional[date] = None,
                          entry: str = 'end') -> List[Dict]:
    """Compute per-quarter baseline price and % gain at each horizon.

    Baseline = Close on the first trading day on/after the entry date.
    For horizon N, exit = Close on the first trading day on/after
    (entry date + N calendar days). A horizon's window only counts as
    complete when that exit trading day exists and is on/before as_of;
    otherwise the gain is None for that quarter.

    Args:
        price_history: DataFrame with 'Date' and 'Close' columns (tz-aware
            or naive dates both accepted, as returned by data_manager).
        blackout_periods: output of get_blackout_periods().
        horizons: calendar-day horizons; defaults to HORIZONS.
        as_of: date treated as "today" (defaults to actual today) —
            injectable for testing.
        entry: 'end' buys at the blackout end (earnings) date — the
            default; 'start' buys at the blackout start date instead,
            flipping the trade to the beginning of the blackout period.

    Returns:
        Chronological list of dicts with keys:
        label, blackout_start, blackout_end, baseline_date, baseline_close,
        gains ({horizon: fractional return or None}),
        exits ({horizon: (exit_date, exit_close) or None})
    """
    if horizons is None:
        horizons = HORIZONS
    if as_of is None:
        as_of = date.today()

    dates, closes = _extract_series(price_history)

    def close_on_or_after(target: date):
        """Return (trading_day, close) for first trading day >= target, or None."""
        idx = bisect_left(dates, target)
        if idx >= len(dates):
            return None
        close = closes[idx]
        if close is None or pd.isna(close):
            return None
        return dates[idx], close

    results = []
    for period in blackout_periods:
        end = period['blackout_end']
        anchor = period['blackout_start'] if entry == 'start' else end
        baseline = close_on_or_after(anchor)
        if baseline is None:
            # No tradable session at/after the entry date within data
            continue
        baseline_date, baseline_close = baseline
        if baseline_date > as_of:
            # Baseline session has not happened yet
            continue

        gains: Dict[int, Optional[float]] = {}
        exits: Dict[int, Optional[tuple]] = {}
        for n in horizons:
            exit_pt = close_on_or_after(anchor + timedelta(days=n))
            if exit_pt is None or exit_pt[0] > as_of:
                gains[n] = None  # window not fully elapsed
                exits[n] = None
            else:
                gains[n] = (exit_pt[1] / baseline_close) - 1.0
                exits[n] = exit_pt

        results.append({
            'label': period['label'],
            'blackout_start': period['blackout_start'],
            'blackout_end': end,
            'baseline_date': baseline_date,
            'baseline_close': baseline_close,
            'gains': gains,
            'exits': exits,
        })
    return results


def compute_compounded_curves(quarter_gains: List[Dict],
                              horizons: List[int] = None) -> Dict[int, Dict]:
    """Compound per-quarter returns over chronological quarters, per horizon.

    For horizon N: cumulative product of (1 + r) across all quarters whose
    N-day window fully elapsed, expressed as cumulative % gain from $1.
    Each horizon's curve spans only its completed windows.

    Returns:
        {horizon: {'labels': [...], 'values': [...%], 'final': %}} where
        values[i] is the compounded % gain through labels[i].
    """
    if horizons is None:
        horizons = HORIZONS
    curves: Dict[int, Dict] = {}
    for n in horizons:
        labels, values, cum = [], [], 1.0
        for row in quarter_gains:
            r = row['gains'].get(n)
            if r is None:
                continue
            cum *= (1.0 + r)
            labels.append(row['label'])
            values.append((cum - 1.0) * 100.0)
        curves[n] = {'labels': labels, 'values': values,
                     'final': (cum - 1.0) * 100.0, 'count': len(labels)}
    return curves


# ---------------------------------------------------------------------------
# Daily equity curves (pure functions)
# ---------------------------------------------------------------------------

def compute_daily_equity_curves(price_history: pd.DataFrame,
                                quarter_gains: List[Dict],
                                horizons: List[int] = None,
                                as_of: Optional[date] = None) -> Dict[int, Dict]:
    """Compute the daily mark-to-market equity of each horizon's strategy.

    Strategy for horizon N: buy at each quarter's blackout-end close, hold
    until the N-day rolled exit close, then sit in cash until the next
    quarter's buy. Equity starts at $1 on the first completed-or-open
    window's baseline date and is marked daily at each close. A window whose
    exit has not elapsed yet is held to the last available trading day
    (live position). If a new window's buy arrives while still holding
    (holding period longer than the earnings gap), the buy signal is
    SKIPPED — the position runs to its own exit and re-entry happens at the
    next blackout end after going flat.

    Args:
        price_history: same frame accepted by compute_quarter_gains().
        quarter_gains: output of compute_quarter_gains().
        horizons: calendar-day horizons; defaults to HORIZONS.
        as_of: date treated as "today" (defaults to actual today).

    Returns:
        {horizon: {'dates': [...], 'values': [...cumulative % gain from $1]}}
    """
    if horizons is None:
        horizons = HORIZONS
    if as_of is None:
        as_of = date.today()

    dates, closes = _extract_series(price_history)
    # Only trading days on/before as_of participate
    last_idx = bisect_left(dates, as_of) - 1
    if last_idx < 0:
        return {n: {'dates': [], 'values': []} for n in horizons}

    curves: Dict[int, Dict] = {}
    for n in horizons:
        # (baseline_date, exit_date or None) per window; incomplete windows
        # (exit None) still open a live position marked to the last session
        windows = [(row['baseline_date'], row['exits'][n][0] if row['exits'].get(n) else None)
                   for row in quarter_gains]
        if not windows:
            curves[n] = {'dates': [], 'values': []}
            continue

        start_idx = bisect_left(dates, windows[0][0])
        if start_idx > last_idx:
            curves[n] = {'dates': [], 'values': []}
            continue

        out_dates, out_values = [], []
        equity, shares = 1.0, 0.0
        w_idx = 0
        current_exit = None
        for i in range(start_idx, last_idx + 1):
            d, c = dates[i], closes[i]
            if c is None or pd.isna(c):
                # Bad quote day: carry yesterday's equity, keep position
                out_dates.append(d)
                out_values.append((equity - 1.0) * 100.0)
                continue
            # New window buys today (baseline dates are trading days)
            while w_idx < len(windows) and windows[w_idx][0] == d:
                if shares > 0:
                    # Still holding (hold longer than the window gap): skip
                    # this buy signal and run the position to its own exit
                    w_idx += 1
                    continue
                shares = equity / c
                current_exit = windows[w_idx][1]
                w_idx += 1
            if shares > 0:
                equity = shares * c
                if current_exit is not None and d == current_exit:
                    shares = 0.0  # sell; equity holds as cash until next buy
            out_dates.append(d)
            out_values.append((equity - 1.0) * 100.0)
        curves[n] = {'dates': out_dates, 'values': out_values}
    return curves


# ---------------------------------------------------------------------------
# Plotting (matplotlib only — no Tk objects here)
# ---------------------------------------------------------------------------

def split_blackout_values(dates: List[date], values: List[float],
                          intervals: List[tuple]):
    """Split a daily series into normal and blackout-highlight parts.

    Returns (normal_values, blackout_values):
    - blackout_values carries the values on days inside a blackout window
      [start, end] — end inclusive so the highlighted segment joins the
      line at the earnings/buy day — and NaN elsewhere.
    - normal_values carries NaN inside [start, end) and the value elsewhere
      (the end day belongs to both, as the join point).
    """
    if not intervals:
        return list(values), [float('nan')] * len(values)
    ordered = sorted(intervals)
    starts = [s for s, _ in ordered]
    ends = [e for _, e in ordered]
    normal, yellow = [], []
    for d, v in zip(dates, values):
        i = bisect_right(starts, d) - 1
        in_blackout = i >= 0 and d < ends[i]   # [start, end)
        at_end = i >= 0 and d == ends[i]       # join point (earnings day)
        normal.append(float('nan') if in_blackout else v)
        yellow.append(v if (in_blackout or at_end) else float('nan'))
    return normal, yellow


def build_blackout_figure(ticker: str,
                          quarter_gains: List[Dict],
                          curves: Dict[int, Dict],
                          horizons: List[int] = None,
                          scatter: bool = False,
                          overlay_series: Optional[Dict] = None,
                          entry_mode: str = 'end'):
    """Build the compounded-gain matplotlib figure for the Blackout tab.

    X-axis = quarters (chronological); one line per horizon; Y = cumulative
    % gain from $1 (buy at the entry close, sell N days later, repeat).

    Args:
        scatter: plot markers only instead of connected lines.
        overlay_series: {'dates': [...], 'closes': [...]} raw price series;
            when given, a normalized buy-and-hold price line is overlaid,
            sampled at each quarter's baseline date.
        entry_mode: 'end' (buy at blackout end / earnings day) or 'start'
            (buy at blackout start) — used for the title text.
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    if horizons is None:
        horizons = HORIZONS

    # Union of quarter labels in chronological order (rows are already sorted;
    # labels can repeat across years only in string form, so keep first-seen order)
    all_labels: List[str] = []
    for row in quarter_gains:
        if row['label'] not in all_labels:
            all_labels.append(row['label'])
    pos = {label: i for i, label in enumerate(all_labels)}

    # Plain Figure (no pyplot): avoids pyplot managers/hidden Tk windows —
    # the figure is embedded in the app's own canvas anyway
    fig = Figure(figsize=(11, 6))
    ax = fig.subplots()
    if not all_labels:
        ax.text(0.5, 0.5, "No completed blackout windows to display",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        fig.suptitle(f"{ticker} — Post-Blackout Performance")
        return fig

    for n in horizons:
        curve = curves.get(n)
        if not curve or not curve['labels']:
            continue
        xs = [pos[lbl] for lbl in curve['labels']]
        if scatter:
            ax.plot(xs, curve['values'], linestyle='None', marker='o',
                    markersize=4, label=f"{n}d ×{curve['count']}")
        else:
            ax.plot(xs, curve['values'], marker='o', markersize=3,
                    linewidth=1.4, label=f"{n}d ×{curve['count']}")

    if ax.lines and overlay_series and overlay_series.get('dates'):
        # Buy-and-hold price, normalized to the first sampled quarter and
        # sampled at each quarter's baseline close.
        o_dates, o_closes = overlay_series['dates'], overlay_series['closes']
        xs, ys, ref = [], [], None
        for row in quarter_gains:
            idx = bisect_left(o_dates, row['baseline_date'])
            if idx >= len(o_dates):
                continue
            c = o_closes[idx]
            if c is None or pd.isna(c):
                continue
            if ref is None:
                ref = c
            xs.append(pos[row['label']])
            ys.append((c / ref - 1.0) * 100.0)
        if ys and ref:
            # Buy & Hold renders as a solid line (never scatter/dotted)
            ax.plot(xs, ys, color='black', linewidth=1.4, linestyle='-',
                    label="Buy & Hold (price)")

    if not ax.lines:
        # No horizon selected / no data for the selected horizons
        ax.text(0.5, 0.5, "No horizons selected",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        fig.suptitle(f"{ticker} — Post-Blackout Performance")
        return fig

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel("Cumulative % gain")
    entry_note = ("blackout start" if entry_mode == 'start'
                  else "blackout end (earnings-day close)")
    ax.set_title(f"{ticker} — Buy at {entry_note}, "
                 f"sell N calendar days later, repeat each quarter")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Horizon (quarters completed)", fontsize=8, title_fontsize=8,
              loc='best')
    fig.tight_layout()
    return fig


def build_daily_equity_figure(ticker: str,
                              daily_curves: Dict[int, Dict],
                              horizons: List[int] = None,
                              scatter: bool = False,
                              overlay_series: Optional[Dict] = None,
                              highlight_blackout: bool = False,
                              blackout_intervals: Optional[List[tuple]] = None,
                              entry_mode: str = 'end',
                              show_blackout_lines: bool = False):
    """Build the daily-equity matplotlib figure for the Blackout tab.

    X-axis = actual calendar dates (all dates shown); one line per horizon;
    Y = cumulative % gain from $1, marked at each close (flat while in cash
    between windows).

    Args:
        scatter: plot small markers only instead of connected lines.
        overlay_series: {'dates': [...], 'closes': [...]} raw price series;
            when given, a buy-and-hold price line normalized to the equity
            start date is overlaid for comparison.
        highlight_blackout: draw the equity segments during blackout windows
            (given by blackout_intervals) in gold/yellow so the flat pre-buy
            runs are easy to spot. All dates stay on the axis.
        blackout_intervals: [(blackout_start, blackout_end), ...] windows.
        entry_mode: 'end' (buy at blackout end / earnings day) or 'start'
            (buy at blackout start) — used for the title text.
        show_blackout_lines: draw vertical lines at each blackout start
            (yellow dashed) and end/earnings (green dotted) date within the
            plotted range, with legend entries for both.
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    if horizons is None:
        horizons = HORIZONS

    # Plain Figure (no pyplot): avoids pyplot managers/hidden Tk windows —
    # the figure is embedded in the app's own canvas anyway
    fig = Figure(figsize=(11, 6))
    ax = fig.subplots()
    plotted = False
    first_date = None
    last_date = None
    colors = plt.get_cmap('tab10')
    for idx, n in enumerate(horizons):
        curve = daily_curves.get(n)
        if not curve or not curve['dates']:
            continue
        xs = pd.to_datetime(curve['dates'])
        if first_date is None:
            first_date = curve['dates'][0]
        last_date = curve['dates'][-1]
        vals = curve['values']
        final = vals[-1] if vals else 0.0
        color = colors(idx % 10)
        label = f"{n}d  ({final:+.0f}%)"
        if highlight_blackout and blackout_intervals:
            normal, yellow = split_blackout_values(curve['dates'], vals,
                                                   blackout_intervals)
            if scatter:
                ax.plot(xs, normal, linestyle='None', marker='.',
                        markersize=1.6, alpha=0.85, color=color)
                ax.plot(xs, yellow, linestyle='None', marker='.',
                        markersize=2.2, alpha=0.95, color='#FFD700')
            else:
                ax.plot(xs, normal, linewidth=1.0, color=color, label=label)
                ax.plot(xs, yellow, linewidth=2.4, color='#FFD700')
        elif scatter:
            ax.plot(xs, vals, linestyle='None', marker='.', markersize=1.6,
                    alpha=0.85, color=color, label=label)
        else:
            ax.plot(xs, vals, linewidth=1.0, color=color, label=label)
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No horizons selected / no completed windows",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        fig.suptitle(f"{ticker} — Post-Blackout Daily Equity")
        return fig

    if overlay_series and overlay_series.get('dates'):
        # Buy-and-hold price normalized to the same start as the equity
        # curves, so both begin at 0% for an apples-to-apples comparison.
        o_dates, o_closes = overlay_series['dates'], overlay_series['closes']
        start_idx = bisect_left(o_dates, first_date) if first_date else 0
        ref = o_closes[start_idx] if start_idx < len(o_closes) else None
        if ref is not None and not pd.isna(ref):
            xs = [pd.to_datetime(d) for d in o_dates[start_idx:]]
            ys = [(c / ref - 1.0) * 100.0 for c in o_closes[start_idx:]]
            # Buy & Hold renders as a solid line (never scatter/dotted)
            ax.plot(xs, ys, color='black', linewidth=1.4, linestyle='-',
                    label="Buy & Hold (price)")

    handles, labels = ax.get_legend_handles_labels()
    if highlight_blackout and blackout_intervals:
        handles.append(Line2D([0], [0], color='#FFD700', lw=3))
        labels.append("Blackout")
    if show_blackout_lines and blackout_intervals and first_date:
        # Vertical markers at each blackout start (yellow dashed) and
        # end/earnings (green dotted) date. Lines outside the plotted
        # range are skipped so the x-axis is not stretched backward.
        n_start = n_end = 0
        for start_d, end_d in blackout_intervals:
            if first_date <= start_d <= last_date:
                ax.axvline(pd.Timestamp(start_d), color='#FFD700',
                           linestyle='--', linewidth=0.7, alpha=0.55)
                n_start += 1
            if first_date <= end_d <= last_date:
                ax.axvline(pd.Timestamp(end_d), color='green',
                           linestyle=':', linewidth=0.9, alpha=0.6)
                n_end += 1
        if n_start:
            handles.append(Line2D([0], [0], color='#FFD700', lw=1.5, ls='--'))
            labels.append("Blackout start")
        if n_end:
            handles.append(Line2D([0], [0], color='green', lw=1.5, ls=':'))
            labels.append("Blackout end")
    ax.legend(handles, labels, title="Horizon (final gain)", fontsize=8,
              title_fontsize=8, loc='best')

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.set_ylabel("Cumulative % gain (daily mark)")
    entry_note = ("blackout start" if entry_mode == 'start'
                  else "blackout end")
    ax.set_title(f"{ticker} — Daily equity: buy at {entry_note}, sell N "
                 f"calendar days later, cash between quarters")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def extend_result_horizon(result: Dict, n: int) -> Dict:
    """Add an arbitrary holding period (any N days, possibly > 1 quarter) to
    an existing analysis result, in place and without re-fetching data.

    Computes per-quarter gains/exits for horizon n for BOTH entry modes,
    merges them into each mode's quarter_gains rows, and rebuilds the
    compounded curve and daily equity curve for that horizon (daily walk
    skips buy signals that arrive while still holding, so long holds span
    multiple quarters naturally).

    Args:
        result: dict returned by analyze_ticker().
        n: holding period in calendar days (> 0).

    Returns:
        The same result dict, for chaining.
    """
    n = int(n)
    if n <= 0:
        return result
    ps = result.get('price_series') or {}
    if not ps.get('dates'):
        return result
    price = pd.DataFrame({'Date': list(ps['dates']),
                          'Close': list(ps['closes'])})
    for entry, mode in result['modes'].items():
        if n in mode['curves']:
            continue  # already computed (e.g. a stock HORIZONS value)
        rows_n = compute_quarter_gains(price, result['periods'],
                                       horizons=[n],
                                       as_of=result.get('as_of'),
                                       entry=entry)
        # rows align 1:1 with mode['quarter_gains'] (same periods/as_of/entry)
        for dst, src in zip(mode['quarter_gains'], rows_n):
            dst['gains'][n] = src['gains'][n]
            dst['exits'][n] = src['exits'][n]
        mode['curves'][n] = compute_compounded_curves(
            mode['quarter_gains'], horizons=[n])[n]
        mode['daily_curves'][n] = compute_daily_equity_curves(
            price, mode['quarter_gains'], horizons=[n],
            as_of=result.get('as_of'))[n]
    return result


# ---------------------------------------------------------------------------
# Orchestrator (network + data manager; not unit-tested)
# ---------------------------------------------------------------------------

def analyze_ticker(ticker: str,
                   manager,
                   blackout_start_days: int = DEFAULT_BLACKOUT_START_DAYS,
                   refresh_prices: bool = False,
                   as_of: Optional[date] = None) -> Dict:
    """Run the full blackout analysis for one ticker.

    Args:
        ticker: symbol to analyze.
        manager: StockDataManager instance (used for full price history).
        blackout_start_days: calendar days between blackout start and earnings.
        refresh_prices: force a fresh max-period price download.
        as_of: override "today" (testing only).

    Returns:
        Dict with keys: ticker, periods, price_series, as_of,
        quarters_analyzed, and modes — {'end': {...}, 'start': {...}} where
        each mode holds quarter_gains, curves, daily_curves and
        blackout_intervals for that entry point (buy at blackout end vs
        blackout start). Both modes are computed up front so the GUI can
        flip between them without re-fetching data; additional holding
        periods (any N days) can be merged later via extend_result_horizon().
    """
    ticker = ticker.upper()
    if as_of is None:
        as_of = date.today()
    import threading
    t0 = time.time()

    def _step(msg):
        logging.info(f"[blackout:{ticker}] {msg} "
                     f"({time.time() - t0:.1f}s elapsed, "
                     f"thread={threading.current_thread().name})")

    _step("fetching earnings dates")
    earnings = get_earnings_dates(ticker)
    periods = get_blackout_periods(earnings, blackout_start_days)
    _step(f"got {len(periods)} blackout periods")

    price = manager.load_full_history(ticker, refresh=refresh_prices)
    if price is None or price.empty or 'Close' not in price.columns:
        raise RuntimeError(f"No price history available for {ticker}")
    _step(f"price history loaded ({len(price)} rows)")

    modes = {}
    for entry in ('end', 'start'):
        quarter_gains = compute_quarter_gains(price, periods, as_of=as_of,
                                              entry=entry)
        modes[entry] = {
            'quarter_gains': quarter_gains,
            'curves': compute_compounded_curves(quarter_gains),
            'daily_curves': compute_daily_equity_curves(price, quarter_gains,
                                                        as_of=as_of),
            'blackout_intervals': [(r['blackout_start'], r['blackout_end'])
                                   for r in quarter_gains],
        }

    # Raw price series (through as_of) for the buy-and-hold overlay
    p_dates, p_closes = _extract_series(price)
    cutoff = bisect_left(p_dates, as_of)
    price_series = {'dates': p_dates[:cutoff], 'closes': p_closes[:cutoff]}

    _step("analysis complete")
    return {
        'ticker': ticker,
        'periods': periods,
        'modes': modes,
        'price_series': price_series,
        'as_of': as_of,
        'quarters_analyzed': len(modes['end']['quarter_gains']),
    }
