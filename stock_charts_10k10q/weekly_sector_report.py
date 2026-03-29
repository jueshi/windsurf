# -*- coding: utf-8 -*-
"""Weekly sector rotation report: analyze sectors and email summary.

Designed to run as a scheduled task (Windows Task Scheduler).
Uses Gemini/OpenAI to summarize sector rotation data, then sends
the report via Outlook COM (pywin32).

Usage:
    python weekly_sector_report.py
    python weekly_sector_report.py --to other@email.com
    python weekly_sector_report.py --dry-run   # print report without sending
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
import pandas as pd

# Ensure project directory is on the path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from data_manager import StockDataManager
from sector_rotation import (
    CORE_SECTOR_ETFS, SECTOR_ETF_MAP, BENCHMARK,
    load_sector_data, build_rotation_table, build_rolling_ranks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DEFAULT_RECIPIENT = "jueshi@gmail.com"


def _update_sector_data(manager):
    """Download/refresh data for all sector ETFs and benchmark."""
    all_tickers = list(CORE_SECTOR_ETFS) + [BENCHMARK]
    for ticker in all_tickers:
        try:
            manager.update_data(ticker)
        except Exception as e:
            logging.warning(f"Failed to update {ticker}: {e}")


def _build_report_text(rotation_table, rolling_ranks):
    """Build a plain-text data summary for the LLM prompt."""
    lines = []
    lines.append("=== SECTOR ROTATION DATA ===")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %A')}")
    lines.append("")

    # Current rankings
    lines.append("SECTOR MOMENTUM RANKINGS (sorted by composite score):")
    lines.append(f"{'Rank':<5} {'ETF':<6} {'Sector':<18} {'1W vs SPY':>10} {'2W vs SPY':>10} {'1M vs SPY':>10} {'3M vs SPY':>10} {'Composite':>10}")
    lines.append("-" * 85)
    for _, row in rotation_table.iterrows():
        lines.append(
            f"{int(row['rank']):<5} {row['ETF']:<6} {row['Sector']:<18} "
            f"{row.get('5d_vs_spy', 0):>+9.2f}% {row.get('10d_vs_spy', 0):>+9.2f}% "
            f"{row.get('21d_vs_spy', 0):>+9.2f}% {row.get('63d_vs_spy', 0):>+9.2f}% "
            f"{row.get('composite', 0):>+9.2f}"
        )
    lines.append("")

    # Recent rank changes (last 5 trading days)
    if not rolling_ranks.empty and len(rolling_ranks) >= 5:
        lines.append("RANK CHANGES (last 5 trading days):")
        recent = rolling_ranks.iloc[-5:]
        for etf in rolling_ranks.columns:
            start_rank = int(recent[etf].iloc[0])
            end_rank = int(recent[etf].iloc[-1])
            change = start_rank - end_rank  # positive = improved
            direction = "UP" if change > 0 else "DOWN" if change < 0 else "FLAT"
            lines.append(f"  {etf} ({SECTOR_ETF_MAP.get(etf, etf)}): {start_rank} -> {end_rank} ({direction} {abs(change)})")
        lines.append("")

    # Absolute returns
    lines.append("ABSOLUTE RETURNS (not relative):")
    for _, row in rotation_table.iterrows():
        lines.append(
            f"  {row['ETF']} ({row['Sector']}): "
            f"1W {row.get('5d_roc', 0):+.2f}%, 2W {row.get('10d_roc', 0):+.2f}%, "
            f"1M {row.get('21d_roc', 0):+.2f}%, 3M {row.get('63d_roc', 0):+.2f}%"
        )

    return "\n".join(lines)


def _get_top_sector_etfs(rotation_table, n=3):
    """Return the top N sector ETFs by composite score."""
    return rotation_table.head(n)[["ETF", "Sector", "composite"]].to_dict("records")


def _analyze_stock_technicals(manager, ticker):
    """Compute technical signals for a single stock to detect breakouts.

    Returns dict with price stats and breakout indicators.
    """
    import numpy as np

    try:
        df = manager.load_data(ticker)
        if df is None or df.empty or len(df) < 60:
            return None

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        close = df["Close"]
        volume = df["Volume"] if "Volume" in df.columns else None

        current = close.iloc[-1]

        # Price performance
        roc_5 = ((current / close.iloc[-5]) - 1) * 100 if len(close) >= 5 else np.nan
        roc_21 = ((current / close.iloc[-21]) - 1) * 100 if len(close) >= 21 else np.nan
        roc_63 = ((current / close.iloc[-63]) - 1) * 100 if len(close) >= 63 else np.nan

        # 52-week high proximity
        high_252 = close.iloc[-252:].max() if len(close) >= 252 else close.max()
        low_252 = close.iloc[-252:].min() if len(close) >= 252 else close.min()
        pct_from_high = ((current / high_252) - 1) * 100
        range_position = ((current - low_252) / (high_252 - low_252) * 100) if high_252 != low_252 else 50

        # Breakout detection: price above 50-day high with volume surge
        high_50 = close.iloc[-50:].max() if len(close) >= 50 else close.max()
        above_50d_high = current >= high_50 * 0.98  # within 2% of 50-day high

        # Consolidation detection: measure volatility compression
        # Compare recent 20-day range to prior 40-day range
        if len(close) >= 60:
            recent_range = (close.iloc[-20:].max() - close.iloc[-20:].min()) / close.iloc[-20:].mean() * 100
            prior_range = (close.iloc[-60:-20].max() - close.iloc[-60:-20].min()) / close.iloc[-60:-20].mean() * 100
            consolidation_ratio = recent_range / prior_range if prior_range > 0 else 1.0
        else:
            recent_range = prior_range = consolidation_ratio = np.nan

        # Volume surge (last 5 days vs 50-day avg)
        vol_surge = np.nan
        if volume is not None and len(volume) >= 50:
            avg_vol_50 = volume.iloc[-50:].mean()
            avg_vol_5 = volume.iloc[-5:].mean()
            vol_surge = avg_vol_5 / avg_vol_50 if avg_vol_50 > 0 else 1.0

        # Moving averages
        sma_20 = close.iloc[-20:].mean() if len(close) >= 20 else np.nan
        sma_50 = close.iloc[-50:].mean() if len(close) >= 50 else np.nan
        above_sma20 = current > sma_20 if not np.isnan(sma_20) else False
        above_sma50 = current > sma_50 if not np.isnan(sma_50) else False

        # Breakout score (higher = stronger breakout signal)
        breakout_score = 0
        if above_50d_high:
            breakout_score += 3
        if above_sma20:
            breakout_score += 1
        if above_sma50:
            breakout_score += 1
        if not np.isnan(consolidation_ratio) and consolidation_ratio < 0.5:
            breakout_score += 3  # tight consolidation before move
        if not np.isnan(vol_surge) and vol_surge > 1.5:
            breakout_score += 2  # volume confirmation

        return {
            "ticker": ticker,
            "price": current,
            "roc_1w": roc_5,
            "roc_1m": roc_21,
            "roc_3m": roc_63,
            "pct_from_52w_high": pct_from_high,
            "range_position": range_position,
            "above_50d_high": above_50d_high,
            "consolidation_ratio": consolidation_ratio,
            "vol_surge": vol_surge,
            "above_sma20": above_sma20,
            "above_sma50": above_sma50,
            "breakout_score": breakout_score,
        }
    except Exception as e:
        logging.warning(f"Technical analysis failed for {ticker}: {e}")
        return None


def _build_holdings_analysis(manager, top_sectors):
    """For each top sector, fetch top 10 holdings and compute breakout signals.

    Returns dict: {etf: {"sector": ..., "holdings": [...]}}
    """
    from sector_rotation import get_sector_top_holdings

    results = {}
    for sector_info in top_sectors:
        etf = sector_info["ETF"]
        sector_name = sector_info["Sector"]
        logging.info(f"Fetching top 10 holdings for {etf} ({sector_name})...")

        holdings = get_sector_top_holdings(etf, n=10)
        if not holdings:
            continue

        # Download missing data
        for h in holdings:
            try:
                data_path = manager._get_data_path(h["symbol"])
                if not os.path.exists(data_path):
                    manager.update_data(h["symbol"])
            except Exception as e:
                logging.warning(f"Failed to download {h['symbol']}: {e}")

        # Analyze each holding
        analyzed = []
        for h in holdings:
            tech = _analyze_stock_technicals(manager, h["symbol"])
            if tech:
                tech["name"] = h["name"]
                tech["weight"] = h["weight"]
                analyzed.append(tech)

        # Sort by breakout score descending
        analyzed.sort(key=lambda x: x["breakout_score"], reverse=True)
        results[etf] = {"sector": sector_name, "holdings": analyzed}

    return results


def _build_holdings_text(holdings_analysis):
    """Build plain-text holdings data for the LLM prompt."""
    lines = []
    lines.append("\n=== TOP SECTOR HOLDINGS ANALYSIS ===")
    for etf, data in holdings_analysis.items():
        lines.append(f"\n--- {etf} ({data['sector']}) Top Holdings ---")
        lines.append(f"{'Ticker':<7} {'Name':<30} {'Weight':>7} {'1W':>7} {'1M':>7} {'3M':>7} "
                     f"{'%From52H':>8} {'VolSurge':>9} {'ConsRatio':>10} {'BkScore':>8}")
        lines.append("-" * 120)
        for h in data["holdings"]:
            cr = h.get('consolidation_ratio')
            cr_str = f"{cr:.2f}" if cr == cr else "N/A"
            vs = h.get('vol_surge')
            vs_str = f"{vs:.2f}x" if vs == vs else "N/A"
            lines.append(
                f"{h['ticker']:<7} {h['name'][:29]:<30} {h['weight']:>6.1%} "
                f"{h.get('roc_1w', 0):>+6.1f}% {h.get('roc_1m', 0):>+6.1f}% {h.get('roc_3m', 0):>+6.1f}% "
                f"{h.get('pct_from_52w_high', 0):>+7.1f}% "
                f"{vs_str:>9} {cr_str:>10} {h['breakout_score']:>8}"
            )
    return "\n".join(lines)


def _generate_stock_picks_summary(report_text, holdings_text):
    """Call LLM to analyze top sector holdings and identify breakout candidates."""
    from gemini_analyzer import _call_llm

    prompt = f"""You are a senior technical analyst specializing in breakout trading
within leading sectors. You are given sector rotation data and detailed technical
analysis of the top 10 holdings in each leading sector.

{report_text}

{holdings_text}

COLUMN DEFINITIONS:
- Weight: holding weight in the sector ETF
- 1W/1M/3M: price return over 1 week / 1 month / 3 months
- %From52H: how far below the 52-week high (0% = at the high)
- VolSurge: recent 5-day avg volume / 50-day avg volume (>1.5 = volume spike)
- ConsRatio: recent 20-day price range / prior 40-day range (<0.5 = tight consolidation before move)
- BkScore: composite breakout score (0-10, higher = stronger breakout signal)
  - 3 pts: near 50-day high
  - 3 pts: tight consolidation (ConsRatio < 0.5)
  - 2 pts: volume surge (>1.5x)
  - 1 pt each: above SMA20, above SMA50

Please provide:

1. **Top Breakout Candidates** (3-5 stocks): Stocks with the highest breakout scores,
   especially those breaking out of a consolidation pattern (low ConsRatio + high BkScore).
   For each, explain:
   - The technical setup (consolidation length, volume pattern)
   - Why the sector tailwind supports this trade
   - Nearest resistance level to watch (based on 52-week high proximity)

2. **Emerging Setups** (2-3 stocks): Stocks that haven't broken out yet but are in
   tight consolidation (low ConsRatio) within a leading sector. These are "watch list" names.

3. **Avoid List** (1-2 stocks): Any top holdings that look weak despite being in a
   strong sector (lagging the sector, below moving averages, etc.)

Use specific numbers from the data. Be concise and actionable.
"""
    return _call_llm(prompt)


def _generate_ai_summary(report_text):
    """Call LLM to generate a sector rotation analysis summary."""
    from gemini_analyzer import _call_llm

    prompt = f"""You are a senior market strategist writing a weekly sector rotation briefing
for an active investor. Analyze the following sector rotation data and produce a concise,
actionable report.

{report_text}

Please structure your response as follows:

1. **Market Regime** (1-2 sentences): What phase of the economic cycle does sector leadership suggest? (early recovery, mid expansion, late cycle, defensive/recession)

2. **Sectors Gaining Momentum** (top 2-3): Which sectors are rotating IN? What's driving it? Any notable acceleration in the past week?

3. **Sectors Losing Momentum** (bottom 2-3): Which sectors are rotating OUT? Is this a gradual fade or sharp reversal?

4. **Key Rotation Signal**: What's the single most important rotation to watch right now? (e.g., "Money moving from Tech to Utilities signals risk-off")

5. **Actionable Ideas** (2-3 bullets): Specific sector ETFs to consider adding or trimming based on the rotation data.

6. **Risk Check**: Any divergences or warning signs in the data? (e.g., all sectors weak vs SPY = broad market stress)

Keep the tone professional but direct. Use the actual numbers from the data.
Write in English.
"""
    return _call_llm(prompt)


def _build_html_email(ai_summary, rotation_table, rolling_ranks,
                      holdings_analysis=None, stock_picks_summary=None,
                      chart_tickers=None):
    """Build a nicely formatted HTML email body."""
    date_str = datetime.now().strftime("%Y-%m-%d %A")

    # Convert AI summary markdown-ish text to basic HTML
    ai_html = ai_summary.replace("\n\n", "</p><p>").replace("\n", "<br>")
    # Bold **text**
    import re
    ai_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', ai_html)
    ai_html = f"<p>{ai_html}</p>"

    # Color helper for values
    def _val_cell(val, fmt="+.2f"):
        if val != val:  # NaN
            return '<td style="padding:6px 10px;text-align:right;color:#999;">N/A</td>'
        color = "#2e7d32" if val > 0 else "#d32f2f" if val < 0 else "#666"
        return f'<td style="padding:6px 10px;text-align:right;color:{color};font-weight:600;">{val:{fmt}}%</td>'

    # Momentum rankings table
    rank_rows = ""
    for _, row in rotation_table.iterrows():
        bg = "#f8f9fa" if int(row["rank"]) % 2 == 0 else "#ffffff"
        rank_rows += f"""<tr style="background:{bg};">
            <td style="padding:6px 10px;text-align:center;">{int(row['rank'])}</td>
            <td style="padding:6px 10px;font-weight:600;">{row['ETF']}</td>
            <td style="padding:6px 10px;">{row['Sector']}</td>
            {_val_cell(row.get('5d_vs_spy', 0))}
            {_val_cell(row.get('10d_vs_spy', 0))}
            {_val_cell(row.get('21d_vs_spy', 0))}
            {_val_cell(row.get('63d_vs_spy', 0))}
            {_val_cell(row.get('composite', 0), '+.1f')}
        </tr>"""

    # Helper function to build rank change rows for a time period
    def _build_rank_changes(rolling_ranks, lookback_days, label):
        """Build rank change rows for a specific lookback period."""
        if rolling_ranks.empty or len(rolling_ranks) < lookback_days:
            return ""

        recent = rolling_ranks.iloc[-lookback_days:]
        changes = []
        for etf in rolling_ranks.columns:
            start_rank = int(recent[etf].iloc[0])
            end_rank = int(recent[etf].iloc[-1])
            delta = start_rank - end_rank
            changes.append((etf, start_rank, end_rank, delta))
        changes.sort(key=lambda x: -x[3])  # biggest improvement first

        rows = ""
        for etf, sr, er, delta in changes:
            if delta > 0:
                arrow, color = "&#9650;", "#2e7d32"
            elif delta < 0:
                arrow, color = "&#9660;", "#d32f2f"
            else:
                arrow, color = "&#9644;", "#999"
            name = SECTOR_ETF_MAP.get(etf, etf)
            rows += (
                f'<tr><td style="padding:4px 10px;font-weight:600;">{etf}</td>'
                f'<td style="padding:4px 10px;">{name}</td>'
                f'<td style="padding:4px 10px;text-align:center;">{sr}</td>'
                f'<td style="padding:4px 10px;text-align:center;">{er}</td>'
                f'<td style="padding:4px 10px;text-align:center;color:{color};font-weight:700;">'
                f'{arrow} {abs(delta)}</td></tr>'
            )
        return rows

    # Rank changes for different periods
    rank_change_rows_5d = _build_rank_changes(rolling_ranks, 5, "5-day")
    rank_change_rows_1w = _build_rank_changes(rolling_ranks, 5, "1-week")
    rank_change_rows_1m = _build_rank_changes(rolling_ranks, 21, "1-month")

    # Absolute returns table
    abs_rows = ""
    for _, row in rotation_table.iterrows():
        bg = "#f8f9fa" if int(row["rank"]) % 2 == 0 else "#ffffff"
        abs_rows += f"""<tr style="background:{bg};">
            <td style="padding:6px 10px;font-weight:600;">{row['ETF']}</td>
            <td style="padding:6px 10px;">{row['Sector']}</td>
            {_val_cell(row.get('5d_roc', 0))}
            {_val_cell(row.get('10d_roc', 0))}
            {_val_cell(row.get('21d_roc', 0))}
            {_val_cell(row.get('63d_roc', 0))}
        </tr>"""

    th_style = 'style="padding:8px 10px;text-align:right;border-bottom:2px solid #dee2e6;background:#e9ecef;"'
    th_left = 'style="padding:8px 10px;text-align:left;border-bottom:2px solid #dee2e6;background:#e9ecef;"'
    th_center = 'style="padding:8px 10px;text-align:center;border-bottom:2px solid #dee2e6;background:#e9ecef;"'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="font-family:Segoe UI,Arial,sans-serif;color:#333;max-width:800px;margin:0 auto;padding:20px;">

<h1 style="color:#1a237e;border-bottom:3px solid #1a237e;padding-bottom:8px;font-size:22px;">
    Weekly Sector Rotation Report</h1>
<p style="color:#666;margin-top:-5px;">{date_str}</p>

<h2 style="color:#1a237e;font-size:16px;margin-top:25px;">AI Analysis</h2>
<div style="background:#f5f7ff;border-left:4px solid #1a237e;padding:12px 16px;margin-bottom:20px;line-height:1.6;">
{ai_html}
</div>

<h2 style="color:#1a237e;font-size:16px;">Sector Momentum vs SPY</h2>
<table style="border-collapse:collapse;width:100%;font-size:13px;margin-bottom:20px;">
<thead><tr>
    <th {th_center}>Rank</th><th {th_left}>ETF</th><th {th_left}>Sector</th>
    <th {th_style}>1W</th><th {th_style}>2W</th><th {th_style}>1M</th><th {th_style}>3M</th>
    <th {th_style}>Score</th>
</tr></thead>
<tbody>{rank_rows}</tbody>
</table>

<h2 style="color:#1a237e;font-size:16px;">Rank Changes Since Last Week (5-day)</h2>
<table style="border-collapse:collapse;width:100%;font-size:13px;margin-bottom:20px;">
<thead><tr>
    <th {th_left}>ETF</th><th {th_left}>Sector</th>
    <th {th_center}>From</th><th {th_center}>To</th><th {th_center}>Change</th>
</tr></thead>
<tbody>{rank_change_rows_5d if rank_change_rows_5d else '<tr><td colspan="5" style="padding:8px 10px;text-align:center;color:#999;">Insufficient data</td></tr>'}</tbody>
</table>

<h2 style="color:#1a237e;font-size:16px;">Rank Changes Since Last Month (21-day)</h2>
<table style="border-collapse:collapse;width:100%;font-size:13px;margin-bottom:20px;">
<thead><tr>
    <th {th_left}>ETF</th><th {th_left}>Sector</th>
    <th {th_center}>From</th><th {th_center}>To</th><th {th_center}>Change</th>
</tr></thead>
<tbody>{rank_change_rows_1m if rank_change_rows_1m else '<tr><td colspan="5" style="padding:8px 10px;text-align:center;color:#999;">Insufficient data</td></tr>'}</tbody>
</table>

<h2 style="color:#1a237e;font-size:16px;">Absolute Returns</h2>
<table style="border-collapse:collapse;width:100%;font-size:13px;margin-bottom:20px;">
<thead><tr>
    <th {th_left}>ETF</th><th {th_left}>Sector</th>
    <th {th_style}>1W</th><th {th_style}>2W</th><th {th_style}>1M</th><th {th_style}>3M</th>
</tr></thead>
<tbody>{abs_rows}</tbody>
</table>

{_build_holdings_html(holdings_analysis, stock_picks_summary, th_style, th_left, th_center) if holdings_analysis else ""}

{_build_live_charts_html(chart_tickers, holdings_analysis) if chart_tickers else ""}

<hr style="border:none;border-top:1px solid #dee2e6;margin:20px 0;">
<p style="color:#999;font-size:11px;">Generated by Personal AI Stock Assistant</p>
</body></html>"""

    return html


def _translate_ticker_for_stockcharts(ticker):
    """Translate ticker symbols to StockCharts.com format.

    Some tickers use different symbols on StockCharts:
    - BRK-A -> BRK/A
    - BRK-B -> BRK/B
    """
    translation_map = {
        "BRK-A": "BRK/A",
        "BRK-B": "BRK/B",
    }
    return translation_map.get(ticker, ticker)


def _build_live_charts_html(tickers, holdings_analysis=None):
    """Build HTML with live StockCharts.com chart images embedded directly.

    Each ticker gets daily (1yr), weekly (5yr), and monthly (max) charts in a row.
    Charts are grouped by sector when holdings_analysis is provided.
    Translates ticker symbols to StockCharts.com format (e.g., BRK-A -> BRK/A).
    """
    if not tickers:
        return ""

    # Build ticker-to-sector mapping
    ticker_sectors = {}
    if holdings_analysis:
        for etf, data in holdings_analysis.items():
            for h in data.get("holdings", []):
                ticker_sectors[h["ticker"]] = f"{etf} ({data['sector']})"

    # Group by sector preserving order
    by_sector = {}
    for t in tickers:
        sector = ticker_sectors.get(t, "Other")
        by_sector.setdefault(sector, []).append(t)

    timeframes = [
        ("D", "Daily (1yr)", 1, 0, 0, "t3784817951c"),
        ("W", "Weekly (5yr)", 5, 0, 0, "t1918602869c"),
        ("M", "Monthly (Max)", 99, 0, 0, "t1918602869c"),
    ]

    sections = []
    sections.append('<h2 style="color:#1a237e;font-size:16px;margin-top:30px;">Technical Charts (Live)</h2>')

    for sector, sector_tickers in by_sector.items():
        sections.append(f'<h3 style="color:#333;font-size:14px;margin-top:20px;border-bottom:1px solid #ddd;'
                        f'padding-bottom:4px;">{sector}</h3>')

        for ticker in sector_tickers:
            chart_ticker = _translate_ticker_for_stockcharts(ticker)
            sections.append(f'<p style="font-weight:700;font-size:13px;margin:12px 0 4px;">{ticker}</p>')
            sections.append('<table style="border-collapse:collapse;"><tr>')
            for p, label, yr, mn, dy, style_id in timeframes:
                url = (f"https://stockcharts.com/c-sc/sc?s={chart_ticker}&p={p}"
                       f"&yr={yr}&mn={mn}&dy={dy}&i={style_id}")
                sections.append(
                    f'<td style="padding:2px;vertical-align:top;">'
                    f'<div style="font-size:10px;color:#666;text-align:center;margin-bottom:2px;">{label}</div>'
                    f'<img src="{url}" style="max-width:280px;height:auto;" alt="{ticker} {label}"></td>'
                )
            sections.append('</tr></table>')

    return "\n".join(sections)


def _build_holdings_html(holdings_analysis, stock_picks_summary, th_style, th_left, th_center):
    """Build HTML sections for holdings deep dive."""
    import re

    def _val_cell(val, fmt="+.1f"):
        if val != val:
            return '<td style="padding:5px 8px;text-align:right;color:#999;">N/A</td>'
        color = "#2e7d32" if val > 0 else "#d32f2f" if val < 0 else "#666"
        return f'<td style="padding:5px 8px;text-align:right;color:{color};font-weight:600;">{val:{fmt}}%</td>'

    def _score_badge(score):
        if score >= 7:
            bg, label = "#2e7d32", "STRONG"
        elif score >= 4:
            bg, label = "#f57c00", "MODERATE"
        else:
            bg, label = "#999", "LOW"
        return (f'<td style="padding:5px 8px;text-align:center;">'
                f'<span style="background:{bg};color:white;padding:2px 6px;border-radius:3px;'
                f'font-size:11px;font-weight:700;">{score} {label}</span></td>')

    sections = []

    # Stock picks AI summary
    if stock_picks_summary:
        picks_html = stock_picks_summary.replace("\n\n", "</p><p>").replace("\n", "<br>")
        picks_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', picks_html)
        picks_html = f"<p>{picks_html}</p>"
        sections.append(f"""
<h2 style="color:#1a237e;font-size:16px;margin-top:30px;">Stock Picks &amp; Breakout Candidates</h2>
<div style="background:#fff8e1;border-left:4px solid #f57c00;padding:12px 16px;margin-bottom:20px;line-height:1.6;">
{picks_html}
</div>""")

    # Holdings tables per sector
    for etf, data in holdings_analysis.items():
        holdings = data["holdings"]
        if not holdings:
            continue

        rows = ""
        for i, h in enumerate(holdings):
            bg = "#f8f9fa" if i % 2 == 0 else "#ffffff"
            cr = h.get("consolidation_ratio")
            cr_val = f"{cr:.2f}" if cr == cr else "N/A"
            # Highlight tight consolidation
            if cr == cr and cr < 0.5:
                cr_cell = f'<td style="padding:5px 8px;text-align:right;color:#2e7d32;font-weight:700;">{cr_val}</td>'
            else:
                cr_cell = f'<td style="padding:5px 8px;text-align:right;">{cr_val}</td>'

            vs = h.get("vol_surge")
            vs_val = f"{vs:.1f}x" if vs == vs else "N/A"
            if vs == vs and vs > 1.5:
                vs_cell = f'<td style="padding:5px 8px;text-align:right;color:#1a237e;font-weight:700;">{vs_val}</td>'
            else:
                vs_cell = f'<td style="padding:5px 8px;text-align:right;">{vs_val}</td>'

            rows += f"""<tr style="background:{bg};">
                <td style="padding:5px 8px;font-weight:600;">{h['ticker']}</td>
                <td style="padding:5px 8px;font-size:12px;">{h['name'][:25]}</td>
                <td style="padding:5px 8px;text-align:right;">{h['weight']:.1%}</td>
                {_val_cell(h.get('roc_1w', 0))}
                {_val_cell(h.get('roc_1m', 0))}
                {_val_cell(h.get('roc_3m', 0))}
                {_val_cell(h.get('pct_from_52w_high', 0))}
                {vs_cell}
                {cr_cell}
                {_score_badge(h['breakout_score'])}
            </tr>"""

        sections.append(f"""
<h2 style="color:#1a237e;font-size:16px;margin-top:25px;">{etf} ({data['sector']}) — Top Holdings</h2>
<table style="border-collapse:collapse;width:100%;font-size:12px;margin-bottom:15px;">
<thead><tr>
    <th {th_left}>Ticker</th><th {th_left}>Name</th>
    <th {th_style}>Wt</th><th {th_style}>1W</th><th {th_style}>1M</th><th {th_style}>3M</th>
    <th {th_style}>%52H</th><th {th_style}>Vol</th><th {th_style}>Cons.</th>
    <th {th_center}>Breakout</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>""")

    return "\n".join(sections)


def _generate_ticker_charts(manager, tickers, output_dir):
    """Generate daily, weekly, and monthly candlestick chart PNGs for each ticker.

    Returns dict: {ticker: {"daily": path, "weekly": path, "monthly": path}}
    Only includes tickers with enough data.
    """
    import matplotlib
    matplotlib.use("Agg")
    import mplfinance as mpf

    os.makedirs(output_dir, exist_ok=True)
    chart_style = mpf.make_mpf_style(
        base_mpf_style="charles",
        rc={"font.size": 8, "axes.titlesize": 10},
    )

    results = {}
    for ticker in tickers:
        try:
            df = manager.load_data(ticker)
            if df is None or df.empty:
                continue
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None) if df.index.tz else df.index

            # Ensure required OHLCV columns
            required = ["Open", "High", "Low", "Close"]
            if not all(c in df.columns for c in required):
                continue

            ticker_charts = {}
            timeframes = {
                "daily": (df.iloc[-120:], None),       # ~6 months daily
                "weekly": (df.iloc[-504:], "W"),        # ~2 years weekly
                "monthly": (df, "ME"),                  # all data monthly
            }

            for tf_name, (tf_data, resample) in timeframes.items():
                if tf_data.empty or len(tf_data) < 5:
                    continue

                if resample:
                    ohlc = tf_data.resample(resample).agg({
                        "Open": "first", "High": "max", "Low": "min", "Close": "last",
                        **({"Volume": "sum"} if "Volume" in tf_data.columns else {}),
                    }).dropna()
                else:
                    ohlc = tf_data

                if len(ohlc) < 5:
                    continue
                # Limit monthly to last 10 years to avoid mplfinance warnings
                if resample and len(ohlc) > 120:
                    ohlc = ohlc.iloc[-120:]

                save_path = os.path.join(output_dir, f"{ticker}_{tf_name}.png")
                # Add volume if available, add moving averages
                plot_kwargs = dict(
                    type="candle",
                    style=chart_style,
                    title=f"{ticker} ({tf_name.title()})",
                    figsize=(5, 3),
                    savefig=dict(fname=save_path, dpi=120, bbox_inches="tight", pad_inches=0.1),
                    tight_layout=True,
                    mav=(10, 20) if tf_name == "daily" else (10,),
                )
                if "Volume" in ohlc.columns and ohlc["Volume"].sum() > 0:
                    plot_kwargs["volume"] = True

                try:
                    mpf.plot(ohlc, **plot_kwargs)
                    ticker_charts[tf_name] = save_path
                    logging.info(f"Generated {tf_name} chart for {ticker}")
                except Exception as e:
                    logging.warning(f"Failed to generate {tf_name} chart for {ticker}: {e}")

            if ticker_charts:
                results[ticker] = ticker_charts

        except Exception as e:
            logging.warning(f"Chart generation failed for {ticker}: {e}")

    return results


def _build_charts_html(chart_paths, holdings_analysis):
    """Build HTML sections with inline chart images for each ticker.

    chart_paths: {ticker: {"daily": cid, "weekly": cid, "monthly": cid}}
    Returns HTML string referencing cid: URLs for inline images.
    """
    if not chart_paths:
        return ""

    sections = []
    sections.append('<h2 style="color:#1a237e;font-size:16px;margin-top:30px;">Technical Charts</h2>')

    # Group tickers by sector
    ticker_to_sector = {}
    for etf, data in (holdings_analysis or {}).items():
        for h in data.get("holdings", []):
            ticker_to_sector[h["ticker"]] = f"{etf} ({data['sector']})"

    # Group charts by sector
    by_sector = {}
    for ticker in chart_paths:
        sector = ticker_to_sector.get(ticker, "Other")
        by_sector.setdefault(sector, []).append(ticker)

    for sector, tickers in by_sector.items():
        sections.append(f'<h3 style="color:#333;font-size:14px;margin-top:20px;">{sector}</h3>')

        for ticker in tickers:
            charts = chart_paths[ticker]
            sections.append(f'<p style="font-weight:700;font-size:13px;margin:10px 0 5px;">{ticker}</p>')
            sections.append('<table style="border-collapse:collapse;"><tr>')
            for tf in ["daily", "weekly", "monthly"]:
                if tf in charts:
                    cid = charts[tf]
                    sections.append(
                        f'<td style="padding:2px;vertical-align:top;">'
                        f'<img src="cid:{cid}" style="max-width:280px;height:auto;" alt="{ticker} {tf}"></td>'
                    )
            sections.append('</tr></table>')

    return "\n".join(sections)


def _generate_stockcharts_gallery(tickers, output_path, holdings_analysis=None):
    """Generate an HTML page with live StockCharts.com charts grouped by sector.

    Each ticker gets daily (1yr), weekly (5yr), and monthly (max) charts in a row.
    Tickers are grouped under sector headers if holdings_analysis is provided.
    """
    # Build ticker-to-sector mapping
    ticker_sectors = {}
    if holdings_analysis:
        for etf, data in holdings_analysis.items():
            for h in data.get("holdings", []):
                ticker_sectors[h["ticker"]] = f"{etf} ({data['sector']})"

    # Group tickers by sector, preserving order
    by_sector = {}
    for t in tickers:
        sector = ticker_sectors.get(t, "Other")
        by_sector.setdefault(sector, []).append(t)

    date_str = datetime.now().strftime("%Y-%m-%d")

    # Build sector sections HTML
    sector_sections = ""
    for sector, sector_tickers in by_sector.items():
        tickers_js = json.dumps(sector_tickers)
        sector_sections += f"""
        <h2 style="color:#1a237e;margin:25px 0 10px;border-bottom:2px solid #1a237e;padding-bottom:5px;">{sector}</h2>
        <script>
        (function() {{
            const tickers = {tickers_js};
            const timeFrames = [
                {{ p: "D", label: "Daily (1yr)", yr: 1, mn: 0, dy: 0, i: "t3784817951c" }},
                {{ p: "W", label: "Weekly (5yr)", yr: 5, mn: 0, dy: 0, i: "t1918602869c" }},
                {{ p: "M", label: "Monthly (Max)", yr: 99, mn: 0, dy: 0, i: "t1918602869c" }}
            ];
            let html = '';
            for (const ticker of tickers) {{
                html += '<div class="stock-row"><h3 class="ticker-header">' + ticker + '</h3>';
                for (const tf of timeFrames) {{
                    const url = 'https://stockcharts.com/c-sc/sc?s=' + ticker + '&p=' + tf.p +
                                '&yr=' + tf.yr + '&mn=' + tf.mn + '&dy=' + tf.dy + '&i=' + tf.i;
                    html += '<div class="chart-cell"><div class="tf-label">' + tf.label +
                            '</div><img src="' + url + '" alt="' + ticker + ' ' + tf.label + '"></div>';
                }}
                html += '</div>';
            }}
            document.currentScript.insertAdjacentHTML('afterend', html);
        }})();
        </script>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sector Rotation Charts - {date_str}</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: Segoe UI, Arial, sans-serif; background: #f4f4f9; margin: 0; padding: 20px; color: #333; }}
h1 {{ text-align: center; color: #1a237e; margin-bottom: 5px; }}
.subtitle {{ text-align: center; color: #666; margin-bottom: 20px; }}
.stock-row {{ display: flex; flex-wrap: wrap; background: #fff; border: 1px solid #e0e0e0;
             border-radius: 5px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
.ticker-header {{ width: 100%; margin: 0; padding: 8px 15px; font-size: 1.3em; color: #1a237e;
                  background: #f8f8f8; border-bottom: 1px solid #e0e0e0; }}
.chart-cell {{ flex: 1 1 calc(100%/3); text-align: center; padding: 0; border-right: 1px solid #f0f0f0;
              min-width: 0; }}
.chart-cell:last-child {{ border-right: none; }}
.tf-label {{ font-size: 0.85em; font-weight: 600; color: #555; padding: 5px 0 2px; }}
img {{ max-width: 100%; height: auto; display: block; }}
@media (max-width: 900px) {{ .chart-cell {{ flex: 1 1 100%; border-right: none; }} }}
</style>
</head>
<body>
<h1>Weekly Sector Rotation Charts</h1>
<p class="subtitle">{date_str} &mdash; {len(tickers)} tickers across {len(by_sector)} sectors</p>
{sector_sections}
<hr style="margin-top:30px;border:none;border-top:1px solid #ccc;">
<p style="color:#999;font-size:11px;text-align:center;">Generated by Personal AI Stock Assistant. Charts from StockCharts.com.</p>
</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def _send_via_gmail_smtp(subject, body, recipient):
    """Send email using Gmail SMTP with app password.

    Requires GMAIL_APP_PASSWORD in .env (16-char app password from Google Account).
    Sender address is the recipient by default (sending to self).
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from dotenv import load_dotenv

    load_dotenv()
    app_password = os.getenv("GMAIL_APP_PASSWORD")
    sender = os.getenv("GMAIL_SENDER", recipient)

    if not app_password:
        raise ValueError(
            "GMAIL_APP_PASSWORD not set in .env.\n"
            "To create one: Google Account -> Security -> 2-Step Verification -> App passwords\n"
            "Then add to .env: GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx"
        )

    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText("See HTML version of this email.", "plain", "utf-8"))
    msg.attach(MIMEText(body, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, app_password)
        server.send_message(msg)

    logging.info(f"Email sent to {recipient} via Gmail SMTP")


def main():
    parser = argparse.ArgumentParser(description="Weekly Sector Rotation Report")
    parser.add_argument("--to", default=DEFAULT_RECIPIENT, help="Recipient email address")
    parser.add_argument("--dry-run", action="store_true", help="Print report without sending email")
    args = parser.parse_args()

    logging.info("Starting weekly sector rotation report...")

    # Initialize data manager
    manager = StockDataManager()

    # Update sector data
    logging.info("Updating sector ETF data...")
    _update_sector_data(manager)

    # Load and analyze
    logging.info("Loading sector data and computing rotation metrics...")
    data = load_sector_data(manager)
    if len(data) < 3:
        logging.error("Not enough sector data loaded. Aborting.")
        sys.exit(1)

    rotation_table = build_rotation_table(data)
    rolling_ranks = build_rolling_ranks(data)

    # Build data summary
    report_text = _build_report_text(rotation_table, rolling_ranks)
    logging.info("Data summary built. Calling LLM for analysis...")

    # Generate AI summary
    try:
        ai_summary = _generate_ai_summary(report_text)
    except Exception as e:
        logging.error(f"LLM call failed: {e}")
        ai_summary = "(AI summary unavailable — see raw data below)"

    # Deep dive into top sector holdings
    logging.info("Analyzing top sector holdings for breakout candidates...")
    top_sectors = _get_top_sector_etfs(rotation_table, n=3)
    holdings_analysis = _build_holdings_analysis(manager, top_sectors)

    holdings_text = _build_holdings_text(holdings_analysis)
    stock_picks_summary = None
    try:
        stock_picks_summary = _generate_stock_picks_summary(report_text, holdings_text)
    except Exception as e:
        logging.error(f"Stock picks LLM call failed: {e}")

    # Collect all discussed tickers for live charts in email
    all_discussed_tickers = []
    for data_item in holdings_analysis.values():
        for h in data_item.get("holdings", []):
            if h["ticker"] not in all_discussed_tickers:
                all_discussed_tickers.append(h["ticker"])
    logging.info(f"Embedding live charts for {len(all_discussed_tickers)} tickers")

    # Compose final email (HTML) with live StockCharts.com images
    date_str = datetime.now().strftime("%Y-%m-%d")
    subject = f"Weekly Sector Rotation Report - {date_str}"
    body = _build_html_email(ai_summary, rotation_table, rolling_ranks,
                             holdings_analysis, stock_picks_summary,
                             chart_tickers=all_discussed_tickers)

    if args.dry_run:
        print(body)
        logging.info("Dry run complete.")
        return

    # Send email
    try:
        _send_via_gmail_smtp(subject, body, args.to)
    except Exception as e:
        logging.error(f"Failed to send email: {e}")
        print(body)
        sys.exit(1)

    logging.info("Weekly sector rotation report complete.")


if __name__ == "__main__":
    main()
