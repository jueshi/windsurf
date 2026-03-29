# -*- coding: utf-8 -*-
"""Generate and send relative strength study email for top 20 big cap stocks.

Features:
- Ranks stocks by relative strength vs SPY
- Shows rank changes over last week and last month
- Embeds daily/weekly/monthly charts for each stock
- Orders stocks by market cap

Usage:
    python relative_strength_email.py
    python relative_strength_email.py --to other@email.com
    python relative_strength_email.py --dry-run
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

# Ensure project directory is on the path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from simple_data_loader import update_data
from relative_strength import (
    load_stock_data, build_relative_strength_table, build_rolling_ranks, BENCHMARK
)
from ticker_lists import mega_tickers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DEFAULT_RECIPIENT = "jueshi@gmail.com"
BIG_CAP_TICKERS = mega_tickers[:20]  # Top 20 mega cap stocks


def _update_stock_data(tickers):
    """Download/refresh data for all stocks and benchmark."""
    all_tickers = list(tickers) + [BENCHMARK]
    for ticker in all_tickers:
        try:
            update_data(ticker)
        except Exception as e:
            logging.warning(f"Failed to update {ticker}: {e}")


def _build_rank_changes_table(rolling_ranks, lookback_days, label):
    """Build rank change rows for a specific lookback period."""
    if rolling_ranks.empty or len(rolling_ranks) < lookback_days:
        return []

    recent = rolling_ranks.iloc[-lookback_days:]
    changes = []

    for ticker in rolling_ranks.columns:
        start_rank = int(recent[ticker].iloc[0])
        end_rank = int(recent[ticker].iloc[-1])
        delta = start_rank - end_rank
        changes.append({
            "Ticker": ticker,
            "From": start_rank,
            "To": end_rank,
            "Change": delta,
        })

    # Sort by biggest improvement first
    changes.sort(key=lambda x: -x["Change"])
    return changes


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


def _build_live_charts_html(tickers):
    """Build HTML with live StockCharts.com chart images embedded directly.

    Each ticker gets daily (1yr), weekly (5yr), and monthly (max) charts in a row.
    Translates ticker symbols to StockCharts.com format (e.g., BRK-A -> BRK/A).
    """
    if not tickers:
        return ""

    timeframes = [
        ("D", "Daily (1yr)", 1, 0, 0, "t3784817951c"),
        ("W", "Weekly (5yr)", 5, 0, 0, "t1918602869c"),
        ("M", "Monthly (Max)", 99, 0, 0, "t1918602869c"),
    ]

    sections = []
    sections.append('<h2 style="color:#1a237e;font-size:16px;margin-top:30px;">Technical Charts (Live)</h2>')

    for ticker in tickers:
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


def _build_html_email(rs_table, rolling_ranks, chart_html):
    """Build HTML email with relative strength rankings and charts."""
    date_str = datetime.now().strftime("%Y-%m-%d %A")

    # Helper for value cells with color coding
    def _val_cell(val, fmt="+.2f"):
        if val != val:  # NaN check (NaN != NaN is True)
            return '<td style="padding:6px 10px;text-align:right;color:#999;">N/A</td>'
        color = "#2e7d32" if val > 0 else "#d32f2f" if val < 0 else "#666"
        return f'<td style="padding:6px 10px;text-align:right;color:{color};font-weight:600;">{val:{fmt}}%</td>'

    # Build rankings table
    rank_rows = ""
    for _, row in rs_table.iterrows():
        bg = "#f8f9fa" if int(row["Rank"]) % 2 == 0 else "#ffffff"
        rank_rows += f"""<tr style="background:{bg};">
            <td style="padding:6px 10px;text-align:center;">{int(row['Rank'])}</td>
            <td style="padding:6px 10px;font-weight:600;">{row['Ticker']}</td>
            {_val_cell(row.get('1W', 0))}
            {_val_cell(row.get('2W', 0))}
            {_val_cell(row.get('1M', 0))}
            {_val_cell(row.get('3M', 0))}
            {_val_cell(row.get('Composite', 0), '+.1f')}
        </tr>"""

    # Build rank change tables
    th_style = 'style="padding:8px 10px;text-align:right;border-bottom:2px solid #dee2e6;background:#e9ecef;"'
    th_left = 'style="padding:8px 10px;text-align:left;border-bottom:2px solid #dee2e6;background:#e9ecef;"'
    th_center = 'style="padding:8px 10px;text-align:center;border-bottom:2px solid #dee2e6;background:#e9ecef;"'

    def _build_change_rows(changes):
        rows = ""
        for change in changes:
            delta = change["Change"]
            if delta > 0:
                arrow, color = "&#9650;", "#2e7d32"
            elif delta < 0:
                arrow, color = "&#9660;", "#d32f2f"
            else:
                arrow, color = "&#9644;", "#999"

            rows += (
                f'<tr><td style="padding:4px 10px;font-weight:600;">{change["Ticker"]}</td>'
                f'<td style="padding:4px 10px;text-align:center;">{change["From"]}</td>'
                f'<td style="padding:4px 10px;text-align:center;">{change["To"]}</td>'
                f'<td style="padding:4px 10px;text-align:center;color:{color};font-weight:700;">'
                f'{arrow} {abs(delta)}</td></tr>'
            )
        return rows

    changes_1w = _build_rank_changes_table(rolling_ranks, 5, "1-week")
    changes_1m = _build_rank_changes_table(rolling_ranks, 21, "1-month")

    change_rows_1w = _build_change_rows(changes_1w) if changes_1w else '<tr><td colspan="4" style="padding:8px 10px;text-align:center;color:#999;">Insufficient data</td></tr>'
    change_rows_1m = _build_change_rows(changes_1m) if changes_1m else '<tr><td colspan="4" style="padding:8px 10px;text-align:center;color:#999;">Insufficient data</td></tr>'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="font-family:Segoe UI,Arial,sans-serif;color:#333;max-width:1000px;margin:0 auto;padding:20px;">

<h1 style="color:#1a237e;border-bottom:3px solid #1a237e;padding-bottom:8px;font-size:22px;">
    Relative Strength Study: Top 20 Big Caps vs SPY</h1>
<p style="color:#666;margin-top:-5px;">{date_str}</p>

<h2 style="color:#1a237e;font-size:16px;margin-top:25px;">Relative Strength Rankings</h2>
<table style="border-collapse:collapse;width:100%;font-size:13px;margin-bottom:20px;">
<thead><tr>
    <th {th_center}>Rank</th><th {th_left}>Ticker</th>
    <th {th_style}>1W</th><th {th_style}>2W</th><th {th_style}>1M</th><th {th_style}>3M</th>
    <th {th_style}>Score</th>
</tr></thead>
<tbody>{rank_rows}</tbody>
</table>

<h2 style="color:#1a237e;font-size:16px;">Rank Changes Since Last Week (5-day)</h2>
<table style="border-collapse:collapse;width:100%;font-size:13px;margin-bottom:20px;">
<thead><tr>
    <th {th_left}>Ticker</th>
    <th {th_center}>From</th><th {th_center}>To</th><th {th_center}>Change</th>
</tr></thead>
<tbody>{change_rows_1w}</tbody>
</table>

<h2 style="color:#1a237e;font-size:16px;">Rank Changes Since Last Month (21-day)</h2>
<table style="border-collapse:collapse;width:100%;font-size:13px;margin-bottom:20px;">
<thead><tr>
    <th {th_left}>Ticker</th>
    <th {th_center}>From</th><th {th_center}>To</th><th {th_center}>Change</th>
</tr></thead>
<tbody>{change_rows_1m}</tbody>
</table>

<h2 style="color:#1a237e;font-size:16px;">Stock Charts (Daily/Weekly/Monthly)</h2>
{chart_html if chart_html else '<p style="color:#999;">Unable to generate charts</p>'}

<hr style="border:none;border-top:1px solid #dee2e6;margin:20px 0;">
<p style="color:#999;font-size:11px;">Generated by Personal AI Stock Assistant</p>
</body></html>"""

    return html


def _send_email(subject, html_body, recipient):
    """Send email via Gmail SMTP with app password.

    Requires GMAIL_APP_PASSWORD in .env (16-char app password from Google Account).
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from dotenv import load_dotenv

    try:
        # Load .env from script directory
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        load_dotenv(env_path)
        app_password = os.getenv("GMAIL_APP_PASSWORD")
        sender = os.getenv("GMAIL_SENDER", recipient)

        if not app_password:
            logging.error(
                "GMAIL_APP_PASSWORD not set in .env.\n"
                "To create one: Google Account -> Security -> 2-Step Verification -> App passwords\n"
                "Then add to .env: GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx"
            )
            return False

        msg = MIMEMultipart("alternative")
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText("See HTML version of this email.", "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, app_password)
            server.send_message(msg)

        logging.info(f"Email sent to {recipient}")
        return True

    except Exception as e:
        logging.error(f"Failed to send email: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Relative Strength Email Report")
    parser.add_argument("--to", default=DEFAULT_RECIPIENT, help="Recipient email address")
    parser.add_argument("--dry-run", action="store_true", help="Print report without sending email")
    args = parser.parse_args()

    logging.info("Starting relative strength analysis...")

    # Update stock data
    logging.info(f"Updating data for {len(BIG_CAP_TICKERS)} stocks...")
    _update_stock_data(BIG_CAP_TICKERS)

    # Load data
    logging.info("Loading stock data...")
    all_data = load_stock_data(BIG_CAP_TICKERS + [BENCHMARK])
    stock_data = {t: all_data[t] for t in BIG_CAP_TICKERS if t in all_data}
    benchmark_data = all_data.get(BENCHMARK)

    if not stock_data or benchmark_data is None:
        logging.error("Failed to load sufficient data")
        sys.exit(1)

    # Build rankings
    logging.info("Building relative strength rankings...")
    rs_table = build_relative_strength_table(stock_data, benchmark_data)
    rolling_ranks = build_rolling_ranks(stock_data, benchmark_data)

    if rs_table.empty:
        logging.error("No relative strength data available")
        sys.exit(1)

    # Generate charts using StockCharts.com live images
    logging.info(f"Generating charts for {len(rs_table)} stocks...")
    chart_tickers = list(rs_table["Ticker"].values) if "Ticker" in rs_table.columns else list(rs_table.index)
    chart_html = _build_live_charts_html(chart_tickers)

    # Build email
    date_str = datetime.now().strftime("%Y-%m-%d")
    subject = f"Relative Strength Study: Top 20 Big Caps - {date_str}"
    html_body = _build_html_email(rs_table, rolling_ranks, chart_html)

    if args.dry_run:
        print(html_body)
        return

    # Send email
    if _send_email(subject, html_body, args.to):
        logging.info("Report completed successfully")
    else:
        logging.error("Failed to send email")
        sys.exit(1)


if __name__ == "__main__":
    main()
