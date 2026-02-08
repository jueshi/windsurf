import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf


def _format_percentage(value: Optional[float | str]) -> str:
    """Return a friendly percentage string regardless of the raw format."""
    if value is None:
        return "N/A"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "N/A"
        if text.endswith("%"):
            return text
        try:
            numeric = float(text.replace(",", ""))
        except ValueError:
            return text
        if abs(numeric) <= 1:
            numeric *= 100
        return f"{numeric:.2f}%"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric) <= 1:
        numeric *= 100
    return f"{numeric:.2f}%"


def _percent_to_ratio(value: Optional[str]) -> Optional[float]:
    """Convert a formatted percent string into a 01 float."""
    if value is None:
        return None
    try:
        cleaned = str(value).replace("%", "").strip()
        if not cleaned:
            return None
        number = float(cleaned)
        if abs(number) > 1:
            number /= 100
        return number
    except (TypeError, ValueError):
        return None


def _extract_major_metrics(major: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Normalize yfinance's major_holders table into a dict."""
    if major is None or major.empty:
        return {}
    df = major.copy()
    if df.shape[1] >= 2:
        labels = df.iloc[:, 1].astype(str).str.strip()
        values = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        return dict(zip(labels, values))
    labels = df.index.astype(str).str.strip()
    values = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    return dict(zip(labels, values))


def _normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Lowercase/slugify holder columns while retaining the original names."""
    to_slug = {}
    for col in df.columns:
        slug = (
            str(col)
            .strip()
            .lower()
            .replace("%", "pct")
            .replace(" ", "_")
        )
        to_slug[col] = slug
    normalized = df.rename(columns=to_slug).copy()
    reverse = {v: k for k, v in to_slug.items()}
    return normalized, reverse


def _format_shares(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{numeric / 1_000_000:.2f}M"


@lru_cache()
def _get_fmp_api_key() -> Optional[str]:
    """Return the FMP API key from environment or the repo-level .env file."""
    env_value = os.getenv("FMP_API_KEY")
    if env_value:
        return env_value.strip()

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return None

    try:
        with env_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.split("=", 1)[0].strip() == "FMP_API_KEY":
                    return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def get_institutional_change(ticker: str, api_key: Optional[str]) -> List[Dict[str, object]]:
    """Fetch institutional position changes from FinancialModelingPrep."""
    if not api_key:
        return []
    url = f"https://financialmodelingprep.com/api/v3/institutional-holder/{ticker}?apikey={api_key}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        print(f"⚠️ Unable to fetch FMP institutional change data: {exc}")
        return []
    if not isinstance(payload, list):
        return []

    cleaned: List[Dict[str, object]] = []
    for entry in payload:
        holder_name = entry.get("holder") or entry.get("institution") or "Unknown"
        change_raw = entry.get("change")
        try:
            change_val = float(change_raw)
        except (TypeError, ValueError):
            continue
        cleaned.append(
            {
                "holder": holder_name,
                "change": change_val,
                "shares": entry.get("shares"),
                "reported_date": entry.get("reportDate") or entry.get("dateReported"),
            }
        )
    cleaned.sort(key=lambda item: abs(item["change"]), reverse=True)
    return cleaned


def get_smart_money_data(ticker_symbol: str):
    print(f"\n---  SMART MONEY ANALYSIS: {ticker_symbol.upper()} ---\n")
    stock = yf.Ticker(ticker_symbol)

    # SECTION 1  Ownership breakdown
    metrics = _extract_major_metrics(stock.major_holders)
    if not metrics:
        print(" Could not retrieve Major Holders data.")
    else:
        insider_raw = (
            metrics.get("% of Shares Held by All Insider")
            or metrics.get("% of Shares Held by All Insider Persons")
        )
        inst_raw = (
            metrics.get("% of Shares Held by Institutions")
            or metrics.get("% of Float Held by Institutions")
        )
        insider_pct = _format_percentage(insider_raw)
        inst_pct = _format_percentage(inst_raw)

        print(" OWNERSHIP STRUCTURE")
        print(f" Insider Ownership:       {insider_pct}")
        print(f" Institutional Ownership: {inst_pct}")

        inst_ratio = _percent_to_ratio(inst_pct)
        if inst_ratio is not None:
            if inst_ratio > 0.8:
                print(" Status: [CROWDED]  very high institutional consensus.")
            elif inst_ratio < 0.4:
                print(" Status: [RETAIL HEAVY]  institutions are scarce.")
    print("-" * 40)

    # SECTION 2  Top institutional holders
    inst_df = stock.institutional_holders
    if inst_df is None or inst_df.empty:
        print(" No institutional holder data found.")
    else:
        cleaned, _ = _normalize_columns(inst_df)
        holder_col = next((c for c in cleaned.columns if c in ("holder", "name")), None)
        shares_col = next((c for c in cleaned.columns if c.startswith("shares")), None)
        pct_col = next(
            (
                c
                for c in cleaned.columns
                if c
                in (
                    "pct_out",
                    "pct_of_shares_out",
                    "pct_outstanding",
                    "percent_out",
                )
            ),
            None,
        )
        date_col = next((c for c in cleaned.columns if "date" in c and "reported" in c), None)

        display = pd.DataFrame()
        if holder_col:
            display["Holder"] = cleaned[holder_col]
        if shares_col:
            display["Shares (M)"] = cleaned[shares_col].apply(_format_shares)
        if pct_col:
            display["% Out"] = cleaned[pct_col].apply(_format_percentage)
        if date_col:
            display["Date Reported"] = cleaned[date_col]

        print(" TOP INSTITUTIONAL HOLDERS (The Whales)")
        if display.empty:
            print(cleaned.head(10).to_string(index=False))
        else:
            print(display.head(10).to_string(index=False))
    print("-" * 40)

    # SECTION 2B ─ Institutional change feed (FMP)
    fmp_key = _get_fmp_api_key()
    changes = get_institutional_change(ticker_symbol, fmp_key)
    if not fmp_key:
        print("⚠️ FMP_API_KEY missing in environment/.env; skipping change feed.")
    elif not changes:
        print("⚠️ No institutional change data returned by FMP.")
    else:
        print("📈 Institutional Position Change (FMP)")
        for row in changes[:10]:
            arrow = "▲" if row["change"] > 0 else "▼"
            change_text = f"{row['change']:.2f}%"
            shares_text = _format_shares(row.get("shares"))
            date_text = row.get("reported_date") or ""
            print(f"{arrow} {row['holder']:<35} {change_text:>8}  {shares_text:>10}  {date_text}")
    print("-" * 40)

    # SECTION 3  Short-interest sentiment
    try:
        info = stock.info or {}
    except Exception:
        info = {}
    short_percent = info.get("shortPercentOfFloat")

    print(" SHORT SELLER SENTIMENT")
    if short_percent is None:
        print(" Short Interest data unavailable.")
    else:
        print(f" Short % of Float: {short_percent * 100:.2f}%")
        if short_percent > 0.10:
            print(" Squeeze Potential: HIGH (bears leaning hard).")
        elif short_percent < 0.02:
            print(" Squeeze Potential: LOW (few dedicated shorts).")
    print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    tickers = ["MRVL", "AVGO", "NVDA"]
    for symbol in tickers:
        get_smart_money_data(symbol)
