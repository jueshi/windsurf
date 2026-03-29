# -*- coding: utf-8 -*-
"""Simple data loader without GUI dependencies.

Lightweight alternative to data_manager.py for non-GUI scripts.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Default data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "stock_data")


def ensure_data_dir():
    """Ensure the data directory exists."""
    os.makedirs(DATA_DIR, exist_ok=True)


def get_data_path(ticker):
    """Get the path for a ticker's CSV file."""
    ensure_data_dir()
    return os.path.join(DATA_DIR, f"{ticker}.csv")


def load_data(ticker):
    """Load data for a ticker from CSV file, with fallback to download."""
    csv_path = get_data_path(ticker)

    # Try to load from CSV first
    if os.path.exists(csv_path):
        try:
            # Try to read without skipping rows first (yfinance format)
            df = pd.read_csv(csv_path)

            # Check if this is a yfinance-format CSV (has Date index, Close column)
            if 'Date' in df.columns:
                df = df.set_index('Date')
            elif df.index.name != 'Date' and len(df.columns) > 0:
                # If first column looks like dates, use it as index
                first_col = df.iloc[:, 0]
                try:
                    pd.to_datetime(first_col)
                    df = df.set_index(df.columns[0])
                except:
                    pass

            # Handle old format with "Price" column
            if 'Price' in df.columns and 'Close' not in df.columns:
                df = df.rename(columns={'Price': 'Close'})

            df.index.name = None
            df.index = pd.to_datetime(df.index)

            # Keep only Close column
            if 'Close' in df.columns:
                df = df[['Close']]

            # Convert data to numeric, drop any rows with all NaNs
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna(how='all')

            # Only return if we have substantial data (at least 63 days for 3-month analysis)
            if not df.empty and len(df) >= 63:
                logging.info(f"Loaded {ticker} from CSV ({len(df)} rows)")
                return df
            logging.warning(f"{ticker} CSV has insufficient data ({len(df) if not df.empty else 0} rows)")
        except Exception as e:
            logging.warning(f"Error loading {ticker} from CSV: {e}")

    # Try to download fresh data as fallback
    logging.info(f"Downloading data for {ticker}...")
    try:
        # Download last 2 years of data (enough for full year of analysis + buffer)
        df = yf.download(ticker, period="2y", progress=False, ignore_tz=True)
        if df is not None and not df.empty:
            # Flatten multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Keep only Close column if multiple columns
            if 'Close' in df.columns:
                df = df[['Close']]

            # Save to CSV
            df.to_csv(csv_path)
            logging.info(f"Downloaded {ticker} ({len(df)} rows)")
            return df
    except Exception as e:
        logging.warning(f"Failed to download {ticker}: {e}")

    return None


def update_data(ticker):
    """Update data for a ticker (download latest data)."""
    logging.info(f"Updating {ticker}...")
    try:
        # Download last 2 years of data (enough for full year of analysis + buffer)
        df = yf.download(ticker, period="2y", progress=False, ignore_tz=True)
        if df is not None and not df.empty:
            csv_path = get_data_path(ticker)
            df.to_csv(csv_path)
            logging.info(f"Updated {ticker} ({len(df)} rows)")
            return True
    except Exception as e:
        logging.warning(f"Failed to update {ticker}: {e}")
    return False


if __name__ == "__main__":
    # Test
    ticker = "SPY"
    print(f"Loading {ticker}...")
    df = load_data(ticker)
    if df is not None:
        print(f"Loaded {len(df)} rows")
        print(df.tail())
