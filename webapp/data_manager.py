import os
import sys
import re
import json
import time
import logging
import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple
from random import uniform

# Define directory for stock data
STOCK_DATA_DIR = 'webapp/stock_data'
os.makedirs(STOCK_DATA_DIR, exist_ok=True)

# Define directory for plots (though we might generate them on the fly for web)
PLOTS_DIR = 'webapp/static/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

class StockDataManager:
    """
    A comprehensive manager for downloading, updating, and analyzing stock data for the web app.
    Refactored from original desktop app version.
    """

    def __init__(self, data_dir: str = STOCK_DATA_DIR, plot_save_path: str = PLOTS_DIR):
        self.data_dir = data_dir
        self.plot_save_path = plot_save_path
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.plot_save_path, exist_ok=True)
        self.start_date = None
        self.end_date = None
        self.last_request_time = 0
        self.min_request_interval = 1.0

    def _get_data_path(self, ticker: str) -> str:
        return os.path.join(self.data_dir, f'{ticker.upper()}_stock_data.tsv')

    def _save_data_with_consistent_format(self, data, file_path):
        if data is None or data.empty:
            logging.warning(f"Cannot save empty or None data to {file_path}")
            return

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data_to_save = data.copy()

        numeric_cols = data_to_save.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        for col in numeric_cols:
            data_to_save[col] = data_to_save[col].round(2)

        data_to_save.to_csv(file_path, sep='\t', index=False, float_format='%.2f')

    def load_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load stock data from a file."""
        try:
            data_path = self._get_data_path(ticker)
            if not os.path.exists(data_path):
                return None

            # Read data
            data = pd.read_csv(data_path, sep='\t')

            # Clean and normalize
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                data = data.dropna(subset=['Date'])
                data = data.sort_values('Date')

                # Ensure numeric
                numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                for col in numeric_cols:
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce')

                data = data.set_index('Date')
                return data
            return None

        except Exception as e:
            logging.error(f"Error loading data for {ticker}: {e}")
            return None

    def _download_with_retry(self, ticker: str, max_retries: int = 3, force_download: bool = False) -> Optional[pd.DataFrame]:
        for attempt in range(max_retries):
            try:
                # Rate limiting
                current_time = time.time()
                if current_time - self.last_request_time < self.min_request_interval:
                    time.sleep(self.min_request_interval)

                logging.info(f"Downloading {ticker}, attempt {attempt+1}")

                # Use Ticker object
                ticker_obj = yf.Ticker(ticker)

                if force_download:
                    stock_data = ticker_obj.history(period="max")
                else:
                    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
                    stock_data = ticker_obj.history(start=start_date)

                if not stock_data.empty:
                    self.last_request_time = time.time()
                    return stock_data

            except Exception as e:
                logging.warning(f"Download failed for {ticker}: {e}")
                time.sleep(uniform(1.0, 3.0))

        return None

    def update_data(self, ticker: str, force_download: bool = False) -> Optional[pd.DataFrame]:
        ticker = ticker.upper()
        data_path = self._get_data_path(ticker)
        existing_data = self.load_data(ticker)

        # Determine if we need to download
        need_download = force_download
        if existing_data is None or existing_data.empty:
            need_download = True
        elif not force_download:
            # Check if data is stale (older than 1 day)
            last_date = existing_data.index.max()
            if (datetime.now() - last_date).days > 1:
                # We could implement incremental update here, but for simplicity let's re-download or fetch recent
                # For this simplified version, let's just re-download recent if force_download is False
                # But to keep it robust like the original, we might want to merge.
                # For web speed, maybe just getting last 1y is enough?
                # Let's stick to full history for consistency.
                pass
                # Actually, let's try to just append new data if possible, or re-download if gap is large.
                # Simplification: If data exists, fetch only recent.
                try:
                    start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                    if start_date < datetime.now().strftime('%Y-%m-%d'):
                        recent_data = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
                        if not recent_data.empty:
                            # Standardize columns
                            if isinstance(recent_data.columns, pd.MultiIndex):
                                recent_data.columns = recent_data.columns.get_level_values(0)

                            # Combine
                            existing_data = existing_data.reset_index()
                            recent_data = recent_data.reset_index()

                            # Align columns
                            cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                            combined = pd.concat([existing_data, recent_data], ignore_index=True)

                            # Deduplicate
                            combined['Date'] = pd.to_datetime(combined['Date'])
                            combined = combined.drop_duplicates(subset='Date', keep='last')
                            combined = combined.sort_values('Date')

                            self._save_data_with_consistent_format(combined, data_path)
                            return combined.set_index('Date')
                except Exception as e:
                    logging.error(f"Incremental update failed for {ticker}: {e}")
                    # Fallback to full download
                    need_download = True

        if need_download:
            stock_data = self._download_with_retry(ticker, force_download=True)
            if stock_data is not None:
                stock_data = stock_data.reset_index()
                # Ensure standard columns
                if 'Date' in stock_data.columns:
                    # Keep only relevant columns
                    keep_cols = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] if c in stock_data.columns]
                    stock_data = stock_data[keep_cols]
                    self._save_data_with_consistent_format(stock_data, data_path)
                    return stock_data.set_index('Date')

        return existing_data

    def get_fundamental_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        try:
            ticker_obj = yf.Ticker(ticker)
            return ticker_obj.info
        except Exception as e:
            logging.error(f"Error fetching fundamental data for {ticker}: {e}")
            return None

# Singleton instance
data_manager = StockDataManager()
