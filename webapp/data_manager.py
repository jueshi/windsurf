import os
import sys
import re
import json
import time
import logging
import math
import hashlib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple
from random import uniform
from functools import lru_cache

# Use absolute paths based on this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOCK_DATA_DIR = os.path.join(BASE_DIR, 'stock_data')
os.makedirs(STOCK_DATA_DIR, exist_ok=True)

# Define directory for plots
PLOTS_DIR = os.path.join(BASE_DIR, 'static', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


# Simple in-memory cache with TTL
class SimpleCache:
    """
    Simple in-memory cache with time-to-live (TTL) support.
    Used for caching API responses like fundamental data.
    """
    def __init__(self, default_ttl: int = 300):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self.default_ttl = default_ttl  # 5 minutes default
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                return value
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        ttl = ttl or self.default_ttl
        self._cache[key] = (value, time.time() + ttl)
    
    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
    
    def remove(self, key: str) -> None:
        """Remove specific key from cache."""
        self._cache.pop(key, None)


# Global cache instance
_cache = SimpleCache(default_ttl=300)  # 5 minute default TTL

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
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce', utc=True)
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
            try:
                # Normalize timezone handling
                if hasattr(last_date, "tzinfo") and last_date.tzinfo is not None:
                    now_dt = datetime.now(timezone.utc)
                else:
                    now_dt = datetime.now()
                if (now_dt.date() - last_date.date()).days > 0:
                    # We could implement incremental update here, but for simplicity let's re-download or fetch recent
                    # For this simplified version, let's just re-download recent if force_download is False
                    # But to keep it robust like the original, we might want to merge.
                    # For web speed, maybe just getting last 1y is enough?
                    # Let's stick to full history for consistency.
                    updated = False
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
                                combined['Date'] = pd.to_datetime(combined['Date'], utc=True)
                                combined = combined.drop_duplicates(subset='Date', keep='last')
                                combined = combined.sort_values('Date')

                                self._save_data_with_consistent_format(combined, data_path)
                                updated = True
                                return combined.set_index('Date')
                    except Exception as e:
                        logging.error(f"Incremental update failed for {ticker}: {e}")
                        # Fallback to full download
                        need_download = True
                    if not updated:
                        need_download = True
            except Exception as e:
                logging.error(f"Staleness check failed for {ticker}: {e}")
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

    def get_fundamental_data(self, ticker: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get fundamental data for a ticker with caching.
        Cache TTL is 5 minutes by default.
        """
        cache_key = f"fundamental_{ticker.upper()}"
        
        # Check cache first
        if use_cache:
            cached = _cache.get(cache_key)
            if cached is not None:
                logging.debug(f"Cache hit for {ticker} fundamental data")
                return cached
        
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.info
            
            # Cache the result
            if data:
                _cache.set(cache_key, data, ttl=300)  # 5 minutes
            
            return data
        except Exception as e:
            logging.error(f"Error fetching fundamental data for {ticker}: {e}")
            return None
    
    def get_chart_data_cached(self, ticker: str, timeframe: str = 'D') -> Optional[pd.DataFrame]:
        """
        Get chart data with short-term caching (1 minute).
        Useful for avoiding repeated API calls during page interactions.
        """
        cache_key = f"chart_{ticker.upper()}_{timeframe}"
        
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached
        
        data = self.load_data(ticker)
        if data is None or data.empty:
            data = self.update_data(ticker)
        
        if data is not None and not data.empty:
            _cache.set(cache_key, data, ttl=60)  # 1 minute cache
        
        return data
    
    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """Clear cache for a specific ticker or all cache."""
        if ticker:
            _cache.remove(f"fundamental_{ticker.upper()}")
            _cache.remove(f"chart_{ticker.upper()}_D")
            _cache.remove(f"chart_{ticker.upper()}_W")
            _cache.remove(f"chart_{ticker.upper()}_M")
        else:
            _cache.clear()


# Singleton instance
data_manager = StockDataManager()
