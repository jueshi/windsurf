import os
import sys
import re
import json
import time
import logging
import importlib
import inspect
import threading
import webbrowser
from queue import Queue
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from tkcalendar import DateEntry
import time
import pytz
import yfinance as yf
from typing import Optional, List, Any
from random import uniform
import math

# Directories and Configurations
STOCK_DATA_DIR = 'stock_data'
os.makedirs(STOCK_DATA_DIR, exist_ok=True)

class StockDataManager:
    """
    A comprehensive manager for downloading, updating, and analyzing stock data.

    Attributes:
        data_dir (str): Directory to store stock data files
        plot_save_path (str): Path to save plots
        start_date (str): Start date for data retrieval
        end_date (str): End date for data retrieval
    """

    def __init__(self, data_dir: str = STOCK_DATA_DIR, plot_save_path: str = STOCK_DATA_DIR):
        """
        Initialize the StockDataManager.

        Args:
            data_dir (str, optional): Directory to store stock data. Defaults to STOCK_DATA_DIR.
            plot_save_path (str, optional): Path to save plots. Defaults to STOCK_DATA_DIR.
        """
        self.data_dir = data_dir
        self.plot_save_path = plot_save_path
        os.makedirs(self.data_dir, exist_ok=True)
        self.start_date = None
        self.end_date = None
        self.last_request_time = 0
        self.min_request_interval = 2.0  # Minimum time between requests in seconds
        
    def _save_data_with_consistent_format(self, data, file_path):
        """
        Save data to file with consistent formatting to avoid spacing issues.
        
        Args:
            data (pd.DataFrame): Data to save
            file_path (str): Path to save the data to
        """
        # Check if data is None or empty
        if data is None or data.empty:
            logging.warning(f"Cannot save empty or None data to {file_path}")
            return
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Make a copy to avoid modifying the original data
        data_to_save = data.copy()
        
        # Round numeric columns to 2 decimal places
        numeric_cols = data_to_save.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        for col in numeric_cols:
            data_to_save[col] = data_to_save[col].round(2)
        
        # Save with consistent float format
        data_to_save.to_csv(file_path, sep='\t', index=False, float_format='%.2f')

    def _get_data_path(self, ticker: str) -> str:
        """
        Generate the full path for a ticker's data file.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            str: Full path to the ticker's data file
        """
        return os.path.join(self.data_dir, f'{ticker.upper()}_stock_data.tsv')

    def _load_stock_data(self, ticker: str) -> pd.DataFrame:
        """
        Load stock data from a file.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            pd.DataFrame: Loaded stock data
        """
        try:
            # Read data
            data_path = self._get_data_path(ticker)
            stock_data = pd.read_csv(data_path, sep='\t', header=None)

            # Handle empty or insufficient data
            if len(stock_data) <= 3:
                logging.warning(f"Insufficient data for {ticker}")
                return pd.DataFrame()

            # Remove header rows
            stock_data = stock_data.iloc[3:]

            # Reset index
            stock_data.reset_index(drop=True, inplace=True)

            # Dynamically create column names
            default_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

            # Use actual columns if they match the expected number
            if len(stock_data.columns) == len(default_columns):
                stock_data.columns = default_columns
            else:
                # Fallback to using available columns
                stock_data.columns = default_columns[:len(stock_data.columns)]

            # Convert Date column to datetime with UTC=True to handle mixed timezones
            stock_data['Date'] = pd.to_datetime(stock_data['Date'], utc=True)

            # Convert numeric columns to float
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for col in numeric_columns:
                if col in stock_data.columns:
                    stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')

            return stock_data

        except Exception as e:
            logging.error(f"Error loading data for {ticker}: {e}")
            return pd.DataFrame()

    def _download_with_retry(self, ticker: str, max_retries: int = 3, force_download: bool = False) -> Optional[pd.DataFrame]:
        """
        Download stock data with retry logic and rate limiting.

        Args:
            ticker (str): Stock ticker symbol
            max_retries (int): Maximum number of retry attempts
            force_download (bool): If True, download all available history

        Returns:
            Optional[pd.DataFrame]: Downloaded stock data or None if all retries fail
        """
        for attempt in range(max_retries):
            try:
                # Ensure minimum time between requests - more aggressive rate limiting
                current_time = time.time()
                time_since_last_request = current_time - self.last_request_time
                min_interval = self.min_request_interval * (attempt + 1)  # Increase interval with each retry
                if time_since_last_request < min_interval:
                    sleep_time = min_interval - time_since_last_request
                    logging.info(f"Rate limiting: Waiting {sleep_time:.1f}s before request for {ticker}")
                    time.sleep(sleep_time)

                # Add more random jitter to avoid synchronized requests and detection
                jitter = uniform(0.5, 2.0)
                logging.info(f"Adding jitter delay of {jitter:.1f}s for {ticker}")
                time.sleep(jitter)

                # Try with a user agent to avoid being blocked
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }

                # First attempt with direct yfinance download
                try:
                    if force_download:
                        # If force_download is True, get all available history
                        logging.info(f"Requesting ALL available history for {ticker} using period='max'")
                        stock_data = yf.download(
                            ticker,
                            period="max",  # Get all available history
                            progress=False,
                            auto_adjust=True  # Explicitly set to avoid FutureWarning
                        )
                    else:
                        # Calculate start date to be 5 years ago
                        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')  # Request 5 years of data
                        end_date = datetime.now().strftime('%Y-%m-%d')

                        logging.info(f"Requesting data for {ticker} from {start_date} to {end_date}")
                        stock_data = yf.download(
                            ticker,
                            start=start_date,
                            end=end_date,
                            progress=False,
                            auto_adjust=True  # Explicitly set to avoid FutureWarning
                        )

                    # Verify we got valid data
                    if not stock_data.empty and len(stock_data) > 0:
                        self.last_request_time = time.time()
                        logging.info(f"Successfully downloaded data for {ticker} using yfinance direct method")
                        return stock_data

                except json.JSONDecodeError as json_err:
                    logging.warning(f"JSONDecodeError with direct method for {ticker}: {json_err}")
                    # Fall through to alternative method
                except Exception as direct_err:
                    logging.warning(f"Error with direct method for {ticker}: {direct_err}")
                    # Fall through to alternative method

                # Alternative method: Use Ticker object
                try:
                    logging.info(f"Trying alternative method for {ticker} using Ticker object")
                    ticker_obj = yf.Ticker(ticker)

                    if force_download:
                        # If force_download is True, get all available history
                        logging.info(f"Alternative method: Requesting ALL available history for {ticker} using period='max'")
                        stock_data = ticker_obj.history(period="max")
                    else:
                        # Request 5 years of data using the start and end parameters
                        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
                        end_date = datetime.now().strftime('%Y-%m-%d')
                        logging.info(f"Alternative method: Requesting data for {ticker} from {start_date} to {end_date}")
                        stock_data = ticker_obj.history(start=start_date, end=end_date)

                    if not stock_data.empty and len(stock_data) > 0:
                        self.last_request_time = time.time()
                        logging.info(f"Successfully downloaded data for {ticker} using Ticker object method")
                        return stock_data

                except Exception as ticker_err:
                    logging.warning(f"Error with Ticker object method for {ticker}: {ticker_err}")
                    # Continue to next retry attempt

                self.last_request_time = time.time()

            except Exception as e:
                wait_time = (2 ** (attempt + 2)) + uniform(0, 2)  # More aggressive exponential backoff with jitter
                logging.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}. Waiting {wait_time:.1f}s before retry.")
                time.sleep(wait_time)

        logging.error(f"All download attempts failed for {ticker}")
        return None

    def initial_download(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Download all available historical stock data for a given ticker.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Optional[pd.DataFrame]: Downloaded stock data or None if download fails
        """
        try:
            # Validate and prepare parameters
            ticker = ticker.upper()

            logging.info(f"Attempting to download data for {ticker}")

            # Download with retry logic
            stock_data = self._download_with_retry(ticker)

            if stock_data is None:
                logging.warning(f"No data downloaded for {ticker} after all retry attempts")
                return None

            logging.info(f"Download completed for {ticker}. Data shape: {stock_data.shape}")

            # Log first and last dates for verification
            if not stock_data.empty:
                first_date = stock_data.index.min()
                last_date = stock_data.index.max()
                logging.info(f"{ticker} data range: {first_date} to {last_date}")

            # Prepare data file path
            data_path = self._get_data_path(ticker)

            # Reset index to make Date a column
            stock_data_reset = stock_data.reset_index()

            # Dynamically create output columns
            columns_order = ['Date', 'Open', 'High', 'Low', 'Close']

            # Add Volume if it exists
            if 'Volume' in stock_data_reset.columns:
                columns_order.append('Volume')

            # Add Adj Close if it exists
            if 'Adj Close' in stock_data_reset.columns:
                columns_order.append('Adj Close')

            # Select columns that exist in the dataframe
            output_data = stock_data_reset[[col for col in columns_order if col in stock_data_reset.columns]]

            # Save data to local file with consistent formatting
            self._save_data_with_consistent_format(output_data, data_path)
            logging.info(f"All available historical data for {ticker} saved to {data_path}")

            return stock_data

        except Exception as e:
            logging.error(f"Error in initial download for {ticker}: {e}")

    def update_data(self, ticker: str, force_download: bool = False) -> Optional[pd.DataFrame]:
        """
        Update existing stock data with the most recent information.
        Ensures at least 3 years of historical data if possible.
        Downloads all missing data since the last update, not just the latest data.

        Args:
            ticker (str): Stock ticker symbol
            force_download (bool, optional): Force download of new data instead of updating existing. Defaults to False.

        Returns:
            Optional[pd.DataFrame]: Updated stock data or None if update fails
        """
        try:
            # Validate and prepare parameters
            ticker = ticker.upper()
            data_path = self._get_data_path(ticker)

            # Initialize existing_data as None to avoid scope issues
            existing_data = None

            # Check if we should use existing data
            if os.path.exists(data_path) and not force_download:
                # Load existing data using the more robust load_data method
                existing_data = self.load_data(ticker)

                if existing_data is None or existing_data.empty:
                    # handle case where file exists but is empty/invalid
                    force_download = True
                else:
                    existing_data = existing_data.reset_index()

                # Convert Date column to datetime with UTC=True
                existing_data['Date'] = pd.to_datetime(existing_data['Date'], utc=True)

                # Get the latest date in the data
                latest_local_date = existing_data['Date'].max()

                # Check if we have at least 3 years of data
                earliest_date = existing_data['Date'].min()
                latest_date = existing_data['Date'].max()
                data_span = latest_date - earliest_date

                if data_span.days >= 365 * 3:
                    logging.info(f"{ticker} has {data_span.days / 365:.1f} years of data, which meets the 3-year minimum requirement")

                    # Check if data is already up to date (within 1 day)
                    # But don't skip if force_download is explicitly requested
                    days_since_update = (datetime.now(timezone.utc).date() - latest_date.date()).days
                    if days_since_update <= 1 and not force_download:
                        logging.info(f"Data for {ticker} is up to date (last update: {latest_date.date()})")
                        return existing_data
                    elif days_since_update <= 1:
                        logging.info(f"Data for {ticker} appears up to date but force_download requested, will fetch latest")
                else:
                    logging.info(f"{ticker} only has {data_span.days / 365:.1f} years of data, which is less than the 3-year minimum. Forcing full download.")
                    force_download = True

            # If force_download is True or we don't have enough historical data, do a full download
            if force_download:
                logging.info(f"Downloading full historical data for {ticker}")
                stock_data = self._download_with_retry(ticker, force_download=force_download)

                if stock_data is not None and not stock_data.empty:
                    # Reset index to make Date a column
                    stock_data = stock_data.reset_index()

                    # Check if we have at least 3 years of data after download
                    if 'Date' in stock_data.columns:
                        earliest_date = pd.to_datetime(stock_data['Date'], utc=True).min()
                        latest_date = pd.to_datetime(stock_data['Date'], utc=True).max()
                        data_span = latest_date - earliest_date
                        logging.info(f"{ticker} downloaded data spans {data_span.days / 365:.1f} years")

                    # Save to file with consistent formatting
                    self._save_data_with_consistent_format(stock_data, data_path)
                    logging.info(f"Saved full historical data for {ticker} to {data_path}")
                    return stock_data
                else:
                    logging.warning(f"No data downloaded for {ticker}")
                    return None

            # If we get here, we need to update existing data with new data
            # Check if existing_data is available
            if existing_data is None:
                logging.info(f"No existing data found for {ticker}. Switching to force download mode.")
                # Recursively call update_data with force_download=True
                return self.update_data(ticker, force_download=True)

            # Check for gaps in the data
            logging.info(f"Checking for gaps in {ticker} data")
            existing_data['Date'] = pd.to_datetime(existing_data['Date'])
            existing_data = existing_data.sort_values('Date')

            # Get the latest date in local data
            latest_local_date = existing_data['Date'].max()

            # Get current date
            current_date = (datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

            # Initialize container for new data frames
            all_new_data_frames = []

            # As per user request, focus only on the most recent data
            # We'll skip historical gap filling and just get the latest data
            logging.info(f"Focusing only on the most recent data for {ticker} as requested")

            # Get data from the day after the latest date to today
            start_date = (latest_local_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

            # Ensure start_date is before current_date
            start_dt = pd.to_datetime(start_date)
            current_dt = pd.to_datetime(current_date)

            if start_dt >= current_dt:
                logging.warning(f"Latest date in {ticker} data ({start_date}) is already current or in the future. No update needed.")
            else:
                logging.info(f"Downloading recent data for {ticker} from {start_date} to {current_date}")

                # Try to download the most recent data
                try:
                    recent_data = yf.download(ticker, start=start_date, end=current_date, progress=False, auto_adjust=True)

                    if not recent_data.empty:
                        logging.info(f"Downloaded {len(recent_data)} rows of recent data for {ticker}")
                        recent_data_reset = recent_data.reset_index()
                        all_new_data_frames.append(recent_data_reset)
                    else:
                        logging.warning(f"No recent data available for {ticker} from {start_date} to {current_date}")
                except Exception as e:
                    logging.error(f"Error downloading recent data for {ticker}: {e}")

                    # If first attempt fails, try just today's data
                    try:
                        logging.info(f"Trying to get just today's data for {ticker}")
                        today_data = yf.download(ticker, start=current_dt.strftime('%Y-%m-%d'), end=None, progress=False, auto_adjust=True)

                        if not today_data.empty:
                            logging.info(f"Downloaded today's data for {ticker}")
                            today_data_reset = today_data.reset_index()
                            all_new_data_frames.append(today_data_reset)
                        else:
                            logging.warning(f"No data available for today for {ticker}")
                    except Exception as e2:
                        logging.error(f"Error downloading today's data for {ticker}: {e2}")

            # If we have no new data at all, return existing data
            if not all_new_data_frames:
                logging.info(f"No new data available for {ticker}")
                return existing_data

            # Process and combine all new data frames
            all_new_data = pd.concat(all_new_data_frames, ignore_index=True) if all_new_data_frames else None

            if all_new_data is None or all_new_data.empty:
                logging.info(f"No new data available for {ticker}")
                return existing_data

            # Ensure consistent column order and names
            columns_order = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

            # Process column names
            new_data_processed = all_new_data.copy()
            
            # Handle tuple column names (multi-index columns) from yfinance
            if isinstance(new_data_processed.columns[0], tuple):
                logging.info("Detected multi-index columns in new data, flattening to standard format")
                # Create a mapping from tuple columns to standard column names
                column_mapping = {}
                standard_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                
                for col in new_data_processed.columns:
                    # For tuple columns like ('Open', 'NVDA'), map to 'Open'
                    if col[0] in standard_columns:
                        column_mapping[col] = col[0]
                    # Special case for Date which might be formatted differently
                    elif 'date' in col[0].lower():
                        column_mapping[col] = 'Date'
                    # Special case for Adj Close which might be formatted differently
                    elif 'adj' in col[0].lower() and 'close' in col[0].lower():
                        column_mapping[col] = 'Adj Close'
                
                # Rename the columns
                new_data_processed.columns = [column_mapping.get(col, col) for col in new_data_processed.columns]
                logging.info(f"Flattened columns: {new_data_processed.columns.tolist()}")

            # Ensure Date is in datetime format for proper comparison later
            if 'Date' in new_data_processed.columns:
                new_data_processed['Date'] = pd.to_datetime(new_data_processed['Date'])

            # Rename columns if needed
            if 'Adj Close' not in new_data_processed.columns and 'Adj_Close' in new_data_processed.columns:
                new_data_processed.rename(columns={'Adj_Close': 'Adj Close'}, inplace=True)

            # Select and order columns
            # First ensure all standard columns exist in new_data_processed
            for col in columns_order:
                if col not in new_data_processed.columns:
                    logging.warning(f"Column {col} not found in new data, adding it with NaN values")
                    new_data_processed[col] = np.nan
            
            # Now select and order the columns
            new_data_processed = new_data_processed[columns_order]
            
            # Log the final column structure
            logging.info(f"Final new_data_processed columns: {new_data_processed.columns.tolist()}")
            logging.info(f"Sample of processed new data:\n{new_data_processed.head(3)}")
            logging.info(f"Debug - After processing, new_data shape: {new_data_processed.shape}")

            # Ensure consistent numeric formatting in both datasets before concatenation
            # Apply rounding to existing data
            if not existing_data.empty:
                numeric_cols = existing_data.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
                for col in numeric_cols:
                    existing_data[col] = existing_data[col].round(2)
            
            # Apply rounding to new data
            if not new_data_processed.empty:
                numeric_cols = new_data_processed.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
                for col in numeric_cols:
                    new_data_processed[col] = new_data_processed[col].round(2)
            
            # Ensure column consistency between existing_data and new_data_processed
            logging.info(f"Existing data columns: {existing_data.columns.tolist() if not existing_data.empty else 'Empty'}") 
            logging.info(f"New data columns: {new_data_processed.columns.tolist() if not new_data_processed.empty else 'Empty'}") 
            
            # Remove any ticker column from new_data_processed if it exists
            if 'Ticker' in new_data_processed.columns:
                new_data_processed = new_data_processed.drop(columns=['Ticker'])
            
            # Handle column alignment based on whether existing_data is empty or not
            if existing_data.empty:
                # If existing_data is empty, ensure new_data_processed has the standard columns
                standard_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                
                # Add any missing standard columns
                for col in standard_columns:
                    if col not in new_data_processed.columns:
                        new_data_processed[col] = float('nan')
                
                # Keep only the standard columns and in the right order
                new_data_processed = new_data_processed[[col for col in standard_columns if col in new_data_processed.columns]]
                
            elif not new_data_processed.empty:
                # If both dataframes have data, align new_data_processed to existing_data
                for col in existing_data.columns:
                    if col not in new_data_processed.columns:
                        # Add missing column with NaN values
                        new_data_processed[col] = float('nan')
                
                # Reorder columns to match existing_data
                new_data_processed = new_data_processed[existing_data.columns]
                
                # Ensure both dataframes have Date in the same format before concatenation
                if 'Date' in existing_data.columns:
                    # First convert to datetime
                    existing_data['Date'] = pd.to_datetime(existing_data['Date'])
                    # Then format to yyyy-mm-dd string format if it's not already
                    if not isinstance(existing_data['Date'].iloc[0], str) or len(existing_data['Date'].iloc[0]) > 10:
                        existing_data['Date'] = existing_data['Date'].dt.strftime('%Y-%m-%d')
                
                if 'Date' in new_data_processed.columns:
                    # First convert to datetime
                    new_data_processed['Date'] = pd.to_datetime(new_data_processed['Date'])
                    # Then format to yyyy-mm-dd string format
                    new_data_processed['Date'] = new_data_processed['Date'].dt.strftime('%Y-%m-%d')
                
                # Debug the column names and data types before concatenation
                logging.info(f"Existing data columns: {existing_data.columns.tolist()}")
                logging.info(f"New data columns: {new_data_processed.columns.tolist()}")
                
                # Print sample data from both dataframes for debugging
                logging.info(f"Existing data sample:\n{existing_data.head(2)}")
                logging.info(f"New data sample:\n{new_data_processed.head(2)}")
                
                # Ensure both dataframes have the same column types
                for col in existing_data.columns:
                    if col in new_data_processed.columns:
                        # Try to convert to the same dtype if possible
                        try:
                            if col != 'Date':  # Skip Date as we handle it separately
                                # First convert any string numeric values to float
                                if new_data_processed[col].dtype == 'object' and existing_data[col].dtype != 'object':
                                    new_data_processed[col] = pd.to_numeric(new_data_processed[col], errors='coerce')
                                # For numeric columns, ensure they're float64 for consistency
                                if pd.api.types.is_numeric_dtype(existing_data[col]):
                                    new_data_processed[col] = new_data_processed[col].astype('float64')
                                    existing_data[col] = existing_data[col].astype('float64')
                                else:
                                    new_data_processed[col] = new_data_processed[col].astype(existing_data[col].dtype)
                        except Exception as e:
                            logging.warning(f"Could not convert column {col} to matching dtype: {e}")
                    else:
                        # If column exists in existing_data but not in new_data_processed, add it
                        logging.info(f"Adding missing column {col} to new_data_processed")
                        if pd.api.types.is_numeric_dtype(existing_data[col]):
                            new_data_processed[col] = np.nan
                        else:
                            new_data_processed[col] = None
                
                # Ensure both dataframes have the same columns before concatenation
                for col in new_data_processed.columns:
                    if col not in existing_data.columns:
                        # Add the column with appropriate data type
                        if pd.api.types.is_numeric_dtype(new_data_processed[col]):
                            existing_data[col] = np.nan
                        else:
                            existing_data[col] = None
                        logging.info(f"Added missing column {col} to existing_data")
                
                # Debug column dtypes
                logging.info(f"Existing data dtypes: {existing_data.dtypes}")
                logging.info(f"New data dtypes: {new_data_processed.dtypes}")
                
                # Combine existing and new data by appending new data below existing data
                # Use axis=0 to stack vertically and ignore_index=True to reset indices
                combined_data = pd.concat([existing_data, new_data_processed], axis=0, ignore_index=True)
                
                # Debug the combined data
                logging.info(f"Combined data columns: {combined_data.columns.tolist()}")
                logging.info(f"Combined data sample:\n{combined_data.tail(3)}")
                
                # Convert Date to datetime for proper sorting and deduplication
                combined_data['Date'] = pd.to_datetime(combined_data['Date'])
                logging.info(f"Debug - After concat, combined_data shape: {combined_data.shape}, existing_data shape: {existing_data.shape}")
                
                # Remove duplicate dates, keeping the last entry
                combined_data = combined_data.drop_duplicates(subset='Date', keep='last')
                logging.info(f"Debug - After drop_duplicates, combined_data shape: {combined_data.shape}")
                
                # Sort by date to ensure chronological order
                combined_data = combined_data.sort_values('Date')
                
                # Fill any NaN values in the Date column
                if combined_data['Date'].isna().any():
                    logging.warning(f"Found NaN values in Date column, dropping those rows")
                    combined_data = combined_data.dropna(subset=['Date'])
                
                # Convert Date back to yyyy-mm-dd string format before saving
                combined_data['Date'] = combined_data['Date'].dt.strftime('%Y-%m-%d')
                
                # Debug the final combined data
                logging.info(f"Final combined data sample:\n{combined_data.tail(3)}")
                logging.info(f"Final combined data shape: {combined_data.shape}")
                
                # Get the latest data points from existing and new data for comparison
                latest_existing = None
                latest_new = None
                
                if not existing_data.empty:
                    # Get the latest row from existing data
                    existing_data_copy = existing_data.copy()
                    if 'Date' in existing_data_copy.columns:
                        existing_data_copy['Date'] = pd.to_datetime(existing_data_copy['Date'])
                        latest_existing = existing_data_copy.sort_values('Date').iloc[-1].to_dict()
                    else:
                        logging.warning("No Date column in existing_data, cannot determine latest entry")
                
                if not new_data_processed.empty:
                    # Get the latest row from new data
                    new_data_copy = new_data_processed.copy()
                    if 'Date' in new_data_copy.columns:
                        new_data_copy['Date'] = pd.to_datetime(new_data_copy['Date'])
                        latest_new = new_data_copy.sort_values('Date').iloc[-1].to_dict()
                    else:
                        logging.warning("No Date column in new_data_processed, cannot determine latest entry")
                
                # Check if we have new data to save
                should_save = False
                
                # Only proceed with comparison if we have both latest points
                if latest_existing is not None and latest_new is not None:
                    existing_date = pd.to_datetime(latest_existing['Date']).date() if isinstance(latest_existing['Date'], str) else latest_existing['Date'].date()
                    new_date = pd.to_datetime(latest_new['Date']).date() if isinstance(latest_new['Date'], str) else latest_new['Date'].date()
                    
                    logging.info(f"Comparing latest existing data: {existing_date} with latest new data: {new_date}")
                    
                    # If new data has a newer date, always save
                    if new_date > existing_date:
                        should_save = True
                        logging.info(f"New data has newer date ({new_date} > {existing_date}), will save")
                    else:
                        # Same date - check if values have changed
                        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                            if col in latest_existing and col in latest_new:
                                try:
                                    existing_val = float(latest_existing[col])
                                    new_val = float(latest_new[col])
                                    if not np.isclose(existing_val, new_val, atol=0.0001):
                                        should_save = True
                                        logging.info(f"Latest {col} value updated: {existing_val} -> {new_val}")
                                except (ValueError, TypeError) as e:
                                    logging.warning(f"Could not compare {col} values: {e}")
                                    if str(latest_existing[col]) != str(latest_new[col]):
                                        should_save = True
                                except Exception as e:
                                    logging.warning(f"Error comparing {col} values: {e}")
                                    continue
                elif latest_new is not None:
                    # No existing data but we have new data
                    should_save = True
                    logging.info("No existing data found, will save new data")

                if should_save:
                    # Save the updated data with consistent formatting
                    self._save_data_with_consistent_format(combined_data, data_path)
                    logging.info(f"Data for {ticker} updated and saved")
                    return combined_data
                else:
                    logging.info(f"No new data to update for {ticker}")
                    return existing_data

        except Exception as e:
            logging.error(f"Error updating data for {ticker}: {e}")
            return None

    def load_data(self, ticker):
        """Load stock data for a given ticker"""
        try:
            data_path = os.path.join(self.data_dir, f"{ticker}_stock_data.tsv")
            if os.path.exists(data_path):
                # First, check if the file is valid and has proper structure
                try:
                    # Try to read the first few lines to check structure
                    with open(data_path, 'r') as f:
                        first_lines = [next(f) for _ in range(5) if f]
                    
                    # Check if file appears to be malformed
                    if len(first_lines) < 2 or not all('\t' in line for line in first_lines):
                        logging.warning(f"Malformed data file for {ticker}, attempting to re-download")
                        # Re-download the data
                        self.update_data(ticker, force_download=True)
                        # If re-download fails, return None
                        if not os.path.exists(data_path):
                            return None
                except Exception as file_error:
                    logging.warning(f"Error checking file structure for {ticker}: {file_error}, attempting to re-download")
                    # Re-download the data
                    self.update_data(ticker, force_download=True)
                    # If re-download fails, return None
                    if not os.path.exists(data_path):
                        return None
                
                # Now try to read the data with error handling
                try:
                    data = pd.read_csv(data_path, sep='\t')
                except Exception as read_error:
                    logging.error(f"Error reading data file for {ticker}: {read_error}, attempting to re-download")
                    # Re-download the data
                    self.update_data(ticker, force_download=True)
                    try:
                        data = pd.read_csv(data_path, sep='\t')
                    except Exception as retry_error:
                        logging.error(f"Failed to read data for {ticker} after re-download: {retry_error}")
                        return None

                # Check if data is empty or has too few rows
                if data.empty or len(data) < 5:
                    logging.warning(f"Empty or insufficient data for {ticker}, attempting to re-download")
                    self.update_data(ticker, force_download=True)
                    try:
                        data = pd.read_csv(data_path, sep='\t')
                        if data.empty or len(data) < 5:
                            logging.error(f"Still insufficient data for {ticker} after re-download")
                            return None
                    except Exception:
                        return None

                # Always normalize columns to ensure consistent structure
                logging.info(f"Normalizing column structure for {ticker} data")
                # Create a new DataFrame with standardized columns
                standard_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                
                # Create a new DataFrame with the same index as the original data
                new_df = pd.DataFrame(index=data.index)
                
                # First ensure we have a Date column
                if 'Date' in data.columns:
                    new_df['Date'] = data['Date'].values  # Use .values to avoid index alignment issues
                else:
                    # Try to find a date column with a different name
                    date_found = False
                    for col in data.columns:
                        if 'date' in str(col).lower():
                            new_df['Date'] = data[col].values  # Use .values to avoid index alignment issues
                            date_found = True
                            break
                    
                    if not date_found:
                        logging.error(f"No Date column found in {ticker} data")
                        # Re-download the data
                        self.update_data(ticker, force_download=True)
                        return self.load_data(ticker)  # Recursive call with fresh data
                
                # Map other columns to standard ones, handling various formats
                for std_col in standard_columns:
                    if std_col == 'Date':  # Already handled Date column
                        continue
                        
                    # Try exact match first
                    if std_col in data.columns:
                        new_df[std_col] = data[std_col].values  # Use .values to avoid index alignment issues
                        continue

                    # Try case-insensitive match
                    col_matched = False
                    for col in data.columns:
                        if std_col.lower() == str(col).lower():
                            new_df[std_col] = data[col].values  # Use .values to avoid index alignment issues
                            col_matched = True
                            break
                    
                    if col_matched:
                        continue

                    # Try partial match in complex columns
                    for col in data.columns:
                        if std_col.lower() in str(col).lower():
                            new_df[std_col] = data[col].values  # Use .values to avoid index alignment issues
                            break
                
                # Ensure we have at least Close price data
                if 'Close' not in new_df.columns and 'Adj Close' not in new_df.columns:
                    # If we don't have Close data, try to find any numeric column
                    for col in data.columns:
                        if col != 'Date' and pd.to_numeric(data[col], errors='coerce').notna().any():
                            new_df['Close'] = data[col]
                            logging.warning(f"Using column '{col}' as 'Close' for {ticker}")
                            break
                    else:
                        logging.error(f"No suitable price data found for {ticker}")
                        # Re-download the data
                        self.update_data(ticker, force_download=True)
                        return self.load_data(ticker)  # Recursive call with fresh data

                # If we have a valid DataFrame with at least Date and one price column, use it
                if 'Date' in new_df.columns and any(col in new_df.columns for col in ['Close', 'Adj Close']):
                    # Convert all price columns to numeric
                    for col in new_df.columns:
                        if col != 'Date':
                            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                    
                    # Ensure Date column is properly formatted as datetime
                    new_df['Date'] = pd.to_datetime(new_df['Date'], errors='coerce')
                    
                    # Drop any rows with invalid dates or all NaN values
                    new_df = new_df.dropna(subset=['Date'])
                    new_df = new_df.dropna(how='all', subset=[col for col in new_df.columns if col != 'Date'])
                    
                    # Reset index to avoid any alignment issues
                    new_df = new_df.reset_index(drop=True)
                    
                    # Set Date as index for easier filtering
                    data = new_df.set_index('Date')

                    # Save the normalized data back to file
                    try:
                        data_to_save = data.reset_index()
                        # Save with consistent formatting
                        self._save_data_with_consistent_format(data_to_save, data_path)
                        logging.info(f"Normalized column structure for {ticker} data while loading")
                    except Exception as save_error:
                        logging.warning(f"Could not save normalized data for {ticker}: {save_error}")

                    return data
                else:
                    logging.error(f"Failed to normalize data for {ticker}")
                    # Re-download as a last resort
                    self.update_data(ticker, force_download=True)
                    return None
            else:
                logging.warning(f"No data file found for {ticker}")
                return None
        except Exception as e:
            logging.error(f"Error loading data for {ticker}: {e}")
            # Try to re-download as a last resort
            try:
                self.update_data(ticker, force_download=True)
                return self.load_data(ticker)  # Recursive call with fresh data
            except:
                return None

    def visualize_data(self,
                       ticker: str,
                       column: str = 'Close',
                       title: Optional[str] = None):
        """
        Create a visualization of stock data.

        Args:
            ticker (str): Stock ticker symbol
            column (str, optional): Column to plot. Defaults to 'Close'.
            title (Optional[str], optional): Custom plot title
        """
        try:
            # Load data with normalization
            data = self.load_data(ticker)

            if data is None or data.empty:
                logging.warning(f"No data available for {ticker}")
                return

            # Make a copy to avoid modifying the original data
            data = data.copy()

            # Check if Date is already the index (from load_data method)
            if data.index.name == 'Date':
                # Data already has Date as index, which is what we want
                pass
            elif 'Date' in data.columns:
                # Convert Date to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(data['Date']):
                    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

                # Handle timezone-aware datetime objects
                try:
                    # Convert to timezone-naive datetime objects
                    data['Date'] = data['Date'].dt.tz_localize(None)
                except (AttributeError, TypeError):
                    # Already timezone-naive or has timezone info
                    try:
                        # Try to convert if it has timezone info
                        data['Date'] = data['Date'].dt.tz_convert(None)
                    except (AttributeError, TypeError):
                        # Can't convert, just continue
                        pass

                # Set Date as index for plotting
                data = data.set_index('Date')
            else:
                logging.error(f"Date column not found in {ticker} data")
                return

            # Double-check that we have the expected columns
            if column not in data.columns:
                # Try to find a matching column
                for col in data.columns:
                    if column.lower() in str(col).lower():
                        logging.info(f"Using column '{col}' instead of '{column}' for {ticker}")
                        column = col
                        break
                else:
                    # If no matching column is found, default to 'Close' if available
                    if 'Close' in data.columns:
                        logging.warning(f"Column '{column}' not found in {ticker} data, using 'Close' instead")
                        column = 'Close'
                    else:
                        logging.error(f"Column '{column}' not found in {ticker} data and no suitable alternative found")
                        return

            # Check if we have valid data after processing
            if data.empty or column not in data.columns:
                logging.error(f"No valid data available for {ticker} after processing")
                return

            # Check for NaN values in the column to plot
            if data[column].isna().all():
                logging.error(f"All values in column '{column}' for {ticker} are NaN")
                return

            # Plot the data
            try:
                plt.ioff()  # Turn off interactive mode
                plt.figure(figsize=(12, 6))
                data[column].plot()
                plt.title(title or f"{ticker} {column} Price History")
                plt.xlabel('Date')
                plt.ylabel(f'{column} Price')
                plt.grid(True)
                plt.tight_layout()

                # Ensure the directory exists
                os.makedirs(self.plot_save_path, exist_ok=True)
                save_path = os.path.join(self.plot_save_path, f'{ticker}_{column}_plot.png')
                plt.savefig(save_path)
                logging.info(f"Generated visualization for {ticker}")
            except Exception as inner_e:
                logging.error(f"Error plotting for {ticker}: {inner_e}")
            finally:
                plt.close('all')  # Close all figures to prevent memory leaks
        except Exception as e:
            logging.error(f"Error visualizing data for {ticker}: {e}")
        finally:
            plt.close('all')  # Ensure figures are closed even if an error occurs

    def visualize_multiple_tickers(self,
                              tickers: List[str],
                              folder_name: str,
                              column: str = 'Close',
                              title: Optional[str] = None):
        """
        Create a subplot visualization of stock data for multiple tickers.

        Args:
            tickers (List[str]): List of stock ticker symbols
            folder_name (str): Folder name for saving plots
            column (str, optional): Column to plot. Defaults to 'Close'.
            title (Optional[str], optional): Custom plot title
        """
        try:
            plt.ioff()  # Turn off interactive mode

            # Determine subplot layout
            n_tickers = len(tickers)
            rows = math.ceil(math.sqrt(n_tickers))
            cols = math.ceil(n_tickers / rows)

            plt.figure(figsize=(15, 10))

            for i, ticker in enumerate(tickers, 1):
                data = self._load_stock_data(ticker)

                if data is None or data.empty:
                    print(f"No data available for {ticker}")
                    continue

                plt.subplot(rows, cols, i)
                plt.plot(data.index, data[column], label=f'{ticker} {column}')
                plt.title(f'{ticker} Stock Price')
                plt.xlabel('Date')
                plt.ylabel(f'{column} Price')
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)

            plt.suptitle(title or f'Stock Prices for {", ".join(tickers)}')
            plt.tight_layout()

            # Ensure the directory exists
            plots_dir = os.path.join(self.plot_save_path, folder_name)
            os.makedirs(plots_dir, exist_ok=True)
            save_path = os.path.join(plots_dir, f'multiple_tickers_{column}_plot.png')
            plt.savefig(save_path)
            print(f"Multiple tickers plot saved to {save_path}")

            plt.close('all')  # Close all figures to prevent memory leaks
        except Exception as e:
            print(f"Error visualizing multiple tickers: {e}")
        finally:
            plt.close('all')  # Ensure figures are closed even if an error occurs

    def resample_data(self,
                       ticker: str,
                       resample_freq: str = 'W',
                       column: str = 'Close'):
        """
        Resample stock data to a different frequency.

        Args:
            ticker (str): Stock ticker symbol
            resample_freq (str, optional): Resampling frequency.
                Defaults to 'W' (weekly).
                Common options:
                - 'D': Daily
                - 'W': Weekly
                - 'ME': Monthly end
                - 'Q': Quarterly
            column (str, optional): Column to resample. Defaults to 'Close'.

        Returns:
            pd.Series: Resampled stock data
        """
        try:
            # Load data with normalization
            data = self.load_data(ticker)

            if data is None or data.empty:
                logging.warning(f"No data available for {ticker}")
                return None

            # Make a copy to avoid modifying the original data
            data = data.copy()

            # Check if Date is already the index (from load_data method)
            if data.index.name == 'Date':
                # Data already has Date as index, which is what we want
                pass
            elif 'Date' in data.columns:
                # Clean data: remove any rows with NaN in Date column
                data = data.dropna(subset=['Date'])

                # Clean data: remove any non-numeric header rows (where Date is string but not a valid date)
                if data.shape[0] > 0 and isinstance(data['Date'].iloc[0], str) and not pd.to_datetime(data['Date'].iloc[0], errors='coerce'):
                    logging.warning(f"Removing header row from {ticker} data")
                    data = data.iloc[1:].reset_index(drop=True)

                # Convert Date to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(data['Date']):
                    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

                # Drop rows with invalid dates
                data = data.dropna(subset=['Date'])

                # Set Date as index
                data = data.set_index('Date')
            else:
                logging.error(f"Date column not found in {ticker} data")
                return None

            # Double-check that we have the expected columns
            if column not in data.columns:
                # Try to find a matching column
                for col in data.columns:
                    if column.lower() in str(col).lower():
                        column = col
                        logging.info(f"Using column '{col}' instead of '{column}' for resampling {ticker} data")
                        break
                else:
                    # If no matching column is found, default to 'Close' if available
                    if 'Close' in data.columns:
                        column = 'Close'
                        logging.warning(f"Column '{column}' not found in {ticker} data, using 'Close' instead")
                    else:
                        logging.error(f"Column '{column}' not found in {ticker} data and no suitable alternative found")
                        return None

            # Convert numeric columns to float
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')

            # Drop rows with NaN in the column we're resampling
            data = data.dropna(subset=[column])

            # Check if we have valid data after cleaning
            if data.empty:
                logging.warning(f"No valid data available for {ticker} after cleaning")
                return None

            # Resample data
            resampled = data[column].resample(resample_freq).last()

            # Drop NaN values from resampled data
            resampled = resampled.dropna()

            return resampled

        except Exception as e:
            logging.error(f"Error resampling data for {ticker}: {e}")
            return None

    def visualize_daily_vs_weekly(self, ticker: str, column: str = 'Close') -> None:
        """
        Visualize daily and weekly stock prices for a given ticker

        Args:
            ticker (str): Stock ticker symbol
            column (str, optional): Price column to visualize. Defaults to 'Close'.
        """
        try:
            # Set matplotlib to use non-interactive backend for thread safety
            # This prevents warnings when called from background threads
            import matplotlib
            original_backend = matplotlib.get_backend()
            matplotlib.use('Agg')  # Use non-interactive backend for thread safety
            # Load daily data with normalization and cleaning
            data = self.load_data(ticker)

            if data is None or data.empty:
                logging.warning(f"No data available for {ticker}")
                return

            # Make a copy to avoid modifying the original data
            daily_data = data.copy()

            # Check if Date is already the index (from load_data method)
            if daily_data.index.name == 'Date':
                # Data already has Date as index, which is what we want
                pass
            elif 'Date' in daily_data.columns:
                # Clean data: remove any rows with NaN in Date column
                daily_data = daily_data.dropna(subset=['Date'])

                # Convert Date to datetime
                daily_data['Date'] = pd.to_datetime(daily_data['Date'], errors='coerce')
                daily_data = daily_data.dropna(subset=['Date'])
                daily_data.set_index('Date', inplace=True)
            else:
                logging.error(f"Date column not found in {ticker} data")
                return

            # Apply date range filtering if specified
            if self.start_date:
                # Convert start_date to a timezone-aware timestamp (UTC)
                start_date = pd.to_datetime(self.start_date, utc=True)
                # Log the date range before filtering
                logging.info(f"Applying start date filter: {start_date}, data range: {daily_data.index.min()} to {daily_data.index.max()}")
                # Filter data (index is already UTC-aware)
                daily_data = daily_data[daily_data.index >= start_date]
                # Log the data range after filtering
                logging.info(f"After start date filter: data range: {daily_data.index.min()} to {daily_data.index.max()}, rows: {len(daily_data)}")

            if self.end_date:
                # Convert end_date to a timezone-aware timestamp (UTC)
                end_date = pd.to_datetime(self.end_date, utc=True)
                # Log the date range before filtering
                logging.info(f"Applying end date filter: {end_date}, data range: {daily_data.index.min()} to {daily_data.index.max()}")
                # Filter data (index is already UTC-aware)
                daily_data = daily_data[daily_data.index <= end_date]
                # Log the data range after filtering
                logging.info(f"After end date filter: data range: {daily_data.index.min()} to {daily_data.index.max()}, rows: {len(daily_data)}")

            # Check if we still have data after filtering
            if daily_data.empty:
                logging.warning(f"No data available for {ticker} in the specified date range")
                return

            # Convert numeric columns to float
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for col in numeric_columns:
                if col in daily_data.columns:
                    daily_data[col] = pd.to_numeric(daily_data[col], errors='coerce')

            # Handle column selection
            if column not in daily_data.columns:
                # Try to find a matching column
                for col in daily_data.columns:
                    if column.lower() in str(col).lower():
                        column = col
                        logging.info(f"Using column '{col}' instead of '{column}' for {ticker}")
                        break
                else:
                    # If no matching column is found, default to 'Close' if available
                    if 'Close' in daily_data.columns:
                        column = 'Close'
                        logging.warning(f"Column '{column}' not found in {ticker} data, using 'Close' instead")
                    else:
                        logging.error(f"Column '{column}' not found in {ticker} data and no suitable alternative found")
                        return

            # Drop rows with NaN in the column we're plotting
            daily_data = daily_data.dropna(subset=[column])

            # Get weekly and monthly data through resampling
            weekly_data = self.resample_data(ticker, resample_freq='W', column=column)
            monthly_data = self.resample_data(ticker, resample_freq='ME', column=column)

            if weekly_data is None or monthly_data is None:
                logging.error(f"Could not generate weekly or monthly data for {ticker}")
                return

            # Ensure datetime indices
            weekly_data.index = pd.to_datetime(weekly_data.index)
            monthly_data.index = pd.to_datetime(monthly_data.index)

            # Get the latest data date for display
            latest_data_date = daily_data.index.max()
            if hasattr(latest_data_date, 'strftime'):
                latest_date_str = latest_data_date.strftime('%Y-%m-%d')
            else:
                latest_date_str = str(latest_data_date)[:10]

            # Create figure with three subplots
            fig, (ax3, ax2, ax1) = plt.subplots(1, 3, figsize=(30, 7))
            
            # Add main title with latest data date
            fig.suptitle(f'{ticker} Stock Price Charts (Data as of {latest_date_str})', fontsize=14, fontweight='bold', y=1.02)

            # Plot daily data for recent year
            if not daily_data.empty:
                recent_year_data = daily_data[daily_data.index > daily_data.index.max() - pd.Timedelta(days=365)]
                if not recent_year_data.empty:
                    ax1.semilogy(recent_year_data.index, recent_year_data[column])
                    ax1.set_title(f'{ticker} Recent Year Daily {column} Prices')
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel(f'{column} Price ($)')
                    ax1.tick_params(axis='x', rotation=45)
                    ax1.grid(True, which='both', ls='-', alpha=0.5)
                else:
                    logging.warning(f"No recent year data available for {ticker}")

            # Plot weekly data for recent 5 years
            if not weekly_data.empty:
                recent_5_years_data = weekly_data[weekly_data.index > weekly_data.index.max() - pd.Timedelta(days=1825)]
                if not recent_5_years_data.empty:
                    ax2.semilogy(recent_5_years_data.index, recent_5_years_data.values)
                    ax2.set_title(f'{ticker} Recent 5 Years Weekly {column} Prices')
                    ax2.set_xlabel('Date')
                    ax2.set_ylabel(f'{column} Price ($)')
                    ax2.tick_params(axis='x', rotation=45)
                    ax2.grid(True, which='both', ls='-', alpha=0.5)
                else:
                    logging.warning(f"No recent 5 years data available for {ticker}")

            # Plot monthly data
            if not monthly_data.empty:
                ax3.semilogy(monthly_data.index, monthly_data.values)
                ax3.set_title(f'{ticker} Monthly {column} Prices')
                ax3.set_xlabel('Date')
                ax3.set_ylabel(f'{column} Price ($)')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, which='both', ls='-', alpha=0.5)
            else:
                logging.warning(f"No monthly data available for {ticker}")

            # Adjust layout and save figure
            plt.tight_layout()

            # Save with a single consistent filename pattern
            plt.savefig(os.path.join(self.plot_save_path, f'{ticker}_daily_weekly_monthly.png'), dpi=300)

            plt.close(fig)

            # Restore original backend
            matplotlib.use(original_backend)

            logging.info(f"Generated visualization for {ticker}")

        except Exception as e:
            logging.error(f"Error visualizing data for {ticker}: {e}")
            plt.close('all')  # Ensure figures are closed even if an error occurs

    def plot_multiple_tickers(self, tickers):
        # First, download initial data for all tickers
        for ticker in tickers:
            self.update_data(ticker)

        # Define time frames
        time_frames = [
            ('1 Year', pd.Timestamp.today() - pd.Timedelta(days=365), pd.Timestamp.today()),
            ('5 Years', pd.Timestamp.today() - pd.Timedelta(days=365*5), pd.Timestamp.today()),
            ('All Available Data', None, None)
        ]

        # Create a figure with three subplots
        fig, (ax3, ax2, ax1) = plt.subplots(1, 3, figsize=(30, 7))

        # Color palette for distinct lines
        colors = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#7f7f7f',  # gray
            '#bcbd22',  # olive
            '#17becf'   # cyan
        ]

        # Extend colors if needed
        if len(tickers) > len(colors):
            import colorsys

            # Generate additional colors
            def generate_distinct_colors(n):
                HSV_tuples = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
                return [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]

            additional_colors = generate_distinct_colors(len(tickers) - len(colors))
            # Convert RGB to hex
            additional_colors = ['#%02x%02x%02x' % tuple(int(x*255) for x in color) for color in additional_colors]
            colors.extend(additional_colors)

        # First pass: load original data
        ticker_data = {}
        global_earliest_start = pd.Timestamp.max

        for ticker in tickers:
            try:
                # Load data from file
                data = self._load_stock_data(ticker)

                # Convert Date column to datetime and set as index
                data['Date'] = pd.to_datetime(data['Date'], utc=True)
                data.set_index('Date', inplace=True)

                # Convert Close column to numeric, removing any non-numeric characters
                data['Close'] = pd.to_numeric(data['Close'].replace({'$': ''}, regex=True), errors='coerce')

                # Update global earliest start date
                global_earliest_start = min(global_earliest_start, data.index.min())

                # Store the original data
                ticker_data[ticker] = data

                # Print original data date range for debugging
                print(f"{ticker} original data date range: {data.index.min()} to {data.index.max()}")

            except Exception as e:
                print(f"Error processing {ticker}: {e}")

        # Second pass: pad data for each ticker
        padded_ticker_data = {}

        for ticker in tickers:
            data = ticker_data[ticker]

            # Find the first trading day of the original data
            first_trading_day = data.index.min()

            # Create a date range from global earliest start to the first trading day
            pre_trading_dates = pd.date_range(start=global_earliest_start, end=first_trading_day - pd.Timedelta(days=1), freq='D')

            # Find the first price in the original data
            first_price = data['Close'].iloc[0]

            # Create a padded series for the pre-trading period
            pre_trading_series = pd.Series(index=pre_trading_dates, data=first_price, dtype=float)

            # Combine pre-trading series with original data
            padded_series = pd.concat([pre_trading_series, data['Close']])

            # Store padded data
            padded_ticker_data[ticker] = padded_series

            # Print padded data info
            print(f"{ticker} padded data date range: {padded_series.index.min()} to {padded_series.index.max()}")
            print(f"{ticker} padded data length: {len(padded_series)}")

            # Save padded data to CSV
            # padded_series.to_csv(f"{ticker}_padded_data.csv")

        # Third pass: plot charts for different time frames
        for idx, (label, start_date, end_date) in enumerate(time_frames):
            # Select the appropriate axis
            ax = [ax1, ax2, ax3][idx]

            # Initialize an empty DataFrame to store aligned data
            df_aligned = pd.DataFrame()

            for ticker in tickers:
                # Filter padded data within specified date range
                if start_date is None or end_date is None:
                    start_date = padded_ticker_data[ticker].index.min()
                    end_date = padded_ticker_data[ticker].index.max()

                filtered_data = padded_ticker_data[ticker].loc[start_date:end_date]

                # Add to aligned DataFrame
                df_aligned[ticker] = filtered_data

            # Normalize prices to starting value of 100
            df_normalized = df_aligned / df_aligned.iloc[0] * 100

            # Plot each ticker's normalized data
            for i, ticker in enumerate(df_normalized.columns):
                # Plot the line
                ax.plot(df_normalized.index, df_normalized[ticker],
                        label=ticker, color=colors[i], linewidth=2)

                # Add ticker name at the end of the line
                last_price = df_normalized[ticker].iloc[-1]
                last_date = df_normalized.index[-1]

                # Add a small offset to the x and y positions to prevent overlap
                x_offset = pd.Timedelta(days=10)
                y_offset = last_price * 0.02  # 2% offset

                comment = tickers_comment_dict.get(ticker, '')
                ticker_or_comment = ticker if comment == '' else f'{comment}'

                ax.annotate(f' {ticker_or_comment} ({last_price:.0f}%) ',
                            xy=(last_date, last_price),
                            xytext=(last_date + x_offset, last_price + y_offset),
                            fontsize=10,
                            color=colors[i],
                            va='bottom')

            ax.set_title(f'{label} Stock Price Performance (Normalized to 100)', fontsize=16)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Normalized Price (Log Scale)', fontsize=12)
            ax.set_yscale('log')  # Set Y-axis to logarithmic scale

            # Create legend labels with comments
            legend_labels = []
            for ticker in df_normalized.columns:
                comment = tickers_comment_dict.get(ticker, '')
                legend_labels.append(f'{ticker} {comment}'.strip())

            ax.legend(legend_labels, fontsize=10, loc='upper left', bbox_to_anchor=(0, 1.1))

            ax.grid(True, linestyle='--', alpha=0.7)

            # Ensure x-axis ticks are not too crowded
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Adjust layout and add overall title
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Make room for the legend
        fig.suptitle('Stock Price Performance Comparison', fontsize=20, y=1.02)

        # Ensure the plots directory exists
        os.makedirs('plots', exist_ok=True)

        # Determine folder name
        ticker_list_name = self._get_folder_name(tickers)
        plot_path = os.path.join('plots', f'{ticker_list_name}_comparison.png')

        # Save the plot
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {plot_path}")

        # Show the plot
        plt.show()


    def generate_html_report(self, plots_dir=None, filename=None, tickers=None):
        """
        Generate an HTML report with embedded stock plots

        Args:
            plots_dir (str, optional): Directory containing plot images. Defaults to current plot_save_path.
            filename (str, optional): Custom filename for the HTML report. Defaults to 'stock_analysis_report.html'.
            tickers (list, optional): List of tickers to include in the report. If None, all tickers found in the plots directory will be included.
        """
        try:
            # Use current plot_save_path if no directory specified
            if plots_dir is None:
                plots_dir = self.plot_save_path

            # Import required libraries
            import os
            import glob
            import webbrowser

            # Create HTML content
            html_content = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>Stock Price Analysis Report</title>
                <style>
                    body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
                    h1 { color: #333; text-align: center; }
                    .plot-container {
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: center;
                        gap: 20px;
                        margin-top: 20px;
                    }
                    .plot-item {
                        text-align: center;
                        max-width: 100%;
                    }
                    .plot-item img {
                        max-width: 100%;
                        height: auto;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                    }
                </style>
            </head>
            <body>
                <h1>Stock Price Analysis Report</h1>
                <div class="plot-container">
            """

            # Find all PNG files in the plots directory
            if tickers:
                # Filter plot files to only include the specified tickers
                plot_files = []
                for ticker in tickers:
                    ticker_upper = ticker.upper()
                    # Add daily/weekly/monthly plots (our standard format)
                    timeframe_plots = glob.glob(os.path.join(plots_dir, f"{ticker_upper}_daily_weekly_monthly.png"))
                    plot_files.extend(timeframe_plots)

                    # If no plots found with the standard format, try legacy formats for backward compatibility
                    if not timeframe_plots:
                        # Try daily vs weekly plots (older format)
                        daily_weekly_plots = glob.glob(os.path.join(plots_dir, f"{ticker_upper}_daily_vs_weekly_price.png"))
                        plot_files.extend(daily_weekly_plots)

                        # Try stock price plots (another older format)
                        if not daily_weekly_plots:
                            price_plots = glob.glob(os.path.join(plots_dir, f"{ticker_upper}_stock_prices.png"))
                            plot_files.extend(price_plots)
            else:
                # Include all plots if no specific tickers are provided
                plot_files = glob.glob(os.path.join(plots_dir, '*_stock_prices.png'))
                plot_files += glob.glob(os.path.join(plots_dir, '*_daily_weekly_monthly.png'))
                plot_files += glob.glob(os.path.join(plots_dir, '*_daily_vs_weekly_price.png'))

            # Add plots to HTML - avoid duplicates by tracking processed tickers
            processed_tickers = set()
            for plot_file in plot_files:
                # Extract ticker name from filename
                ticker = os.path.basename(plot_file).split('_')[0]

                # Skip if we've already processed this ticker
                if ticker in processed_tickers:
                    continue

                # Add ticker to processed set
                processed_tickers.add(ticker)
                html_content += f"""
                    <div class="plot-item">
                        <h2>{ticker} Stock Prices</h2>
                        <img src="{os.path.basename(plot_file)}" alt="{ticker} Stock Price Plot">
                    </div>
                """

            # Close HTML tags
            html_content += """
                </div>
            </body>
            </html>
            """

            # Use custom filename if provided, otherwise use default
            if filename is None:
                filename = 'stock_analysis_report.html'

            # Save HTML report
            report_path = os.path.abspath(os.path.join(plots_dir, filename))
            with open(report_path, 'w') as f:
                f.write(html_content)

            logging.info(f"Generated stock analysis report at {report_path}")

            # Return the report path for the caller to use
            return report_path

        except Exception as e:
            logging.error(f"Error generating HTML report: {e}")

    def process_stock_data(self, tickers=[], name=None, force_download=False):
        """
        Process and visualize stock data for multiple tickers

        Args:
            tickers (list): List of stock tickers to process
            name (str, optional): Custom name for the data folder. Defaults to None.
            force_download (bool): Force re-download of data. Defaults to False.
        """
        # Validate input
        if not tickers:
            logging.warning("No tickers provided for processing")
            return

        # Determine folder name
        folder_name = self._get_folder_name(tickers, name)

        # Create subfolder
        folder_path = os.path.join('stock_data', folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Create a new stock manager with the specific plot save path
        stock_manager = StockDataManager(plot_save_path=folder_path)

        try:
            # Update and process data for each ticker
            for ticker in tickers:
                # Download and update stock data (force re-download if specified)
                stock_manager.update_data(ticker, force_download=force_download)

                # Visualize daily, weekly, and monthly data
                stock_manager.visualize_daily_vs_weekly(ticker)

            # Visualize stock prices
            stock_manager.visualize_multiple_tickers(tickers, folder_name)

            # Generate HTML report after processing all tickers
            stock_manager.generate_html_report()

        except Exception as e:
            logging.error(f"Error processing stock data: {e}")

    def get_data(self, ticker, start_date=None, end_date=None):
        """
        Wrapper method to handle get_data calls by redirecting to appropriate methods.

        Args:
            ticker (str): Stock ticker symbol
            start_date (str, optional): Start date for data retrieval
            end_date (str, optional): End date for data retrieval

        Returns:
            pd.DataFrame: Stock data for the specified ticker
        """
        logging.info(f"get_data wrapper called for {ticker}")

        try:
            # Try to use _load_stock_data if it exists
            if hasattr(self, '_load_stock_data'):
                return self._load_stock_data(ticker, start_date, end_date)
            # Try to use load_data if it exists
            elif hasattr(self, 'load_data'):
                data = self.load_data(ticker)
                if data is None or data.empty:
                    # If data is empty, try force downloading
                    logging.warning(f"No data found for {ticker}, attempting force download")
                    self.update_data(ticker, force_download=True)
                    data = self.load_data(ticker)
                return data
            # Try to use update_data if it exists
            elif hasattr(self, 'update_data'):
                return self.update_data(ticker)
            # Fallback to direct yfinance download
            else:
                logging.warning(f"No data loading method found, using direct yfinance download for {ticker}")
                import yfinance as yf
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(period="max")
                return data
        except AttributeError as e:
            logging.error(f"Method not found: {e}")
            raise AttributeError(f"Could not find a suitable method to get data for {ticker}")

        except Exception as e:
            logging.error(f"Error in get_data wrapper: {str(e)}")
            # Try force download as a last resort
            try:
                logging.warning(f"Attempting force download for {ticker} after error")
                self.update_data(ticker, force_download=True)
                return self.load_data(ticker)
            except Exception as download_error:
                logging.error(f"Force download failed for {ticker}: {download_error}")
                # Return empty DataFrame as fallback
                return pd.DataFrame()

    def get_fundamental_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get fundamental data for a given ticker.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Optional[Dict[str, Any]]: A dictionary with fundamental data or None if it fails.
        """
        try:
            logging.info(f"Fetching fundamental data for {ticker}")
            ticker_obj = yf.Ticker(ticker)
            return ticker_obj.info
        except Exception as e:
            logging.error(f"Error fetching fundamental data for {ticker}: {e}")
            return None

    def _get_folder_name(self, tickers, name=None):
        """
        Get a folder name for storing ticker data

        Args:
            tickers (list): List of tickers
            name (str, optional): Custom name. Defaults to None.

        Returns:
            str: Folder name
        """
        if name:
            return name
        else:
            # Try to get the variable name from globals
            try:
                for var_name, var_value in globals().items():
                    if var_value is tickers:
                        return var_name
                # Fallback to sorted tickers
                return '_'.join(sorted(tickers[:3])) + f"_etc_{len(tickers)}"
            except Exception:
                # Most conservative fallback
                return '_'.join(sorted([str(t) for t in tickers[:3]])) + f"_etc_{len(tickers)}"
