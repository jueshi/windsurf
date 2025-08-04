"""
Stock Data Retriever and Manager

This script provides functionality for downloading, updating, and visualizing stock data.

CHANGELOG:
---------
v1.13.0 (2025-01-04):
- Enhanced AI_ticker_extractor.py to automatically parse and process stock tickers
- Added support for dynamically extracting and saving stock tickers from clipboard
- Improved regex pattern for ticker extraction
- Renamed extracted tickers to follow naming convention
- Added robust error handling for ticker parsing

v1.12.0 (2025-01-04):
- Created separate ticker_lists.py for managing stock ticker lists
- Implemented dynamic ticker list processing in main function
- Added support for automatically discovering and processing stock ticker lists
- Renamed existing ticker lists to follow _stocks or _tickers naming convention
- Added new lists: bitcoin_tickers, canslim_tickers
- Improved code modularity and maintainability
- Simplified stock data processing workflow

v1.11.0 (2025-01-02):
- Enhanced data update mechanism to prevent unnecessary file writes
- Added strict checks to only update local files when genuinely new data is available
- Improved logging to distinguish between no new data and update scenarios
- Reduced file I/O operations and potential disk wear

v1.10.0 (2025-01-02):
- Enhanced data update mechanism to check local data freshness
- Implemented intelligent data update strategy
- Added logic to only download new data if local data is outdated
- Reduced unnecessary API calls and improved data retrieval efficiency
- Maintained data continuity by appending only new data points

v1.9.0 (2025-01-02):
- Enhanced main function to visualize daily, weekly, and monthly charts for all tickers
- Automated visualization process for multiple stock tickers
- Improved script flexibility for comprehensive stock data analysis

v1.8.0 (2025-01-02):
- Extended stock price visualization to include monthly price charts
- Enhanced multi-frequency price visualization with daily, weekly, and monthly views
- Increased figure width to accommodate three subplots
- Maintained logarithmic y-axis scaling across all frequency views

v1.7.0 (2025-01-02):
- Introduced logarithmic (semilogy) scaling for stock price visualizations
- Enhanced price chart readability by using logarithmic y-axis
- Improved visualization of percentage changes and exponential trends
- Updated daily and weekly price charts to use logarithmic scaling

v1.6.0 (2025-01-02):
- Significantly improved data loading robustness
- Added comprehensive checks for data sufficiency
- Enhanced error handling for various data loading scenarios
- Implemented more intelligent column mapping and fallback mechanisms
- Added index reset to ensure consistent data processing

v1.5.0 (2025-01-02):
- Improved data loading robustness with dynamic column handling
- Added flexible column selection in data loading
- Enhanced error handling for inconsistent data formats
- Implemented more resilient data parsing mechanism

v1.4.0 (2025-01-02):
- Added `resample_data()` method to support different time frequency aggregations
- Implemented `visualize_daily_vs_weekly()` method for side-by-side price comparisons
- Added flexible resampling with support for daily, weekly, monthly, and quarterly views
- Enhanced visualization capabilities with multi-frequency plotting

v1.3.0 (2025-01-02):
- Added `visualize_multiple_tickers()` method to create subplots for multiple stocks
- Implemented dynamic subplot layout with max 3 columns
- Enhanced visualization to support multiple tickers in a single plot
- Improved error handling for multi-ticker visualization

v1.2.0 (2025-01-02):
- Improved column handling to dynamically support different data formats
- Switched to tab-separated (.tsv) files for data storage
- Enhanced error handling and logging for column variations
- Made visualization method more robust to handle different column sets

v1.1.0 (Previous Version):
- Implemented class-based design for stock data management
- Added initial download and update functionalities
- Created visualization method for stock data
- Implemented logging for tracking operations

v1.0.0 (Initial Release):
- Basic stock data retrieval functionality
- Simple CSV-based data storage

Dependencies:
- pandas
- yfinance
- matplotlib
- logging

Author: Codeium AI Assistant
Last Updated: 2025-01-04
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging
import numpy as np
from typing import Optional, List, Any
import webbrowser
import re
from ticker_lists import *
import math
import time
from datetime import datetime, timezone, timedelta
from random import uniform
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import inspect
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Directories and Configurations
STOCK_DATA_DIR = 'stock_data'
os.makedirs(STOCK_DATA_DIR, exist_ok=True)

class StockDataManager:
    """
    A comprehensive manager for downloading, updating, and analyzing stock data.
    
    Attributes:
        data_dir (str): Directory to store stock data files
        plot_save_path (str): Path to save plots
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
        self.last_request_time = 0
        self.min_request_interval = 2.0  # Minimum time between requests in seconds
    
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
                            progress=False
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
                            progress=False
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
            
            # Save data to local file using tab separator
            output_data.to_csv(data_path, sep='\t', index=False)
            logging.info(f"All available historical data for {ticker} saved to {data_path}")
            
            return stock_data
        
        except Exception as e:
            logging.error(f"Error in initial download for {ticker}: {e}")

    def update_data(self, ticker: str, force_download: bool = False) -> Optional[pd.DataFrame]:
        """
        Update existing stock data with the most recent information.
        Ensures at least 3 years of historical data if possible.
        
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
                # Load existing data
                existing_data = pd.read_csv(data_path, sep='\t')
                
                # Convert Date column to datetime with UTC=True
                existing_data['Date'] = pd.to_datetime(existing_data['Date'], utc=True)
                
                # Check if we have at least 3 years of data
                earliest_date = existing_data['Date'].min()
                latest_date = existing_data['Date'].max()
                data_span = latest_date - earliest_date
                
                if data_span.days >= 365 * 3:
                    logging.info(f"{ticker} has {data_span.days / 365:.1f} years of data, which meets the 3-year minimum requirement")
                    
                    # Check if data is already up to date (within 1 day)
                    if (datetime.now(timezone.utc).date() - latest_date.date()).days <= 1:
                        logging.info(f"Data for {ticker} is up to date")
                        return existing_data
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
                    
                    # Save to file
                    os.makedirs(os.path.dirname(data_path), exist_ok=True)
                    stock_data.to_csv(data_path, sep='\t', index=False)
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
                
            # Get the latest date in local data
            latest_local_date = existing_data['Date'].max()
            
            # Download new data from the day after the latest local date
            start_date = (latest_local_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            new_data = yf.download(ticker, start=start_date, progress=False)
            
            # Validate downloaded data
            if new_data.empty:
                logging.info(f"No new data available for {ticker}")
                return existing_data
            
            # Reset index to make Date a column
            new_data_reset = new_data.reset_index()
            
            # Ensure consistent column order and names
            columns_order = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            
            # Prepare new data with consistent columns
            new_data_processed = new_data_reset.rename(columns={
                'Date': 'Date', 
                'Open': 'Open', 
                'High': 'High', 
                'Low': 'Low', 
                'Close': 'Close', 
                'Adj Close': 'Adj Close', 
                'Volume': 'Volume'
            })
            
            # Select and order columns
            new_data_processed = new_data_processed[[col for col in columns_order if col in new_data_processed.columns]]
            
            # Combine existing and new data
            combined_data = pd.concat([existing_data, new_data_processed], ignore_index=True)
            
            # Remove duplicate dates, keeping the last entry
            combined_data = combined_data.drop_duplicates(subset='Date', keep='last')
            
            # Sort by date
            combined_data = combined_data.sort_values('Date')
            
            # Only save if there are actually new data points
            if len(combined_data) > len(existing_data):
                # Save data to local file using tab separator
                combined_data.to_csv(data_path, sep='\t', index=False)
                logging.info(f"Data for {ticker} updated successfully")
                return combined_data
            else:
                logging.info(f"No new data to update for {ticker}")
                return existing_data
                
        except Exception as e:
            logging.error(f"Error updating data for {ticker}: {e}")
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
            plt.ioff()  # Turn off interactive mode
            data = self._load_stock_data(ticker)
            
            if data is None or data.empty:
                print(f"No data available for {ticker}")
                return
            
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data[column], label=f'{ticker} {column} Price')
            plt.title(title or f'{ticker} Stock Price - {column}')
            plt.xlabel('Date')
            plt.ylabel(f'{column} Price')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Ensure the directory exists
            os.makedirs(self.plot_save_path, exist_ok=True)
            save_path = os.path.join(self.plot_save_path, f'{ticker}_{column}_plot.png')
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            
            plt.close('all')  # Close all figures to prevent memory leaks
        except Exception as e:
            print(f"Error visualizing data for {ticker}: {e}")
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
                       column: str = 'Close') -> pd.Series:
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
            # Load stock data
            stock_data = self._load_stock_data(ticker)
            
            # Set Date as index for resampling
            stock_data.set_index('Date', inplace=True)
            
            # Resample data
            resampled_data = stock_data[column].resample(resample_freq).last()
            
            return resampled_data
        
        except Exception as e:
            logging.error(f"Error resampling data for {ticker}: {e}")
            return pd.Series()

    def visualize_daily_vs_weekly(self, ticker: str, column: str = 'Close') -> None:
        """
        Visualize daily and weekly stock prices for a given ticker
        
        Args:
            ticker (str): Stock ticker symbol
            column (str, optional): Price column to visualize. Defaults to 'Close'.
        """
        try:
            # Load daily and weekly data
            daily_data = self._load_stock_data(ticker)
            weekly_data = self.resample_data(ticker, resample_freq='W', column=column)
            
            # Ensure datetime indices
            daily_data['Date'] = pd.to_datetime(daily_data['Date'])
            daily_data.set_index('Date', inplace=True)
            weekly_data.index = pd.to_datetime(weekly_data.index)
            
            # Create figure with three subplots
            fig, (ax3, ax2, ax1) = plt.subplots(1, 3, figsize=(30, 7))
            
            # Plot daily data for recent year
            recent_year_data = daily_data[daily_data.index > daily_data.index.max() - pd.Timedelta(days=365)]
            ax1.semilogy(recent_year_data.index, recent_year_data[column])
            ax1.set_title(f'{ticker} Recent Year Daily {column} Prices')
            ax1.set_xlabel('Date')
            ax1.set_ylabel(f'{column} Price ($)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, which='both', ls='-', alpha=0.5)
            
            # Plot weekly data for recent 5 years
            recent_5_years_data = weekly_data[weekly_data.index > weekly_data.index.max() - pd.Timedelta(days=1825)]
            ax2.semilogy(recent_5_years_data.index, recent_5_years_data.values)
            ax2.set_title(f'{ticker} Recent 5 Years Weekly {column} Prices')
            ax2.set_xlabel('Date')
            ax2.set_ylabel(f'{column} Price ($)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, which='both', ls='-', alpha=0.5)
            
            # Plot monthly data
            monthly_data = self.resample_data(ticker, resample_freq='ME', column=column)
            monthly_data.index = pd.to_datetime(monthly_data.index, utc=True)
            ax3.semilogy(monthly_data.index, monthly_data.values)
            ax3.set_title(f'{ticker} Monthly {column} Prices')
            ax3.set_xlabel('Date')
            ax3.set_ylabel(f'{column} Price ($)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, which='both', ls='-', alpha=0.5)
            
            # Adjust layout and save figure
            plt.tight_layout()
            
            # Save with a single consistent filename pattern
            plt.savefig(os.path.join(self.plot_save_path, f'{ticker}_daily_weekly_monthly.png'), dpi=300)
            
            plt.close(fig)
            
            logging.info(f"Generated visualization for {ticker}")
        
        except Exception as e:
            logging.error(f"Error visualizing data for {ticker}: {e}")

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

class StockDataGUI:
    """GUI for Stock Data Manager"""
    
    def __init__(self, root, manager):
        """Initialize the GUI"""
        self.root = root
        self.root.title("Stock Data Manager")
        self.root.geometry("800x600")
        self.manager = manager
        
        # Get all ticker lists
        self.ticker_lists = self._get_ticker_lists()
        self.current_tickers = []
        self.watch_list = []  # Initialize watch list
        
        # Load watch list from ticker_lists.py if it exists
        try:
            import ticker_lists
            if hasattr(ticker_lists, 'watch_list'):
                self.watch_list = ticker_lists.watch_list.copy()
                logging.info(f"Loaded {len(self.watch_list)} tickers from watch list")
        except Exception as e:
            logging.error(f"Error loading watch list: {e}")
        
        self._create_widgets()
        
    def _get_ticker_lists(self):
        """Get all ticker lists from ticker_lists module"""
        ticker_lists = {}
        current_module = sys.modules['ticker_lists']
        
        for name in dir(current_module):
            obj = getattr(current_module, name)
            # Find lists that contain 'ticker' or 'stock' in their name and are actually lists
            if (isinstance(obj, list) and 
                ('ticker' in name.lower() or 'stock' in name.lower()) and 
                len(obj) > 0 and 
                isinstance(obj[0], str)):
                ticker_lists[name] = obj
        
        return ticker_lists
    
    def _create_widgets(self):
        """Create all GUI widgets"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create top frame for ticker list selection
        top_frame = ttk.Frame(main_frame, padding="10")
        top_frame.pack(fill=tk.X, pady=5)
        
        # Ticker list selection with filter
        ttk.Label(top_frame, text="Ticker List:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Create a frame for the dropdown and its filter
        dropdown_frame = ttk.Frame(top_frame)
        dropdown_frame.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Add filter entry for ticker list dropdown
        self.list_filter_var = tk.StringVar()
        list_filter_entry = ttk.Entry(dropdown_frame, textvariable=self.list_filter_var, width=20)
        list_filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        list_filter_entry.bind("<KeyRelease>", self._filter_ticker_lists)
        
        # Create the combobox for ticker lists
        self.ticker_list_var = tk.StringVar()
        self.ticker_list_combo = ttk.Combobox(dropdown_frame, textvariable=self.ticker_list_var, 
                                        values=list(self.ticker_lists.keys()), width=60)
        self.ticker_list_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        self.ticker_list_combo.bind("<<ComboboxSelected>>", self._on_list_selected)
        
        ttk.Button(top_frame, text="Load List", command=self._load_ticker_list).grid(row=0, column=2, padx=5, pady=5)
        
        # Add manual ticker entry
        ttk.Label(top_frame, text="Add Ticker:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.manual_ticker_var = tk.StringVar()
        manual_ticker_entry = ttk.Entry(top_frame, textvariable=self.manual_ticker_var, width=80)
        manual_ticker_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(top_frame, text="Add", command=self._add_manual_ticker).grid(row=1, column=2, padx=5, pady=5)
        
        # Add list name entry and save button
        ttk.Label(top_frame, text="New List Name:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.list_name_var = tk.StringVar()
        list_name_entry = ttk.Entry(top_frame, textvariable=self.list_name_var, width=80)
        list_name_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(top_frame, text="Save List", command=self._save_ticker_list).grid(row=2, column=2, padx=5, pady=5)
        
        # Create middle frame with three sections: available tickers, watch list, and chart display
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Left section for available tickers (limited width)
        left_frame = ttk.LabelFrame(middle_frame, text="Available Tickers", padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # Add filter entry for ticker list
        filter_frame = ttk.Frame(left_frame)
        filter_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar()
        filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var, width=8)
        filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Bind filter entry to update the list as user types
        self.filter_var.trace_add("write", self._apply_ticker_filter)
        
        # Create ticker listbox with scrollbar
        ticker_frame = ttk.Frame(left_frame)
        ticker_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(ticker_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Limit width to 5 letters (approximately 40 pixels)
        self.ticker_listbox = tk.Listbox(ticker_frame, selectmode=tk.EXTENDED, height=20, width=10)
        self.ticker_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.ticker_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.ticker_listbox.yview)
        
        # Middle section for watch list (limited width)
        middle_list_frame = ttk.LabelFrame(middle_frame, text="Watch List", padding="5")
        middle_list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # Create watch list listbox with scrollbar
        watch_frame = ttk.Frame(middle_list_frame)
        watch_frame.pack(fill=tk.BOTH, expand=True)
        
        watch_scrollbar = ttk.Scrollbar(watch_frame)
        watch_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Limit width to 5 letters (approximately 40 pixels)
        self.watch_listbox = tk.Listbox(watch_frame, selectmode=tk.EXTENDED, height=20, width=10)
        self.watch_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right section for chart display (takes remaining space)
        self.chart_frame = ttk.LabelFrame(middle_frame, text="Chart Display", padding="5")
        self.chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 0))
        
        # Create a label to display chart
        self.chart_label = ttk.Label(self.chart_frame)
        self.chart_label.pack(fill=tk.BOTH, expand=True)
        
        self.watch_listbox.config(yscrollcommand=watch_scrollbar.set)
        watch_scrollbar.config(command=self.watch_listbox.yview)
        
        # Populate watch list listbox with loaded watch list
        for ticker in self.watch_list:
            self.watch_listbox.insert(tk.END, ticker)
        
        # Create right-click context menu for ticker listbox
        self.ticker_context_menu = tk.Menu(self.ticker_listbox, tearoff=0)
        self.ticker_context_menu.add_command(label="Copy to Watch List", command=self._copy_to_watch_list)
        
        # Create right-click context menu for watch list
        self.watch_context_menu = tk.Menu(self.watch_listbox, tearoff=0)
        self.watch_context_menu.add_command(label="Delete from Watch List", command=self._delete_from_watch_list)
        
        # Bind right-click events
        self.ticker_listbox.bind("<Button-3>", self._show_ticker_context_menu)
        self.watch_listbox.bind("<Button-3>", self._show_watch_context_menu)
        
        # Bind selection events to display charts
        self.ticker_listbox.bind("<<ListboxSelect>>", self._on_ticker_selected)
        self.watch_listbox.bind("<<ListboxSelect>>", self._on_watch_ticker_selected)
        
        # Create bottom frame for actions
        bottom_frame = ttk.Frame(main_frame, padding="10")
        bottom_frame.pack(fill=tk.X, pady=5)
        
        # Force download toggle
        self.force_download_var = tk.BooleanVar(value=False)
        force_download_check = ttk.Checkbutton(bottom_frame, text="Force Download", variable=self.force_download_var)
        force_download_check.pack(side=tk.RIGHT, padx=5)
        ttk.Label(bottom_frame, text="Options:").pack(side=tk.RIGHT, padx=5)
        
        # Action buttons
        ttk.Button(bottom_frame, text="Download/Update Data", command=self._download_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Visualize Daily/Weekly/Monthly", command=self._visualize_all_timeframes).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="View HTML Report", command=self._view_html_report).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _on_list_selected(self, event):
        """Handle ticker list selection and auto-load the selected list"""
        selected_list = self.ticker_list_var.get()
        if selected_list in self.ticker_lists:
            self.status_var.set(f"Selected list: {selected_list} with {len(self.ticker_lists[selected_list])} tickers")
            # Auto-load the selected ticker list
            self._load_ticker_list()
    
    def _filter_ticker_lists(self, event):
        """Filter the ticker list dropdown based on filter text"""
        filter_text = self.list_filter_var.get().strip().upper()
        
        # Get all available ticker lists
        all_lists = list(self.ticker_lists.keys())
        
        # Apply filter
        if filter_text:
            filtered_lists = [lst for lst in all_lists if filter_text in lst.upper()]
            self.ticker_list_combo['values'] = filtered_lists
            
            # If we have exactly one match, select it
            if len(filtered_lists) == 1:
                self.ticker_list_var.set(filtered_lists[0])
                self._on_list_selected(None)  # Trigger list selection event
        else:
            # Reset to show all lists
            self.ticker_list_combo['values'] = all_lists
        
        # Update status
        if filter_text:
            self.status_var.set(f"List filter: '{filter_text}' - {len(self.ticker_list_combo['values'])} matches")
    
    def _apply_ticker_filter(self, *args):
        """Filter the ticker list based on filter text"""
        filter_text = self.filter_var.get().strip().upper()
        
        # If no current tickers or no filter, don't do anything
        if not hasattr(self, 'current_tickers') or not self.current_tickers:
            return
            
        # Get the currently selected list
        selected_list = self.ticker_list_var.get()
        if not selected_list or selected_list not in self.ticker_lists:
            return
            
        # Get the full list of tickers
        tickers = self.current_tickers
        
        # Clear the listbox
        self.ticker_listbox.delete(0, tk.END)
        
        # Apply filter and update listbox
        filtered_count = 0
        for ticker in tickers:
            # Apply filter
            if filter_text and filter_text not in ticker.upper():
                continue
                
            # Add ticker to listbox
            if 'tickers_comment_dict' in globals() and ticker in tickers_comment_dict:
                self.ticker_listbox.insert(tk.END, f"{ticker} - {tickers_comment_dict[ticker]}")
            else:
                self.ticker_listbox.insert(tk.END, ticker)
            filtered_count += 1
        
        # Update status
        if filter_text:
            self.status_var.set(f"Filter '{filter_text}': showing {filtered_count}/{len(tickers)} tickers from {selected_list}")
        else:
            self.status_var.set(f"Showing all {len(tickers)} tickers from {selected_list}")
    
    def _load_ticker_list(self):
        """Load selected ticker list into listbox"""
        selected_list = self.ticker_list_var.get()
        if not selected_list:
            messagebox.showwarning("No List Selected", "Please select a ticker list first.")
            return
        
        if selected_list in self.ticker_lists:
            tickers = self.ticker_lists[selected_list]
            self.current_tickers = tickers
            
            # Reset filter when loading a new list
            if hasattr(self, 'filter_var'):
                self.filter_var.set('')
            
            # Update listbox
            self.ticker_listbox.delete(0, tk.END)
            for ticker in tickers:
                # Check if we have a comment for this ticker
                if 'tickers_comment_dict' in globals() and ticker in tickers_comment_dict:
                    self.ticker_listbox.insert(tk.END, f"{ticker} - {tickers_comment_dict[ticker]}")
                else:
                    self.ticker_listbox.insert(tk.END, ticker)
            
            self.status_var.set(f"Loaded {len(tickers)} tickers from {selected_list}")
    
    def _add_manual_ticker(self):
        """Add manually entered ticker(s)"""
        ticker_input = self.manual_ticker_var.get().strip()
        if not ticker_input:
            return
        
        # Remove brackets and split by commas
        ticker_input = ticker_input.replace('[', '').replace(']', '')
        tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
        
        if not tickers:
            return
        
        added_count = 0
        for ticker in tickers:
            # Remove any quotes around the ticker
            ticker = ticker.strip('\'"')
            
            # Skip empty tickers
            if not ticker:
                continue
                
            # Add to current tickers if not already present
            if ticker not in self.current_tickers:
                self.current_tickers.append(ticker)
                self.ticker_listbox.insert(tk.END, ticker)
                added_count += 1
        
        if added_count == 1:
            self.status_var.set(f"Added ticker: {tickers[0]}")
        else:
            self.status_var.set(f"Added {added_count} tickers")
        
        # Clear entry field
        self.manual_ticker_var.set("")
    
    def _save_ticker_list(self):
        """Save current tickers as a new list in ticker_lists.py"""
        list_name = self.list_name_var.get().strip()
        if not list_name:
            messagebox.showwarning("No List Name", "Please enter a name for the ticker list.")
            return
        
        if not self.current_tickers:
            messagebox.showwarning("No Tickers", "Please add tickers to the list before saving.")
            return
        
        # Format list name to be a valid Python variable name
        list_name = list_name.replace(" ", "_").replace("-", "_")
        if not list_name[0].isalpha() and list_name[0] != '_':
            list_name = "ticker_" + list_name
        
        # Create Python code for the new list
        tickers_str = ", ".join([f"\"{ticker}\"" for ticker in self.current_tickers])
        new_list_code = f"\n{list_name}_stocks = [{tickers_str}]\n"
        
        try:
            # Read the current content of ticker_lists.py
            ticker_lists_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_lists.py")
            with open(ticker_lists_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Find the position of the first function definition
            function_pattern = re.compile(r'\n# Function to')
            match = function_pattern.search(content)
            
            if match:
                # Insert the new list before the function definition
                insert_position = match.start()
                new_content = content[:insert_position] + new_list_code + content[insert_position:]
                
                # Write the modified content back to the file
                with open(ticker_lists_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
            else:
                # If no function definition found, append to the end of the file
                with open(ticker_lists_path, "a", encoding="utf-8") as f:
                    f.write(new_list_code)
            
            # Update the ticker lists dictionary
            self.ticker_lists[list_name + "_stocks"] = self.current_tickers
            self.ticker_list_var.set(list_name + "_stocks")
            
            # Update the dropdown menu
            self.ticker_list_dropdown['values'] = list(self.ticker_lists.keys())
            
            self.status_var.set(f"Saved {len(self.current_tickers)} tickers as '{list_name}_stocks'")
            messagebox.showinfo("List Saved", f"Ticker list saved as '{list_name}_stocks' in ticker_lists.py")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving ticker list: {str(e)}")
            logging.error(f"Error saving ticker list: {e}")
    
    def _show_ticker_context_menu(self, event):
        """Show context menu on right-click in ticker listbox"""
        # Only show context menu if there are selected items
        if self.ticker_listbox.curselection():
            try:
                self.ticker_context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.ticker_context_menu.grab_release()
                
    def _show_watch_context_menu(self, event):
        """Show context menu on right-click in watch list"""
        # Only show context menu if there are selected items
        if self.watch_listbox.curselection():
            try:
                self.watch_context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.watch_context_menu.grab_release()
                
    def _delete_from_watch_list(self):
        """Delete selected tickers from watch list"""
        selected_indices = self.watch_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select at least one ticker to delete.")
            return
            
        # Get selected tickers
        selected_tickers = [self.watch_listbox.get(i) for i in selected_indices]
        
        # Confirm deletion
        if len(selected_tickers) == 1:
            confirm = messagebox.askyesno("Confirm Delete", f"Delete {selected_tickers[0]} from watch list?")
        else:
            confirm = messagebox.askyesno("Confirm Delete", f"Delete {len(selected_tickers)} tickers from watch list?")
            
        if not confirm:
            return
            
        # Delete from watch list (in reverse order to maintain correct indices)
        for i in sorted(selected_indices, reverse=True):
            ticker = self.watch_listbox.get(i)
            self.watch_listbox.delete(i)
            if ticker in self.watch_list:
                self.watch_list.remove(ticker)
                
        # Save the updated watch list
        self._save_watch_list()
        
        if len(selected_tickers) == 1:
            self.status_var.set(f"Deleted {selected_tickers[0]} from watch list")
        else:
            self.status_var.set(f"Deleted {len(selected_tickers)} tickers from watch list")
    
    def _copy_to_watch_list(self):
        """Copy selected tickers to watch list and save to ticker_lists.py"""
        selected_tickers = self._get_selected_tickers()
        if not selected_tickers:
            return
            
        # Add selected tickers to watch list if not already present
        added_count = 0
        for ticker in selected_tickers:
            if ticker not in self.watch_list:
                self.watch_list.append(ticker)
                self.watch_listbox.insert(tk.END, ticker)
                added_count += 1
                
        if added_count > 0:
            # Save the updated watch list to ticker_lists.py
            self._save_watch_list()
            
            if added_count == 1:
                self.status_var.set(f"Added {selected_tickers[0]} to watch list and saved")
            else:
                self.status_var.set(f"Added {added_count} tickers to watch list and saved")
        else:
            self.status_var.set("All selected tickers already in watch list")
    
    def _save_watch_list(self):
        """Save the watch list to ticker_lists.py"""
        if not self.watch_list:
            return
            
        # Use 'watch_list' as the name for the list in ticker_lists.py
        list_name = "watch_list"
        
        # Create Python code for the watch list
        tickers_str = ", ".join([f"\"{ticker}\"" for ticker in self.watch_list])
        new_list_code = f"\n{list_name} = [{tickers_str}]\n"
        
        try:
            # Read the current content of ticker_lists.py
            ticker_lists_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_lists.py")
            with open(ticker_lists_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Check if watch_list already exists in the file
            watch_list_pattern = re.compile(r'\nwatch_list\s*=\s*\[.*?\]', re.DOTALL)
            match = watch_list_pattern.search(content)
            
            if match:
                # Replace the existing watch_list
                new_content = content[:match.start()] + new_list_code + content[match.end():]
            else:
                # Find the position of the first function definition
                function_pattern = re.compile(r'\n# Function to')
                func_match = function_pattern.search(content)
                
                if func_match:
                    # Insert the watch list before the function definition
                    insert_position = func_match.start()
                    new_content = content[:insert_position] + new_list_code + content[insert_position:]
                else:
                    # If no function definition found, append to the end of the file
                    new_content = content + new_list_code
            
            # Write the modified content back to the file
            with open(ticker_lists_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            # Update the ticker lists dictionary if it's not already there
            if list_name not in self.ticker_lists:
                self.ticker_lists[list_name] = self.watch_list
                # Update the dropdown menu
                self.ticker_list_dropdown['values'] = list(self.ticker_lists.keys())
            else:
                # Just update the existing entry
                self.ticker_lists[list_name] = self.watch_list
                
            logging.info(f"Saved watch list with {len(self.watch_list)} tickers to ticker_lists.py")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving watch list: {str(e)}")
            logging.error(f"Error saving watch list: {e}")
    
    def _get_selected_tickers(self):
        """Get selected tickers from listbox"""
        selected_indices = self.ticker_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select at least one ticker.")
            return []
        
        selected_tickers = []
        for i in selected_indices:
            # Extract ticker symbol (it might include a comment after a dash)
            ticker_text = self.ticker_listbox.get(i)
            ticker = ticker_text.split(' - ')[0].strip()
            selected_tickers.append(ticker)
        
        return selected_tickers
    
    def _download_data(self):
        """Download or update data for selected tickers"""
        selected_tickers = self._get_selected_tickers()
        if not selected_tickers:
            return
        
        # Get force download setting
        force_download = self.force_download_var.get()
        mode_text = "force downloading" if force_download else "updating"
        
        self.status_var.set(f"{mode_text.capitalize()} data for {len(selected_tickers)} tickers...")
        self.root.update_idletasks()
        
        success_count = 0
        for ticker in selected_tickers:
            try:
                data = self.manager.update_data(ticker, force_download=force_download)
                if data is not None and not data.empty:
                    success_count += 1
                    self.status_var.set(f"{mode_text.capitalize()} data for {ticker} ({success_count}/{len(selected_tickers)})")
                else:
                    self.status_var.set(f"No data available for {ticker}")
                self.root.update_idletasks()
            except Exception as e:
                messagebox.showerror("Error", f"Error {mode_text} data for {ticker}: {str(e)}")
        
        self.status_var.set(f"Completed: {mode_text.capitalize()} data for {success_count}/{len(selected_tickers)} tickers")
    
    def _visualize_daily_weekly(self):
        """Visualize daily vs weekly charts for selected tickers"""
        selected_tickers = self._get_selected_tickers()
        if not selected_tickers:
            return
        
        for ticker in selected_tickers:
            try:
                self.status_var.set(f"Visualizing daily vs weekly for {ticker}...")
                self.root.update_idletasks()
                
                self.manager.visualize_daily_vs_weekly(ticker)
                
                # Open the saved chart in the default web browser
                chart_path = os.path.join(self.manager.plot_save_path, f"{ticker}_daily_vs_weekly_price.png")
                if os.path.exists(chart_path):
                    webbrowser.open(f"file:///{os.path.abspath(chart_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error visualizing {ticker}: {str(e)}")
        
        self.status_var.set(f"Completed visualization for {len(selected_tickers)} tickers")
    
    def _visualize_all_timeframes(self):
        """Visualize daily, weekly, and monthly charts for selected tickers"""
        selected_tickers = self._get_selected_tickers()
        if not selected_tickers:
            return
        
        for ticker in selected_tickers:
            try:
                self.status_var.set(f"Visualizing all timeframes for {ticker}...")
                self.root.update_idletasks()
                
                # Use the existing visualize_daily_vs_weekly method which already shows daily, weekly, and monthly data
                self.manager.visualize_daily_vs_weekly(ticker)
                
                # Open the saved chart in the default web browser
                chart_path = os.path.join(self.manager.plot_save_path, f"{ticker}_daily_vs_weekly_price.png")
                if os.path.exists(chart_path):
                    webbrowser.open(f"file:///{os.path.abspath(chart_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error visualizing {ticker}: {str(e)}")
        
        self.status_var.set(f"Completed visualization for {len(selected_tickers)} tickers")
    
    def _on_ticker_selected(self, event):
        """Handle ticker selection from available tickers list"""
        selected_indices = self.ticker_listbox.curselection()
        if not selected_indices:
            return
            
        # Get the selected ticker
        ticker_text = self.ticker_listbox.get(selected_indices[0])
        ticker = ticker_text.split(' - ')[0].strip()
        
        # Display chart for the selected ticker
        self._display_chart(ticker)
    
    def _on_watch_ticker_selected(self, event):
        """Handle ticker selection from watch list"""
        selected_indices = self.watch_listbox.curselection()
        if not selected_indices:
            return
            
        # Get the selected ticker
        ticker = self.watch_listbox.get(selected_indices[0])
        
        # Display chart for the selected ticker
        self._display_chart(ticker)
    
    def _display_chart(self, ticker):
        """Display chart for the selected ticker"""
        try:
            # Check if data exists for this ticker
            data_path = self.manager._get_data_path(ticker)
            if not os.path.exists(data_path):
                # Download data if it doesn't exist
                self.status_var.set(f"Downloading data for {ticker}...")
                self.root.update_idletasks()
                self.manager.update_data(ticker, force_download=True)
            
            # Generate or update chart if needed
            plots_dir = self.manager.plot_save_path
            os.makedirs(plots_dir, exist_ok=True)
            
            timeframe_plot_path = os.path.join(plots_dir, f"{ticker}_daily_weekly_monthly.png")
            chart_outdated = False
            
            # If chart doesn't exist, it needs to be generated
            if not os.path.exists(timeframe_plot_path):
                chart_outdated = True
            # If chart exists, check if data file is newer than chart file
            elif os.path.exists(data_path):
                chart_mod_time = os.path.getmtime(timeframe_plot_path)
                data_mod_time = os.path.getmtime(data_path)
                
                # If data file is newer, chart is outdated
                if data_mod_time > chart_mod_time:
                    chart_outdated = True
            
            # Generate chart if needed
            if chart_outdated:
                self.status_var.set(f"Generating chart for {ticker}...")
                self.root.update_idletasks()
                self.manager.visualize_daily_vs_weekly(ticker)
            
            # Display the chart in the chart_label
            if os.path.exists(timeframe_plot_path):
                # Load and resize the image
                img = Image.open(timeframe_plot_path)
                
                # Get the chart frame size
                chart_width = self.chart_frame.winfo_width()
                chart_height = self.chart_frame.winfo_height()
                
                # If the frame hasn't been rendered yet, use default size
                if chart_width <= 1:
                    chart_width = 800
                if chart_height <= 1:
                    chart_height = 600
                
                # Resize image to fit the frame while maintaining aspect ratio
                img_width, img_height = img.size
                aspect_ratio = img_width / img_height
                
                if chart_width / chart_height > aspect_ratio:
                    # Frame is wider than image
                    new_height = chart_height
                    new_width = int(new_height * aspect_ratio)
                else:
                    # Frame is taller than image
                    new_width = chart_width
                    new_height = int(new_width / aspect_ratio)
                
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Convert to PhotoImage and display
                photo = ImageTk.PhotoImage(img)
                self.chart_label.config(image=photo)
                self.chart_label.image = photo  # Keep a reference to prevent garbage collection
                
                self.status_var.set(f"Displaying chart for {ticker}")
            else:
                self.status_var.set(f"Error: Chart for {ticker} not found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying chart for {ticker}: {str(e)}")
            self.status_var.set(f"Error displaying chart for {ticker}")
    
    def cleanup(self):
        """Clean up resources before application exit"""
        try:
            # First, clear any image references which often cause issues
            if hasattr(self, 'chart_label') and hasattr(self.chart_label, 'image'):
                self.chart_label.image = None
            
            # Clear listbox contents
            if hasattr(self, 'ticker_listbox'):
                self.ticker_listbox.delete(0, tk.END)
            if hasattr(self, 'watch_listbox'):
                self.watch_listbox.delete(0, tk.END)
                
            # Destroy all widgets explicitly to prevent reference cycles
            for widget in self.root.winfo_children():
                if widget.winfo_exists():
                    widget.destroy()
            
            # Set Tkinter variables to None instead of deleting them
            # This helps prevent 'main thread is not in main loop' errors
            if hasattr(self, 'status_var'):
                self.status_var.set('')
                self.status_var = None
            if hasattr(self, 'ticker_list_var'):
                self.ticker_list_var.set('')
                self.ticker_list_var = None
            if hasattr(self, 'force_download_var'):
                self.force_download_var.set(False)
                self.force_download_var = None
            if hasattr(self, 'manual_ticker_var'):
                self.manual_ticker_var.set('')
                self.manual_ticker_var = None
            if hasattr(self, 'list_name_var'):
                self.list_name_var.set('')
                self.list_name_var = None
                
            # Clear other references
            if hasattr(self, 'ticker_context_menu'):
                self.ticker_context_menu = None
            if hasattr(self, 'watch_context_menu'):
                self.watch_context_menu = None
                
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            # Don't re-raise the exception as we're already in cleanup
    
    def _view_html_report(self):
        """Generate and view HTML report for the current ticker list"""
        selected_tickers = self._get_selected_tickers()
        if not selected_tickers:
            messagebox.showwarning("No Tickers Selected", "Please select at least one ticker from the list.")
            return
            
        try:
            # Create plots directory if it doesn't exist
            plots_dir = self.manager.plot_save_path
            os.makedirs(plots_dir, exist_ok=True)
            
            # Check for missing data and download automatically
            missing_tickers = []
            for ticker in selected_tickers:
                data_path = self.manager._get_data_path(ticker)
                if not os.path.exists(data_path):
                    missing_tickers.append(ticker)
            
            # If there are missing tickers, download their data automatically
            if missing_tickers:
                self.status_var.set(f"Downloading missing data for {len(missing_tickers)} tickers...")
                self.root.update_idletasks()
                
                # Download data for missing tickers with force download enabled
                for ticker in missing_tickers:
                    self.status_var.set(f"Downloading data for {ticker}...")
                    self.root.update_idletasks()
                    self.manager.update_data(ticker, force_download=True)
                
                self.status_var.set(f"Downloaded data for {len(missing_tickers)} tickers")
                self.root.update_idletasks()
                
            # Check for missing or outdated visualizations and generate them
            for ticker in selected_tickers:
                timeframe_plot_path = os.path.join(plots_dir, f"{ticker}_daily_weekly_monthly.png")
                data_path = self.manager._get_data_path(ticker)
                
                # Check if chart needs to be generated or updated
                chart_outdated = False
                
                # If chart doesn't exist, it needs to be generated
                if not os.path.exists(timeframe_plot_path):
                    chart_outdated = True
                # If chart exists, check if data file is newer than chart file
                elif os.path.exists(data_path):
                    chart_mod_time = os.path.getmtime(timeframe_plot_path)
                    data_mod_time = os.path.getmtime(data_path)
                    
                    # If data file is newer, chart is outdated
                    if data_mod_time > chart_mod_time:
                        chart_outdated = True
                        self.status_var.set(f"Chart for {ticker} is outdated. Regenerating...")
                        self.root.update_idletasks()
                
                # Generate chart if needed
                if chart_outdated:
                    self.status_var.set(f"Generating visualizations for {ticker}...")
                    self.root.update_idletasks()
                    self.manager.visualize_daily_vs_weekly(ticker)
            
            # Get the current ticker list name
            current_list_name = self.ticker_list_var.get() or "custom_list"
            
            # Get current date in YYYY-MM-DD format
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Create a filename with list name and date
            report_filename = f"stock_analysis_{current_list_name}_{current_date}.html"
            
            # Generate HTML report with custom filename and selected tickers
            self.status_var.set(f"Generating HTML report for {current_list_name}...")
            self.root.update_idletasks()
            report_path = self.manager.generate_html_report(plots_dir, report_filename, selected_tickers)
            
            # Open the HTML report in Microsoft Edge
            if os.path.exists(report_path):
                try:
                    # Register and use Microsoft Edge
                    webbrowser.register('edge', None, webbrowser.BackgroundBrowser(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'))
                    webbrowser.get('edge').open(f"file:///{os.path.abspath(report_path)}")
                    self.status_var.set(f"HTML report for {current_list_name} opened in Edge browser")
                except Exception as browser_error:
                    # Fall back to default browser if Edge registration fails
                    logging.warning(f"Could not open Edge browser: {browser_error}. Using default browser.")
                    webbrowser.open(f"file:///{os.path.abspath(report_path)}")
                    self.status_var.set(f"HTML report for {current_list_name} opened in default browser")
            else:
                self.status_var.set("Error: HTML report not found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error generating HTML report: {str(e)}")
            self.status_var.set("Error generating HTML report")
    
    def _get_selected_tickers(self):
        """Get selected tickers from listbox"""
        selected_indices = self.ticker_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select at least one ticker.")
            return []
        
        selected_tickers = []
        for i in selected_indices:
            # Extract ticker symbol (it might include a comment after a dash)
            ticker_text = self.ticker_listbox.get(i)
            ticker = ticker_text.split(' - ')[0].strip()
            selected_tickers.append(ticker)
        
        return selected_tickers
    
    def _download_data(self):
        """Download or update data for selected tickers"""
        selected_tickers = self._get_selected_tickers()
        if not selected_tickers:
            return
            
        # Get force download setting
        force_download = self.force_download_var.get()
        mode_text = "force downloading" if force_download else "updating"
            
        self.status_var.set(f"{mode_text.capitalize()} data for {len(selected_tickers)} tickers...")
        self.root.update_idletasks()
        
        success_count = 0
        for ticker in selected_tickers:
            try:
                data = self.manager.update_data(ticker, force_download=force_download)
                if data is not None and not data.empty:
                    success_count += 1
                    self.status_var.set(f"{mode_text.capitalize()} data for {ticker} ({success_count}/{len(selected_tickers)})")
                else:
                    self.status_var.set(f"No data available for {ticker}")
                self.root.update_idletasks()
            except Exception as e:
                messagebox.showerror("Error", f"Error {mode_text} data for {ticker}: {str(e)}")
        
        self.status_var.set(f"Completed: {mode_text.capitalize()} data for {success_count}/{len(selected_tickers)} tickers")
def main():
    """Main function to launch the Stock Data Manager GUI."""
    root = tk.Tk()
    root.title("Stock Data Manager")
    
    # Maximize the window
    root.state('zoomed')  # Windows-specific command to maximize

    # Create the StockDataManager instance
    manager = StockDataManager()
    
    # Create the application
    app = StockDataGUI(root, manager)
    
    # Define the on_closing handler
    def on_closing():
        try:
            print("Cleaning up resources...")
            app.cleanup()
            root.destroy()
        except Exception as e:
            print(f"Error during application shutdown: {str(e)}")
    
    # Set the protocol handler
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the main loop
    root.mainloop()
    


if __name__ == '__main__':
    main()