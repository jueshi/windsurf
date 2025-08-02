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
from ticker_lists import *
import math
import time
import json
from datetime import datetime, timedelta
from random import uniform

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
            plt.savefig(os.path.join(self.plot_save_path, f'{ticker}_{column}_daily_weekly_monthly.png'), dpi=300)
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


    def generate_html_report(self, plots_dir=None):
        """
        Generate an HTML report with embedded stock plots
        
        Args:
            plots_dir (str, optional): Directory containing plot images. Defaults to current plot_save_path.
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
            plot_files = glob.glob(os.path.join(plots_dir, '*_stock_prices.png'))
            plot_files += glob.glob(os.path.join(plots_dir, '*_daily_weekly_monthly.png'))

            # Add plots to HTML
            for plot_file in plot_files:
                # Extract ticker name from filename
                ticker = os.path.basename(plot_file).split('_')[0]
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

            # Save HTML report
            report_path = os.path.abspath(os.path.join(plots_dir, 'stock_analysis_report.html'))
            with open(report_path, 'w') as f:
                f.write(html_content)

            # Open the report in Microsoft Edge
            webbrowser.register('edge', None, webbrowser.BackgroundBrowser(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'))
            webbrowser.get('edge').open(f'file://{report_path}')

            logging.info(f"Generated stock analysis report at {report_path}")

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
        if name:
            return name
        else:
            # Try to get the variable name from the caller's locals
            try:
                # frame = inspect.currentframe().f_back
                # for var_name, var_value in frame.f_locals.items():
                for var_name, var_value in globals().items(): #ticker_list is globale variable now
                    if var_value is tickers:
                        return var_name
                else:
                    # Fallback to sorted tickers
                    return '_'.join(sorted(tickers))
            except Exception:
                # Most conservative fallback
                return '_'.join(sorted(tickers))

def main():
    r'''use AI to gennerate a pythhon list of stock tickers from content in clipboard.
    C:\Users\juesh\OneDrive\Documents\cursor\AI_ticker_extractor.py'''

    # Initialize stock manager
    stock_manager = StockDataManager()
    
    # Dynamically find all stock ticker lists
    ticker_lists = [var for var in globals() 
                    if isinstance(globals()[var], list) 
                    and (var.endswith('_stocks') or var.endswith('_tickers'))
                    and var != 'ticker_lists']
    
    print(f"Found {len(ticker_lists)} ticker lists")
    for ticker_list in ticker_lists:
        print(f"Processing {ticker_list}: {len(globals()[ticker_list])} tickers")
        # Uncomment the next line when ready to process
        # stock_manager.process_stock_data(globals()[ticker_list],force_download=True)
        
    # test_tickers = ['BRK-B','LRCX','MRVL']   
    
    # Process stock data
    # stock_manager.process_stock_data(tickers=top_sectors)
    # stock_manager.process_stock_data(tickers=recent_analyst_upgrades)
    # stock_manager.process_stock_data(tickers=ibd_50_stocks)
    # stock_manager.process_stock_data(tickers=zacks_rank_1_stocks)
    # stock_manager.process_stock_data(tickers=positive_earnings_surprise_stocks)
    # Option to force download all data
    force_download = True

    stock_manager.process_stock_data(tickers=Jues401k_stocks, force_download=force_download)
    
    # stock_manager.process_stock_data(tickers=new_highs, name='new_highs')
    # stock_manager.process_stock_data(tickers=new_lows)
    # stock_manager.process_stock_data(tickers=test_tickers)
    # stock_manager.process_stock_data(tickers=bitcoin_tickers)
    # stock_manager.process_stock_data(tickers=canslim_tickers) 
    # stock_manager.process_stock_data(tickers=chinese_stocks_tickers, force_download=force_download)
    # stock_manager.process_stock_data(tickers=daily_watch_tickers)
    # stock_manager.process_stock_data(tickers=index_tickers,force_download=True)
    


if __name__ == '__main__':
    main()