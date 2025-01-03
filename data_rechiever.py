"""
Stock Data Retriever and Manager

This script provides functionality for downloading, updating, and visualizing stock data.

CHANGELOG:
---------
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
Last Updated: 2025-01-02
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import logging
import numpy as np
from typing import Optional, List, Any

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
    """
    
    def __init__(self, data_dir: str = STOCK_DATA_DIR):
        """
        Initialize the StockDataManager.
        
        Args:
            data_dir (str, optional): Directory to store stock data. Defaults to STOCK_DATA_DIR.
        """
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
    
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
            
            # Convert Date column to datetime
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            
            # Convert numeric columns to float
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for col in numeric_columns:
                if col in stock_data.columns:
                    stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
            
            return stock_data
        
        except Exception as e:
            logging.error(f"Error loading data for {ticker}: {e}")
            return pd.DataFrame()
    
    def initial_download(self, 
                         ticker: str, 
                         start_date: Optional[str] = '1980-01-01', 
                         end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Download initial stock data for a given ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (Optional[str], optional): Start date for data retrieval. Defaults to '1980-01-01'.
            end_date (Optional[str], optional): End date for data retrieval. Defaults to today.
        
        Returns:
            Optional[pd.DataFrame]: Downloaded stock data or None if download fails
        """
        try:
            # Validate and prepare parameters
            ticker = ticker.upper()
            
            # Set end date to today if not specified
            if end_date is None:
                end_date = datetime.today()
            
            # Download stock data
            stock_data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date,
                progress=False
            )
            
            # Validate downloaded data
            if stock_data.empty:
                logging.warning(f"No data downloaded for {ticker}")
                return None
            
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
            logging.info(f"Initial data for {ticker} saved to {data_path}")
            
            return stock_data
        
        except Exception as e:
            logging.error(f"Error in initial download for {ticker}: {e}")
            return None

    def update_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Update existing stock data with the most recent information.
        
        Args:
            ticker (str): Stock ticker symbol
        
        Returns:
            Optional[pd.DataFrame]: Updated stock data or None if update fails
        """
        try:
            # Validate and prepare parameters
            ticker = ticker.upper()
            data_path = self._get_data_path(ticker)
            
            # Download new data
            initial_data = yf.download(
                ticker, 
                start='1980-01-01',
                end=datetime.today(),
                progress=False
            )
            
            # Validate downloaded data
            if initial_data.empty:
                logging.warning(f"No data downloaded for {ticker}")
                return None
            
            # Reset index to make Date a column
            stock_data_reset = initial_data.reset_index()
            
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
            logging.info(f"Data for {ticker} updated successfully")
            
            return initial_data
        
        except Exception as e:
            logging.error(f"Error updating data for {ticker}: {e}")
            return None

    def visualize_data(self, 
                       ticker: str, 
                       column: str = 'Close', 
                       title: Optional[str] = None) -> None:
        """
        Create a visualization of stock data.
        
        Args:
            ticker (str): Stock ticker symbol
            column (str, optional): Column to plot. Defaults to 'Close'.
            title (Optional[str], optional): Custom plot title
        """
        try:
            # Load data
            stock_data = self._load_stock_data(ticker)
            
            # Validate data
            if stock_data.empty:
                logging.warning(f"No data found for {ticker}")
                return
            
            # If specified column is not available, use the first numeric column
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            if column not in stock_data.columns:
                if not numeric_columns:
                    logging.warning("No numeric columns available for plotting")
                    return
                column = numeric_columns[0]
                logging.info(f"Defaulting to column: {column}")
            
            # Create plot
            plt.figure(figsize=(15, 7))
            
            # Plot the data
            plt.plot(stock_data['Date'], stock_data[column])
            
            # Set title and labels
            plt.title(title or f'{ticker} Stock Prices')
            plt.xlabel('Date')
            plt.ylabel(f'{column} Price ($)')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            logging.error(f"Error visualizing data for {ticker}: {e}")

    def visualize_multiple_tickers(self, 
                              tickers: List[str], 
                              column: str = 'Close', 
                              title: Optional[str] = None) -> None:
        """
        Create a subplot visualization of stock data for multiple tickers.
        
        Args:
            tickers (List[str]): List of stock ticker symbols
            column (str, optional): Column to plot. Defaults to 'Close'.
            title (Optional[str], optional): Custom plot title
        """
        try:
            # Determine subplot layout
            n_tickers = len(tickers)
            rows = (n_tickers + 2) // 3  # Ceiling division to get rows
            cols = min(n_tickers, 3)  # Max 3 columns
            
            # Create subplot figure
            plt.figure(figsize=(15, 5 * rows))
            
            # Plot each ticker
            for idx, ticker in enumerate(tickers, 1):
                # Create subplot
                plt.subplot(rows, cols, idx)
                
                # Load data
                stock_data = self._load_stock_data(ticker)
                
                # Validate data
                if stock_data.empty:
                    logging.warning(f"No data found for {ticker}")
                    continue
                
                # If specified column is not available, use the first numeric column
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                if column not in stock_data.columns:
                    if not numeric_columns:
                        logging.warning("No numeric columns available for plotting")
                        continue
                    column = numeric_columns[0]
                    logging.info(f"Defaulting to column: {column}")
                
                # Plot the data
                plt.plot(stock_data['Date'], stock_data[column], label=ticker)
                
                # Set title and labels
                plt.title(f'{ticker} Stock {column} Prices')
                plt.xlabel('Date')
                plt.ylabel(f'{column} Price ($)')
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.legend()
            
            # Adjust layout and add overall title
            plt.tight_layout()
            if title:
                plt.suptitle(title, fontsize=16)
            
            # Show the plot
            plt.show()
        
        except Exception as e:
            logging.error(f"Error visualizing data for multiple tickers: {e}")

    def resample_data(self, 
                       ticker: str, 
                       resample_freq: str = 'W', 
                       column: str = 'Close') -> pd.DataFrame:
        """
        Resample stock data to a different frequency.
        
        Args:
            ticker (str): Stock ticker symbol
            resample_freq (str, optional): Resampling frequency. 
                Defaults to 'W' (weekly). 
                Common options:
                - 'D': Daily
                - 'W': Weekly
                - 'M': Monthly
                - 'Q': Quarterly
            column (str, optional): Column to resample. Defaults to 'Close'.
        
        Returns:
            pd.DataFrame: Resampled stock data
        """
        try:
            # Load data
            stock_data = self._load_stock_data(ticker)
            
            # Validate data
            if stock_data.empty:
                logging.warning(f"No data found for {ticker}")
                return pd.DataFrame()
            
            # Set Date as index
            stock_data.set_index('Date', inplace=True)
            
            # Resample the data
            resampled_data = stock_data[column].resample(resample_freq).last()
            
            return resampled_data
        
        except Exception as e:
            logging.error(f"Error resampling data for {ticker}: {e}")
            return pd.DataFrame()

    def visualize_daily_vs_weekly(self, 
                                  ticker: str, 
                                  column: str = 'Close', 
                                  title: Optional[str] = None) -> None:
        """
        Create a visualization of daily, weekly, and monthly stock data.
        
        Args:
            ticker (str): Stock ticker symbol
            column (str, optional): Column to plot. Defaults to 'Close'.
            title (Optional[str], optional): Custom plot title
        """
        try:
            # Load daily data
            daily_data = self._load_stock_data(ticker)
            
            # Validate data
            if daily_data.empty:
                logging.warning(f"No data found for {ticker}")
                return
            
            # Resample to weekly and monthly data
            weekly_data = self.resample_data(ticker, resample_freq='W', column=column)
            monthly_data = self.resample_data(ticker, resample_freq='M', column=column)
            
            # Create side-by-side plots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 7))
            
            # Plot daily data
            ax1.semilogy(daily_data['Date'], daily_data[column])
            ax1.set_title(f'{ticker} Daily {column} Prices')
            ax1.set_xlabel('Date')
            ax1.set_ylabel(f'{column} Price ($)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True)
            
            # Plot weekly data
            ax2.semilogy(weekly_data.index, weekly_data.values)
            ax2.set_title(f'{ticker} Weekly {column} Prices')
            ax2.set_xlabel('Date')
            ax2.set_ylabel(f'{column} Price ($)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True)
            
            # Plot monthly data
            ax3.semilogy(monthly_data.index, monthly_data.values)
            ax3.set_title(f'{ticker} Monthly {column} Prices')
            ax3.set_xlabel('Date')
            ax3.set_ylabel(f'{column} Price ($)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True)
            
            # Adjust layout and add overall title
            plt.tight_layout()
            if title:
                plt.suptitle(title, fontsize=16)
            
            # Show the plot
            plt.show()
        
        except Exception as e:
            logging.error(f"Error visualizing data for {ticker}: {e}")

def main():
    """
    Demonstrate stock data management functionalities.
    """
    # Initialize stock data manager
    stock_manager = StockDataManager()
    
    # List of tickers to process
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    # Process each ticker
    for ticker in tickers:
        # Perform initial download (if not already done)
        initial_data = stock_manager.initial_download(ticker)
        
        # Update data
        updated_data = stock_manager.update_data(ticker)
    
    # Visualize multiple tickers
    # stock_manager.visualize_multiple_tickers(tickers)
    
    # Visualize daily vs weekly for first ticker
    stock_manager.visualize_daily_vs_weekly(tickers[0])

if __name__ == '__main__':
    main()
