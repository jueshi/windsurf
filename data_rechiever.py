"""
Stock Data Retriever and Manager

This script provides functionality for downloading, updating, and visualizing stock data.

CHANGELOG:
---------
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
from typing import Optional, List

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
            # Read data
            data_path = self._get_data_path(ticker)
            stock_data = pd.read_csv(data_path, sep='\t')
            
            # Convert Date column to datetime
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            
            # Convert numeric columns to float
            numeric_columns = [
                col for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] 
                if col in stock_data.columns
            ]
            
            for col in numeric_columns:
                stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
            
            # Validate data
            if stock_data.empty:
                logging.warning(f"No data found for {ticker}")
                return
            
            # If specified column is not available, use the first numeric column
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
        
        # Visualize data
        if updated_data is not None:
            stock_manager.visualize_data(ticker)

if __name__ == '__main__':
    main()
