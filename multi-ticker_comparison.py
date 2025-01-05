"""
Stock Ticker Multi-Comparison Visualization

This script provides a comprehensive visualization of stock performance 
for multiple tickers across different time frames.

Changelog:
- 2025-01-05: v1.1.0
  * Added support for 1-year and 5-year stock price performance comparison
  * Implemented subplot visualization for easy comparison
  * Enhanced data preprocessing and normalization
  * Improved error handling and data validation

- 2025-01-03: v1.0.0
  * Initial implementation of stock price comparison script
  * Basic data loading and visualization functionality

Dependencies:
- pandas
- matplotlib
- data_rechiever module

Usage:
Run the script to generate a comparative plot of stock performances.
Customize the stock_tickers list to compare different stocks.
"""

import pandas as pd
import matplotlib.pyplot as plt
from data_rechiever import StockDataManager
import os

# Create a list of tickers
stock_tickers = ['AAPL', 'MSFT', 'GOOG', 'TSLA']

# Process the tickers

class StockDataManagerExt(StockDataManager):
    def plot_multiple_tickers(self, tickers):
        # First, download initial data for all tickers
        for ticker in tickers:
            self.initial_download(ticker)
        
        # Define time frames
        time_frames = [
            ('1 Year', pd.Timestamp.today() - pd.Timedelta(days=365), pd.Timestamp.today()),
            ('5 Years', pd.Timestamp.today() - pd.Timedelta(days=365*5), pd.Timestamp.today())
        ]
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
        
        # Iterate through time frames
        for idx, (label, start_date, end_date) in enumerate(time_frames):
            # Select the appropriate axis
            ax = ax1 if idx == 0 else ax2
            
            # Initialize an empty DataFrame
            df = pd.DataFrame()
            
            for ticker in tickers:
                try:
                    # Load data from file
                    data = self._load_stock_data(ticker)
                    
                    # Convert Date column to datetime and set as index
                    data['Date'] = pd.to_datetime(data['Date'])
                    data.set_index('Date', inplace=True)
                    
                    # Convert Close column to numeric, removing any non-numeric characters
                    data['Close'] = pd.to_numeric(data['Close'].replace({'$': ''}, regex=True), errors='coerce')
                    
                    # Print data date range for debugging
                    print(f"{ticker} {label} data date range: {data.index.min()} to {data.index.max()}")
                    
                    # Filter data within date range
                    filtered_data = data.loc[(data.index >= start_date) & (data.index <= end_date)]
                    
                    # Print filtered data info
                    print(f"{ticker} {label} filtered data length: {len(filtered_data)}")
                    
                    # Only add if filtered data is not empty
                    if not filtered_data.empty:
                        df[ticker] = filtered_data['Close']
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
            
            # Check if DataFrame is empty
            if df.empty:
                print(f"No data available for the {label} time frame.")
                continue
            
            # Normalize prices to starting value of 100
            df_normalized = df / df.iloc[0] * 100
            
            # Plot on the selected axis
            df_normalized.plot(ax=ax)
            ax.set_title(f'{label} Stock Price Performance (Normalized to 100)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Normalized Price')
            ax.legend(tickers)
            ax.grid(True)
        
        # Adjust layout and add overall title
        plt.tight_layout()
        fig.suptitle('Stock Price Performance Comparison', fontsize=16, y=1.02)
        
        # Ensure the plots directory exists
        os.makedirs('plots', exist_ok=True)
        
        # Save the plot
        plot_path = os.path.join('plots', 'multi_ticker_performance_comparison.png')
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
        
        # Show the plot
        plt.show()

stock_manager = StockDataManagerExt()

stock_manager.plot_multiple_tickers(tickers=stock_tickers)