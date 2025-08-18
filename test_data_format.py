import os
import sys
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the stock_charts_10k10q directory to the path
sys.path.append('stock_charts_10k10q')

# Import the StockDataManager class
from stock_charts_10k10q.data_manager import StockDataManager

def test_data_formatting():
    """Test the data formatting fix by downloading and saving stock data"""
    # Create a StockDataManager instance
    manager = StockDataManager()
    
    # Test tickers
    test_tickers = ['GOOG', 'AAPL', 'MSFT']
    
    for ticker in test_tickers:
        logging.info(f"Testing data formatting for {ticker}")
        
        # Force download new data
        data = manager.update_data(ticker, force_download=True)
        
        if data is not None:
            logging.info(f"Successfully downloaded data for {ticker}")
            
            # Check the saved file
            data_path = manager._get_data_path(ticker)
            logging.info(f"Data saved to {data_path}")
            
            # Read the first few lines of the file to check formatting
            with open(data_path, 'r') as f:
                lines = [f.readline().strip() for _ in range(10)]
                
            logging.info(f"First few lines of {ticker} data file:")
            for line in lines:
                logging.info(line)
        else:
            logging.error(f"Failed to download data for {ticker}")

if __name__ == "__main__":
    test_data_formatting()
