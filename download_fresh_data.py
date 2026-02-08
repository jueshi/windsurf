import os
import sys
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the stock_charts_10k10q directory to the path
sys.path.insert(0, os.path.abspath('stock_charts_10k10q'))

# Import the StockDataManager class
try:
    from data_manager import StockDataManager
    logging.info("Successfully imported StockDataManager")
except ImportError as e:
    logging.error(f"Error importing StockDataManager: {e}")
    sys.exit(1)

def download_fresh_stock_data(tickers):
    """Download fresh stock data with proper formatting for the specified tickers"""
    # Create a StockDataManager instance
    manager = StockDataManager()
    
    logging.info(f"Starting fresh download for {len(tickers)} tickers: {', '.join(tickers)}")
    
    results = []
    for ticker in tickers:
        try:
            logging.info(f"Downloading data for {ticker}...")
            
            # Force download new data
            data = manager.update_data(ticker, force_download=True)
            
            if data is not None and not data.empty:
                logging.info(f"Successfully downloaded data for {ticker}")
                
                # Get the file path
                data_path = manager._get_data_path(ticker)
                
                # Check the file format
                with open(data_path, 'r') as f:
                    header = f.readline().strip()
                    first_data_line = f.readline().strip() if f.readline() else ""
                
                results.append({
                    'ticker': ticker,
                    'status': 'success',
                    'rows': len(data),
                    'file_path': data_path,
                    'header': header,
                    'sample': first_data_line
                })
            else:
                logging.error(f"Failed to download data for {ticker}")
                results.append({
                    'ticker': ticker,
                    'status': 'failed',
                    'error': 'No data returned'
                })
        except Exception as e:
            logging.error(f"Error downloading data for {ticker}: {e}")
            results.append({
                'ticker': ticker,
                'status': 'error',
                'error': str(e)
            })
    
    # Print summary
    print("\nDownload Summary:")
    print("-" * 80)
    for result in results:
        if result['status'] == 'success':
            print(f"{result['ticker']}: SUCCESS - {result['rows']} rows")
            print(f"  File: {result['file_path']}")
            print(f"  Header: {result['header']}")
            print(f"  Sample: {result['sample']}")
        else:
            print(f"{result['ticker']}: FAILED - {result.get('error', 'Unknown error')}")
        print("-" * 80)
    
    return results

if __name__ == "__main__":
    # Default tickers to download
    default_tickers = ['GOOG', 'AAPL', 'MSFT', 'AMZN', 'META']
    
    # Get tickers from command line arguments or use defaults
    tickers = sys.argv[1:] if len(sys.argv) > 1 else default_tickers
    
    # Download fresh data
    download_fresh_stock_data(tickers)
