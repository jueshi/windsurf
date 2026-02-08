import yfinance as yf
import pandas as pd
import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Function to check if data exists for a specific date range
def check_data_for_dates(ticker, start_date, end_date=None):
    logging.info(f"Checking data for {ticker} from {start_date} to {end_date or 'today'}")
    
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Check if data exists
    if data.empty:
        logging.info(f"No data available for {ticker} from {start_date} to {end_date or 'today'}")
        return False
    else:
        logging.info(f"Data available for {ticker} from {start_date} to {end_date or 'today'}")
        logging.info(f"Date range: {data.index.min()} to {data.index.max()}")
        logging.info(f"Number of data points: {len(data)}")
        return True

# Check NVDA data for specific dates
ticker = "NVDA"

# Check data for August 2, 2025
check_data_for_dates(ticker, "2025-08-02", "2025-08-03")

# Check data for August 3, 2025
check_data_for_dates(ticker, "2025-08-03", "2025-08-04")

# Check data for August 4, 2025
check_data_for_dates(ticker, "2025-08-04", "2025-08-05")

# Check data for the entire range
check_data_for_dates(ticker, "2025-08-02", "2025-08-05")

# Get market status
try:
    nvda = yf.Ticker(ticker)
    info = nvda.info
    if 'marketState' in info:
        market_state = info['marketState']
        logging.info(f"Current market state for {ticker}: {market_state}")
    else:
        logging.info(f"Market state information not available for {ticker}")
except Exception as e:
    logging.error(f"Error getting market information: {e}")
