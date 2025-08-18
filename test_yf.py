import yfinance as yf
import pandas as pd
import datetime
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Ticker to test
ticker = "NVDA"

try:
    # Test 1: Get data specifically for Aug 2-4, 2025
    logging.info("Test 1: Downloading data for NVDA from 2025-08-02 to 2025-08-04")
    data1 = yf.download(ticker, start="2025-08-02", end="2025-08-05", progress=False)
    if not data1.empty:
        logging.info(f"Data shape: {data1.shape}")
        logging.info(f"Date range: {data1.index.min()} to {data1.index.max()}")
        logging.info(f"Data:\n{data1}")
    else:
        logging.info("No data returned for Aug 2-4, 2025")
    
    time.sleep(1)  # Add delay between API calls
    
    # Test 2: Check if today's data is available
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    logging.info(f"\nTest 2: Checking if today's data ({today}) is available")
    today_data = yf.download(ticker, start=yesterday, end=today, progress=False)
    if not today_data.empty:
        logging.info(f"Today's data is available. Last date: {today_data.index.max()}")
        logging.info(f"Data:\n{today_data}")
    else:
        logging.info(f"No data available for today ({today})")
    
    time.sleep(1)  # Add delay between API calls
    
    # Test 3: Check market status and latest info
    logging.info("\nTest 3: Getting the latest ticker info for NVDA")
    nvda = yf.Ticker("NVDA")
    latest_info = nvda.info
    
    if 'regularMarketTime' in latest_info:
        market_time = datetime.datetime.fromtimestamp(latest_info['regularMarketTime'])
        logging.info(f"Latest market data time: {market_time}")
        logging.info(f"Latest price: {latest_info.get('regularMarketPrice', 'N/A')}")
        logging.info(f"Current market status: {latest_info.get('marketState', 'Unknown')}")
    else:
        logging.info("No regular market time information available")

except Exception as e:
    logging.error(f"Error during testing: {e}")
