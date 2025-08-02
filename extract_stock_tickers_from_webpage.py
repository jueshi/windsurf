import pandas as pd
import logging
import yfinance as yf

logging.basicConfig(level=logging.INFO)

'''use AI to gennerate a pythhon list of stock tickers from the table below.'''

def get_sp500_tickers():
    """
    Get S&P 500 tickers using yfinance
    
    Returns:
        list: List of S&P 500 stock tickers
    """
    try:
        # Use Wikipedia's list of S&P 500 companies
        sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        
        # Extract tickers
        tickers = sp500_table['Symbol'].tolist()
        
        logging.info(f"Extracted {len(tickers)} tickers from Wikipedia")
        return tickers
    
    except Exception as e:
        logging.error(f"Error extracting tickers from Wikipedia: {e}")
        return []

def validate_and_filter_tickers(tickers):
    """
    Validate tickers using yfinance and filter out invalid ones
    
    Args:
        tickers (list): List of stock tickers to validate
    
    Returns:
        list: List of valid stock tickers
    """
    valid_tickers = []
    for ticker in tickers:
        try:
            # Fetch stock info
            stock_info = yf.Ticker(ticker).info
            
            # Check if basic info exists
            if stock_info and all(key in stock_info for key in ['regularMarketPrice', 'symbol', 'longName']):
                valid_tickers.append(ticker)
        except Exception:
            continue
    
    logging.info(f"Validated {len(valid_tickers)} tickers")
    return valid_tickers

def test_extract_stock_tickers():
    """
    Test the stock ticker extraction function
    """
    try:
        # Extract tickers from Wikipedia
        tickers = get_sp500_tickers()
        
        # Validate tickers
        valid_tickers = validate_and_filter_tickers(tickers)
        
        # Print and save results
        print(f"Number of valid tickers: {len(valid_tickers)}")
        print("First 20 tickers:", valid_tickers[:20])
        
        # Save to CSV
        df = pd.DataFrame(valid_tickers, columns=['Ticker'])
        df.to_csv('sp500_tickers.csv', index=False)
        
        # Ensure we have a reasonable number of tickers
        assert len(valid_tickers) > 400, "Not enough valid tickers found"
        
        return valid_tickers
    
    except Exception as e:
        logging.error(f"Ticker extraction failed: {e}")
        raise

if __name__ == '__main__':
    test_extract_stock_tickers()