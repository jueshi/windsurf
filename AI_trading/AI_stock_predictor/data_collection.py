import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None
    print("Warning: TextBlob not installed. News sentiment analysis will be skipped.")
import requests

def get_stock_data(ticker, period='5y', interval='1d'):
    """Fetch historical stock data from Yahoo Finance"""
    stock = yf.Ticker(ticker)
    return stock.history(period=period, interval=interval)

def collect_stock_data(stock_symbol, period='5y', interval='1d'):
    """
    Collect historical stock data using multiple data sources as fallback
    
    Args:
        stock_symbol (str): Stock ticker symbol
        period (str, optional): Data collection period. Defaults to '5y'.
        interval (str, optional): Data collection interval. Defaults to '1d'.
    
    Returns:
        pd.DataFrame: Historical stock price data
    """
    # List of data sources to try
    data_sources = [
        # yfinance (primary source)
        lambda: yf.download(stock_symbol, period=period, interval=interval),
        
        # Alpha Vantage (backup source)
        lambda: get_alpha_vantage_data(stock_symbol, period, interval),
        
        # Pandas Datareader (alternative source)
        lambda: get_pandas_datareader_data(stock_symbol, period, interval)
    ]
    
    # Try each data source
    for source in data_sources:
        try:
            stock_data = source()
            
            # Validate data
            if stock_data is not None and not stock_data.empty:
                # Ensure required columns exist
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in stock_data.columns for col in required_columns):
                    # Reset index to make date a column
                    stock_data = stock_data.reset_index()
                    
                    # Rename columns to standard format
                    stock_data.columns = [col.capitalize() for col in stock_data.columns]
                    
                    print(f"Successfully retrieved stock data for {stock_symbol}")
                    print(f"Data range: {stock_data['Date'].min()} to {stock_data['Date'].max()}")
                    print(f"Number of data points: {len(stock_data)}")
                    
                    return stock_data
        except Exception as e:
            print(f"Error retrieving data from source: {e}")
    
    # If all sources fail
    raise ValueError(f"Could not retrieve stock data for {stock_symbol} from any source")

def get_alpha_vantage_data(stock_symbol, period, interval):
    """
    Retrieve stock data from Alpha Vantage
    
    Args:
        stock_symbol (str): Stock ticker symbol
        period (str): Data collection period
        interval (str): Data collection interval
    
    Returns:
        pd.DataFrame or None: Stock price data
    """
    try:
        from alpha_vantage.timeseries import TimeSeries
        import os
        
        # Load API key from environment variable
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            print("Alpha Vantage API key not found")
            return None
        
        # Initialize Alpha Vantage client
        ts = TimeSeries(key=api_key, output_format='pandas')
        
        # Retrieve daily data
        data, _ = ts.get_daily_adjusted(symbol=stock_symbol, outputsize='full')
        
        return data
    except ImportError:
        print("Alpha Vantage library not installed")
        return None
    except Exception as e:
        print(f"Alpha Vantage data retrieval error: {e}")
        return None

def get_pandas_datareader_data(stock_symbol, period, interval):
    """
    Retrieve stock data using Pandas Datareader
    
    Args:
        stock_symbol (str): Stock ticker symbol
        period (str): Data collection period
        interval (str): Data collection interval
    
    Returns:
        pd.DataFrame or None: Stock price data
    """
    try:
        import pandas_datareader as pdr
        from datetime import datetime, timedelta
        
        # Convert period to datetime
        end_date = datetime.now()
        years = int(period[:-1]) if period.endswith('y') else 5
        start_date = end_date - timedelta(days=years*365)
        
        # Retrieve data
        data = pdr.get_data_yahoo(stock_symbol, start=start_date, end=end_date)
        
        return data
    except ImportError:
        print("Pandas Datareader library not installed")
        return None
    except Exception as e:
        print(f"Pandas Datareader data retrieval error: {e}")
        return None

def get_news_sentiment(ticker, days=7):
    if TextBlob is None:
        print("Skipping news sentiment analysis due to missing TextBlob.")
        return pd.DataFrame()  # Return empty DataFrame
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Placeholder for news API integration
    # news_data = fetch_news_from_api(ticker, start_date, end_date)
    # sentiment_scores = [TextBlob(article).sentiment.polarity for article in news_data]
    
    # Return dummy data for now
    return pd.Series([0.5] * 7)  # Neutral sentiment

def save_data(data, filename):
    """Save collected data to file"""
    data.to_csv(f'data/{filename}.csv')
