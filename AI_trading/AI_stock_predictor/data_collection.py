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
