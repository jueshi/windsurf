import os
import logging
import pandas as pd
import plotly.graph_objects as go
from data_manager import StockDataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def generate_charts():
    """
    Generate timeframe charts and save them as images.
    """
    manager = StockDataManager()
    ticker = "AAPL"

    # Ensure data is available
    if not os.path.exists(manager._get_data_path(ticker)):
        logging.info(f"Data for {ticker} not found, downloading...")
        manager.update_data(ticker, force_download=True)

    df = manager.load_data(ticker)

    if df is None or df.empty:
        logging.error(f"No data available for {ticker}.")
        return

    # --- Generate Daily Chart (Last 6 Months) ---
    try:
        daily_df_filtered = df[df.index >= (df.index.max() - pd.DateOffset(months=6))]
        fig_daily = go.Figure(data=[go.Candlestick(
            x=daily_df_filtered.index, open=daily_df_filtered['Open'], high=daily_df_filtered['High'],
            low=daily_df_filtered['Low'], close=daily_df_filtered['Close']
        )])
        fig_daily.update_layout(title_text=f"{ticker} Daily", xaxis_rangeslider_visible=False, margin=dict(t=30, b=10, l=20, r=20))
        fig_daily.update_xaxes(type='date', tickformat='%b %d, %Y')
        fig_daily.write_html("daily_chart.html")
        logging.info("Daily chart saved as daily_chart.html")
    except Exception as e:
        logging.error(f"Failed to generate daily chart: {e}")

    # --- Generate Weekly Chart (Last 3 Years) ---
    try:
        weekly_df = df.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
        weekly_df_filtered = weekly_df[weekly_df.index >= (weekly_df.index.max() - pd.DateOffset(years=3))]
        fig_weekly = go.Figure(data=[go.Candlestick(
            x=weekly_df_filtered.index, open=weekly_df_filtered['Open'], high=weekly_df_filtered['High'],
            low=weekly_df_filtered['Low'], close=weekly_df_filtered['Close']
        )])
        fig_weekly.update_layout(title_text=f"{ticker} Weekly", xaxis_rangeslider_visible=False, margin=dict(t=30, b=10, l=20, r=20))
        fig_weekly.update_xaxes(type='date', tickformat='%b %d, %Y')
        fig_weekly.write_html("weekly_chart.html")
        logging.info("Weekly chart saved as weekly_chart.html")
    except Exception as e:
        logging.error(f"Failed to generate weekly chart: {e}")

    # --- Generate Monthly Chart (All Time) ---
    try:
        monthly_df = df.resample('M').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
        fig_monthly = go.Figure(data=[go.Candlestick(
            x=monthly_df.index, open=monthly_df['Open'], high=monthly_df['High'],
            low=monthly_df['Low'], close=monthly_df['Close']
        )])
        fig_monthly.update_layout(title_text=f"{ticker} Monthly (Log Scale)", xaxis_rangeslider_visible=False,
                                  margin=dict(t=30, b=10, l=20, r=20), yaxis_type='log')
        fig_monthly.update_xaxes(type='date', tickformat='%b %Y')
        fig_monthly.write_html("monthly_chart.html")
        logging.info("Monthly chart saved as monthly_chart.html")
    except Exception as e:
        logging.error(f"Failed to generate monthly chart: {e}")

if __name__ == '__main__':
    generate_charts()
