from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from ..data_manager import data_manager
import pandas as pd
import json

router = APIRouter(
    prefix="/charts",
    tags=["charts"]
)

templates = Jinja2Templates(directory="webapp/templates")

@router.get("/data/{ticker}")
async def get_chart_data(ticker: str, timeframe: str = "D"):
    """
    Get stock data for charting.
    timeframe: D (Daily), W (Weekly), M (Monthly)
    """
    ticker = ticker.upper()

    # Check if data exists, if not try to download
    data = data_manager.load_data(ticker)

    if data is None or data.empty:
        # Trigger download
        data = data_manager.update_data(ticker)

    if data is None or data.empty:
        return JSONResponse({"error": "No data found"}, status_code=404)

    # Resample if needed
    if timeframe == "W":
        data = data.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    elif timeframe == "M":
        data = data.resample('ME').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

    # Format for Plotly
    # Plotly expects arrays for x, open, high, low, close
    reset_data = data.reset_index()

    result = {
        "ticker": ticker,
        "dates": reset_data['Date'].dt.strftime('%Y-%m-%d').tolist(),
        "open": reset_data['Open'].tolist(),
        "high": reset_data['High'].tolist(),
        "low": reset_data['Low'].tolist(),
        "close": reset_data['Close'].tolist(),
        "volume": reset_data['Volume'].tolist() if 'Volume' in reset_data.columns else []
    }

    return JSONResponse(result)
