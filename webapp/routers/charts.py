from datetime import datetime
from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse
from sqlalchemy.orm import Session
from ..data_manager import data_manager
from ..database import get_db
from .. import models
from . import portfolios as portfolios_router
import pandas as pd
import json

router = APIRouter(
    prefix="/charts",
    tags=["charts"]
)

templates = Jinja2Templates(directory="webapp/templates")

def _parse_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None


def _build_snapshot_series_from_positions(portfolio: models.Portfolio) -> list[dict]:
    """Create a minimal equity series from the portfolio's current positions."""
    positions = getattr(portfolio, "positions", []) or []
    total_value = 0.0
    for pos in positions:
        qty = float(pos.quantity or 0.0)
        price = pos.current_price or pos.avg_cost or 0.0
        total_value += qty * float(price or 0.0)

    timestamp = portfolio.updated_at or portfolio.created_at or datetime.utcnow()
    if not isinstance(timestamp, datetime):
        timestamp = datetime.utcnow()

    return [
        {
            "timestamp": timestamp.isoformat(),
            "equity": round(total_value, 2),
        }
    ]


def _build_synthetic_chart_payload(
    portfolio: models.Portfolio,
    timeframe: str,
    from_date: datetime | None,
    to_date: datetime | None,
    db: Session,
):
    analytics = portfolios_router._build_equity_analytics(portfolio, db)
    series = analytics.get("series", {}).get("overall", [])
    if not series:
        series = _build_snapshot_series_from_positions(portfolio)
    if not series:
        return None

    dates: list[str] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []
    prev_close: float | None = None

    for point in series:
        timestamp = point.get("timestamp")
        if not timestamp:
            continue
        try:
            dt = datetime.fromisoformat(timestamp)
        except ValueError:
            continue
        if from_date and dt.date() < from_date.date():
            continue
        if to_date and dt.date() > to_date.date():
            continue

        close_price = float(point.get("equity") or 0.0)
        open_price = prev_close if prev_close is not None else close_price
        high_price = max(open_price, close_price)
        low_price = min(open_price, close_price)

        dates.append(dt.strftime("%Y-%m-%d"))
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(0.0)
        prev_close = close_price

    if not dates:
        return None

    return {
        "ticker": portfolio.synthetic_symbol,
        "dates": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }


@router.get("/data/{ticker}")
async def get_chart_data(
    ticker: str,
    timeframe: str = "D",
    from_date: str = None,
    to_date: str = None,
    db: Session = Depends(get_db),
):
    """
    Get stock data for charting.
    timeframe: D (Daily), W (Weekly), M (Monthly)
    from_date: Start date in YYYY-MM-DD format (optional)
    to_date: End date in YYYY-MM-DD format (optional)
    """
    import logging
    ticker = ticker.upper()

    from_dt = _parse_date(from_date)
    to_dt = _parse_date(to_date)

    if ticker.startswith("PORT-"):
        portfolio = (
            db.query(models.Portfolio)
            .filter(models.Portfolio.synthetic_symbol == ticker)
            .first()
        )
        if not portfolio:
            return JSONResponse({"error": "Portfolio synthetic ticker not found"}, status_code=404)
        synthetic_payload = _build_synthetic_chart_payload(portfolio, timeframe, from_dt, to_dt, db)
        if not synthetic_payload:
            return JSONResponse({"error": "No portfolio equity data available"}, status_code=404)
        return JSONResponse(synthetic_payload)

    try:
        # Check if data exists, if not try to download
        data = data_manager.load_data(ticker)

        if data is None or data.empty:
            # Trigger download
            data = data_manager.update_data(ticker)

        if data is None or data.empty:
            return JSONResponse({"error": "No data found"}, status_code=404)

        # Ensure index is DatetimeIndex for resampling
        if not isinstance(data.index, pd.DatetimeIndex):
            # Try to convert index to datetime
            if 'Date' in data.columns:
                data = data.set_index('Date')
            elif 'Datetime' in data.columns:
                data = data.set_index('Datetime')
            # Convert index to datetime if it's not already
            data.index = pd.to_datetime(data.index, utc=True)
        
        # Ensure timezone-aware index is converted to UTC for resampling
        if data.index.tz is not None:
            data.index = data.index.tz_convert('UTC')
        
        # Apply date range filter if specified
        if from_dt:
            data = data[data.index >= pd.to_datetime(from_dt, utc=True)]
        if to_dt:
            cutoff = pd.to_datetime(to_dt + pd.Timedelta(days=1), utc=True)
            data = data[data.index < cutoff]
        
        if data.empty:
            return JSONResponse({"error": "No data in selected date range"}, status_code=404)

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
            data = data.resample('M').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

        # Format for Plotly
        reset_data = data.reset_index()
        
        # Handle different index column names (Date vs Datetime)
        date_col = 'Date' if 'Date' in reset_data.columns else reset_data.columns[0]
        
        # Convert dates, handling timezone-aware datetimes
        dates = pd.to_datetime(reset_data[date_col], utc=True).dt.strftime('%Y-%m-%d').tolist()
        
        result = {
            "ticker": ticker,
            "dates": dates,
            "open": reset_data['Open'].tolist(),
            "high": reset_data['High'].tolist(),
            "low": reset_data['Low'].tolist(),
            "close": reset_data['Close'].tolist(),
            "volume": reset_data['Volume'].tolist() if 'Volume' in reset_data.columns else []
        }

        return JSONResponse(result)
    except Exception as e:
        logging.error(f"Error fetching chart data for {ticker}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/gallery/{list_id}", response_class=HTMLResponse)
async def get_multi_timeframe_gallery(
    request: Request, 
    list_id: int, 
    db: Session = Depends(get_db)
):
    """
    Generate a multi-timeframe gallery page for all tickers in a list.
    Shows Daily, Weekly, and Monthly charts side-by-side for each ticker.
    Uses Finviz chart images for fast loading.
    """
    ticker_list = db.query(models.TickerList).filter(models.TickerList.id == list_id).first()
    if not ticker_list:
        return HTMLResponse("<h3>Ticker list not found</h3>", status_code=404)
    
    tickers = [t.symbol for t in ticker_list.tickers]
    list_name = ticker_list.name
    
    return templates.TemplateResponse("gallery.html", {
        "request": request,
        "tickers": tickers,
        "list_name": list_name,
        "list_id": list_id
    })


@router.get("/gallery/tickers", response_class=HTMLResponse)
async def get_multi_timeframe_gallery_by_tickers(
    request: Request,
    tickers: str
):
    """
    Generate a multi-timeframe gallery page for specified tickers.
    tickers: Comma-separated list of ticker symbols.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        return HTMLResponse("<h3>No tickers provided</h3>", status_code=400)
    
    return templates.TemplateResponse("gallery.html", {
        "request": request,
        "tickers": ticker_list,
        "list_name": "Custom",
        "list_id": None
    })


@router.get("/gallery", response_class=HTMLResponse)
async def get_chart_gallery(
    request: Request,
    tickers: str,
    timeframe: str = "D",
    style: str = "candle"
):
    """
    Generate a chart gallery page for specified tickers.
    
    Args:
        tickers: Comma-separated list of ticker symbols
        timeframe: D (Daily), W (Weekly), M (Monthly), or 'multi' for all three
        style: 'candle' for candlestick or 'line' for line charts
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        return HTMLResponse("<h3>No tickers provided</h3>", status_code=400)
    
    # Map timeframe to display name
    timeframe_names = {
        'D': 'Daily',
        'W': 'Weekly', 
        'M': 'Monthly',
        'multi': 'Multi-Timeframe'
    }
    
    return templates.TemplateResponse("chart_gallery.html", {
        "request": request,
        "tickers": ticker_list,
        "timeframe": timeframe,
        "timeframe_name": timeframe_names.get(timeframe, timeframe),
        "style": style,
        "is_multi": timeframe == 'multi'
    })
