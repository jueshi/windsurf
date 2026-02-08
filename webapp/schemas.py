"""
Pydantic schemas for request validation.
"""
import re
from pydantic import BaseModel, field_validator


class TickerSymbol(BaseModel):
    """Validates a stock ticker symbol."""
    symbol: str

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """
        Validate ticker symbol format.
        - Must be 1-5 uppercase letters
        - Allows common suffixes like .A, .B for share classes
        """
        v = v.upper().strip()
        # Match 1-5 letters, optionally followed by .A, .B, etc.
        if not re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', v):
            raise ValueError('Invalid ticker symbol. Must be 1-5 letters (e.g., AAPL, BRK.A)')
        return v


class TickerListCreate(BaseModel):
    """Validates ticker list creation."""
    name: str

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """
        Validate list name.
        - Must be 1-50 characters
        - No leading/trailing whitespace
        """
        v = v.strip()
        if not v:
            raise ValueError('List name cannot be empty')
        if len(v) > 50:
            raise ValueError('List name must be 50 characters or less')
        return v


# Key fundamental metrics to display (filtered from yfinance's 100+ fields)
DISPLAY_METRICS = {
    'longName': 'Company Name',
    'sector': 'Sector',
    'industry': 'Industry',
    'marketCap': 'Market Cap',
    'trailingPE': 'P/E (TTM)',
    'forwardPE': 'Forward P/E',
    'priceToBook': 'Price/Book',
    'priceToSalesTrailing12Months': 'Price/Sales',
    'dividendYield': 'Dividend Yield',
    'payoutRatio': 'Payout Ratio',
    'beta': 'Beta',
    'fiftyTwoWeekHigh': '52-Week High',
    'fiftyTwoWeekLow': '52-Week Low',
    'fiftyDayAverage': '50-Day Avg',
    'twoHundredDayAverage': '200-Day Avg',
    'averageVolume': 'Avg Volume',
    'revenueGrowth': 'Revenue Growth',
    'earningsGrowth': 'Earnings Growth',
    'profitMargins': 'Profit Margin',
    'operatingMargins': 'Operating Margin',
    'returnOnEquity': 'ROE',
    'returnOnAssets': 'ROA',
    'debtToEquity': 'Debt/Equity',
    'currentRatio': 'Current Ratio',
    'freeCashflow': 'Free Cash Flow',
    'totalRevenue': 'Total Revenue',
    'netIncomeToCommon': 'Net Income',
}


def filter_fundamental_data(data: dict) -> dict:
    """
    Filter fundamental data to only include key metrics.
    Also formats values for display.
    """
    if not data:
        return {}
    
    filtered = {}
    for key, label in DISPLAY_METRICS.items():
        value = data.get(key)
        if value is not None:
            # Format the value based on type
            formatted = format_metric_value(key, value)
            if formatted is not None:
                filtered[label] = formatted
    
    return filtered


def format_metric_value(key: str, value) -> str:
    """Format a metric value for display."""
    if value is None or value == 'N/A':
        return None
    
    try:
        # Percentage fields - dividendYield from yfinance is already in % form (0.37 = 0.37%)
        if key == 'dividendYield':
            if isinstance(value, (int, float)):
                return f"{value:.2f}%"
        
        # Other percentage fields need to be multiplied by 100
        if key in ['payoutRatio', 'revenueGrowth', 'earningsGrowth', 
                   'profitMargins', 'operatingMargins', 'returnOnEquity', 'returnOnAssets']:
            if isinstance(value, (int, float)):
                return f"{value * 100:.2f}%"
        
        # Large number fields (market cap, revenue, etc.)
        if key in ['marketCap', 'totalRevenue', 'netIncomeToCommon', 'freeCashflow', 'averageVolume']:
            if isinstance(value, (int, float)):
                if abs(value) >= 1e12:
                    return f"${value / 1e12:.2f}T"
                elif abs(value) >= 1e9:
                    return f"${value / 1e9:.2f}B"
                elif abs(value) >= 1e6:
                    return f"${value / 1e6:.2f}M"
                else:
                    return f"${value:,.0f}"
        
        # Price fields
        if key in ['fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 'fiftyDayAverage', 'twoHundredDayAverage']:
            if isinstance(value, (int, float)):
                return f"${value:.2f}"
        
        # Ratio fields
        if key in ['trailingPE', 'forwardPE', 'priceToBook', 'priceToSalesTrailing12Months', 
                   'beta', 'debtToEquity', 'currentRatio']:
            if isinstance(value, (int, float)):
                return f"{value:.2f}"
        
        # Default: return as string
        return str(value)
    
    except (ValueError, TypeError):
        return str(value)
