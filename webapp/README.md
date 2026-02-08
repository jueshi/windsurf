# Stock Toolbox Web - Quick Start Guide

A FastAPI-based web application for stock analysis with interactive charts, AI-powered insights, news aggregation, and SEC filings integration.

## Features

| Feature | Description |
|---------|-------------|
| **Interactive Charts** | Candlestick charts with volume (Daily/Weekly/Monthly) via Plotly |
| **Ticker Management** | Create/manage ticker lists with search functionality |
| **Fundamental Data** | Company info from Yahoo Finance (P/E, market cap, etc.) |
| **AI Analysis** | Business analysis powered by Google Gemini (Chinese + English) |
| **News Feed** | Stock news via Tavily API with AI sentiment analysis |
| **SEC Filings** | Latest 10-K/10-Q filings with table extraction |

---

## Prerequisites

- **Python 3.9+**
- API Keys (optional but recommended):
  - `GEMINI_API_KEY` - For AI analysis features
  - `TAVILY_API_KEY` - For news fetching
  - `SEC_EDGAR_EMAIL` - For SEC API access (your email)

---

## Installation

```bash
# 1. Navigate to the webapp directory
cd webapp

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create a .env file with your API keys
```

### `.env` File Template

```env
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
SEC_EDGAR_EMAIL=your_email@example.com
GEMINI_MODEL_NAME=gemini-2.0-flash  # Optional, defaults to gemini-2.0-flash
```

---

## Running the Application

```bash
# From the project root directory (parent of webapp/)
uvicorn webapp.main:app --reload --host 0.0.0.0 --port 8000
```

Then open: **http://localhost:8000**

---

## Usage Guide

### 1. Managing Ticker Lists

1. **Create a list**: In the sidebar, click to create a new ticker list
2. **Add tickers**: Type a ticker symbol (e.g., `AAPL`) and add it to your list
3. **Search**: Use the search bar to find tickers across all lists

### 2. Viewing Charts

1. Select a ticker from your list
2. The **Chart** tab displays an interactive candlestick chart
3. Toggle between **Daily**, **Weekly**, or **Monthly** timeframes

### 3. Fundamental Analysis

1. Select a ticker
2. Click the **Fundamentals** tab
3. View company info: sector, market cap, P/E ratios, etc.

### 4. AI Business Analysis

1. Select a ticker
2. Go to **Analysis (AI)** tab
3. Click to generate a comprehensive business analysis
4. Output includes: business model, competitive landscape, financials, growth prospects, and risks

### 5. News & Sentiment

1. Select a ticker
2. Go to **News** tab
3. View recent news articles
4. Click to analyze news sentiment (bullish/bearish classification)

### 6. SEC Filings

1. Select a ticker
2. Go to **SEC Filings** tab
3. View latest 10-K and 10-Q filings
4. Extract and view financial tables from filings

---

## Project Structure

```
webapp/
├── main.py              # FastAPI app entry point
├── database.py          # SQLite database setup
├── models.py            # SQLAlchemy models (TickerList, Ticker)
├── data_manager.py      # Stock data download/caching (yfinance)
├── gemini_analyzer.py   # Google Gemini AI integration
├── news_fetcher.py      # Tavily news API wrapper
├── sec_api.py           # SEC EDGAR API wrapper
├── requirements.txt     # Python dependencies
├── routers/
│   ├── tickers.py       # Ticker list CRUD endpoints
│   ├── charts.py        # Chart data endpoints
│   ├── analysis.py      # Fundamental & AI analysis
│   ├── news.py          # News feed endpoints
│   └── sec.py           # SEC filings endpoints
├── templates/
│   ├── base.html        # Base template (sidebar, layout)
│   ├── index.html       # Main dashboard
│   └── components/      # HTMX partial templates
└── static/              # Static files (plots, etc.)
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/tickers/list` | GET | Get all ticker lists |
| `/tickers/list` | POST | Create new ticker list |
| `/tickers/add/{list_id}` | POST | Add ticker to list |
| `/tickers/{ticker_id}` | DELETE | Remove ticker |
| `/tickers/search?q=` | GET | Search tickers |
| `/charts/data/{ticker}?timeframe=` | GET | Get OHLCV data |
| `/analysis/fundamental/{ticker}` | GET | Get fundamental data |
| `/analysis/business/{ticker}` | POST | Run AI analysis |
| `/news/feed/{ticker}` | GET | Get news articles |
| `/news/analyze/{ticker}` | POST | Analyze news sentiment |
| `/sec/filings/{ticker}` | GET | Get SEC filings |
| `/sec/extract/{ticker}/{accession}` | POST | Extract filing tables |

---

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy, SQLite
- **Frontend**: Bootstrap 5, HTMX, Plotly.js
- **Data**: yfinance, Tavily API, SEC EDGAR API
- **AI**: Google Gemini API

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Charts not loading | Check if `yfinance` can access Yahoo Finance |
| AI analysis fails | Verify `GEMINI_API_KEY` is set correctly |
| News not loading | Verify `TAVILY_API_KEY` is set correctly |
| SEC filings fail | Set `SEC_EDGAR_EMAIL` to a valid email |
| Rate limit errors | The app has built-in retry logic; wait and retry |

---

## Notes

- Stock data is cached locally in `webapp/stock_data/` as TSV files
- Database (`stock_toolbox.db`) stores ticker lists only
- AI analysis outputs are bilingual (Chinese + English)
