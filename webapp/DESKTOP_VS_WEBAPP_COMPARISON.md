# Desktop vs Webapp Feature Comparison

This document compares the Python desktop application (`stock_charts_10k10q`) with the web application (`webapp`).

## Executive Summary

| Aspect | Desktop (Tkinter) | Webapp (FastAPI) |
|--------|-------------------|------------------|
| **Framework** | Tkinter + Matplotlib | FastAPI + HTMX + Plotly.js |
| **Architecture** | Monolithic GUI | Client-Server (REST API) |
| **Charts** | StockCharts.com images + Plotly | Plotly.js (interactive) |
| **Data Storage** | TSV files + Python module | SQLite + TSV files |
| **Deployment** | Local only | Can be hosted |

---

## Feature Comparison Matrix

### ✅ Features in BOTH

| Feature | Desktop | Webapp | Notes |
|---------|---------|--------|-------|
| Stock data download (yfinance) | ✅ | ✅ | Same core logic |
| Fundamental data display | ✅ | ✅ | Webapp filters to 25 key metrics |
| AI Business Analysis (Gemini) | ✅ | ✅ | Same prompts |
| News fetching (Tavily) | ✅ | ✅ | Same API |
| SEC filings (10-K, 10-Q) | ✅ | ✅ | Webapp has improved rate limiting |
| Ticker list management | ✅ | ✅ | Different storage (module vs SQLite) |
| Daily/Weekly/Monthly timeframes | ✅ | ✅ | Same concept |
| Watch list | ✅ | ✅ | Desktop: module, Webapp: DB |

---

### ❌ Features MISSING from Webapp (Remaining)

| Feature | Desktop Location | Priority | Complexity |
|---------|------------------|----------|------------|
| ~~**Multi-Timeframe Gallery**~~ | ~~`multi_tf_charts.py`~~ | ✅ DONE | ~~Medium~~ |
| **StockCharts.com Integration** | `generate_stockcharts_gallery.py` | MEDIUM | Low |
| **Line Chart Comparison Gallery** | `multi_tf_charts.py` | MEDIUM | Medium |
| ~~**Buffett/CANSLIM Scoring**~~ | ~~`buffett_canslim.py`~~ | ✅ DONE | ~~High~~ |
| **Seasonality Charts** | `gui.py` (StockCharts.com) | LOW | Low |
| ~~**Finviz v=3 News Feed**~~ | ~~`gui.py`~~ | ✅ DONE | ~~Medium~~
| **Ticker extraction from webpage** | `extract_stock_tickers_from_webpage.py` | MEDIUM | Medium |
| **AI Ticker Extractor** | `AI_ticker_extractor.py` | LOW | Medium |
| **Custom URL Manager** | `gui.py` (custom_urls.json) | LOW | Low |
| **Keyboard Shortcuts** | `gui.py` | LOW | Low |
| **Tooltip System** | `tooltip_manager.py` | LOW | Low |
| **Edit ticker_lists.py in Notepad++** | `gui.py` | N/A | N/A (desktop-only) |
| **Copy tickers to clipboard** | `gui.py` | LOW | Low |
| **Prev/Next list navigation** | `gui.py` | LOW | Low |
| **SEC Filing Table Extraction** | `sec_filing_extractor.py` | MEDIUM | Medium |
| **Mock SEC Data for Testing** | `mock_sec_data.py` | LOW | Low |
| **SEC API Caching (disk)** | `sec_api_cache.py` | MEDIUM | Medium |

---

### 🆕 Features UNIQUE to Webapp

| Feature | Location | Notes |
|---------|----------|-------|
| **Compare Tab** | `templates/index.html` | Multi-ticker overlay chart |
| **In-memory caching with TTL** | `data_manager.py` | 5min fundamental, 1min chart |
| **Loading spinners** | `templates/index.html` | UX improvement |
| **Input validation** | `routers/tickers.py`, `schemas.py` | Pydantic + regex |
| **REST API** | `routers/*.py` | Can be consumed by other apps |
| **Yahoo Finance quick link** | `templates/index.html` | Header button |
| **Test suite** | `tests/` | pytest-based |

---

## Detailed Feature Analysis

### 1. Chart Generation

**Desktop:**
```python
# live_chart_generator.py - Generates HTML with StockCharts.com images
def generate_chart_html(tickers, columns, output_filename, time_frame):
    # Uses StockCharts.com dynamic chart URLs
    # Opens in browser as standalone HTML file
```

**Webapp:**
```python
# routers/charts.py - Returns JSON for Plotly.js
@router.get("/data/{ticker}")
async def get_chart_data(ticker: str, timeframe: str = "D"):
    # Returns OHLCV data as JSON
    # Frontend renders with Plotly.js
```

**Gap:** Webapp lacks:
- Multi-timeframe gallery (D/W/M side-by-side)
- StockCharts.com image integration
- Line chart comparison gallery

---

### 2. Ticker List Management

**Desktop:**
```python
# Loads from ticker_lists.py module
def _load_ticker_lists_from_module(self):
    import ticker_lists
    for name in dir(ticker_lists):
        obj = getattr(ticker_lists, name)
        if isinstance(obj, list):
            self.ticker_lists[name] = obj
```

**Webapp:**
```python
# Uses SQLite database
class TickerList(Base):
    __tablename__ = "ticker_lists"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    tickers = relationship("Ticker", back_populates="ticker_list")
```

**Gap:** Webapp lacks:
- Prev/Next list navigation buttons
- Filter ticker lists by name
- Refresh from disk
- Edit in external editor

---

### 3. Buffett/CANSLIM Analysis

**Desktop:**
```python
# buffett_canslim.py
def analyze_stock_scores(ticker: str) -> Dict:
    # Returns buffett_scores (8 dimensions)
    # Returns canslim_scores (7 dimensions)
    # Generates radar chart visualization
```

**Webapp:** ❌ Not implemented

**Priority:** HIGH - This is a key differentiating feature

---

### 4. SEC Filing Handling

**Desktop:**
```python
# sec_api_wrapper.py - Supports mock data for testing
# sec_api_cache.py - Disk-based caching
# sec_filing_extractor.py - Advanced table extraction with retry logic
```

**Webapp:**
```python
# sec_api.py - Basic implementation
# In-memory CIK cache only
# No mock data support
```

**Gap:** Webapp lacks:
- Disk-based SEC cache
- Mock data for testing
- Advanced table extraction
- Retry with exponential backoff

---

### 5. News Features

**Desktop:**
```python
# gui.py - Finviz v=3 integration
self.stock_news_temp_tickers = []  # Tickers detected from Finviz
# Can fetch market news without selecting a ticker first
```

**Webapp:**
```python
# news_fetcher.py - Tavily only
# Requires ticker to be selected
```

**Gap:** Webapp lacks:
- Finviz v=3 feed integration
- Ticker extraction from news pages
- Market-wide news (no ticker required)

---

## Implementation Priority

### Phase 4: High Priority Features

1. **Multi-Timeframe Gallery**
   - Port `multi_tf_charts.py` logic
   - Create `/charts/gallery/{list_id}` endpoint
   - Generate HTML with D/W/M charts per ticker

2. **Buffett/CANSLIM Scoring**
   - Port `buffett_canslim.py` and `stock_radar_batch.py`
   - Create `/analysis/buffett/{ticker}` endpoint
   - Add radar chart visualization

3. **Finviz News Feed**
   - Add `/news/market` endpoint (no ticker required)
   - Extract tickers from Finviz v=3 page
   - Populate temporary ticker list

### Phase 5: Medium Priority Features

4. **StockCharts.com Integration**
   - Add StockCharts image URLs to chart options
   - Port `generate_stockcharts_gallery.py`

5. **SEC Improvements**
   - Add disk-based caching
   - Port advanced table extraction
   - Add mock data for testing

6. **Ticker List Enhancements**
   - Prev/Next navigation
   - List name filter
   - Import from Python module

### Phase 6: Low Priority Features

7. **UI Polish**
   - Keyboard shortcuts
   - Tooltips
   - Copy to clipboard

---

## Code Reuse Opportunities

| Desktop File | Webapp Equivalent | Reuse Potential |
|--------------|-------------------|-----------------|
| `data_manager.py` | `data_manager.py` | 80% shared |
| `gemini_analyzer.py` | `gemini_analyzer.py` | 90% shared |
| `news_fetcher.py` | `news_fetcher.py` | 95% shared |
| `sec_api_wrapper.py` | `sec_api.py` | 60% shared |
| `buffett_canslim.py` | (missing) | 100% portable |
| `multi_tf_charts.py` | (missing) | 70% portable |
| `live_chart_generator.py` | (missing) | 50% portable |

---

## Recommended Next Steps

1. **Immediate:** Port Buffett/CANSLIM scoring (high value, self-contained)
2. **Short-term:** Add multi-timeframe gallery view
3. **Medium-term:** Integrate Finviz news feed
4. **Long-term:** Full feature parity with desktop

---

*Generated: December 2024*
