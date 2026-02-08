# Stock Toolbox Web - Product Requirements Document
## Issues Identified & Proposed Fixes

**Version:** 1.0  
**Date:** December 1, 2025  
**Status:** Draft

---

## Executive Summary

This document identifies issues in the Stock Toolbox Web application codebase and proposes fixes. Issues are categorized by severity: **Critical**, **High**, **Medium**, **Low**.

---

## 1. Critical Issues

### 1.1 Missing Static Directory Structure
**File:** `main.py` line 12  
**Issue:** App mounts `/static` but the `webapp/static/` directory only contains an empty `plots/` folder. No CSS/JS files exist.  
**Impact:** Static file serving will fail if custom assets are added.  
**Fix:** Create proper static directory structure or remove mount if not needed.

```
webapp/static/
├── css/
├── js/
└── plots/
```

### 1.2 Deprecated SQLAlchemy Import
**File:** `database.py` line 2  
**Issue:** `declarative_base` imported from `sqlalchemy.ext.declarative` is deprecated.  
**Impact:** Future SQLAlchemy versions will break.  
**Fix:** 
```python
# Change from:
from sqlalchemy.ext.declarative import declarative_base
# To:
from sqlalchemy.orm import declarative_base
```

### 1.3 Database Path Hardcoded
**File:** `database.py` line 5  
**Issue:** Database path `sqlite:///./stock_toolbox.db` is relative and creates DB in CWD, not webapp directory.  
**Impact:** Database location varies based on where uvicorn is started.  
**Fix:** Use absolute path based on module location:
```python
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLALCHEMY_DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'stock_toolbox.db')}"
```

---

## 2. High Priority Issues

### 2.1 No Error Handling for Missing API Keys
**Files:** `gemini_analyzer.py`, `news_fetcher.py`, `sec_api.py`  
**Issue:** API key checks return error strings but don't prevent further execution gracefully.  
**Impact:** Confusing error messages, potential crashes.  
**Fix:** Add consistent error handling pattern:
- Return structured error responses
- Add startup validation for required keys
- Add `.env.example` template file

### 2.2 Fundamental Data Shows Raw Dict
**File:** `templates/components/fundamental.html` lines 13-18  
**Issue:** Iterates over entire `yfinance.info` dict (100+ fields), showing raw keys like `longBusinessSummary`.  
**Impact:** Poor UX, overwhelming data display.  
**Fix:** Filter to key metrics only:
```python
DISPLAY_METRICS = [
    'longName', 'sector', 'industry', 'marketCap', 'trailingPE', 
    'forwardPE', 'dividendYield', 'beta', 'fiftyTwoWeekHigh', 
    'fiftyTwoWeekLow', 'averageVolume', 'revenueGrowth'
]
```

### 2.3 Compare Tab Not Implemented
**File:** `templates/index.html` line 73-74  
**Issue:** Compare tab exists in UI but has no functionality.  
**Impact:** Broken feature, user confusion.  
**Fix:** Either implement comparison feature or remove tab.

### 2.4 No Ticker List Deletion
**File:** `routers/tickers.py`  
**Issue:** Can create lists and delete tickers, but cannot delete entire ticker lists.  
**Impact:** Users cannot clean up unused lists.  
**Fix:** Add DELETE endpoint:
```python
@router.delete("/list/{list_id}", response_class=HTMLResponse)
async def delete_ticker_list(request: Request, list_id: int, db: Session = Depends(get_db)):
    ...
```

### 2.5 No Input Validation
**Files:** All routers  
**Issue:** No validation on ticker symbols (length, format, special chars).  
**Impact:** Potential for invalid data, injection risks.  
**Fix:** Add Pydantic models for request validation:
```python
from pydantic import BaseModel, validator
import re

class TickerCreate(BaseModel):
    symbol: str
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not re.match(r'^[A-Z]{1,5}$', v.upper()):
            raise ValueError('Invalid ticker symbol')
        return v.upper()
```

---

## 3. Medium Priority Issues

### 3.1 Hardcoded Template Paths
**Files:** All routers  
**Issue:** Each router creates its own `Jinja2Templates` instance with hardcoded path.  
**Impact:** Duplication, maintenance burden.  
**Fix:** Create shared templates instance in `main.py` and pass to routers.

### 3.2 No Loading States for All Tabs
**File:** `templates/index.html`  
**Issue:** Chart has loading spinner, but Fundamentals/News/SEC load without visual feedback.  
**Impact:** User doesn't know if request is pending.  
**Fix:** Add HTMX indicators to all tab loads:
```html
<div id="fundamental-container" hx-indicator="#fundamental-spinner">
```

### 3.3 No Caching for Fundamental Data
**File:** `data_manager.py`  
**Issue:** `get_fundamental_data()` calls yfinance API every time.  
**Impact:** Slow responses, rate limiting risk.  
**Fix:** Add simple TTL cache:
```python
from functools import lru_cache
from datetime import datetime

@lru_cache(maxsize=100)
def get_fundamental_data_cached(ticker: str, cache_key: str):
    return yf.Ticker(ticker).info
```

### 3.4 SEC Table Extraction Fragile
**File:** `sec_api.py` lines 100-109  
**Issue:** Document name matching logic is brittle (`if name.endswith('.htm') and ...`).  
**Impact:** May fail to find primary document for some filings.  
**Fix:** Improve document detection:
1. Look for `primaryDocument` field in index.json
2. Fallback to largest .htm file
3. Add error message when document not found

### 3.5 No Pagination for Ticker Lists
**File:** `routers/tickers.py`  
**Issue:** All tickers loaded at once.  
**Impact:** Performance issues with large lists.  
**Fix:** Add pagination or virtual scrolling for lists > 50 items.

### 3.6 Duplicate Ticker Check Only Per-List
**File:** `routers/tickers.py` line 42  
**Issue:** Same ticker can exist in multiple lists (may be intentional).  
**Impact:** Potential data redundancy.  
**Fix:** Document as feature or add global uniqueness option.

---

## 4. Low Priority Issues

### 4.1 Unused Buttons in Header
**File:** `templates/index.html` lines 8-14  
**Issue:** Share, Export, "This week" buttons have no functionality.  
**Impact:** UI clutter.  
**Fix:** Remove or implement.

### 4.2 No Dark Mode Toggle
**File:** `templates/base.html`  
**Issue:** Sidebar is dark but main content is light.  
**Impact:** Inconsistent visual design.  
**Fix:** Add CSS variables for theming or make consistent.

### 4.3 Mobile Responsiveness
**File:** `templates/base.html` line 27  
**Issue:** Sidebar hidden on mobile (`d-none d-md-block`) with no alternative navigation.  
**Impact:** App unusable on mobile.  
**Fix:** Add mobile hamburger menu or bottom navigation.

### 4.4 No Favicon
**Issue:** No favicon configured.  
**Fix:** Add favicon to static and reference in base.html.

### 4.5 Console Logging Only
**Files:** All Python files  
**Issue:** Uses `logging` module but no file handler configured.  
**Impact:** Logs lost on restart.  
**Fix:** Add file logging configuration.

### 4.6 No Tests
**Issue:** No test files exist.  
**Impact:** No automated verification of functionality.  
**Fix:** Add pytest tests for:
- API endpoints
- Data manager functions
- Template rendering

---

## 5. Security Issues

### 5.1 No CSRF Protection
**Issue:** Forms use HTMX POST without CSRF tokens.  
**Impact:** Vulnerable to CSRF attacks.  
**Fix:** Add CSRF middleware or use FastAPI's built-in protection.

### 5.2 API Keys in Environment
**Status:** Correctly implemented  
**Note:** API keys loaded from `.env` which is good practice. Ensure `.env` is in `.gitignore`.

### 5.3 No Rate Limiting
**Issue:** No rate limiting on API endpoints.  
**Impact:** Vulnerable to abuse.  
**Fix:** Add `slowapi` or similar rate limiting middleware.

---

## 6. Performance Issues

### 6.1 yfinance Downloads Full History
**File:** `data_manager.py` line 100  
**Issue:** Downloads full history (`period="max"`) even for simple chart views.  
**Impact:** Slow initial loads, unnecessary data.  
**Fix:** Default to 2 years, offer "Load More" option.

### 6.2 No Database Indexes
**File:** `models.py`  
**Issue:** Only basic indexes on id and symbol.  
**Impact:** Slow queries as data grows.  
**Fix:** Add composite index on `(list_id, symbol)`.

### 6.3 Synchronous yfinance Calls
**File:** `data_manager.py`  
**Issue:** yfinance calls are synchronous, blocking the event loop.  
**Impact:** Poor concurrency under load.  
**Fix:** Use `run_in_executor` or async yfinance wrapper.

---

## 7. Code Quality Issues

### 7.1 Inconsistent Error Response Format
**Issue:** Some endpoints return HTML, some return JSON errors.  
**Fix:** Standardize on JSON for API endpoints, HTML for HTMX endpoints.

### 7.2 No Type Hints in Some Functions
**Files:** Various  
**Fix:** Add type hints for better IDE support and documentation.

### 7.3 Magic Numbers
**File:** `sec_api.py` line 45  
**Issue:** `if i >= 10` - magic number for table limit.  
**Fix:** Define as constant: `MAX_TABLES_DISPLAY = 10`

---

## 8. Implementation Priority

### Phase 1 - Critical (Week 1)
1. Fix deprecated SQLAlchemy import
2. Fix database path
3. Add `.env.example` template

### Phase 2 - High (Week 2)
1. Filter fundamental data display
2. Add ticker list deletion
3. Add input validation
4. Implement or remove Compare tab

### Phase 3 - Medium (Week 3-4)
1. Add loading states
2. Implement fundamental data caching
3. Improve SEC document detection
4. Consolidate template instances

### Phase 4 - Low/Polish (Week 5+)
1. Remove unused UI elements
2. Add mobile navigation
3. Add favicon
4. Add tests
5. Add rate limiting

---

## 9. Files to Create

| File | Purpose |
|------|---------|
| `.env.example` | Template for required environment variables |
| `webapp/static/css/custom.css` | Custom styles |
| `webapp/tests/__init__.py` | Test package |
| `webapp/tests/test_api.py` | API endpoint tests |
| `webapp/schemas.py` | Pydantic validation models |

---

## 10. Dependencies to Add

```txt
# requirements.txt additions
slowapi          # Rate limiting
pytest           # Testing
pytest-asyncio   # Async test support
httpx            # Already present, for testing
```

---

## Appendix: Quick Wins (< 1 hour each)

1. ✅ Fix timezone datetime error in charts (DONE)
2. ✅ Fix loadTicker global scope issue (DONE)
3. ✅ Add `.env.example` file (DONE)
4. ✅ Fix deprecated SQLAlchemy import (DONE)
5. ✅ Add ticker list delete button (DONE)
6. ✅ Fix database path to absolute (DONE)
7. ✅ Filter fundamental data to key metrics (DONE)
8. ✅ Add input validation for tickers (DONE)
9. ✅ Implement Compare tab (DONE)
10. ✅ Add in-memory caching for API data (DONE)
11. ✅ Add loading states for all tabs (DONE)
12. ✅ Improve SEC API with rate limiting and better doc detection (DONE)
13. ✅ Fix data_manager paths to absolute (DONE)
14. ✅ Replace unused header buttons with Yahoo Finance link (DONE)
15. ✅ Add basic test suite (DONE)
16. ✅ Add Buffett/CANSLIM investment analysis (DONE)
17. ✅ Add radar chart visualization for investment scores (DONE)
18. ✅ Add Multi-Timeframe Gallery (D/W/M Finviz charts) (DONE)
19. ✅ Add Finviz Market News feed (no ticker required) (DONE)
20. ✅ Extract and display mentioned tickers from news (DONE)
