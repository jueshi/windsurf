# Product Requirements Document: SEC Features Refactor

**Version:** 1.0  
**Date:** November 30, 2025  
**Author:** Development Team  
**Status:** Draft

---

## 1. Executive Summary

This PRD outlines the plan to refactor and consolidate the SEC-related features in the stock_charts_10k10q application. The current implementation has significant code duplication, thread safety issues, and inconsistent behavior across multiple modules. This refactor will improve reliability, maintainability, and user experience.

---

## 2. Problem Statement

### 2.1 Current State

The application has **6 different files** implementing SEC API functionality:

- `sec_api_cache.py` - Cached API with rate limiting
- `sec_api_wrapper.py` - Mock/real API switcher
- `sec_filing_extractor.py` - Table extraction with its own API calls
- `sec_filing_downloader.py` - Simple downloader without caching
- `sec_edgar_tables.py` - Alternative table extraction
- `sec_edgar_helper.py` - External package wrapper

### 2.2 Key Problems

| Problem | Impact | Severity |
|---------|--------|----------|
| Code duplication | Maintenance burden, inconsistent fixes | High |
| Thread safety violations | Potential GUI crashes | High |
| Inconsistent rate limiting | 403/429 errors from SEC | Medium |
| Missing email headers | Blocked requests | Medium |
| No centralized error handling | Poor user experience | Medium |

---

## 3. Goals and Objectives

### 3.1 Primary Goals

1. **Consolidate** all SEC API calls into a single, well-tested module
2. **Eliminate** thread safety issues in the GUI
3. **Standardize** rate limiting and request headers
4. **Improve** error handling and user feedback

### 3.2 Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| SEC-related Python files | 6 | 3 |
| Thread safety violations | 15+ | 0 |
| Rate limit errors (per 100 requests) | ~10% | <1% |
| Lines of duplicated code | ~500 | <50 |

### 3.3 Non-Goals

- Adding new SEC filing types (8-K, DEF 14A, etc.)
- Changing the UI layout
- Adding new data extraction features

---

## 4. Proposed Solution

### 4.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         gui.py                               │
│                    (Thread-safe calls)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   sec_api_wrapper.py                         │
│              (Public API - Mock/Real switch)                 │
│                                                              │
│  • get_company_cik(ticker)                                   │
│  • get_latest_filing_info(cik, form_type)                    │
│  • download_filing(filing_info)                              │
│  • extract_tables(html_content)                              │
│  • identify_financial_tables(tables)                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│  sec_api_cache.py   │       │   mock_sec_data.py  │
│  (Real SEC API)     │       │   (Test data)       │
│                     │       │                     │
│  • Rate limiting    │       │  • Sample filings   │
│  • Caching          │       │  • Sample tables    │
│  • Retry logic      │       │                     │
└─────────────────────┘       └─────────────────────┘
```

### 4.2 Module Responsibilities

#### 4.2.1 `sec_api_wrapper.py` (Enhanced)

**Purpose:** Single public interface for all SEC operations

**New Responsibilities:**
- Table extraction (moved from `sec_filing_extractor.py`)
- Financial table identification
- Thread-safe status callbacks

**Interface:**

```python
class SECAPIWrapper:
    def __init__(self, use_mock=False):
        """Initialize with mock or real API"""
    
    def get_company_cik(self, ticker: str) -> Optional[str]:
        """Get CIK for a ticker symbol"""
    
    def get_latest_filing_info(self, cik: str, form_type: str) -> Optional[dict]:
        """Get latest filing metadata"""
    
    def download_filing(self, filing_info: dict) -> Optional[str]:
        """Download filing HTML content"""
    
    def extract_tables(self, html_content: str) -> List[pd.DataFrame]:
        """Extract all tables from HTML"""
    
    def identify_financial_tables(self, tables: List[pd.DataFrame]) -> dict:
        """Identify balance sheet, income statement, cash flow"""
    
    def save_tables_to_excel(self, tables: dict, ticker: str, output_dir: str) -> bool:
        """Save extracted tables to Excel files"""
```

#### 4.2.2 `sec_api_cache.py` (Refactored)

**Purpose:** Handle all real SEC API requests with caching and rate limiting

**Changes:**
- Add function to clear in-memory cache
- Make cache directory configurable
- Standardize headers in one place
- Add specific exception types

**Interface:**

```python
# Configuration
def configure(cache_dir: str = None, email: str = None):
    """Configure cache settings"""

def clear_cache(include_memory: bool = True):
    """Clear file and optionally in-memory cache"""

# Core functions (internal use)
def get_company_cik(ticker: str) -> Optional[str]
def get_company_submissions(cik: str) -> Optional[dict]
def get_latest_filing_info(cik: str, form_type: str) -> Optional[dict]
def download_filing(filing_info: dict) -> Optional[str]
```

#### 4.2.3 `gui.py` (SEC-related changes)

**Purpose:** Thread-safe UI for SEC operations

**Changes:**
- Use `safe_update_text_widget` for all text updates
- Use `safe_update_status` for status bar updates
- Add progress callback support
- Proper error display

---

## 5. Detailed Requirements

### 5.1 Thread Safety (P0 - Critical)

#### REQ-TS-001: Use Thread-Safe Widget Updates

**Current Code (gui.py:6593-6594):**
```python
self.business_analysis_text.delete("1.0", tk.END)
self.business_analysis_text.insert(tk.END, f"Extracting...")
```

**Required Change:**
```python
safe_update_text_widget(
    self.business_analysis_text, 
    f"Extracting {form_type} tables for {ticker}...", 
    append=False
)
```

#### REQ-TS-002: Thread-Safe Status Updates

All `self.sec_status_var.set()` calls in background threads must use:
```python
self.root.after(0, lambda: self.sec_status_var.set(message))
```

#### REQ-TS-003: Progress Callback Pattern

Implement a callback pattern for progress updates:
```python
def _extract_sec_filing(self, form_type):
    def on_progress(step: str, message: str):
        self.root.after(0, lambda: self._update_sec_progress(step, message))
    
    def extraction_thread():
        api = sec_api_wrapper.sec_api
        api.extract_with_progress(ticker, form_type, callback=on_progress)
    
    threading.Thread(target=extraction_thread, daemon=True).start()
```

### 5.2 Code Consolidation (P0 - Critical)

#### REQ-CC-001: Deprecate Redundant Files

Mark the following files as deprecated with warnings:
- `sec_filing_downloader.py`
- `sec_edgar_tables.py`

Add deprecation notice:
```python
import warnings
warnings.warn(
    "sec_filing_downloader is deprecated. Use sec_api_wrapper instead.",
    DeprecationWarning,
    stacklevel=2
)
```

#### REQ-CC-002: Move Table Extraction

Move `extract_tables()` and `identify_financial_tables()` from `sec_filing_extractor.py` to `sec_api_wrapper.py`.

#### REQ-CC-003: Single Header Definition

Define headers once in `sec_api_cache.py`:
```python
SEC_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9...",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "From": None  # Set dynamically from config
}

def get_headers() -> dict:
    headers = SEC_HEADERS.copy()
    headers["From"] = os.getenv("SEC_EDGAR_EMAIL", "user@example.com")
    return headers
```

### 5.3 Rate Limiting (P1 - High)

#### REQ-RL-001: Standardized Rate Limiting

All SEC requests must use these parameters:
```python
MIN_DELAY = 10  # seconds between requests
MAX_RETRIES = 5
BASE_BACKOFF = 10  # seconds
MAX_BACKOFF = 300  # 5 minutes
```

#### REQ-RL-002: Request Queue

Implement a request queue to prevent concurrent requests:
```python
import threading
from queue import Queue

class SECRequestQueue:
    def __init__(self):
        self._queue = Queue()
        self._lock = threading.Lock()
        self._last_request_time = 0
    
    def execute(self, request_func, *args, **kwargs):
        with self._lock:
            self._wait_for_rate_limit()
            result = request_func(*args, **kwargs)
            self._last_request_time = time.time()
            return result
```

### 5.4 Error Handling (P1 - High)

#### REQ-EH-001: Custom Exception Types

```python
class SECAPIError(Exception):
    """Base exception for SEC API errors"""
    pass

class SECRateLimitError(SECAPIError):
    """Raised when rate limited by SEC"""
    pass

class SECNotFoundError(SECAPIError):
    """Raised when ticker/filing not found"""
    pass

class SECNetworkError(SECAPIError):
    """Raised on network failures"""
    pass
```

#### REQ-EH-002: User-Friendly Error Messages

Map exceptions to user-friendly messages:
```python
ERROR_MESSAGES = {
    SECRateLimitError: "SEC is temporarily limiting requests. Please wait a few minutes and try again.",
    SECNotFoundError: "Could not find {ticker} in SEC database. Please verify the ticker symbol.",
    SECNetworkError: "Network error connecting to SEC. Please check your internet connection.",
}
```

### 5.5 Cache Management (P2 - Medium)

#### REQ-CM-001: Configurable Cache Directory

```python
def configure_cache(cache_dir: str = None):
    global CACHE_DIR
    if cache_dir:
        CACHE_DIR = Path(cache_dir)
    else:
        CACHE_DIR = Path(os.getenv("SEC_CACHE_DIR", "sec_cache"))
    CACHE_DIR.mkdir(exist_ok=True)
```

#### REQ-CM-002: Clear In-Memory Cache

```python
def clear_all_cache():
    global company_tickers_cache, company_tickers_last_update
    
    # Clear in-memory cache
    company_tickers_cache = None
    company_tickers_last_update = None
    
    # Clear file cache
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(exist_ok=True)
```

### 5.6 Mock Data Sync (P2 - Medium)

#### REQ-MD-001: Sync Mock Toggle with API Instance

```python
def _extract_sec_filing(self, form_type):
    # Sync mock setting BEFORE getting API instance
    use_mock = self.use_mock_data_var.get()
    sec_api_wrapper.use_mock_sec_api(use_mock)
    api = sec_api_wrapper.sec_api
    
    # Now proceed with extraction...
```

---

## 6. Implementation Plan

### Phase 1: Thread Safety Fixes (Week 1)

| Task | File | Effort | Owner |
|------|------|--------|-------|
| Replace direct widget updates with safe methods | gui.py | 4h | - |
| Add progress callback pattern | gui.py | 2h | - |
| Test thread safety | - | 2h | - |

**Deliverables:**
- All 15+ thread safety violations fixed
- No GUI crashes during SEC operations

### Phase 2: Code Consolidation (Week 2)

| Task | File | Effort | Owner |
|------|------|--------|-------|
| Move table extraction to wrapper | sec_api_wrapper.py | 3h | - |
| Add deprecation warnings | sec_filing_downloader.py, sec_edgar_tables.py | 1h | - |
| Standardize headers | sec_api_cache.py | 1h | - |
| Update all imports in gui.py | gui.py | 2h | - |
| Remove duplicate code | Multiple | 2h | - |

**Deliverables:**
- Single source of truth for SEC API calls
- Deprecated files marked with warnings

### Phase 3: Error Handling & Rate Limiting (Week 3)

| Task | File | Effort | Owner |
|------|------|--------|-------|
| Create custom exceptions | sec_api_cache.py | 2h | - |
| Implement request queue | sec_api_cache.py | 3h | - |
| Add user-friendly error messages | gui.py | 2h | - |
| Test rate limiting | - | 2h | - |

**Deliverables:**
- Consistent rate limiting across all requests
- Clear error messages for users

### Phase 4: Cache & Configuration (Week 4)

| Task | File | Effort | Owner |
|------|------|--------|-------|
| Make cache configurable | sec_api_cache.py | 2h | - |
| Fix in-memory cache clearing | sec_api_cache.py, gui.py | 1h | - |
| Fix mock data toggle sync | gui.py | 1h | - |
| Final testing | - | 4h | - |
| Documentation | - | 2h | - |

**Deliverables:**
- Configurable cache directory
- Complete cache clearing functionality
- Updated documentation

---

## 7. Testing Requirements

### 7.1 Unit Tests

```python
# test_sec_api_wrapper.py

def test_get_company_cik_valid_ticker():
    """Test CIK lookup for valid ticker"""
    api = SECAPIWrapper(use_mock=True)
    cik = api.get_company_cik("AAPL")
    assert cik == "0000320193"

def test_get_company_cik_invalid_ticker():
    """Test CIK lookup for invalid ticker"""
    api = SECAPIWrapper(use_mock=True)
    cik = api.get_company_cik("INVALID123")
    assert cik is None

def test_rate_limiting():
    """Test that requests are properly rate limited"""
    api = SECAPIWrapper(use_mock=False)
    start = time.time()
    api.get_company_cik("AAPL")
    api.get_company_cik("MSFT")
    elapsed = time.time() - start
    assert elapsed >= 10  # MIN_DELAY

def test_cache_clearing():
    """Test that cache is properly cleared"""
    api = SECAPIWrapper(use_mock=False)
    api.get_company_cik("AAPL")  # Populate cache
    sec_api_cache.clear_all_cache()
    assert sec_api_cache.company_tickers_cache is None
```

### 7.2 Integration Tests

```python
# test_sec_integration.py

def test_full_extraction_workflow():
    """Test complete extraction from ticker to Excel"""
    api = SECAPIWrapper(use_mock=True)
    
    cik = api.get_company_cik("AAPL")
    assert cik is not None
    
    filing_info = api.get_latest_filing_info(cik, "10-K")
    assert filing_info is not None
    
    html = api.download_filing(filing_info)
    assert html is not None
    
    tables = api.extract_tables(html)
    assert len(tables) > 0

def test_thread_safety():
    """Test concurrent extractions don't crash"""
    import concurrent.futures
    
    api = SECAPIWrapper(use_mock=True)
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(api.get_company_cik, t) for t in tickers]
        results = [f.result() for f in futures]
    
    assert all(r is not None for r in results)
```

### 7.3 Manual Testing Checklist

- [ ] Extract 10-K for AAPL with real API
- [ ] Extract 10-Q for MSFT with real API
- [ ] Toggle mock data and verify behavior changes
- [ ] Clear cache and verify fresh data is fetched
- [ ] Trigger rate limit and verify graceful handling
- [ ] Test with invalid ticker
- [ ] Test with network disconnected
- [ ] Verify no GUI freezing during extraction
- [ ] Verify progress updates display correctly

---

## 8. Rollback Plan

If issues are discovered after deployment:

1. **Immediate:** Revert to previous version of affected files
2. **Short-term:** Re-enable deprecated files without warnings
3. **Communication:** Update users via status message

Git tags for rollback:
- `pre-sec-refactor` - Tag before any changes
- `phase-1-complete` - After thread safety fixes
- `phase-2-complete` - After consolidation
- `phase-3-complete` - After error handling
- `phase-4-complete` - Final release

---

## 9. Documentation Updates

### 9.1 Code Documentation

- Add docstrings to all public methods in `sec_api_wrapper.py`
- Update inline comments explaining rate limiting logic
- Add type hints to all function signatures

### 9.2 User Documentation

- Update README with SEC feature usage
- Document environment variables (`SEC_EDGAR_EMAIL`, `SEC_CACHE_DIR`)
- Add troubleshooting section for common SEC errors

---

## 10. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking existing functionality | Medium | High | Comprehensive testing, rollback plan |
| SEC API changes | Low | High | Abstract API calls, monitor SEC announcements |
| Performance regression | Low | Medium | Benchmark before/after |
| User confusion during transition | Medium | Low | Clear deprecation warnings |

---

## 11. Appendix

### A. Files to Modify

| File | Changes |
|------|---------|
| `sec_api_wrapper.py` | Add table extraction, progress callbacks |
| `sec_api_cache.py` | Add cache clearing, configurable directory |
| `gui.py` | Thread-safe updates, mock sync fix |
| `sec_filing_extractor.py` | Keep for backward compatibility, add deprecation |
| `sec_filing_downloader.py` | Add deprecation warning |
| `sec_edgar_tables.py` | Add deprecation warning |

### B. Files to Delete (Future)

After deprecation period (3 months):
- `sec_filing_downloader.py`
- `sec_edgar_tables.py`
- `sec_edgar_helper.py` (if unused)

### C. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SEC_EDGAR_EMAIL` | `user@example.com` | Email for SEC API requests |
| `SEC_CACHE_DIR` | `./sec_cache` | Directory for cached responses |
| `SEC_MIN_DELAY` | `10` | Minimum seconds between requests |

---

## 12. Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Owner | | | |
| Tech Lead | | | |
| QA Lead | | | |
