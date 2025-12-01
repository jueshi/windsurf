# SEC Features Review - Potential Issues

## Overview
This document identifies potential issues in the SEC-related features of the stock_charts_10k10q application.

---

## 1. Code Duplication (HIGH PRIORITY)

### Issue
There are **multiple files** implementing the same SEC functionality with slight variations:

| File | Purpose | Has CIK Lookup | Has Filing Download | Has Caching |
|------|---------|----------------|---------------------|-------------|
| `sec_api_cache.py` | Main cached API | ✅ | ✅ | ✅ |
| `sec_api_wrapper.py` | Wrapper for mock/real | ✅ (via cache) | ✅ | ✅ |
| `sec_filing_extractor.py` | Table extraction | ✅ | ✅ | ❌ |
| `sec_filing_downloader.py` | Simple downloader | ✅ | ✅ | ❌ |
| `sec_edgar_tables.py` | Table extraction | ✅ | ✅ | ❌ |
| `sec_edgar_helper.py` | Uses sec_edgar_downloader pkg | ❌ | ✅ | ❌ |

### Impact
- **Inconsistent behavior**: Different files have different rate limiting strategies
- **Maintenance burden**: Bug fixes need to be applied in multiple places
- **Confusion**: Unclear which module to use for what purpose

### Recommendation
Consolidate to use `sec_api_wrapper.py` + `sec_api_cache.py` as the single source of truth. Other modules should import from these.

---

## 2. Thread Safety Issues (MEDIUM-HIGH PRIORITY)

### Issue
In `gui.py`, the `_extract_sec_filing` method runs in a background thread but directly modifies Tkinter widgets:

```python
# Line 6593-6594 in gui.py
self.business_analysis_text.delete("1.0", tk.END)
self.business_analysis_text.insert(tk.END, f"Extracting {form_type} tables for {ticker}...\n\n")
```

### Impact
- **Potential crashes**: Tkinter is not thread-safe; modifying widgets from non-main threads can cause crashes
- **Race conditions**: Multiple extractions could conflict

### Recommendation
Use the existing `safe_update_text_widget` and `safe_update_status` functions from `thread_safe_tkinter.py`:

```python
# Instead of direct widget modification:
safe_update_text_widget(self.business_analysis_text, "Extracting...", append=False)
```

---

## 3. Inconsistent User-Agent Headers (MEDIUM PRIORITY)

### Issue
Different files use different User-Agent strings:

**sec_api_cache.py (line 54):**
```python
"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.67"
```

**sec_filing_downloader.py (line 12):**
```python
"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
```

**sec_edgar_tables.py (line 15):**
```python
"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
```

### Impact
- SEC may block requests if they detect inconsistent patterns
- Harder to maintain

### Recommendation
Define headers in one place (`sec_api_cache.py`) and import everywhere else.

---

## 4. Missing Email Header in Some Files (MEDIUM PRIORITY)

### Issue
SEC requires an email in the `From` header for API requests. This is present in `sec_api_cache.py` and `sec_filing_extractor.py` but **missing** in:
- `sec_filing_downloader.py`
- `sec_edgar_tables.py`

### Impact
- Requests may be rate-limited or blocked more aggressively

### Recommendation
Add `"From": email` header to all SEC request functions.

---

## 5. Rate Limiting Inconsistency (MEDIUM PRIORITY)

### Issue
Different rate limiting strategies across files:

| File | Min Delay | Max Retries | Backoff Strategy |
|------|-----------|-------------|------------------|
| `sec_api_cache.py` | 10s | 5 | Exponential with jitter |
| `sec_filing_extractor.py` | 1s | 5 | Exponential (2^n) |
| `sec_filing_downloader.py` | 1s | 0 | None |
| `sec_edgar_tables.py` | 0s | 0 | None |

### Impact
- Files without proper rate limiting may get 403/429 errors
- Inconsistent user experience

### Recommendation
All SEC requests should go through `sec_api_cache.py` which has proper rate limiting.

---

## 6. Mock Data Not Synced with GUI Toggle (LOW-MEDIUM PRIORITY)

### Issue
In `gui.py` line 6589-6590:
```python
api = sec_api_wrapper.sec_api
using_mock = self.use_mock_data_var.get()
```

The `api` is fetched **before** checking the mock toggle. If the user toggles mock data **after** the extraction starts, the wrong API might be used.

### Impact
- Potential confusion if user toggles during extraction

### Recommendation
Fetch the API instance **after** checking the toggle, or use `sec_api_wrapper.use_mock_sec_api(use_mock)` at the start of extraction.

---

## 7. Hardcoded Debug File in sec_edgar_tables.py (LOW PRIORITY)

### Issue
Line 157-159 in `sec_edgar_tables.py`:
```python
with open("filing.html", "w", encoding="utf-8") as f:
    f.write(html_content)
print("Saved HTML content to filing.html")
```

### Impact
- Creates unwanted files in the working directory
- Overwrites previous debug files

### Recommendation
Remove or make this optional via a debug flag.

---

## 8. Missing Error Handling for Network Issues (LOW PRIORITY)

### Issue
Some functions don't handle specific network exceptions:
- `ConnectionError`
- `Timeout`
- `SSLError`

### Impact
- Generic error messages that don't help users troubleshoot

### Recommendation
Add specific exception handling with user-friendly messages.

---

## 9. Cache Directory Not Configurable (LOW PRIORITY)

### Issue
Cache directory is hardcoded in `sec_api_cache.py`:
```python
CACHE_DIR = Path("sec_cache")
```

### Impact
- Users can't customize cache location
- May cause issues if running from different directories

### Recommendation
Make cache directory configurable via environment variable or config file.

---

## 10. In-Memory Cache Not Cleared on Toggle (LOW PRIORITY)

### Issue
In `sec_api_cache.py`, there's an in-memory cache for company tickers:
```python
company_tickers_cache = None
company_tickers_last_update = None
```

When user clicks "Clear Cache" in GUI, only the file cache is cleared, not the in-memory cache.

### Impact
- Stale data may persist until application restart

### Recommendation
Add a function to clear in-memory cache and call it from `_clear_sec_cache`.

---

## Summary of Priorities

| Priority | Issue | Effort |
|----------|-------|--------|
| HIGH | Code duplication | High |
| MEDIUM-HIGH | Thread safety | Medium |
| MEDIUM | Inconsistent headers | Low |
| MEDIUM | Missing email header | Low |
| MEDIUM | Rate limiting inconsistency | Medium |
| LOW-MEDIUM | Mock data toggle sync | Low |
| LOW | Hardcoded debug file | Low |
| LOW | Network error handling | Medium |
| LOW | Cache directory config | Low |
| LOW | In-memory cache clear | Low |

---

## Recommended Actions

1. **Short-term**: Fix thread safety issues in `gui.py` to prevent crashes
2. **Medium-term**: Consolidate all SEC API calls to use `sec_api_wrapper.py`
3. **Long-term**: Refactor to have a single, well-tested SEC module
