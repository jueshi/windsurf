# Worklog

## 2025-11-05

- **Fix syntax error in `ticker_lists.py` (line ~519)**
  - Issue: Invalid identifier `S&P_500_Dividend_Aristocrats_stocks` (ampersand is not allowed in Python identifiers) caused parse error: "cannot assign to expression here".
  - Change: Renamed to `SP500_Dividend_Aristocrats_stocks`.
  - File: `stock_charts_10k10q/ticker_lists.py`.

- **Fix Tkinter thread patching error in `main.py`**
  - Issue: Crash on startup with "type object 'Tk' has no attribute 'call'".
  - Root cause: Attempted to patch `tk.Tk.call`. The Tkinter `call` method lives on `tk.Misc`, not the `Tk` class.
  - Change: Patch `tk.Misc.call` instead of `tk.Tk.call` while preserving thread check.
  - File: `stock_charts_10k10q/main.py`.

- **Review SEC API usage (real API with caching and rate limiting)**
  - Confirmed real SEC endpoints are used via `sec_api_cache.py` and wrapped by `sec_api_wrapper.py`.
  - Caching: On-disk JSON cache under `sec_cache/` with per-endpoint expirations (tickers: 7d, submissions: 1d, filings: 30d). In-memory cache for company tickers also present.
  - Rate limiting: Enforces minimum 10s delay between requests and implements retries with exponential backoff and jitter; handles 403/429 with longer waits.
  - Toggle: GUI provides a mock/real toggle via `sec_api_wrapper.use_mock_sec_api`.

## Validation steps

1. Launch the app normally.
2. Verify no startup error about `Tk.call`.
3. Open any feature that loads ticker lists and the watch list; confirm no parse error from `ticker_lists.py`.
4. In the GUI, ensure status shows "Using real SEC API with caching" when mock is off. Optionally toggle to mock and back.
5. Optionally clear the SEC cache from the GUI and re-fetch to validate caching/rate limiting still behave.

## Notes

- No comments or docstrings were removed during these fixes.
- No external dependencies changed.

## 2025-11-27

- **Surfaced available Gemini models on unsupported/404 errors**
  - Issue: Users saw `An error occurred... models/gemini-3.0-pro is not found` without guidance on available models.
  - Change: Added `_list_supported_gemini_models()` and enhanced `_format_gemini_error()` to call `genai.list_models()` and include accessible model names when we hit 404/unsupported responses.
  - Files: `stock_charts_10k10q/gemini_analyzer.py`.
  - Validation: Triggered the error path manually to confirm the returned message now lists available models when `genai.list_models()` succeeds, or advises running the command if listing fails.

## 2025-11-28

- **Beautified Business Analysis output to infographic-style layout**
  - Added `_beautify_business_analysis` helper plus banner, snapshot, and section-parsing utilities to `gui.py`.
  - Hooked beautifier into BA runs, cached loads, 10-K/10-Q studies, and markdown caching so the text widget shows a structured layout similar to the provided reference image.
  - Updated WORKLOG per user request.

- **Reduced blank space under action buttons**
  - Replaced the wrapper grid with a vertical `PanedWindow`, tightened the action pane padding, and bound sash initialization to `<Configure>` so the divider stays near the bottom yet remains draggable.
  - File: `stock_charts_10k10q/gui.py`.

- **Added Finviz ETF news summarizer**
  - Replicated the stock-news workflow for ETFs: new GUI button, dedicated fetcher for the v=4 feed, shared ticker persistence, and Gemini summarization hook.
  - Refactored Finviz parsing helpers for reuse and added `summarize_etf_news` in `gemini_analyzer.py`.
  - Files: `stock_charts_10k10q/gui.py`, `stock_charts_10k10q/gemini_analyzer.py`.

- **Added Finviz Crypto news summarizer**
  - Mirrored the stock/ETF flow for v=5 crypto headlines: button, threaded workflow, feed parser, and ticker persistence.
  - Implemented `summarize_crypto_news` in `gemini_analyzer.py` leveraging the existing Gemini pipeline.
  - Files: `stock_charts_10k10q/gui.py`, `stock_charts_10k10q/gemini_analyzer.py`.
