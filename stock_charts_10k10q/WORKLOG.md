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

- **Built tooltip infrastructure and initial coverage**
  - Added reusable `TooltipManager` module with hover delay control, Shift+F1 activation, and weakref registry cleanup.
  - Integrated manager into `StockDataGUI`, added Show Tooltips toggle, and instrumented ticker list controls plus chart date range widgets with descriptive tooltips aligned to the new PRD.
  - Files: `stock_charts_10k10q/tooltip_manager.py`, `stock_charts_10k10q/gui.py`.

## 2025-12-19

- **Added insider tooltip support**
  - Goal: Identify which insider sold each block directly on the chart.
  - Change: Wrapped the scatter plot with `mplcursors` hover tooltips (with graceful fallback if the library isn’t installed) showing name, official title, transaction type, date, shares, value, and price when the user hovers over a bubble.
  - Files: `stock_charts_10k10q/insider_sale_data.py`.

- **Added trend line to volume dot chart**
  - Goal: Complement the bubble plot with a continuous view of price action.
  - Change: Plotted a semi-transparent midnight-blue line connecting the closing prices before drawing the scatter markers so viewers can visually follow the trend between dots.
  - File: `stock_charts_10k10q/volumn_dot_chart.py`.

- **Fixed matplotlib color array shape in volume dot chart**
  - Issue: `plt.scatter` raised `ValueError: 'c' argument must be a color...` because `np.where` on pandas Series returned a column vector (`(n, 1)`) instead of a flat list, so matplotlib rejected it.
  - Change: Convert Open/Close/Volume Series to flattened NumPy arrays before computing colors, then compare those arrays so `np.where` yields a 1-D list of `'green'/'red'` strings that matplotlib accepts.
  - File: `stock_charts_10k10q/volumn_dot_chart.py`.

- **Restored Insider Sales visualization data parsing**
  - Issue: Finviz copy/paste delivered a single line containing every row, so `parse_line` never matched and the chart rendered empty.
  - Change: Added `prepare_lines` to split entries on timestamps and drop the trailing `"Dec 11 04:26 PM"` timestamps before parsing transaction details. The pipeline now produces 100 parsed entries with 64 real sales for plotting.
  - Validation: Imported the module with a headless Matplotlib backend to ensure the DataFrame is populated (parsed=100, sales=64) before plotting.

- **Hardened insider parser for variable date formats**
  - Issue: Some Finviz rows omit the leading zero in the day (e.g., `Dec 5 '25`), causing our strict regex to skip entries and leave the scatter empty.
  - Change: Replaced the ad-hoc line splitter with a true regex-based `parse_transactions` pipeline that supports one- or two-digit days both in the date and trailing timestamp, with a line-by-line fallback for unexpected layouts.
  - Validation: Re-imported the script under Agg backend to confirm we still parse 100 rows / 64 sales and the chart renders once `plt.show()` is allowed.

- **Fixed yfinance insider sale parser when column names drift**
  - Issue: `sales_df = df[df['Transaction'].str.contains(...)]` crashed because current yfinance frames expose `Transaction` as a column of dictionaries instead of strings.
  - Change: Normalize whatever combination of `Transaction`, `Text`, and `Transaction` fields exist (renaming to `TransactionType` / `TransactionText`) and build the sale filter by scanning every available descriptor so the code keeps working even if only one of them is textual.
  - Validation: Unable to re-run the plot locally because `seaborn` is missing in this environment; logic verified by re-importing the module to ensure no `.str` errors occur. Please install seaborn (`pip install seaborn`) to render the chart.

- **Added retry/backoff + cache fallback for insider fetches**
  - Issue: Yahoo Finance intermittently returns HTTP 429, leaving the script with an empty DataFrame and no visualization.
  - Change: Implemented exponential backoff retries, on-disk caching of the most recent successful sales DataFrame under `insider_cache/`, and graceful fallback to cached data whenever live fetches fail or return no sale rows.
  - Validation: Triggered the fetch with network calls blocked to ensure we now surface “Loaded cached insider sales...” instead of aborting; real fetch still depends on Yahoo lifting the rate limit.

- **Added Finviz HTML fallback when Yahoo blocks us**
  - Issue: Persistent 429s meant even the cache couldn’t help on first run.
  - Change: When yfinance is empty or missing transaction columns, fetch the Finviz quote page directly, parse the insider table via `read_html(StringIO(...))`, downcast multi-index headers, filter for Sale rows, and normalize the numeric fields before feeding the rest of the pipeline.
  - Validation: Re-ran `py -3 insider_sale_data.py` while Yahoo was throttling; script now logs “Trying Finviz fallback...” and proceeds once Finviz responds. FutureWarning about literal HTML is benign and now suppressed by StringIO.

## 2025-12-21

- **Fixed Powerball CSV path resolution**
  - Issue: Running `powerball_number_rank.py` from outside its folder raised `FileNotFoundError` because it relied on the working directory to locate `Lottery_Powerball_Winning_Numbers__Beginning_2010.csv`.
  - Change: Build the CSV path relative to the script (`Path(__file__).with_name(...)`) so the data file is found regardless of where the script is launched.
  - Files: `stock_charts_10k10q/powerball_number_rank.py`.
  - Validation: Script now reads the CSV successfully when executed from the repo root (`py stock_charts_10k10q/powerball_number_rank.py`).

## 2025-12-24

## 2026-03-15

- **Fixed startup crash when `tkcalendar` is not installed**
  - Issue: App crashed at import time with `ModuleNotFoundError: No module named 'tkcalendar'`.
  - Change: Made `tkcalendar.DateEntry` optional by falling back to `custom_widgets.CustomDateEntry` when `tkcalendar` is missing.
  - Files: `stock_charts_10k10q/gui.py`, `stock_charts_10k10q/data_manager.py`.

- **Fixed startup crash when `sec_edgar_downloader` is not installed**
  - Issue: App crashed at import time with `ModuleNotFoundError: No module named 'sec_edgar_downloader'`.
  - Change: Made `sec_edgar_downloader.Downloader` optional with try/except import and graceful fallback to URL-based SEC filing retrieval.
  - File: `stock_charts_10k10q/gemini_analyzer.py`.

- **Fixed startup crash when `serpapi` is not installed**
  - Issue: App crashed at import time with `ModuleNotFoundError: No module named 'serpapi'`.
  - Change: Made `serpapi.GoogleSearch` optional with try/except import; `_get_filing_url_via_serpapi()` now returns SEC company search page URL when serpapi unavailable.
  - File: `stock_charts_10k10q/gemini_analyzer.py`.

- **Fixed startup crash when `google.generativeai` is not installed**
  - Issue: App crashed at import time with `ModuleNotFoundError: No module named 'google.generativeai'`.
  - Change: Made `google.generativeai` optional with try/except import; added `GEMINI_AVAILABLE` flag and guards to all Gemini-dependent functions.
  - File: `stock_charts_10k10q/gemini_analyzer.py`.

## Validation steps (2026-03-15)

1. Run the app without installing `tkcalendar`, `sec_edgar_downloader`, `serpapi`, or `google.generativeai`; confirm it launches.
2. (Optional) Install missing packages and re-run; confirm native implementations are used.

- **Overlaid price trend on PE chart**
  - Added a twin-y axis so closing price renders alongside the trailing PE series, updated legend/title, and switched to `Series.ffill()` to avoid the pandas deprecation warning.
  - File: `stock_charts_10k10q/PE_chart.py`.
  - Validation: Requires local `yfinance` install; user confirmed chart renders via IDE run.

- **Extended EPS history using annual + quarterly financials**
  - New helpers `_extract_eps_row` and `_build_eps_series` combine quarterly and annual EPS data (diluted/basic) with timezone normalization, deduplicate overlapping periods, and back/forward-fill earlier dates so the entire `period` window gets PE coverage.
  - File: `stock_charts_10k10q/PE_chart.py`.
  - Validation: Running `python PE_chart.py` still depends on `yfinance`; environment currently missing the package in this shell, so user should rerun after installing.

- **Annualized quarterly EPS for consistent PE**
  - Applied a 4-quarter rolling sum to quarterly EPS before merging with annual data so the PE calculation always uses a trailing-twelve-month denominator; kept annual entries as-is.
  - File: `stock_charts_10k10q/PE_chart.py`.
  - Validation: Script requires `yfinance`; rerun locally once available to visualize the updated PE series.

- **Plotted annualized EPS in dedicated pane**
  - Reworked `plot_pe` to use a two-row layout: top panel retains PE + price overlay, bottom panel shows the TTM EPS line with annotation for the latest value.
  - File: `stock_charts_10k10q/PE_chart.py`.
  - Validation: User ran the script via IDE (AMD, 10y) and confirmed summary stats printed.

## 2025-12-25

- **Rebuilt institutional ownership visualization**
  - Replaced the PE/EPS logic with an institutional ownership dot chart: aggregate Yahoo Finance institutional holdings by report date, compute share deltas, attach the nearest closing price, and scatter green/red dots at those prices with share-change annotations (above price for increases, below for reductions).
  - Files: `stock_charts_10k10q/insitutional_ownership.py`.
  - Validation: Run `python insitutional_ownership.py` after installing `yfinance` to generate the updated plot.

- **Hardened smart-money summary script**
  - Rewrote `institutional_ownership_data.py` to normalize yfinance schemas: resilient percentage parsing, slugified column headers, formatted share counts, and short-interest messaging so the CLI summary no longer crashes when `% Out` is missing.
  - Integrated FinancialModelingPrep institutional-change feed (auto-loads `FMP_API_KEY` from environment/.env) so we print top change percentages alongside YF data.
  - File: `stock_charts_10k10q/institutional_ownership_data.py`.
  - Validation: Requires `yfinance` + `requests`; run `python institutional_ownership_data.py` to print the MRVL/AVGO/NVDA summaries.

## 2025-12-29

- **Bundled FFmpeg fallback + clearer guidance for audio tool**
  - Added `typing.Optional` import plus helper routines inside `ensure_ffmpeg_in_path`.
  - New behavior tries existing PATH, common Windows install folders, and finally a portable binary from `imageio-ffmpeg`, wiring its bin directory into PATH automatically.
  - When all attempts fail, the script now prints the checked locations, mentions the portable attempt, and exits with instructions instead of throwing `FileNotFoundError`.
  - File: `stock_charts_10k10q/audio2text_v5.2.1_rename_MD.py`.
  - Validation: Static inspection; please rerun the script to confirm the bundled fallback message appears when FFmpeg is absent.
