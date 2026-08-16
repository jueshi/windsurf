# PRD: Blackout Period Performance Tab

**Project:** Personal AI Stock Assistant (stock_charts_10k10q)
**Branch:** web-app-conversion
**Date:** 2026-08-14
**Status:** Draft — awaiting implementation

---

## Problem Statement

Insider trading blackout periods end at (or shortly after) quarterly earnings announcements. The user believes there may be a repeatable post-blackout price effect — a drift that unfolds over the days and weeks after insiders are allowed to trade again — but currently has no way to test this hypothesis against history. Manually collecting earnings dates, aligning price data, and computing returns at multiple horizons for every past quarter is tedious and error-prone. The app already shows charts, relative strength, seasonality, and insider sales, but nothing connects *when blackouts ended* to *what the price did afterward*.

## Solution

Add a new top-level tab ("⏳ Blackout") to the main chart notebook. For a single selected ticker, the tab:

1. Retrieves all historical quarterly earnings announcement dates (blackout end = earnings date; blackout start = a configurable number of calendar days before earnings, default 30).
2. Fetches the ticker's full available daily price history.
3. For each past quarter, measures price performance at N = 1, 3, 7, 14, 21, 28, 35, 42 calendar days after blackout end, using the close on the blackout-end date as baseline (buy price) and the close on the first trading day on/after each horizon date as the exit price.
4. Plots, for each horizon N, the cumulative compounded percentage gain over chronological quarters — i.e., "what would $1 become if you bought at blackout-end close and sold N days later, every quarter, reinvesting each time."
5. Offers a second chart view, **Daily Equity**: the same strategies marked to market every trading day (rising while holding during each N-day window, flat while in cash between quarters), plotted against actual calendar dates with each horizon's final % in the legend. A radio toggle switches between the two views instantly (both are rendered once per run).
6. Shows a per-quarter data table (blackout start, blackout end, baseline close, % gain at each of the 8 horizons) so the chart can be verified against raw numbers.

Only fully-elapsed windows are included in each horizon's compounding line, so the 42-day line automatically stops earlier than the 1-day line for the most recent quarters.

## User Stories

1. As a retail investor, I want to select a ticker I already follow and immediately see all its historical blackout periods, so that I don't have to look up earnings dates manually.
2. As a retail investor, I want the app to compute price performance 1, 3, 7, 14, 21, 28, 35, and 42 calendar days after each blackout end, so that I can see whether a post-blackout drift exists at short and medium horizons.
3. As a retail investor, I want the chart to show cumulative compounded gains over chronological quarters with one line per horizon, so that I can judge whether a hypothetical "buy at blackout end, sell N days later" strategy would have compounded wealth over the years.
4. As a retail investor, I want a per-quarter table showing blackout start/end, baseline close, and % gain at each horizon, so that I can verify the chart and spot outlier quarters driving the curve.
5. As a retail investor, I want weekends and market holidays handled by rolling forward to the next trading day, so that horizon returns are well-defined for every quarter.
6. As a retail investor, I want only fully-elapsed windows included per horizon, so that recent, incomplete quarters don't distort the longer-horizon lines.
7. As a retail investor, I want the blackout start offset (days before earnings) to be configurable in the UI, so that I can experiment with different blackout-length assumptions.
8. As a retail investor, I want analysis over all available history, so that the compounding covers as many quarters as the data allows.
9. As a retail investor, I want the analysis to run in a background thread with a progress/status indicator, so that the UI stays responsive while data is fetched.
10. As a retail investor, I want price data cached on disk, so that re-running the analysis for the same ticker is fast and doesn't re-hit the network.
11. As a retail investor, I want clear error messaging when earnings dates or price data are unavailable for a ticker, so that I know why a chart is empty.
12. As a retail investor, I want the tab to tell me how many quarters were analyzed and each horizon's window count, so that I know how much statistical weight to give the curves.
13. As a developer, I want the analysis logic in a standalone pure-Python module, so that I can unit-test window math, horizon returns, and compounding without a GUI or network.
14. As a developer, I want the tab to reuse the app's existing ticker-selection helper, so that it behaves consistently with every other tab.
15. As a developer, I want full-history price fetching added to the existing data manager, so that other future features can also use max-period data.

## Implementation Decisions

- **New tab, top-level.** A "⏳ Blackout" tab is added to the main chart notebook alongside Chart/Compare/Sectors/Seasonal/etc. The tab-creation pattern, `active_tab` assignment in the tab-changed handler, and ticker-reactivity hooks follow the existing tab conventions.
- **Standalone analysis module.** All computation lives in a new module (e.g., `blackout_analysis.py`) with pure, GUI-free functions following the `relative_strength.py` pattern:
  - `get_blackout_periods(ticker, blackout_start_days) → list of {quarter_label, blackout_start, blackout_end (= earnings date)}`
  - `compute_quarter_gains(price_history, blackout_periods, horizons) → per-quarter baseline close + % gain at each horizon`
  - `compute_compounded_curves(quarter_gains, horizons) → per-horizon compounded % gain series over chronological quarters`
- **Blackout end = earnings announcement date.** Sourced from yfinance's historical earnings dates (exact timestamps, including EPS estimate/reported/surprise — the surprise column is fetched but optional to display later). Blackout start = configurable calendar days before earnings (default 30, adjustable via a spinbox in the tab). These are *derived* windows, not recovered from Form 4 filings.
- **Baseline price** = close on the blackout-end date (first trading day on/after the earnings date). **Exit price** for horizon N = close on the first trading day on/after (blackout_end + N calendar days).
- **Compounding semantics.** For horizon N: cumulative product of (1 + quarterly return) across all completed windows in chronological order, expressed as % gain from $1. Each horizon's line spans only the quarters where that horizon has fully elapsed.
- **All available history**, bounded in practice by how far back the earnings-dates API returns data (typically a few years of quarters). The UI displays the number of quarters and window counts per horizon so coverage is transparent.
- **Price data via data_manager.** The data manager gains the ability to fetch/serve max-period daily history (reusing its retry/rate-limit logic and the per-ticker TSV cache in `stock_data/`), because the default cached window is ~5 years. A force-refresh control on the tab triggers re-download.
- **Chart rendering.** matplotlib figure embedded in the tab as a live canvas (`FigureCanvasTkAgg` + `NavigationToolbar2Tk`), so the chart is directly zoomable: drag a rectangle to zoom into a specific date period, pan with the pan tool, and step back/forward or return home with the toolbar buttons. Re-renders (view/horizon/entry toggles) swap the figure in place and reset zoom to home; merely switching tabs away and back redraws without rebuilding, preserving the current zoom. Figures are closed on swap to avoid accumulation. (Earlier iterations rendered a static PNG — replaced by the live canvas when zoom was requested.)
- **Worker threading.** Analysis runs in a daemon worker thread that makes ZERO Tkinter calls (calling `root.after()` from a worker caused `Tcl_AsyncDelete: async handler deleted by the wrong thread` crashes). Results are passed through a `queue.Queue` polled by the UI thread every 100 ms — the same pattern as the Relative Strength tab. Additionally, `setup_thread_safe_tkinter()` now patches `tk.Misc.after` app-wide so any remaining legacy worker that calls `root.after()` is bridged onto the main thread instead of creating a cross-thread Tcl timer — the same crash existed in the Buffett/business-analysis/SEC workers.
- **Data table.** A read-only `ttk.Treeview` with one row per quarter: quarter label, blackout start, blackout end, baseline close, then % gain for each of the 8 horizons; color-coded gains (green/red) consistent with the Relative Strength table's tag styling. The chart and the table live in sub-tabs ("📈 Chart" / "📋 Per-Quarter Detail", notebook pattern like the Compare tab) so each gets the full pane width; switching sub-tabs redraws the canvas without rebuilding, preserving zoom.
- **Daily equity view.** `compute_daily_equity_curves()` walks the daily price series per horizon: buy at each quarter's baseline close, mark equity at every close while holding, sell at the rolled exit, hold cash until the next buy. A window whose exit hasn't elapsed is held live to the last session (so curves extend to the present). If a new buy arrives while still holding (irregular earnings spacing), the old position is force-liquidated at that day's close.
- **Horizon filtering.** One checkbox per holding period (21d checked by default, others off) filters which horizons are plotted — in both chart views. Figures are built on the UI thread from the cached analysis result (data is fetched once per run), so toggling a checkbox or the view radio only costs one matplotlib render; with no horizons checked, the chart shows a "No horizons selected" placeholder.
- **Scatter style.** A "Scatter" checkbox switches the horizon series from connected lines to markers only (small dots in the dense daily view) — useful for seeing the density of observations without the visual weight of lines.
- **Buy-and-hold overlay.** An "Overlay Price (Buy & Hold)" checkbox adds the stock's raw price series as a solid black line, normalized to the same start as the equity curves (0% at the first window's baseline) so strategy vs buy-and-hold is directly comparable. In the By-Quarter view it is sampled at each quarter's baseline close. The overlay always renders as a solid line regardless of Scatter mode.
- **Entry mode.** "Buy at: Blackout End / Blackout Start" radios flip the trade's entry point. End (default) buys at the earnings-day close; Start buys at the blackout-start close, with the N-day exit counted from there. Both modes are precomputed per run, so flipping is instant with no refetch, and the per-quarter table and status line follow the selected mode.
- **Blackout date markers (Daily view).** A "V-Lines (Daily)" checkbox draws vertical lines at each blackout start (yellow dashed) and end/earnings (green dotted) date, with matching legend entries. Lines outside the plotted range are skipped so the x-axis is not stretched.
- **Range selection (Daily view).** "Show last [N] quarters" dropdown (4/6/8/12/16/20/All, default 8) zooms to the blackout-start date of the Nth-from-last quarter through the data end (N clamps to available quarters; uses the current entry mode's data). The From/To entries + Zoom/Reset remain for exact custom ranges — a hand-typed zoom flips the dropdown to "Custom" (display-only; not selectable), and re-renders re-apply whichever control is active (dropdown takes precedence unless All/Custom). Dates are converted explicitly with date2num — set_xlim with Timestamps is unreliable on date axes. The toolbar's current view is pushed before zooming so its Back button undoes it; Reset clears everything and restores the full range. Invalid/reversed dates only update the status line (guard-then-act).
- **Default chart state.** Daily Equity view, Scatter on, Overlay Price on, Yellow in Blackout on, V-Lines on, holding period 21d only, entry at Blackout End, range last 8 quarters.
- **Custom holding period.** A "Custom (days)" entry accepts any positive N — including periods longer than one quarter. The value is merged into the cached result for both entry modes without re-fetching (extend_result_horizon) — automatically on every fresh Analyze run when the entry is non-empty and valid, or on Enter; it appears as an extra legend series in both chart views and an extra column in the per-quarter table (columns reconfigure dynamically). Long-hold semantics: the daily equity walk SKIPS buy signals that arrive while still holding — the position runs to its own exit, then re-enters at the next blackout end. Note the By-Quarter compounded curve still compounds every quarter's window (event-study view), so for long N the two views intentionally differ: the daily curve is the executable strategy.
- **Blackout highlight (Daily view).** A "Yellow in Blackout (Daily)" checkbox (on by default) draws the equity segments during each blackout window — the flat pre-buy cash runs — in gold/yellow, drawn thicker than the base line and with a "Blackout" legend entry, so they are easy to spot. All dates stay on the calendar x-axis (an earlier iteration compressed blackouts off the axis; replaced by this highlight at the user's request). Roughly a third of days fall inside blackouts for a typical quarterly-earnings stock.
- **Ticker selection** reuses the app's single-ticker helper (main list, then watch list), consistent with other single-stock tabs.

## Testing Decisions

- **What gets tested:** the standalone analysis module only, via unit tests with synthetic data — no network, no GUI.
- **Good tests assert external behavior:** given a constructed daily price series and a list of earnings dates, the functions' outputs (windows, baseline picks, horizon returns, compounded curves) are checked — never internal intermediate state.
- **Key cases to cover:**
  - Horizon date landing on a weekend/holiday → rolls to the next trading day's close.
  - Earnings date itself on a non-trading day → baseline rolls forward.
  - Partial latest quarter → excluded from horizons that haven't elapsed, included in shorter ones.
  - Compounding math: e.g., returns of +10% then −10% → compounded −1%, not 0%.
  - Missing price data around a quarter → that quarter is skipped, not silently treated as 0%.
- **Prior art:** the repo's `test_data_manager.py` and `test_relative_strength`-style script tests (plain scripts / unittest, synthetic inputs, no network).

## Implementation Checklist

- [x] Extend data manager to fetch and cache max-period daily history (with force-refresh).
- [x] Create standalone analysis module with pure functions (blackout periods, quarter gains, compounded curves).
- [x] Write unit tests for the analysis module (edge cases listed under Testing Decisions).
- [x] Add "⏳ Blackout" tab: controls (ticker display, blackout-start-days spinbox, Run, Refresh data), status label.
- [x] Implement worker-thread run → PNG chart embedding (8 horizon lines over quarters).
- [x] Implement per-quarter Treeview data table with color-coded gains.
- [x] Wire tab-changed handler, ticker reactivity, and error/status messaging.
- [x] Manual end-to-end verification on a real ticker (AAPL: 100 earnings quarters retrieved, 99 analyzed, partial windows handled correctly).
- [x] Add daily-equity view: compute_daily_equity_curves() + daily figure + By-Quarter/Daily radio toggle (3 new unit tests; AAPL daily chart verified).
- [x] Add per-horizon checkboxes filtering both chart views (subset render verified on AAPL; empty-selection placeholder handled).
- [x] Add Scatter style toggle and Overlay Price (Buy & Hold) checkbox for both chart views (render verified on AAPL).
- [x] Add "Skip Blackout (Daily)" axis compression removing blackout windows from the daily chart's time axis (verified on AAPL: 32% of days removed). Superseded: replaced by gold blackout highlighting on the full date axis, and default holding period changed to 21d only.
- [x] Buy & Hold overlay changed to a solid line; added "Buy at: Blackout End / Blackout Start" entry-mode radios (both modes precomputed; verified on AAPL — end vs start differ materially).
- [x] Added "V-Lines (Daily)" checkbox marking blackout start (gray dashed) and end (purple dotted) dates (verified on AAPL zoom render).
- [x] Replaced static PNG with embedded matplotlib canvas + navigation toolbar for interactive date-period zoom/pan (embedding + figure-swap smoke tested).

## Out of Scope

- Deriving blackouts from actual SEC Form 4 insider-filing gaps.
- Buy/sell signals, alerts, or automated trading integration.
- Multi-ticker comparison or benchmark-relative performance in this tab.
- Dividend/split-adjusted total-return precision beyond what the existing adjusted-close cache provides.
- Backtesting position sizing, transaction costs, or slippage.
- Intraday price granularity (daily closes only).

## Further Notes

- yfinance's historical earnings dates typically cover a bounded recent window (roughly the last few years); the number of analyzed quarters will vary by ticker and is surfaced in the UI. If deeper history is wanted later, a fallback (deriving older quarters from fiscal quarter-end + a typical earnings-offset) can be a follow-up.
- The existing `earning_dates.py` script demonstrates the earnings-dates API but is standalone; the new module should implement its own robust fetch (with retry/rate-limit behavior consistent with `insider_sale_data.py`) rather than importing the print-oriented script.
- The per-quarter table intentionally exposes the surprise % column source (earnings dates payload) as a future enhancement (e.g., color rows by beat/miss) without changing the data layer.
