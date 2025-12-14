# Product Requirements Document (PRD): Stock Toolbox Web Application

## 1. Introduction

This document outlines the requirements for converting the existing desktop stock toolbox into a web-based application. The goal is to provide a user-friendly interface for stock analysis, charting, news aggregation, and SEC filing retrieval.

## 2. Core Features

### 2.1. Dashboard & Ticker Management

* **Sidebar Navigation:** A persistent sidebar for navigating ticker lists and searching for tickers.
* **Ticker Lists:** Users can create, view, and manage lists of stock tickers (e.g., "Watchlist", "Tech Stocks").
* **Ticker Search:** A search bar to quickly find tickers and add them to lists or view their details.
* **Ticker Selection:** Clicking a ticker loads its data into the main content area.
* **Pseudo Portfolio Mode:** Any ticker list can be treated as an equal-weight “pseudo portfolio” by allocating fake equal shares/position sizes per constituent, allowing AI strategy playbooks and comparison views to behave consistently even before a formal portfolio is saved.

### 2.2. Charting

* **Interactive Charts:** Display interactive candlestick charts using Plotly.js.
* **Timeframes:** Support for Daily (D), Weekly (W), and Monthly (M) timeframes.
* **Indicators:** Technical indicators (MA, EMA, RSI, MACD, Bollinger Bands) rendered as optional overlays. Users can toggle indicators per chart instance.
* **Volume:** Display volume bars below the price chart.
* **Lunar Phases Overlay:** Comparison charts offer toggles for full moon, new moon, and half moon markers. Lunar data is precomputed server-side and exposed alongside chart payloads so the UI can annotate key dates.
* **AI Technical Insight:** When indicators are toggled, Gemini summarizes notable technical setups (e.g., “RSI oversold” or “MACD bullish crossover”) in a side panel.
* **Trade Markers:** Users can toggle overlays that plot individual buy/sell executions pulled from the trading log, showing entry/exit arrows plus P&L tooltips directly on the primary chart.

### 2.3. Fundamental Analysis

* **Data Table:** Display key fundamental metrics (P/E, Market Cap, Dividend Yield, etc.) fetched from Yahoo Finance.
* **AI Analysis:** Integration with Google Gemini API to generate qualitative business analysis, plus targeted callouts (valuation, profitability, growth, risk) for each metric group.
* **Comparative AI Toggle:** When comparing multiple tickers, users can request AI to highlight divergences and recommend which stocks best fit predefined personas (e.g., “value investor”, “growth investor”).

### 2.4. News Aggregation

* **News Feed:** Display recent news articles for the selected ticker using Tavily API.
* **AI Sentiment Analysis:** Analyze news headlines and content using Gemini to determine sentiment (Positive/Negative/Neutral) and summarize key points.

### 2.5. SEC Filings

* **Filing List:** Retrieve and list recent SEC filings (10-K, 10-Q) for the selected ticker.
* **Table Extraction:** Extract and display financial tables (Balance Sheet, Income Statement, Cash Flow) from filings using `pandas` and `lxml`.
* **AI Filing Insights:** Gemini generates section-specific insights (risk factors, MD&A signals) and can cross-reference themes against user portfolios to flag impacted holdings.

### 2.6. Watchlist Alerts & AI Signal Digest

* **Threshold Alerts:** Users define price or percentage thresholds per ticker. Alerts are stored in SQLite and evaluated whenever new price data loads.
* **Delivery Surface:** Triggered alerts appear in an in-app notification drawer (HTMX component) and optionally via browser notifications.
* **AI Digest:** Gemini clusters triggered alerts into themes (e.g., “semis breakout”) and proposes follow-up actions (rebalance, research, compare peers).

### 2.7. Portfolio Management & Trading Log Automation

* **Portfolio Entities:** Introduce `Portfolio` and `Position` models linked to tickers. Each position tracks quantity, cost basis, acquisition date, and optional notes.
* **Trading Log Import:** Users upload CSV or paste a structured trading log (buy/sell records). The backend parses transactions, either creating a new portfolio or updating an existing one. Duplicate detection prevents double-counting, and each ingested row becomes an editable journal entry so users can amend thesis notes, tag strategies, or delete errant trades after import.
* **Trading Journal Fields:** Each trade (real or simulated) captures optional planning metadata—setup name, thesis, entry/exit rules, confidence, risk budgeting notes, and post-trade reflections. Inputs feed both the structured ledger and a free-form journal entry that Gemini can reference later.
* **Incremental Updates:** Additional trading logs can be ingested later to adjust quantities, cost basis, and realized P&L.
* **Account Snapshot Import:** Provide a workflow to create or refresh a portfolio from current brokerage statements or exported position reports (CSV/Excel). The system maps tickers, quantities, and cost basis automatically, optionally merging with manual adjustments before AI analysis.
* **Screenshot/OCR Intake:** Allow users to paste or upload a screenshot of their brokerage positions page; Gemini extracts tickers, quantities, and cost basis via OCR + LLM parsing, then pre-populates a draft portfolio for review before saving.
* **Broker Connectors:** Users can register read-only connectors (Schwab, Fidelity, E*TRADE, Robinhood, IBKR, TDA, or custom) that store encrypted credentials locally. Connectors track last sync status/message so future background sync jobs know which portfolios are eligible for automated refresh.
* **AI Strategy Surfaces:** Provide in-app controls (scenario, timeframe, benchmark) to trigger Gemini playbooks for individual tickers and full portfolios, rendering bull/bear/neutral plans plus markdown tables directly inside the Analysis and Portfolio tabs.
* **Synthesized Portfolio Ticker:** Each portfolio generates a synthetic “ticker” (e.g., `PORT-123`) representing the weighted aggregate of its positions. This synthetic ticker is:
  * Stored alongside portfolio metadata.
  * Eligible for charting, AI analysis, and comparisons just like real tickers (data derived from constituent weights).
* **AI Portfolio Copilot:** Gemini analyzes holdings to surface allocation imbalances, performance attribution, risk concentrations, and suggests rebalancing or hedging ideas.
* **Scenario Analysis:** Users can simulate hypothetical trades; Gemini projects potential portfolio impact and updates the synthesized ticker preview accordingly.
* **Strategy Simulation Ledger:** A sandbox lets users follow an AI-generated playbook (or name their own strategy) by queuing simulated trades with timestamps, fills, and notes. Each simulation can be promoted into the live portfolio or reset, and Gemini provides guardrails (position sizing, risk checks) before recording the virtual fills.
* **Journal Enhancements:** Provide quick templates (“Momentum Breakout Plan”, “Mean Reversion Checklist”) plus nudges to log emotion/discipline metrics. Offer AI prompts that auto-summarize streaks (“3 consecutive wins following the AI Tech Momentum plan—do you want to increase size?”) and pull context-aware reminders when similar setups reappear.
* **Simulation Workflows:** Beyond the journal, add a dedicated "Strategy Lab" view where users can bucket simulations by hypothesis, auto-clone AI playbooks into task lists, and schedule follow-up validations. Include bulk actions (promote, archive, compare) plus guardrails such as per-strategy exposure caps and risk alerts before committing simulated trades to live portfolios.
* **Backtest Hand-offs:** Allow simulations to be exported into lightweight CSV/JSON specs that the backtesting engine can consume, and surface a summary of backtest KPIs back inside the Strategy Lab so traders can iterate without leaving the app.
* **Portfolio Workspace Tabs:** Break the single-page portfolio view into secondary tabs (Overview, Analytics, Strategy Lab, Journal, Automation) so traders can jump directly to positions, dashboards, lab experiments, journal entries, or automation surfaces without scrolling through the entire surface.

### 2.8. Portfolio Visualization & Comparative Tools

* **Performance Dashboard:** Plotly-based widgets show total equity curve, realized/unrealized gains, sector allocation, and best/worst performers.
* **Strategy Equity Curves:** In addition to aggregate equity, render disaggregated curves per trading strategy (e.g., "AI Playbook – Tech Momentum", "User Strategy – Income Rotation") with hover tooltips for individual simulated trades. Users can compare strategy-level P&L contribution over time and toggle overlays like drawdown bands and benchmark spreads.
* **AI Commentary:** Each dashboard refresh triggers an optional Gemini summary highlighting notable changes since the last review.
* **AI Trading Journey Review:** Gemini ingests the trade ledger (real + simulated), equity curves, and per-strategy stats to narrate the trading journey—calling out inflection points, discipline/adherence to playbooks, and recommending next experiments or risk mitigations.
* **Comparison Enhancements:** Multi-ticker comparison charts accept both real and synthetic tickers, honor lunar phase overlays, and allow AI to annotate divergence points (e.g., “Portfolio vs. SPY outperformance during new moon windows”).
* **Analytics Polish:** Add toggleable overlays for drawdown bands, realized vs. unrealized P&L heatmaps, and “discipline score” callouts derived from journal metadata (e.g., tagging trades that broke risk rules). Provide export-to-PDF/PNG buttons for every chart card plus permalinkable snapshot URLs for async reviews.
* **Tabbed Navigation:** Within a portfolio, persist which sub-tab the user last viewed and ensure all metrics/cards inside each tab refresh when activated so analytics and journal content stay in sync.

### 2.10. Multi-user Readiness & Governance

* **Workspace Isolation:** Namespaces portfolios, journal entries, caches, and AI responses by user/org so multi-tenant deployments keep data siloed. Include per-user rate limits for AI calls and long-running analytics jobs.
* **Role-based Controls:** Introduce roles such as Viewer, Trader, Strategist, and Admin. Roles gate features like promoting simulations, editing live portfolios, triggering bulk ingest, or inviting teammates.
* **Audit & Notifications:** Persist an audit log of portfolio changes, simulations promoted, and AI recommendations accepted/ignored. Allow admins to subscribe to email/slack notifications when guardrails trip (e.g., position size breach) or when Gemini flags risky behavior.
* **Scalable Caching:** Document shared caches (strategy playbooks, equity analytics) with eviction policies, and ensure caches are keyed by user/org + scenario to prevent cross-tenant leakage.

### 2.9. AI Strategy Recommender (Ticker + Portfolio)

* **Scenario Engine:** Users choose a market regime (bullish/bearish/neutral) or let the system infer it from volatility scores; helper aliases normalize inputs like "rally" or "rangebound".
* **Ticker Playbooks:** For any selected ticker, Gemini produces a three-path plan (bullish/bearish/neutral) covering equity tactics, options overlays (spreads, collars, etc.), risk controls, and catalyst checklists. Output includes a markdown playbook table for quick scanning.
* **Portfolio Playbooks:** For saved portfolios (including trading-log generated ones), Gemini ingests position-level data plus the synthesized ticker to recommend allocation tweaks, hedges, and concentration fixes under each regime. It suggests index vs. single-name hedges, ratio spreads, and beta targets, then summarizes in a scenario table.
* **UI Integration:** Add “AI Strategy” buttons within the Analysis tab for current ticker and inside the portfolio dashboard. Results render in markdown-ready cards with copy-to-clipboard and “save to notes” actions.
* **Extensibility:** Strategy requests accept optional context (support/resistance levels, timeframe focus, benchmark). Back-end helpers format fundamentals/price data to keep prompts concise and consistent across modules.

## 3. Technical Architecture

### 3.1. Backend

* **Framework:** FastAPI (Python) for high performance and easy API development.
* **Database:** SQLite using SQLAlchemy ORM for storing ticker lists and user preferences.
* **Data Sources:**
  * `yfinance`: Stock price and fundamental data.
  * `Tavily`: News search.
  * `SEC EDGAR`: Filing retrieval.
  * `Google Gemini`: AI analysis.

### 3.2. Frontend

* **Templating:** Jinja2 for server-side rendering.
* **Interactivity:** HTMX for dynamic content loading without full page reloads.
* **Styling:** Bootstrap 5 for responsive design.
* **Charting:** Plotly.js for client-side rendering of charts.

## 4. User Flow

1.  **Landing:** User sees the dashboard with a default ticker list.
2.  **Selection:** User clicks a ticker from the sidebar.
3.  **Loading:** The main area updates via HTMX to show the "Chart" tab by default.
4.  **Analysis:** User switches tabs to view "Fundamentals", "News", or "SEC Filings".
5.  **Interaction:** User interacts with specific features (e.g., clicks "Run AI Analysis", changes chart timeframe).

## 5. Deployment

* The application is designed to run locally using `uvicorn` or deployed to a containerized environment (Docker).
* Environment variables are used for API keys (`TAVILY_API_KEY`, `GEMINI_API_KEY`, `SEC_EDGAR_EMAIL`).
