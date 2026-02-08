# Pitch Deck Content: Stock Toolbox Web App

## Slide 1: Title Slide
*   **Title:** Stock Toolbox Web App
*   **Subtitle:** The All-in-One Platform for Intelligent Investment Research
*   **Presenter:** [Your Name]

## Slide 2: The Problem
*   **Fragmented Tools:** Investors currently switch between 5 different tabs:
    *   Yahoo Finance for prices.
    *   TradingView for charts.
    *   SEC.gov for filings.
    *   Google News for headlines.
    *   ChatGPT for summaries.
*   **Information Overload:** Too much noise, not enough actionable signal.
*   **Time Consuming:** Manually parsing 10-K filings and financial tables is slow and prone to error.

## Slide 3: The Solution
*   **Unified Interface:** A single dashboard combining Charts, Data, News, and Documents.
*   **AI-Powered:** Built-in AI to summarize news sentiment and analyze business fundamentals instantly.
*   **Automated SEC Retrieval:** One-click access to financial tables extracted directly from official SEC filings.
*   **Focus on Workflow:** Designed specifically for the retail investor's research process.

## Slide 4: Key Features (The "Why Us")
1.  **Interactive Charting:** Professional-grade Plotly charts with multi-timeframe analysis.
2.  **AI Analyst:** Your personal research assistant that reads news and interprets balance sheets.
3.  **Filing Parser:** Don't just read the 10-K; extract the data you need immediately.
4.  **Ticker Management:** Organize your watchlists and portfolio in one place.

## Slide 5: Market Opportunity
*   **Target Audience:** Retail Investors, Swing Traders, Financial Analysts.
*   **Trend:** The rise of retail investing (post-2020) created a demand for "Prosumer" tools that bridge the gap between free websites and expensive Bloomberg terminals.
*   **Value Prop:** Institutional-grade data accessibility at zero marginal cost for the user (local deployment).

## Slide 6: Technical Architecture
*   **Modern Stack:**
    *   **Backend:** FastAPI (High performance Python).
    *   **Frontend:** HTMX & Jinja2 (Fast, dynamic, no complex React build step).
    *   **Database:** SQLite & SQLAlchemy (Lightweight, zero-config).
*   **Integrations:**
    *   **Google Gemini:** LLM for text analysis.
    *   **Tavily:** Search API for real-time news.
    *   **SEC EDGAR:** Regulatory data source.

## Slide 7: Roadmap
*   **Q1:** Launch MVP (Current State) - Core charting, News, Basic SEC.
*   **Q2:** Advanced Technicals - Add Moving Averages, RSI, MACD indicators.
*   **Q3:** Portfolio Tracking - Integration with brokerage APIs for live P&L.
*   **Q4:** Cloud Sync - User accounts and cloud database for cross-device access.

## Slide 8: Call to Action
*   **Download & Use:** Clone the repo and start researching smarter today.
*   **Contribute:** Open source project welcoming PRs for new indicators and data sources.
*   **Contact:** [Your Contact Info]
