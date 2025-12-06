# Product Requirements Document (PRD): Stock Toolbox Web Application

## 1. Introduction
This document outlines the requirements for converting the existing desktop stock toolbox into a web-based application. The goal is to provide a user-friendly interface for stock analysis, charting, news aggregation, and SEC filing retrieval.

## 2. Core Features

### 2.1. Dashboard & Ticker Management
*   **Sidebar Navigation:** A persistent sidebar for navigating ticker lists and searching for tickers.
*   **Ticker Lists:** Users can create, view, and manage lists of stock tickers (e.g., "Watchlist", "Tech Stocks").
*   **Ticker Search:** A search bar to quickly find tickers and add them to lists or view their details.
*   **Ticker Selection:** Clicking a ticker loads its data into the main content area.

### 2.2. Charting
*   **Interactive Charts:** Display interactive candlestick charts using Plotly.js.
*   **Timeframes:** Support for Daily (D), Weekly (W), and Monthly (M) timeframes.
*   **Indicators:** (Future Scope) Technical indicators like MA, RSI, MACD.
*   **Volume:** Display volume bars below the price chart.

### 2.3. Fundamental Analysis
*   **Data Table:** Display key fundamental metrics (P/E, Market Cap, Dividend Yield, etc.) fetched from Yahoo Finance.
*   **AI Analysis:** Integration with Google Gemini API to generate a qualitative business analysis based on fundamental data.

### 2.4. News Aggregation
*   **News Feed:** Display recent news articles for the selected ticker using Tavily API.
*   **AI Sentiment Analysis:** Analyze news headlines and content using Gemini to determine sentiment (Positive/Negative/Neutral) and summarize key points.

### 2.5. SEC Filings
*   **Filing List:** Retrieve and list recent SEC filings (10-K, 10-Q) for the selected ticker.
*   **Table Extraction:** Extract and display financial tables (Balance Sheet, Income Statement, Cash Flow) from filings using `pandas` and `lxml`.

## 3. Technical Architecture

### 3.1. Backend
*   **Framework:** FastAPI (Python) for high performance and easy API development.
*   **Database:** SQLite using SQLAlchemy ORM for storing ticker lists and user preferences.
*   **Data Sources:**
    *   `yfinance`: Stock price and fundamental data.
    *   `Tavily`: News search.
    *   `SEC EDGAR`: Filing retrieval.
    *   `Google Gemini`: AI analysis.

### 3.2. Frontend
*   **Templating:** Jinja2 for server-side rendering.
*   **Interactivity:** HTMX for dynamic content loading without full page reloads.
*   **Styling:** Bootstrap 5 for responsive design.
*   **Charting:** Plotly.js for client-side rendering of charts.

## 4. User Flow
1.  **Landing:** User sees the dashboard with a default ticker list.
2.  **Selection:** User clicks a ticker from the sidebar.
3.  **Loading:** The main area updates via HTMX to show the "Chart" tab by default.
4.  **Analysis:** User switches tabs to view "Fundamentals", "News", or "SEC Filings".
5.  **Interaction:** User interacts with specific features (e.g., clicks "Run AI Analysis", changes chart timeframe).

## 5. Deployment
*   The application is designed to run locally using `uvicorn` or deployed to a containerized environment (Docker).
*   Environment variables are used for API keys (`TAVILY_API_KEY`, `GEMINI_API_KEY`, `SEC_EDGAR_EMAIL`).
