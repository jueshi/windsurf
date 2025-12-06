# Stock Toolbox Web Application - User Guide

## Introduction
The Stock Toolbox Web Application is a comprehensive platform for stock analysis, combining interactive charting, fundamental data, AI-powered news sentiment analysis, and SEC filing retrieval into a single, user-friendly interface.

## 1. Getting Started

### Prerequisites
*   Python 3.8+
*   Internet connection

### Installation
1.  Clone the repository.
2.  Navigate to the `webapp` directory:
    ```bash
    cd webapp
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration
You need to set up the following environment variables for full functionality:
*   `TAVILY_API_KEY`: For fetching news.
*   `GEMINI_API_KEY`: For AI analysis of news and fundamentals.
*   `SEC_EDGAR_EMAIL`: For identifying yourself to the SEC EDGAR system when fetching filings (User-Agent).

You can set these in your terminal or create a `.env` file.

### Running the Application
Run the application using Uvicorn:
```bash
uvicorn main:app --reload
```
Access the application in your browser at `http://127.0.0.1:8000`.

## 2. Dashboard & Navigation

### Sidebar
*   **Ticker Lists**: The sidebar displays your managed lists of tickers. You can create new lists (e.g., "Watchlist", "Portfolio") and switch between them.
*   **Ticker Search**: Use the search bar at the top of the sidebar to find specific stock tickers.
*   **Add Ticker**: Found a ticker? Add it to your currently selected list for easy access.

### Main View
The main area of the screen displays detailed information for the currently selected ticker. It is divided into four main tabs:
1.  **Chart**: Technical analysis.
2.  **Fundamentals**: Key metrics and business analysis.
3.  **News**: Recent articles and sentiment.
4.  **SEC Filings**: Regulatory documents.

## 3. Features

### Charting
*   **Interactive Charts**: Pan, zoom, and hover over data points on the candlestick chart.
*   **Timeframes**: Toggle between Daily (D), Weekly (W), and Monthly (M) views to analyze trends over different horizons.
*   **Volume**: Analyze trading activity with volume bars displayed below the price chart.

### Fundamental Analysis
*   **Key Metrics**: Quickly view essential ratios like P/E, Market Cap, Dividend Yield, and EPS.
*   **AI Business Analysis**: Click the "Run AI Analysis" button to generate a qualitative summary of the company's business model and financial health based on the available data.

### News Aggregation
*   **News Feed**: See a curated list of the most recent news articles relevant to the stock.
*   **AI Sentiment Analysis**: The "Analyze Sentiment" feature uses Google Gemini to read headlines and summaries, categorizing the overall sentiment as Positive, Negative, or Neutral, and providing a brief summary of market buzz.

### SEC Filings
*   **Recent Filings**: Access a list of the latest 10-K (Annual Report) and 10-Q (Quarterly Report) filings directly from the SEC.
*   **Data Extraction**: The tool automatically extracts and displays key financial tables (Balance Sheet, Income Statement, Cash Flow) from the filings, saving you the time of manually parsing these documents.
