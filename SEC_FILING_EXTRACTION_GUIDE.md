# SEC Filing Extraction Guide

This guide explains how to use the SEC filing extraction functionality in the stock data application.

## Overview

The application can extract tables from SEC filings (10-K and 10-Q) for publicly traded companies. It supports two modes:

1. **Real SEC API Mode** - Fetches data from the actual SEC EDGAR database (with caching and rate limiting)
2. **Mock Data Mode** - Uses pre-generated sample data for testing without hitting the SEC API

## Using the SEC Filing Extraction Feature

### From the Main Interface

1. Select a ticker from the "Available Tickers" or "Watch List"
2. Click either "Extract 10-K Tables" or "Extract 10-Q Tables" button
3. The application will switch to the SEC Filings tab and begin the extraction process

### From the SEC Filings Tab

1. Enter a ticker symbol in the "Ticker" field
2. Select the form type (10-K or 10-Q) from the dropdown
3. Toggle "Use Mock Data" checkbox if you want to use mock data instead of real SEC API
4. Click "Extract Tables" button
5. The application will display the extraction progress and results

## Mock Data vs. Real SEC API

### Real SEC API (Default)

- Fetches actual filing data from the SEC EDGAR database
- Subject to SEC API rate limits (10 requests per second)
- Requires internet connection
- Results are cached to avoid repeated downloads
- Provides the most up-to-date and accurate filing data

### Mock Data (For Testing)

- Uses pre-generated sample data for AAPL, MSFT, GOOGL, AMZN, and META
- Works offline without internet connection
- No rate limits or delays
- Useful for testing and development
- Contains simplified sample financial tables

## Handling SEC API Rate Limits

The application includes several features to handle SEC API rate limits gracefully:

1. **Caching** - Downloaded filings are cached to avoid repeated requests
2. **Rate Limiting** - Requests are automatically spaced out to comply with SEC limits
3. **Status Updates** - The application displays detailed status updates during extraction
4. **Mock Data Option** - Switch to mock data when testing functionality

## Troubleshooting

If you encounter issues with SEC filing extraction:

1. **SEC API Rate Limits** - If you see rate limit errors, wait a few minutes and try again
2. **Clear Cache** - Use the "Clear SEC Cache" button to remove cached data if it becomes stale
3. **Switch to Mock Data** - Toggle "Use Mock Data" to test functionality without hitting the SEC API
4. **Check Logs** - Review the application logs for detailed error information

## Advanced Features

- **Export to Excel** - Extracted tables are automatically saved to Excel files
- **Table Identification** - The application attempts to identify key financial tables (Income Statement, Balance Sheet, Cash Flow)
- **Table Viewing** - Select tables from the list to view their contents in the SEC Filings tab
