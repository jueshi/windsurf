# Personal AI Stock Assistant - User Guide

A comprehensive desktop application for stock analysis, charting, news summarization, and SEC filing extraction powered by AI.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Main Interface Overview](#main-interface-overview)
3. [Ticker List Management](#ticker-list-management)
4. [Chart Generation](#chart-generation)
5. [Analysis Tabs](#analysis-tabs)
6. [News &amp; AI Features](#news--ai-features)
7. [SEC Filings](#sec-filings)
8. [Quick Reference URLs](#quick-reference-urls)
9. [Settings &amp; Persistence](#settings--persistence)
10. [Keyboard Shortcuts &amp; Tips](#keyboard-shortcuts--tips)

---

## Getting Started

### Launching the Application

```bash
python main.py
```

The application opens maximized with all features ready to use.

### First-Time Setup

1. The app automatically loads ticker lists from `ticker_lists.py`
2. Your watch list is persisted across sessions
3. Settings (like StockCharts style ID) are saved to `gui_settings.json`

---

## Main Interface Overview

The interface is divided into several key areas:

### Top Toolbar (Row 1)

| Element                 | Description                                             |
| ----------------------- | ------------------------------------------------------- |
| **List Filter**   | Type to filter available ticker lists by name           |
| **List Dropdown** | Select from saved ticker lists                          |
| **Load**          | Load selected list into Available Tickers pane          |
| **↻**            | Refresh ticker lists from `ticker_lists.py`           |
| **◀ / ▶**       | Navigate to previous/next ticker list                   |
| **✕**            | Remove current list from `ticker_lists.py`            |
| **D / W / M**     | Open Daily/Weekly/Monthly candlestick charts in browser |
| **Multi-TF**      | Generate multi-timeframe HTML gallery (D/W/M)           |
| **Lines**         | Generate line chart comparison gallery                  |
| **SC**            | Open StockCharts.com gallery view                       |
| **Style ID**      | StockCharts style code (auto-saved)                     |
| **SC-Line**       | Generate StockCharts gallery with custom style          |
| **📝**            | Open `ticker_lists.py` in text editor                 |
| **📋**            | Copy all tickers to clipboard                           |
| **Tips**          | Toggle tooltips on/off                                  |

### Top Toolbar (Row 2)

| Element              | Description                                     |
| -------------------- | ----------------------------------------------- |
| **Add**        | Enter ticker(s) to add (comma-separated)        |
| **+**          | Add ticker(s) to current list                   |
| **New List**   | Enter name for new ticker list                  |
| **Create**     | Create new list and save to `ticker_lists.py` |
| **🔗 URLs ▾** | Quick access to financial websites              |

### Main Panes (Left to Right)

1. **Available Tickers** - Tickers from the loaded list
2. **Watch List** - Personal watch list (persisted)
3. **Chart Display** - Charts and analysis tabs

### Bottom Action Bar

| Button                | Description                                   |
| --------------------- | --------------------------------------------- |
| **⬇Download**  | Download stock data for selected tickers      |
| **⬇All**       | Download data for ALL tickers in current list |
| **📊Visualize** | Generate D/W/M charts for all tickers         |
| **📄Report**    | Open HTML report with generated charts        |
| **📈Compare**   | Compare percentage performance                |
| **Market**      | Summarize market news                         |
| **Stock**       | Summarize stock-specific news (Finviz)        |
| **ETF**         | Summarize ETF news                            |
| **Crypto**      | Summarize cryptocurrency news                 |
| **📋**          | Summarize clipboard content with AI           |
| **Force DL**    | Force re-download even if cached              |

---

## Ticker List Management

### Loading a Ticker List

1. Use the **List Filter** to search for a list by name
2. Select from the dropdown
3. Click **Load** or simply select (auto-loads)

### Navigating Lists

- Use **◀** and **▶** buttons to cycle through lists
- Lists are stored in `ticker_lists.py`

### Creating a New List

1. Select tickers in the Available Tickers pane
2. Enter a name in **New List** field (Python variable name, no spaces)
3. Click **Create**

### Adding Tickers

1. Enter ticker symbol(s) in the **Add** field
2. Use commas for multiple: `AAPL,MSFT,GOOGL`
3. Click **+** to add

### Managing Tickers

- **A-Z**: Sort tickers alphabetically
- **↑ / ↓**: Move selected ticker up/down in list
- **Right-click** on ticker for context menu:
  - Copy to Watch List
  - Remove Ticker

### Watch List

- Right-click to add tickers from Available Tickers
- Watch list is automatically saved to `ticker_lists.py`
- Persists across application restarts

---

## Chart Generation

### Individual Charts

1. Select a ticker from Available Tickers or Watch List
2. Chart displays automatically in the **Individual Chart** tab
3. Use date range controls to adjust timeframe

### Date Range Controls

| Control                     | Description                         |
| --------------------------- | ----------------------------------- |
| **Start Date**        | Beginning of chart date window      |
| **End Date**          | End of chart date window            |
| **Apply Date Range**  | Rebuild charts with specified dates |
| **Reset Date Range**  | Show full data history              |
| **6M / 1Y / 3Y / 5Y** | Quick preset date ranges            |

### Browser-Based Charts

#### Single Timeframe Charts

- **D** - Daily candlestick charts
- **W** - Weekly candlestick charts
- **M** - Monthly candlestick charts

Opens an HTML gallery in your browser with all tickers from the current list.

#### Multi-Timeframe Gallery

Click **Multi-TF** to generate a gallery showing Daily, Weekly, and Monthly charts for each ticker.

#### Line Charts Gallery

Click **Lines** to generate a comparison line chart gallery.

#### StockCharts.com Integration

- **SC** - Opens StockCharts.com gallery with default style
- **SC-Line** - Opens StockCharts gallery with custom style
- **Style ID** - Enter your StockCharts style code (auto-saved)
  - Get style codes from stockcharts.com/sc3/ui
  - Default: `t3327397499c`

### Comparison Charts

1. Select multiple tickers (Ctrl+Click)
2. Switch to **Comparison Chart** tab
3. Click **📈Compare** to compare percentage performance

### Seasonality Charts

1. Select a ticker
2. Switch to **Seasonality Chart** tab
3. Select years to compare using the dropdown
4. Click **Open in Browser** for full view

---

## Analysis Tabs

### Individual Chart

- Displays price chart for selected ticker
- Supports zoom and pan
- Date range controls apply here

### Comparison Chart

- Compare multiple tickers' performance
- Percentage-based comparison

### Seasonality Chart

- Historical seasonal patterns
- Multi-year comparison
- Year selection dropdown

### Fundamental Analysis

- Displays key metrics from Yahoo Finance
- **Filter Metric**: Search for specific metrics
  - Multiple terms use OR logic
  - Use `!` for exclusion (e.g., `!debt`)
  - Use `*` to show all
- **Save Filter / Load Filter**: Persist filter settings
- Key metrics highlighted:
  - Market Cap, P/E Ratios, Dividend Yield
  - Beta, 52-Week High/Low
  - Sector, Industry, Business Summary

### Business Analysis

| Button                        | Description                         |
| ----------------------------- | ----------------------------------- |
| **Run BA**              | Run comprehensive business analysis |
| **Conduct News Search** | Search for recent news              |
| **10-Q Study**          | Analyze quarterly SEC filings       |
| **10K Study**           | Analyze annual SEC filings          |
| **Extract 10-K Tables** | Extract tables from 10-K filings    |
| **Extract 10-Q Tables** | Extract tables from 10-Q filings    |
| **AI Search**           | Custom AI-powered search query      |

#### BA Tuning Options

- **Freshness (days)**: How recent data should be (1-365)
- **History (max)**: Maximum historical items to show
- **Show Change Over Time**: Toggle trend analysis

### Market News Blog

- Latest market news summary
- AI-generated insights
- Filterable content

### Buffett & CANSLIM

Analyzes stocks using Warren Buffett's value investing principles and CANSLIM methodology.

1. Select a ticker
2. Click **Analyze Selected**
3. View:
   - **Radar & Trend** chart (left pane)
   - **Explanation** text (right pane)

---

## News & AI Features

### News Summarization

All news features use AI (Gemini) to summarize content:

| Button           | Source           | Description                             |
| ---------------- | ---------------- | --------------------------------------- |
| **Market** | Multiple sources | General market news summary             |
| **Stock**  | Finviz           | Stock-specific news for selected ticker |
| **ETF**    | Various          | ETF-related news                        |
| **Crypto** | Various          | Cryptocurrency news                     |

### Clipboard Summarization

1. Copy any text to clipboard
2. Click **📋** button in bottom bar
3. AI summarizes the content

### AI Search

1. Enter search query in Business Analysis tab
2. Click **AI Search**
3. Get AI-powered analysis results

---

## SEC Filings

### SEC Filings Tab

Extract and analyze SEC filings (10-K, 10-Q).

#### Controls

| Control                      | Description                     |
| ---------------------------- | ------------------------------- |
| **Ticker**             | Enter company ticker symbol     |
| **Form Type**          | Select 10-K or 10-Q             |
| **Use Mock Data**      | Toggle for testing              |
| **Extract Tables**     | Fetch and parse filing tables   |
| **Open Output Folder** | Open folder with extracted data |
| **Clear SEC Cache**    | Clear cached SEC data           |

#### Working with Tables

1. Enter ticker and select form type
2. Click **Extract Tables**
3. Select table from **Available Tables** list
4. View content in **Table Content** pane
5. Export options:
   - **Export to Excel** - Save as .xlsx file
   - **Copy to Clipboard** - Copy for pasting

### Financial Tables

Tables marked with `[Financial]` contain structured financial data:

- Balance Sheet
- Income Statement
- Cash Flow Statement

---

## Quick Reference URLs

Access via **🔗 URLs ▾** dropdown:

### Market Data

- Finviz Screener
- TradingView
- StockCharts
- Yahoo Finance
- Koyfin

### News

- MarketWatch
- Bloomberg
- CNBC
- Reuters

### Fundamental Analysis

- Seeking Alpha
- Simply Wall St
- GuruFocus
- Morningstar
- TipRanks

### Earnings & Events

- Earnings Whispers
- Zacks
- Economic Calendar

### Insider & Institutional

- OpenInsider
- WhaleWisdom
- Dataroma

### Options

- Unusual Whales
- CBOE

### Research

- SEC EDGAR
- Macrotrends
- Barchart
- FRED Economic Data

### Ticker-Specific

- Stock Forecast (uses selected ticker)
- StockCharts UI (uses selected ticker)

### Custom URLs

- Add your own frequently used URLs
- Managed via **➕ Add Custom URL...** and **🗑️ Remove Custom URL...**

---

## Settings & Persistence

### Automatically Saved

| Setting              | File                  | Description             |
| -------------------- | --------------------- | ----------------------- |
| StockCharts Style ID | `gui_settings.json` | Custom chart style code |
| Custom URLs          | `custom_urls.json`  | User-added quick links  |
| Ticker Lists         | `ticker_lists.py`   | All saved ticker lists  |
| Watch List           | `ticker_lists.py`   | Personal watch list     |

### Manual Configuration

- Edit `ticker_lists.py` directly for bulk changes
- Click **📝** to open in text editor

---

## Keyboard Shortcuts & Tips

### Selection

- **Click** - Select single ticker
- **Ctrl+Click** - Add to selection
- **Shift+Click** - Select range

### Right-Click Menus

- **Available Tickers**: Copy to Watch List, Remove Ticker
- **Watch List**: Delete from Watch List

### Pro Tips

1. **Quick List Navigation**: Use ◀/▶ to rapidly browse through lists
2. **Batch Analysis**: Select multiple tickers, then use Compare for side-by-side analysis
3. **Custom StockCharts Styles**:

   - Visit stockcharts.com/sc3/ui
   - Customize your chart style
   - Copy the style ID and paste into the Style ID field
   - It auto-saves for future sessions
4. **Filter Power Users**:

   - Fundamental Analysis: `pe ratio dividend` (OR logic)
   - Business Analysis: `revenue !debt` (AND logic with exclusion)
5. **Date Range Shortcuts**: Use 6M/1Y/3Y/5Y buttons for quick timeframe changes
6. **Clipboard AI**: Copy any financial article and use 📋 for instant AI summary
7. **SEC Filing Analysis**: Extract 10-K tables first, then use "10K Study" for AI analysis

---

## Troubleshooting

### GUI Not Showing

- Ensure all dependencies are installed
- Check console for error messages
- Try running with `python -u main.py` for unbuffered output

### Data Not Loading

- Check internet connection
- Try **Force DL** checkbox for fresh download
- Clear SEC cache if filing data is stale

### Charts Not Displaying

- Ensure ticker has valid data
- Check date range isn't too restrictive
- Try Reset Date Range

### Browser Not Opening

- Default browser is used as fallback
- Edge is preferred but not required

---

## Dependencies

Key libraries used:

- `tkinter` - GUI framework
- `yfinance` - Stock data
- `plotly` / `matplotlib` - Charts
- `pandas` - Data manipulation
- `google-generativeai` - AI features (Gemini)
- `requests` / `beautifulsoup4` - Web scraping
- `tkcalendar` - Date picker widgets

---

## File Structure

```
stock_charts_10k10q/
├── main.py                 # Application entry point
├── gui.py                  # Main GUI implementation
├── data_manager.py         # Stock data management
├── ticker_lists.py         # Saved ticker lists
├── gui_settings.json       # Persistent settings
├── custom_urls.json        # Custom URL bookmarks
├── gemini_analyzer.py      # AI analysis module
├── buffett_canslim.py      # Investment analysis
├── news_fetcher.py         # News aggregation
├── sec_filing_extractor.py # SEC filing parser
├── sec_api_wrapper.py      # SEC API interface
└── output/                 # Generated reports
```

---

## Recommended Stock Research Workflow

A systematic approach to analyzing stocks using this tool's features.

### Phase 1: Discovery & Screening

**Goal**: Identify potential investment candidates

1. **Start with Market Overview**

   - Click **Market** news button to get AI-summarized market conditions
   - Check sector rotation and market sentiment
2. **Browse Curated Lists**

   - Use **◀ / ▶** to navigate through pre-built lists:
     - `mag7` - Magnificent 7 tech giants
     - `sp500_top50` - Largest S&P 500 companies
     - `dividend_aristocrats` - Reliable dividend payers
     - `growth_stocks` - High-growth companies
   - Use **D** button to open daily charts gallery for quick visual scan
3. **External Screening**

   - Open **🔗 URLs ▾** → **Finviz Screener** for custom filters
   - Use **AI Search** in Business Analysis tab: "Find undervalued tech stocks with strong cash flow"

### Phase 2: Technical Analysis

**Goal**: Evaluate price action and chart patterns

1. **Multi-Timeframe Analysis**

   - Select your target ticker list
   - Click **Multi-TF** for Daily/Weekly/Monthly view
   - Look for alignment across timeframes (trend confirmation)
2. **StockCharts Deep Dive**

   - Click **SC** for professional charting
   - Use **SC-Line** with custom style for specific indicators
   - Check support/resistance levels, moving averages, volume
3. **Seasonality Check**

   - Select ticker → **Seasonality Chart** tab
   - Compare current year vs historical patterns
   - Identify seasonal strength/weakness periods
4. **Comparison Analysis**

   - Select multiple tickers (Ctrl+Click)
   - Click **📈Compare** to see relative performance
   - Compare against sector ETF (e.g., XLK for tech)

### Phase 3: Fundamental Analysis

**Goal**: Understand the business and valuation

1. **Quick Metrics Review**

   - Select ticker → **Fundamental Analysis** tab
   - Filter for key metrics: `pe ratio market cap revenue`
   - Check: P/E, P/S, Debt/Equity, ROE, Profit Margins
2. **Business Deep Dive**

   - Go to **Business Analysis** tab
   - Click **Run BA** for comprehensive AI analysis
   - Review: Business model, competitive advantages, risks
3. **News & Sentiment**

   - Click **Conduct News Search** for recent developments
   - Click **Stock** news button for Finviz headlines
   - Use **📋** to summarize any article you find online
4. **Investment Framework Analysis**

   - Go to **Buffett & CANSLIM** tab
   - Click **Analyze Selected**
   - Review radar chart for strengths/weaknesses
   - Check if it meets value investing criteria

### Phase 4: SEC Filing Analysis

**Goal**: Verify financials and identify risks

1. **Annual Report (10-K)**

   - **Business Analysis** tab → Click **10K Study**
   - Or go to **SEC Filings** tab → Select 10-K → **Extract Tables**
   - Review: Revenue trends, margin changes, debt levels
2. **Quarterly Report (10-Q)**

   - Click **10-Q Study** for recent quarter analysis
   - Compare to previous quarters for trend changes
   - Look for management commentary changes
3. **Financial Tables**

   - Extract specific tables (Balance Sheet, Income Statement)
   - **Export to Excel** for detailed analysis
   - Track key metrics over multiple periods

### Phase 5: Decision & Monitoring

**Goal**: Make informed decision and track position

1. **Build Your Thesis**

   - Summarize findings using AI: Copy your notes → **📋**
   - Document: Why buy? At what price? What's the risk?
2. **Add to Watch List**

   - Right-click ticker → **Copy to Watch List**
   - Watch list persists across sessions
3. **Set Price Alerts** (External)

   - Use **🔗 URLs ▾** → **TradingView** for alerts
   - Or **Yahoo Finance** for email notifications
4. **Ongoing Monitoring**

   - Check **Stock** news regularly for your watch list
   - Re-run **Fundamental Analysis** after earnings
   - Update **10-Q Study** each quarter

---

## Quick Research Checklists

### 5-Minute Stock Check

- [ ] Open **D** chart - Is it trending up/down/sideways?
- [ ] **Fundamental Analysis** - P/E reasonable? Growing revenue?
- [ ] **Stock** news - Any red flags?

### 30-Minute Deep Dive

- [ ] **Multi-TF** charts - Trend alignment?
- [ ] **Seasonality** - Good entry timing?
- [ ] **Run BA** - Business quality?
- [ ] **Buffett & CANSLIM** - Investment grade?
- [ ] **10-Q Study** - Recent quarter healthy?

### Pre-Earnings Checklist

- [ ] Review last 4 quarters via **10-Q Study**
- [ ] Check **Conduct News Search** for analyst expectations
- [ ] Review **Seasonality** for historical earnings reactions
- [ ] Set position size based on volatility

### Portfolio Review (Weekly)

- [ ] Open Watch List
- [ ] Click **📊Visualize** for all charts
- [ ] Check **Market** news for macro changes
- [ ] Review any **Stock** news for holdings

---

## Research Templates

### Value Investing Template

```
1. Fundamental Analysis → Filter: "book value debt equity roe"
2. Buffett & CANSLIM → Check value metrics
3. 10-K Study → Verify balance sheet strength
4. AI Search → "What are the competitive advantages of [TICKER]?"
```

### Growth Investing Template

```
1. Fundamental Analysis → Filter: "revenue growth earnings"
2. Multi-TF Charts → Confirm uptrend
3. Run BA → Analyze market opportunity
4. 10-Q Study → Verify growth acceleration
```

### Dividend Investing Template

```
1. Fundamental Analysis → Filter: "dividend yield payout"
2. 10-K Study → Check dividend history
3. AI Search → "Is [TICKER] dividend sustainable?"
4. Seasonality → Best entry points historically
```

### Momentum Trading Template

```
1. D/W/M Charts → Identify breakout candidates
2. SC-Line Gallery → Check volume confirmation
3. Compare → Relative strength vs sector
4. Stock News → Catalyst identification
```

---

*Last updated: November 2025*
