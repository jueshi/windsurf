# OpenAI API Integration for LLM Requests

## Overview

The `gemini_analyzer.py` module has been updated to support **OpenAI as the first choice** for all LLM requests, with **Gemini as a fallback** if OpenAI is not available or fails.

## Configuration

### Environment Variables

Set one or both of the following environment variables in your `.env` file:

```bash
# OpenAI API Key (First Choice)
OPENAI_API_KEY=your_openai_api_key_here

# OpenAI Model Name (Optional, defaults to gpt-4o-mini)
OPENAI_MODEL_NAME=gpt-4o-mini

# Gemini API Key (Fallback)
GEMINI_API_KEY=your_gemini_api_key_here

# Gemini Model Name (Optional, defaults to gemini-2.5-flash)
GEMINI_MODEL_NAME=gemini-2.5-flash
```

### Priority Order

1. **OpenAI** - If `OPENAI_API_KEY` is set and the `openai` package is installed
2. **Gemini** - Falls back to Gemini if OpenAI is unavailable or fails

## Installation

To use OpenAI, install the OpenAI Python package:

```bash
pip install openai
```

If the `openai` package is not installed, the system will automatically use Gemini only.

## Updated Functions

The following functions now use the unified LLM interface with OpenAI as first choice:

### Analysis Functions

- `analyze_ticker(ticker, company_info)` - Business analysis for a stock ticker
- `analyze_10k_report(ticker)` - 10-K annual report analysis
- `analyze_10q_report(ticker)` - 10-Q quarterly report analysis
- `analyze_news(news_articles)` - News article sentiment analysis
- `general_search(ticker, company_info, query)` - General AI search about a company
- `general_ai_search(query)` - General AI search without ticker

### News Summarization Functions

- `summarize_market_news(articles, tickers)` - Finviz market news summary
- `summarize_stock_news(articles, tickers)` - Finviz v=3 stock news summary
- `summarize_etf_news(articles, tickers)` - Finviz v=4 ETF news summary
- `summarize_crypto_news(articles, tickers)` - Finviz v=5 crypto news summary
- `summarize_clipboard_content(content, urls)` - Clipboard content summary

## Helper Functions

### `get_active_llm_provider() -> str`

Returns the name of the LLM provider that will be used. Useful for displaying to users which API is active.

```python
from gemini_analyzer import get_active_llm_provider

provider = get_active_llm_provider()
print(f"Active LLM: {provider}")
# Output: "OpenAI (gpt-4o-mini)" or "Gemini (gemini-2.5-flash)" or "No LLM configured"
```

### `_call_llm(prompt, use_openai_first=True) -> str`

Internal unified LLM interface that tries OpenAI first, then falls back to Gemini.

## Error Handling

- If OpenAI fails, the system automatically falls back to Gemini
- If both fail, an appropriate error message is returned
- Rate limiting is handled for both APIs

## Date

Updated: December 2024
