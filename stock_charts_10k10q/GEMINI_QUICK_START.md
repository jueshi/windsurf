# Gemini API Quick Start Guide

## 🚀 Quick Setup

### 1. Get Your Gemini API Key
- Visit: https://makersuite.google.com/app/apikey
- Create a new API key
- Copy the key

### 2. Add to `.env` File
```bash
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 3. Install Required Package (if not already installed)
```bash
pip install google-genai
```

### 4. Test the Connection
```bash
python test_gemini_priority.py
```

## 📝 Usage Examples

### Basic LLM Call
```python
from gemini_analyzer import _call_llm

response = _call_llm("Explain quantum computing in simple terms")
print(response)
```

### Analyze a Stock
```python
from gemini_analyzer import analyze_ticker

company_info = {
    'longName': 'Apple Inc.',
    'sector': 'Technology',
    'industry': 'Consumer Electronics',
    'marketCap': 2500000000000,
    'trailingPE': 28.5,
    'longBusinessSummary': 'Apple designs and sells smartphones...'
}

analysis = analyze_ticker('AAPL', company_info)
print(analysis)
```

### Analyze SEC Filings
```python
from gemini_analyzer import analyze_10k_report, analyze_10q_report

# 10-K Annual Report
analysis_10k = analyze_10k_report('AAPL')

# 10-Q Quarterly Report
analysis_10q = analyze_10q_report('AAPL')
```

### News Analysis
```python
from gemini_analyzer import analyze_news

articles = [
    {
        'title': 'Apple announces new iPhone',
        'content': 'Apple unveiled its latest iPhone...',
        'url': 'https://example.com/article'
    }
]

summary, impacted_tickers = analyze_news(articles)
print(summary)
print(f"Impacted tickers: {impacted_tickers}")
```

## 🔧 Configuration Options

### Available Gemini Models
```bash
# Default (recommended)
GEMINI_MODEL_NAME=gemini-2.5-flash

# Alternative models
GEMINI_MODEL_NAME=gemini-2.0-flash
GEMINI_MODEL_NAME=gemini-1.5-flash-8b
```

### Rate Limiting Settings
Edit in `gemini_analyzer.py` if needed:
```python
class GeminiRateLimiter:
    MIN_INTERVAL = 10.0  # Seconds between API calls
    MAX_RETRIES = 5      # Maximum retry attempts
    BASE_DELAY = 10.0    # Base delay for exponential backoff
```

## 🎯 Key Features

### ✅ Automatic Fallback to OpenAI
If Gemini fails, the system automatically tries OpenAI (if `OPENAI_API_KEY` is set).

### ✅ Smart Rate Limiting
- Respects API rate limits
- Exponential backoff on errors
- Extracts retry delays from error messages

### ✅ Bilingual Output
Most analysis functions provide both Chinese and English responses.

### ✅ Structured JSON Output
News analysis returns structured JSON with sentiment analysis and impacted tickers.

## 🐛 Troubleshooting

### "No Gemini SDK installed" Error
```bash
pip install google-genai
```

### "API has not been used in project" Error
1. Visit: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com
2. Enable the Generative Language API
3. Ensure billing is enabled
4. Wait a few minutes for propagation

### Rate Limit Errors
- Free tier: 10 seconds minimum between calls
- Wait 10-15 seconds between requests
- Consider upgrading to paid tier for higher limits

### "GEMINI_API_KEY not set" Error
1. Check `.env` file exists in project root
2. Verify `GEMINI_API_KEY=...` line is present
3. Restart your Python interpreter after adding the key

## 📊 Monitoring

### Check Which LLM is Active
```python
from gemini_analyzer import get_active_llm_provider

provider = get_active_llm_provider()
print(f"Active LLM: {provider}")
# Output: "Active LLM: Gemini (gemini-2.5-flash)"
```

### View API Call Logs
The system logs important events:
```
INFO: Using Gemini API as primary LLM...
INFO: Rate limiting: Waiting 2.34 seconds before next API call
WARNING: Gemini API call failed: RATE_LIMIT_EXCEEDED
INFO: Falling back to OpenAI...
```

## 💡 Best Practices

1. **Start with Gemini**: It's cost-effective and fast
2. **Use OpenAI as backup**: Set `OPENAI_API_KEY` for reliability
3. **Respect rate limits**: Wait 10+ seconds between calls
4. **Cache results**: Store API responses to avoid redundant calls
5. **Handle errors**: Always wrap API calls in try-except blocks

## 🔗 Useful Links

- [Gemini API Documentation](https://ai.google.dev/docs)
- [Google AI Studio](https://makersuite.google.com/)
- [Rate Limit Policies](https://ai.google.dev/docs/rate_limits)
- [Model Comparison](https://ai.google.dev/models)

---

**Need Help?** Check the main documentation in [`GEMINI_PRIORITY_UPDATE.md`](GEMINI_PRIORITY_UPDATE.md)
