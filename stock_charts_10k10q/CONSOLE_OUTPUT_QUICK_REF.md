# 🖥️ Console Output Quick Reference

## What You'll See

### Before Every API Call
```
🤖 Using Gemini API with model: gemini-2.5-flash
```
or
```
🤖 Using OpenAI API with model: gpt-4o-mini
```

### Fallback Scenario
```
🤖 Using Gemini API with model: gemini-2.5-flash
⚠️  Gemini API call failed, falling back to OpenAI...
🤖 Using OpenAI API with model: gpt-4o-mini
```

## Test It

```bash
# Quick test
python test_model_console.py

# See all patterns
python demo_console_output.py

# Check configuration
python show_active_model.py
```

## Functions That Print Model

All these functions automatically print the model before calling the API:
- ✅ `analyze_ticker()`
- ✅ `analyze_10k_report()`
- ✅ `analyze_10q_report()`
- ✅ `analyze_news()`
- ✅ `general_ai_search()`
- ✅ `summarize_market_news()`
- ✅ `summarize_crypto_news()`
- ✅ `summarize_etf_news()`
- ✅ `summarize_stock_news()`
- ✅ `summarize_clipboard_content()`
- ✅ `_call_llm()` (core function)

## Example Output

```bash
$ python your_analysis_script.py

📊 Analyzing ticker: AAPL (Apple Inc.)
🤖 Using Gemini API with model: gemini-2.5-flash
✅ Analysis completed for AAPL

📈 Starting 10-K analysis for AAPL...
Using edgartools to extract key 10-K sections...
Successfully extracted key sections: 125,432 characters
🤖 Using Gemini API with model: gemini-2.5-flash
✅ Analysis completed
```

## Customize

Edit [`gemini_analyzer.py`](gemini_analyzer.py) lines 299-361 to change the output format.

---

**Full documentation:** [`CONSOLE_OUTPUT_UPDATE.md`](CONSOLE_OUTPUT_UPDATE.md)
