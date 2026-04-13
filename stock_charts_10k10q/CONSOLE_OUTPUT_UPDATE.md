# Console Output: Model Display Before API Calls

## ✅ What Changed

The codebase now **automatically prints the model name to console** before every API call. You'll see output like:

```
🤖 Using Gemini API with model: gemini-2.5-flash
```

or

```
🤖 Using OpenAI API with model: gpt-4o-mini
```

## 📝 Where Model Is Printed

### 1. All LLM Function Calls
Every time you call any of these functions, the model is printed:
- `_call_llm()` - Core LLM interface
- `analyze_ticker()` - Stock analysis
- `analyze_10k_report()` - 10-K SEC filings
- `analyze_10q_report()` - 10-Q quarterly reports
- `analyze_news()` - News analysis
- `general_ai_search()` - General search
- `summarize_*_news()` - News summarization functions

### 2. Console Output Format

**Gemini API (Primary):**
```
📊 Analyzing ticker: AAPL (Apple Inc.)
🤖 Using Gemini API with model: gemini-2.5-flash
✅ Analysis completed for AAPL
```

**OpenAI API (Fallback):**
```
⚠️  Gemini API call failed, falling back to OpenAI...
🤖 Using OpenAI API with model: gpt-4o-mini
✅ Analysis completed for AAPL
```

### 3. Analysis Function Output

**Ticker Analysis:**
```
📊 Analyzing ticker: AAPL (Apple Inc.)
🤖 Using Gemini API with model: gemini-2.5-flash
✅ Analysis completed for AAPL
```

**SEC Filing Analysis:**
```
📈 Starting 10-K analysis for AAPL...
Using edgartools to extract key 10-K sections...
Successfully extracted key sections: 125,432 characters
🤖 Using Gemini API with model: gemini-2.5-flash
✅ Analysis completed
```

## 🧪 Test It

Run the test script to see console output:
```bash
python test_model_console.py
```

Run the demo to see all output patterns:
```bash
python demo_console_output.py
```

## 💡 Benefits

1. **Transparency:** See which API is being used
2. **Debugging:** Easy to verify model selection
3. **Cost Tracking:** Know when Gemini vs OpenAI is used
4. **Fallback Visibility:** See when fallback occurs
5. **User Feedback:** Users know what's happening

## 🔍 Implementation Details

### Core Function (`_call_llm`)
The model is printed at the start of the API call flow:

```python
# Gemini (Primary)
if api_key:
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    print(f"🤖 Using Gemini API with model: {model_name}")

# OpenAI (Fallback)
if openai_client:
    model_name = _get_openai_model_name()
    print(f"🤖 Using OpenAI API with model: {model_name}")
```

### Analysis Functions
Each analysis function prints context before calling the LLM:

```python
print(f"\n📊 Analyzing ticker: {ticker} ({company_name})")
# ... model printed here by _call_llm()
print(f"✅ Analysis completed for {ticker}")
```

## 🎯 Customization

### Change the Output Format
Edit [`gemini_analyzer.py`](gemini_analyzer.py) lines 299-361:
```python
# Current format
print(f"🤖 Using Gemini API with model: {model_name}")

# Custom format (example)
print(f"[LLM] {model_name} - Processing...")
```

### Disable Console Output
If you want to disable the console output, you can:
1. Comment out the `print()` statements in `_call_llm()`
2. Or use logging level to control output
3. Or add an environment variable flag

## 📊 Example Output

Here's what you'll see when analyzing a stock:

```bash
$ python your_script.py

📊 Analyzing ticker: AAPL (Apple Inc.)
🤖 Using Gemini API with model: gemini-2.5-flash
✅ Analysis completed for AAPL

[Analysis results here...]
```

When fallback occurs:
```bash
$ python your_script.py

📊 Analyzing ticker: TSLA (Tesla Inc.)
🤖 Using Gemini API with model: gemini-2.5-flash
⚠️  Gemini API call failed, falling back to OpenAI...
🤖 Using OpenAI API with model: gpt-4o-mini
✅ Analysis completed for TSLA

[Analysis results here...]
```

## 🆚 Before vs After

**Before:**
```bash
$ python analyze_stock.py
Analyzing AAPL...
[Silent processing...]
[Results appear]
```

**After:**
```bash
$ python analyze_stock.py
📊 Analyzing ticker: AAPL (Apple Inc.)
🤖 Using Gemini API with model: gemini-2.5-flash
✅ Analysis completed for AAPL
[Results appear]
```

## 📁 Related Files

- [`gemini_analyzer.py`](gemini_analyzer.py) - Updated with console output
- [`test_model_console.py`](test_model_console.py) - Test script
- [`demo_console_output.py`](demo_console_output.py) - Demo all output patterns
- [`show_active_model.py`](show_active_model.py) - Show current configuration

---

**Updated:** 2026-04-12
**Status:** ✅ Active
