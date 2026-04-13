# Gemini API Priority Update

## Summary

The LLM integration has been updated to prioritize **Gemini API first**, with **OpenAI as fallback**.

## Changes Made

### 1. Modified `_call_llm()` Function
- **File:** [`gemini_analyzer.py`](gemini_analyzer.py)
- **Changed parameter default:** `use_openai_first: bool = False` (was `True`)
- **New logic:**
  1. Try Gemini API first (if `GEMINI_API_KEY` is set)
  2. If Gemini fails, fall back to OpenAI (if `OPENAI_API_KEY` is set)
  3. Raise exception if both fail

### 2. Updated `get_active_llm_provider()` Function
- **New priority order:** Check Gemini first, then OpenAI
- Returns the appropriate provider name based on available API keys

### 3. Updated Documentation
- Changed all function docstrings from "OpenAI (first choice) or Gemini" to "Gemini (first choice) or OpenAI"
- Updated the unified LLM interface comment header

## API Priority Order

```
1. Gemini API (primary)
   ├─ Requires: GEMINI_API_KEY in .env
   └─ Models: gemini-2.5-flash (default), gemini-2.0-flash, gemini-1.5-flash-8b

2. OpenAI API (fallback)
   ├─ Requires: OPENAI_API_KEY in .env
   └─ Model: gpt-4o-mini (default, configurable via OPENAI_MODEL_NAME)
```

## Usage

### Default Behavior (Gemini First)
```python
from gemini_analyzer import _call_llm, get_active_llm_provider

# Check which provider will be used
provider = get_active_llm_provider()
print(f"Using: {provider}")  # e.g., "Using: Gemini (gemini-2.5-flash)"

# Make LLM call - Gemini will be tried first
response = _call_llm("Your prompt here")
```

### Force OpenAI First (Optional)
```python
# To use OpenAI first instead of Gemini
response = _call_llm("Your prompt", use_openai_first=True)
```

## Configuration

### Required Environment Variables

Add to your `.env` file:

```bash
# Primary: Gemini API (recommended)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash  # Optional, uses default if not set

# Fallback: OpenAI API (optional)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4o-mini  # Optional, uses default if not set
```

## Rate Limiting

The Gemini integration includes sophisticated rate limiting:
- **Minimum interval:** 10 seconds between API calls
- **Max retries:** 5 attempts for rate limit errors
- **Exponential backoff:** With jitter for better distribution
- **Smart retry delay:** Extracts suggested delays from API error messages

## Testing

Run the test script to verify the changes:

```bash
python test_gemini_priority.py
```

This will:
1. Show which LLM provider is active
2. Display API key status
3. Test a simple LLM call
4. Confirm the priority order is working correctly

## Benefits of Gemini First

1. **Cost Efficiency:** Gemini API is generally more cost-effective than OpenAI
2. **Rate Limits:** Better rate limiting for free tier usage
3. **Performance:** Fast response times with Gemini 2.5 Flash
4. **Reliability:** OpenAI serves as a robust fallback option

## Migration Notes

- **No breaking changes:** All existing code continues to work
- **Automatic fallback:** If Gemini fails, OpenAI is tried automatically
- **Backward compatible:** Can still force OpenAI first with `use_openai_first=True`

## Files Modified

- [`gemini_analyzer.py`](gemini_analyzer.py) - Main LLM integration module

## Files Created

- [`test_gemini_priority.py`](test_gemini_priority.py) - Test script for verification
- [`GEMINI_PRIORITY_UPDATE.md`](GEMINI_PRIORITY_UPDATE.md) - This documentation

---

**Date:** 2026-04-12
**Author:** Claude Code
**Version:** 1.0
