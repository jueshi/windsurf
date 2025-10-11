# Gemini Model Fallback Fix

This document explains the fix for the 404/unsupported Gemini model error encountered during Buffett & CANSLIM analysis and how the application now handles model selection robustly.

## Summary

- **Issue**: Calls to Gemini returned errors like: `404 models/gemini-1.5-flash is not found ... or is not supported for generateContent`.
- **Root cause**: The configured model may be unavailable in the current API version/region or not support the invoked method.
- **Fix**: Implemented a fallback sequence of supported Gemini models and prefer an environment-configured model when present.

## Files Changed

- `stock_charts_10k10q/stock_radar_batch.py`
  - Reads model from env `GEMINI_MODEL_NAME` with default `gemini-2.5-flash`.
  - Tries candidates in order until one works: `[env value, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-8b"]`.
  - If a candidate raises a "not found / not supported" style error, it automatically tries the next model.
  - Other errors (e.g., auth, rate limit) are re-raised to surface actionable issues.

- `stock_charts_10k10q/gemini_analyzer.py`
  - Added shared helpers:
    - `_get_gemini_model_candidates()` builds the ordered candidate list using `GEMINI_MODEL_NAME` and fallbacks.
    - `_init_gemini_model_with_fallback()` initializes `genai.GenerativeModel` with automatic fallback on 404/unsupported errors.
  - Updated callers to use the same fallback list:
    - `analyze_ticker()` (Business Analysis)
    - `general_search()` (AI Search)

## Configuration

- Set your API key via environment variable:
  - `GEMINI_API_KEY`
- Optionally pin a specific model:
  - `GEMINI_MODEL_NAME` (example: `gemini-2.5-flash`)

Add these to your `.env` in `stock_charts_10k10q/`:

```dotenv
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash
```

## Behavior

- When the analysis runs (`analyze_stock()` in `stock_radar_batch.py`), the app will attempt the configured model first.
- If that model is not found/unsupported, it tries the next fallback model automatically.
- Once a response is successfully generated, subsequent parsing proceeds as before.

## Testing Steps

1. Ensure `GEMINI_API_KEY` is set in your environment.
2. (Optional) Temporarily set `GEMINI_MODEL_NAME` to an unavailable model to exercise the fallback (e.g., `gemini-nonexistent`).
3. In the app, test both:
   - Buffett & CANSLIM: click "Analyze Selected" with a valid ticker.
   - Business Analysis tab: Run BA and also try AI Search.
4. Confirm that generation completes successfully and no 404/not supported error is shown.

## Notes

- If you still encounter errors, verify that the Generative Language API is enabled for your Google Cloud project and that billing is active.
- Region/feature availability can vary; using the fallbacks helps adapt without code changes.
