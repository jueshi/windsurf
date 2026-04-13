# 🤖 Model Display Quick Reference

## One-Line Commands

### Display Current Model
```python
from gemini_analyzer import get_active_llm_provider
print(get_active_llm_provider())
```

### Display Full Configuration
```python
from gemini_analyzer import get_llm_config
import json
print(json.dumps(get_llm_config(), indent=2))
```

### Run from Terminal
```bash
python show_active_model.py
```

## Return Values

### `get_active_llm_provider()`
- ✅ `"Gemini (gemini-2.5-flash)"`
- ✅ `"OpenAI (gpt-4o-mini)"`
- ❌ `"No LLM configured"`

### `get_llm_config()`
```python
{
    'primary_provider': str or None,
    'fallback_provider': str or None,
    'gemini': {
        'available': bool,
        'api_key_set': bool,
        'model': str,
        'sdk_type': str
    },
    'openai': {
        'available': bool,
        'api_key_set': bool,
        'model': str or None
    }
}
```

## Common Patterns

### Log Model Before API Call
```python
from gemini_analyzer import get_active_llm_provider, _call_llm
import logging

provider = get_active_llm_provider()
logging.info(f"Using {provider}")
result = _call_llm("Your prompt")
```

### Check If Using Gemini
```python
from gemini_analyzer import get_llm_config

config = get_llm_config()
if 'gemini' in config['primary_provider'].lower():
    print("✅ Using Gemini API")
```

### Display in GUI
```python
from gemini_analyzer import get_active_llm_provider
import tkinter as tk

root = tk.Tk()
provider = get_active_llm_provider()
tk.Label(root, text=f"🤖 {provider}").pack()
root.mainloop()
```

## Files Created

- ✅ [`show_active_model.py`](show_active_model.py) - Display configuration
- ✅ [`test_model_display.py`](test_model_display.py) - Usage examples
- ✅ [`DISPLAY_MODEL_GUIDE.md`](DISPLAY_MODEL_GUIDE.md) - Complete guide
- ✅ Updated [`gemini_analyzer.py`](gemini_analyzer.py) with `get_llm_config()` function

## Test It

```bash
# Quick test
python -c "from gemini_analyzer import get_active_llm_provider; print(get_active_llm_provider())"

# Full configuration
python show_active_model.py

# Example usage
python test_model_display.py
```

---

**Need more details?** See [`DISPLAY_MODEL_GUIDE.md`](DISPLAY_MODEL_GUIDE.md)
