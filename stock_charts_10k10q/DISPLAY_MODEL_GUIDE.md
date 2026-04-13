# How to Display Active LLM Model

## Quick Reference

### Method 1: Simple Display (Recommended)
```python
from gemini_analyzer import get_active_llm_provider

provider = get_active_llm_provider()
print(f"Using: {provider}")
# Output: "Using: Gemini (gemini-2.5-flash)"
```

### Method 2: Detailed Configuration
```python
from gemini_analyzer import get_llm_config

config = get_llm_config()
print(f"Primary: {config['primary_provider']}")
print(f"Fallback: {config['fallback_provider']}")
print(f"Gemini Model: {config['gemini']['model']}")
print(f"OpenAI Model: {config['openai']['model']}")
```

## Complete Example

```python
from gemini_analyzer import get_active_llm_provider, get_llm_config, _call_llm

def analyze_with_model_display(ticker):
    """Analyze a stock and show which model was used"""

    # Display active model
    provider = get_active_llm_provider()
    print(f"🤖 Using: {provider}")
    print(f"📊 Analyzing {ticker}...")

    # Make the API call
    prompt = f"Analyze {ticker} stock"
    response = _call_llm(prompt)

    return response

# Usage
result = analyze_with_model_display("AAPL")
```

## Available Functions

### `get_active_llm_provider()`
**Returns:** String with provider name and model

```python
provider = get_active_llm_provider()
# Returns: "Gemini (gemini-2.5-flash)" or "OpenAI (gpt-4o-mini)"
```

### `get_llm_config()`
**Returns:** Dictionary with detailed configuration

```python
config = get_llm_config()
# Returns:
# {
#     'primary_provider': 'Gemini (gemini-2.5-flash)',
#     'fallback_provider': 'OpenAI (gpt-4o-mini)',
#     'gemini': {
#         'available': True,
#         'api_key_set': True,
#         'model': 'gemini-2.5-flash',
#         'sdk_type': 'google.genai (new)'
#     },
#     'openai': {
#         'available': True,
#         'api_key_set': True,
#         'model': 'gpt-4o-mini'
#     }
# }
```

## Standalone Scripts

### Run from Command Line

```bash
# Show active model and configuration
python show_active_model.py

# Test with a simple API call
python test_gemini_priority.py

# Quick model display example
python test_model_display.py
```

### Output Example

```
======================================================================
🤖 ACTIVE LLM CONFIGURATION
======================================================================

📦 SDK Status:
  • Gemini SDK: google.genai (new)

🔑 API Keys:
  • GEMINI_API_KEY: ✅ Set
    └─ Key: AIzaSyAb...xyz1
  • OPENAI_API_KEY: ✅ Set
    └─ Key: sk-proj...abc2

⚙️  Model Configuration:
  • Gemini Model: gemini-2.5-flash
  • OpenAI Model: gpt-4o-mini

🎯 ACTIVE LLM PROVIDER:
  → Gemini (gemini-2.5-flash)

📊 PRIORITY ORDER:
  1. Gemini (gemini-2.5-flash) (Primary)
  2. OpenAI (gpt-4o-mini) (Fallback)

💡 RECOMMENDATIONS:
  ✅ Both APIs configured (optimal setup)

======================================================================
```

## Integration Examples

### GUI Application
```python
import tkinter as tk
from gemini_analyzer import get_active_llm_provider

class StockAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        self.update_model_display()

    def setup_ui(self):
        # Model status label
        self.model_label = tk.Label(root, text="Initializing...", fg="blue")
        self.model_label.pack(pady=5)

    def update_model_display(self):
        provider = get_active_llm_provider()
        self.model_label.config(text=f"🤖 {provider}")

# Usage
root = tk.Tk()
app = StockAnalyzerGUI(root)
root.mainloop()
```

### CLI Application
```python
import argparse
from gemini_analyzer import get_active_llm_provider, _call_llm

def main():
    parser = argparse.ArgumentParser(description="Stock Analysis Tool")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--show-model", action="store_true",
                       help="Display which model is being used")
    args = parser.parse_args()

    if args.show_model:
        provider = get_active_llm_provider()
        print(f"Using: {provider}\n")

    print(f"Analyzing {args.ticker}...")
    result = _call_llm(f"Analyze {args.ticker} stock")
    print(result)

if __name__ == "__main__":
    main()
```

### Web Application (Flask)
```python
from flask import Flask, jsonify
from gemini_analyzer import get_active_llm_provider, get_llm_config

app = Flask(__name__)

@app.route('/api/model')
def get_model_info():
    """API endpoint to get current model info"""
    return jsonify({
        'provider': get_active_llm_provider(),
        'config': get_llm_config()
    })

# Usage: curl http://localhost:5000/api/model
```

## Logging Model Usage

```python
import logging
from gemini_analyzer import get_active_llm_provider, _call_llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_with_logging(ticker):
    """Analyze with model logging"""

    provider = get_active_llm_provider()
    logger.info(f"Starting analysis for {ticker} using {provider}")

    try:
        result = _call_llm(f"Analyze {ticker}")
        logger.info(f"Analysis completed successfully")
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

# Usage
analyze_with_logging("AAPL")
```

## Model Selection Logic

The system follows this priority:

1. **Gemini API** (if `GEMINI_API_KEY` is set)
   - Model: `gemini-2.5-flash` (default, configurable via `GEMINI_MODEL_NAME`)
   - Falls back to OpenAI if fails

2. **OpenAI API** (if only `OPENAI_API_KEY` is set)
   - Model: `gpt-4o-mini` (default, configurable via `OPENAI_MODEL_NAME`)

3. **No LLM** (if no API keys configured)
   - Returns: "No LLM configured"

## Environment Variables

```bash
# .env file
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash  # Optional

OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL_NAME=gpt-4o-mini  # Optional
```

## Troubleshooting

### Model Not Showing Correctly
```python
# Check your configuration
from gemini_analyzer import get_llm_config
import json

config = get_llm_config()
print(json.dumps(config, indent=2))
```

### Force Specific Model
```python
import os
from dotenv import load_dotenv

load_dotenv()

# Temporarily override model
os.environ['GEMINI_MODEL_NAME'] = 'gemini-2.0-flash'

from gemini_analyzer import _call_llm
result = _call_llm("Your prompt")
```

---

**See also:**
- [`GEMINI_QUICK_START.md`](GEMINI_QUICK_START.md) - Setup guide
- [`GEMINI_PRIORITY_UPDATE.md`](GEMINI_PRIORITY_UPDATE.md) - Priority configuration
