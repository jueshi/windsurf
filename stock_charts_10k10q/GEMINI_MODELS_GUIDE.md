# Gemini Models - Quick Test Guide

## 🚀 Quick Test

### Method 1: Environment Variable (Recommended)

```python
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['GEMINI_MODEL_NAME'] = 'gemini-2.0-flash-exp'

from gemini_analyzer import _call_llm, get_active_llm_provider

print(f"Using: {get_active_llm_provider()}")
response = _call_llm("What is 2+2?")
print(response)
```

### Method 2: .env File

Add to your `.env` file:
```bash
GEMINI_MODEL_NAME=gemini-2.0-flash-exp
```

Then use normally:
```python
from gemini_analyzer import _call_llm

response = _call_llm("Your prompt here")
```

## 📊 Available Gemini Models

### Experimental Models (Latest)
```bash
gemini-2.0-flash-exp              # Latest experimental flash
gemini-2.0-flash-thinking-exp     # Thinking mode experimental
gemini-exp-1206                   # Experimental from Dec 2024
```

### Stable Models (Production Ready)
```bash
gemini-2.5-flash                  # Fast, cost-effective (recommended)
gemini-2.0-flash                  # Fast, good performance
gemini-1.5-flash-8b               # Smallest, fastest
gemini-1.5-flash                  # Balanced
gemini-1.5-pro                    # Larger context
```

### Legacy Models
```bash
gemini-1.0-pro                    # Original model
gemini-1.5-pro-experimental       # Experimental pro
```

## 🧪 Test Scripts

### Quick Test
```bash
python test_gemini_models.py
```

### Full Test Suite
```bash
python test_gemini_experimental_model.py
```

### Test Multiple Models
```bash
# Run option 2 from the menu
python test_gemini_experimental_model.py
# Choose: 2. Test multiple models
```

## 💻 Code Snippets

### Test gemini-2.0-flash-exp
```python
import os
os.environ['GEMINI_MODEL_NAME'] = 'gemini-2.0-flash-exp'

from gemini_analyzer import _call_llm

response = _call_llm("Write a haiku about AI")
print(response)
```

### Test gemini-2.0-flash-thinking-exp
```python
import os
os.environ['GEMINI_MODEL_NAME'] = 'gemini-2.0-flash-thinking-exp'

from gemini_analyzer import _call_llm

response = _call_llm("Solve: What comes next in 2, 4, 8, 16, ...?")
print(response)
```

### Test with Your Analysis Functions
```python
import os
os.environ['GEMINI_MODEL_NAME'] = 'gemini-2.0-flash-exp'

from gemini_analyzer import analyze_ticker

company_info = {
    'longName': 'Apple Inc.',
    'sector': 'Technology',
    'industry': 'Consumer Electronics',
    'marketCap': 2500000000000,
    'trailingPE': 28.5,
    'longBusinessSummary': 'Apple designs and sells smartphones...'
}

result = analyze_ticker('AAPL', company_info)
print(result)
```

## 🎯 Model Comparison

### Speed (Fastest to Slowest)
1. `gemini-1.5-flash-8b` - Fastest
2. `gemini-2.0-flash-exp` - Very fast
3. `gemini-2.5-flash` - Fast
4. `gemini-2.0-flash` - Fast
5. `gemini-1.5-flash` - Moderate
6. `gemini-1.5-pro` - Slower but smarter
7. `gemini-2.0-flash-thinking-exp` - Slowest (thinking mode)

### Cost (Cheapest to Most Expensive)
1. `gemini-1.5-flash-8b` - Cheapest
2. `gemini-2.0-flash-exp` - Very cheap
3. `gemini-2.5-flash` - Cheap
4. `gemini-2.0-flash` - Cheap
5. `gemini-1.5-flash` - Moderate
6. `gemini-1.5-pro` - More expensive
7. `gemini-2.0-flash-thinking-exp` - Most expensive

### Quality (Basic to Advanced)
1. `gemini-1.5-flash-8b` - Basic tasks
2. `gemini-2.0-flash-exp` - Good quality
3. `gemini-2.5-flash` - Good quality
4. `gemini-2.0-flash` - Good quality
5. `gemini-1.5-flash` - Better quality
6. `gemini-1.5-pro` - High quality
7. `gemini-2.0-flash-thinking-exp` - Best quality (complex reasoning)

## 🔍 Troubleshooting

### Model Not Found
```
Error: Model 'gemini-2.0-flash-exp' not found
```

**Solution:** The model might not be available to your API key. Try:
```bash
GEMINI_MODEL_NAME=gemini-2.5-flash  # Fallback to stable model
```

### Permission Denied
```
Error: API key does not have access to this model
```

**Solution:** Some experimental models require specific API permissions. Use stable models instead.

### Rate Limiting
```
Error: Rate limit exceeded
```

**Solution:** Wait 10-15 seconds between calls when testing experimental models.

## 📝 Recommended Models for Different Use Cases

### Stock Analysis (Your Use Case)
```bash
# Recommended
GEMINI_MODEL_NAME=gemini-2.5-flash  # Fast, good quality, cost-effective

# Alternative (if you want latest features)
GEMINI_MODEL_NAME=gemini-2.0-flash-exp  # Experimental but good
```

### Simple Tasks
```bash
GEMINI_MODEL_NAME=gemini-1.5-flash-8b  # Fastest, cheapest
```

### Complex Reasoning
```bash
GEMINI_MODEL_NAME=gemini-2.0-flash-thinking-exp  # Best for complex tasks
```

### Production (Stable)
```bash
GEMINI_MODEL_NAME=gemini-2.5-flash  # Stable, reliable
```

## 🧪 Run Tests Now

```bash
# Test the experimental model
python test_gemini_models.py

# Full test suite with menu
python test_gemini_experimental_model.py

# Check current configuration
python show_active_model.py
```

## 💡 Tips

1. **Start with stable models** (`gemini-2.5-flash`) for production
2. **Test experimental models** (`gemini-2.0-flash-exp`) for new features
3. **Monitor costs** - Experimental models can change pricing
4. **Check availability** - Not all models are available to all API keys
5. **Rate limits** - Experimental models may have stricter limits

---

**Updated:** 2026-04-12
**Models tested:** gemini-2.0-flash-exp, gemini-2.5-flash, gemini-1.5-flash-8b
