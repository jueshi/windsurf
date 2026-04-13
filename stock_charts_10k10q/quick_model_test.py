#!/usr/bin/env python3
"""
Ultra-simple test: Just change the model name and run.
"""

# STEP 1: Change this to test different models
MODEL_NAME = "gemini-2.0-flash-exp"  # Options: gemini-2.0-flash-exp, gemini-2.5-flash, gemini-1.5-flash-8b

# STEP 2: Run the script
import os
os.environ['GEMINI_MODEL_NAME'] = MODEL_NAME

from gemini_analyzer import _call_llm, get_active_llm_provider

print(f"🤖 Testing model: {MODEL_NAME}")
print(f"📋 Active provider: {get_active_llm_provider()}")
print(f"\n⏳ Testing simple prompt...")

try:
    response = _call_llm("What is 2+2? Answer in one sentence.")
    print(f"\n✅ Success!")
    print(f"📝 Response: {response}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"\n💡 The model '{MODEL_NAME}' might not be available.")
    print(f"   Try: gemini-2.5-flash (stable)")
