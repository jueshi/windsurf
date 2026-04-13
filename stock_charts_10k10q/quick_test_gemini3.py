#!/usr/bin/env python3
"""
Quick test for gemini-3-flash-preview
Just run this script to test if the model works.
"""

import os
from dotenv import load_dotenv

# Set the model
MODEL = "gemini-3-flash-preview"
os.environ['GEMINI_MODEL_NAME'] = MODEL

load_dotenv()

# Import and test
from gemini_analyzer import _call_llm, get_active_llm_provider

print(f"🧪 Testing: {MODEL}")
print(f"📋 Active: {get_active_llm_provider()}")
print(f"\n⏳ Testing simple prompt...")

try:
    response = _call_llm("What is 2+2? Answer in one word.")
    print(f"\n✅ SUCCESS!")
    print(f"📝 Response: {response}")
    print(f"\n🎉 {MODEL} works!")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print(f"\n💡 Model '{MODEL}' may not be available.")
    print(f"   Try: gemini-2.0-flash-exp or gemini-2.5-flash")
