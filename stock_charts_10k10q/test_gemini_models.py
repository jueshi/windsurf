#!/usr/bin/env python3
"""
Simple code snippet to test Gemini models.
Copy and modify this to test different models.
"""

import os
from dotenv import load_dotenv

# METHOD 1: Set model via environment variable (recommended)
load_dotenv()
os.environ['GEMINI_MODEL_NAME'] = 'gemini-2.0-flash-exp'  # Change this to test different models

# Then import and use normally
from gemini_analyzer import _call_llm, get_active_llm_provider

def test_gemini_model():
    """Test a specific Gemini model"""

    # Show which model will be used
    provider = get_active_llm_provider()
    print(f"🤖 Using: {provider}")

    # Test prompt
    prompt = "Explain quantum computing in one sentence."

    print(f"\n📝 Prompt: {prompt}")
    print(f"\n⏳ Calling API...")

    try:
        response = _call_llm(prompt)

        print(f"\n✅ Response:")
        print(f"─" * 70)
        print(response)
        print(f"─" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    test_gemini_model()
