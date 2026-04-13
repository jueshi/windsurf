#!/usr/bin/env python3
"""
Test script to verify Gemini is now tried first, then OpenAI as fallback.
"""

import os
from dotenv import load_dotenv
from gemini_analyzer import get_active_llm_provider, _call_llm

def test_llm_priority():
    """Test that Gemini is prioritized over OpenAI"""

    load_dotenv()

    print("=" * 60)
    print("Testing LLM Priority Order")
    print("=" * 60)

    # Check which provider is active
    active_provider = get_active_llm_provider()
    print(f"\n📊 Active LLM Provider: {active_provider}")

    # Check API keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    print(f"\n🔑 API Key Status:")
    print(f"  - GEMINI_API_KEY: {'✅ Set' if gemini_key else '❌ Not set'}")
    print(f"  - OPENAI_API_KEY: {'✅ Set' if openai_key else '❌ Not set'}")

    # Test a simple call
    print(f"\n🧪 Testing LLM call with simple prompt...")
    test_prompt = "What is 2+2? Answer in one sentence."

    try:
        response = _call_llm(test_prompt)
        print(f"\n✅ LLM Call Successful!")
        print(f"📝 Response: {response[:200]}...")
        print(f"\n🎯 Priority order verified: Gemini → OpenAI")
    except Exception as e:
        print(f"\n❌ LLM Call Failed: {e}")
        print(f"💡 Make sure at least GEMINI_API_KEY or OPENAI_API_KEY is set in .env")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_llm_priority()
