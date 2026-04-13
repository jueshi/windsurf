#!/usr/bin/env python3
"""
Test script to demonstrate console output of model information.
"""

from gemini_analyzer import _call_llm, get_active_llm_provider

def test_console_output():
    """Test that model is printed to console before API call"""

    print("=" * 70)
    print("🧪 Testing Model Console Output")
    print("=" * 70)

    # Show which provider is active
    provider = get_active_llm_provider()
    print(f"\n📋 Configuration: {provider}")

    # Test with a simple prompt
    print(f"\n🔬 Testing API call with simple prompt...")
    test_prompt = "What is 2+2? Answer in one sentence."

    try:
        response = _call_llm(test_prompt)

        print(f"\n✅ Test completed successfully!")
        print(f"📝 Response: {response}")
        print(f"\n👆 Notice the model was printed to console above before the API call")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print(f"💡 Make sure GEMINI_API_KEY or OPENAI_API_KEY is set in .env")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_console_output()
