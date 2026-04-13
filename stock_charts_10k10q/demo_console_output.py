#!/usr/bin/env python3
"""
Demo script showing console output for various analysis functions.
This demonstrates how model information is printed before each API call.
"""

from gemini_analyzer import (
    get_active_llm_provider,
    analyze_ticker,
    general_ai_search,
    _call_llm
)

def demo_ticker_analysis():
    """Demo: Analyze a stock ticker"""
    print("\n" + "=" * 70)
    print("🎯 DEMO 1: Stock Ticker Analysis")
    print("=" * 70)

    provider = get_active_llm_provider()
    print(f"📋 Active Provider: {provider}")

    # Sample company info
    company_info = {
        'longName': 'Apple Inc.',
        'sector': 'Technology',
        'industry': 'Consumer Electronics',
        'marketCap': 2500000000000,
        'trailingPE': 28.5,
        'longBusinessSummary': 'Apple Inc. designs, manufactures...'
    }

    print("\n👀 Watch for the model output below:")
    print("   The model name will be printed before the API call starts\n")

    # This will print the model to console before making the API call
    # result = analyze_ticker('AAPL', company_info)
    # print(result)

    print("\n⚠️  Actual API call commented out to avoid using quota")
    print("   Uncomment the analyze_ticker() lines above to test for real")


def demo_general_search():
    """Demo: General AI search"""
    print("\n" + "=" * 70)
    print("🎯 DEMO 2: General AI Search")
    print("=" * 70)

    provider = get_active_llm_provider()
    print(f"📋 Active Provider: {provider}")

    print("\n👀 Watch for the model output below:")
    print("   The model name will be printed before the API call starts\n")

    # This will print the model to console before making the API call
    # result = general_ai_search("What is the stock market?")
    # print(result)

    print("\n⚠️  Actual API call commented out to avoid using quota")
    print("   Uncomment the general_ai_search() lines above to test for real")


def demo_direct_llm_call():
    """Demo: Direct LLM call"""
    print("\n" + "=" * 70)
    print("🎯 DEMO 3: Direct LLM Call")
    print("=" * 70)

    provider = get_active_llm_provider()
    print(f"📋 Active Provider: {provider}")

    print("\n👀 Watch for the model output below:")
    print("   The model name will be printed before the API call starts\n")

    # This will print the model to console before making the API call
    # result = _call_llm("Explain quantum computing in one sentence")
    # print(result)

    print("\n⚠️  Actual API call commented out to avoid using quota")
    print("   Uncomment the _call_llm() lines above to test for real")


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("🎬 CONSOLE OUTPUT DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows how model information is printed to console")
    print("before each API call. Look for the 🤖 emoji with model name.")

    demo_ticker_analysis()
    demo_general_search()
    demo_direct_llm_call()

    print("\n" + "=" * 70)
    print("✅ Demo Complete")
    print("=" * 70)
    print("\n💡 To see actual API calls, uncomment the function calls in each demo")
    print("   The model name will be printed like this:")
    print("   🤖 Using Gemini API with model: gemini-2.5-flash")
    print("   🤖 Using OpenAI API with model: gpt-4o-mini")
    print()


if __name__ == "__main__":
    main()
