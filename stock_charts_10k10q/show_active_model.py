#!/usr/bin/env python3
"""
Display which LLM provider and model will be used for API calls.
"""

import os
from dotenv import load_dotenv
from gemini_analyzer import get_active_llm_provider, get_llm_config

def show_active_model():
    """Display the current LLM configuration and active model"""

    load_dotenv()

    # Get detailed configuration
    config = get_llm_config()

    print("=" * 70)
    print("🤖 ACTIVE LLM CONFIGURATION")
    print("=" * 70)

    # SDK Status
    print("\n📦 SDK Status:")
    print(f"  • Gemini SDK: {config['gemini']['sdk_type']}")

    # API Keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    print("\n🔑 API Keys:")
    print(f"  • GEMINI_API_KEY: {'✅ Set' if gemini_key else '❌ Not set'}")
    if gemini_key:
        masked_key = gemini_key[:8] + "..." + gemini_key[-4:] if len(gemini_key) > 12 else "***"
        print(f"    └─ Key: {masked_key}")

    print(f"  • OPENAI_API_KEY: {'✅ Set' if openai_key else '❌ Not set'}")
    if openai_key:
        masked_key = openai_key[:8] + "..." + openai_key[-4:] if len(openai_key) > 12 else "***"
        print(f"    └─ Key: {masked_key}")

    # Model Configuration
    print("\n⚙️  Model Configuration:")
    print(f"  • Gemini Model: {config['gemini']['model']}")
    if config['openai']['model']:
        print(f"  • OpenAI Model: {config['openai']['model']}")

    # Active Provider
    active_provider = get_active_llm_provider()

    print("\n🎯 ACTIVE LLM PROVIDER:")
    print(f"  → {active_provider}")

    # Priority Order
    print("\n📊 PRIORITY ORDER:")
    if config['primary_provider']:
        print(f"  1. {config['primary_provider']} (Primary)")
        if config['fallback_provider']:
            print(f"  2. {config['fallback_provider']} (Fallback)")
    else:
        print("  ❌ No API keys configured!")

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if not gemini_key and not openai_key:
        print("  ⚠️  Configure at least one API key in .env file:")
        print("     - GEMINI_API_KEY=your_key_here")
        print("     - OPENAI_API_KEY=your_key_here")
    elif gemini_key and not openai_key:
        print("  ✅ Using Gemini API (recommended)")
        print("  💡 Consider adding OPENAI_API_KEY as fallback")
    elif not gemini_key and openai_key:
        print("  ⚠️  Using OpenAI API only")
        print("  💡 Consider adding GEMINI_API_KEY for better performance")
    else:
        print("  ✅ Both APIs configured (optimal setup)")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    show_active_model()
