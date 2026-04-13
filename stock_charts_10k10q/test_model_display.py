#!/usr/bin/env python3
"""
Simple example of how to display the active model in your code.
"""

from gemini_analyzer import get_active_llm_provider, get_llm_config

def example_usage():
    """Example: Display active model in your application"""

    # Method 1: Simple string (for quick display)
    provider = get_active_llm_provider()
    print(f"Using: {provider}")

    # Method 2: Detailed configuration (for debugging)
    config = get_llm_config()
    print(f"\nDetailed Config:")
    print(f"  Primary: {config['primary_provider']}")
    print(f"  Fallback: {config['fallback_provider']}")
    print(f"  Gemini Model: {config['gemini']['model']}")
    print(f"  OpenAI Model: {config['openai']['model']}")

    # Method 3: Check if specific model is being used
    if "gemini-2.5-flash" in config['gemini']['model']:
        print("\n✅ Using latest Gemini 2.5 Flash model")

if __name__ == "__main__":
    example_usage()
