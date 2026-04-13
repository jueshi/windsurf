#!/usr/bin/env python3
"""
Test script for Gemini experimental models.
Tests gemini-2.0-flash-exp and other experimental models.
"""

import os
from dotenv import load_dotenv

# Set the experimental model BEFORE importing gemini_analyzer
# This ensures the module picks up the model name
os.environ['GEMINI_MODEL_NAME'] = 'gemini-2.0-flash-exp'

# Now import after setting the environment variable
from gemini_analyzer import _call_llm, get_active_llm_provider, get_llm_config

def test_experimental_model():
    """Test the Gemini experimental model"""

    print("=" * 70)
    print("🧪 Testing Gemini Experimental Model")
    print("=" * 70)

    # Show configuration
    config = get_llm_config()
    print(f"\n📋 Configuration:")
    print(f"  • Provider: {config['primary_provider']}")
    print(f"  • Gemini Model: {config['gemini']['model']}")
    print(f"  • SDK Type: {config['gemini']['sdk_type']}")

    # Test prompts
    test_prompts = [
        {
            'name': 'Simple Math',
            'prompt': 'What is 25 * 4? Answer in one sentence.'
        },
        {
            'name': 'Creative Writing',
            'prompt': 'Write a haiku about stock trading.'
        },
        {
            'name': 'Code Generation',
            'prompt': 'Write a Python function to calculate fibonacci numbers.'
        }
    ]

    for i, test in enumerate(test_prompts, 1):
        print(f"\n{'=' * 70}")
        print(f"📝 Test {i}: {test['name']}")
        print(f"{'=' * 70}")
        print(f"Prompt: {test['prompt']}")
        print(f"\n⏳ Calling API...")

        try:
            response = _call_llm(test['prompt'])

            print(f"\n✅ Success!")
            print(f"📄 Response:")
            print(f"─" * 70)
            print(response[:500])  # Show first 500 chars
            if len(response) > 500:
                print(f"\n... (truncated, total {len(response)} chars)")
            print(f"─" * 70)

        except Exception as e:
            print(f"\n❌ Error: {e}")

    print(f"\n{'=' * 70}")
    print("✅ Testing Complete")
    print(f"{'=' * 70}")


def test_different_models():
    """Test different Gemini models"""

    print("\n" + "=" * 70)
    print("🔬 Testing Multiple Gemini Models")
    print("=" * 70)

    models_to_test = [
        'gemini-2.0-flash-exp',
        'gemini-2.5-flash',
        'gemini-2.0-flash-thinking-exp',
        'gemini-1.5-flash-8b'
    ]

    simple_prompt = "What is 2+2? Answer in one word."

    for model in models_to_test:
        print(f"\n{'─' * 70}")
        print(f"🧪 Testing: {model}")
        print(f"{'─' * 70}")

        # Set the model
        os.environ['GEMINI_MODEL_NAME'] = model

        try:
            response = _call_llm(simple_prompt)
            print(f"✅ {model}: {response[:100]}")

        except Exception as e:
            print(f"❌ {model}: {str(e)[:100]}")

        # Wait for rate limiting
        import time
        time.sleep(2)

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════╗
║          Gemini Experimental Model Test Suite                      ║
╚════════════════════════════════════════════════════════════════════╝

This script tests Gemini experimental models.
Current test model: gemini-2.0-flash-exp

Available experimental models to try:
• gemini-2.0-flash-exp (latest experimental)
• gemini-2.0-flash-thinking-exp (thinking mode)
• gemini-2.5-flash (stable, fast)
• gemini-1.5-flash-8b (small, fast)
• gemini-1.5-pro (larger context)
    """)

    choice = input("""
Choose test:
1. Test gemini-2.0-flash-exp (default)
2. Test multiple models
3. Exit

Enter choice (1-3): """).strip()

    if choice == '2':
        test_different_models()
    elif choice == '3':
        print("Goodbye!")
    else:
        test_experimental_model()

    print("\n💡 To test a specific model, set GEMINI_MODEL_NAME in .env:")
    print("   GEMINI_MODEL_NAME=gemini-2.0-flash-exp")
    print()
