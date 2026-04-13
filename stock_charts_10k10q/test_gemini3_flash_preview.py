#!/usr/bin/env python3
"""
Test script for gemini-3-flash-preview model.
"""

import os
import sys
from dotenv import load_dotenv

# Set the model to test
MODEL_TO_TEST = "gemini-3-flash-preview"

print("=" * 70)
print(f"🧪 Testing Gemini Model: {MODEL_TO_TEST}")
print("=" * 70)

# Set the model environment variable
os.environ['GEMINI_MODEL_NAME'] = MODEL_TO_TEST
load_dotenv()

# Now import after setting the model
try:
    from gemini_analyzer import (
        _call_llm,
        get_active_llm_provider,
        get_llm_config,
        GEMINI_AVAILABLE,
        OPENAI_AVAILABLE
    )

    # Check availability
    print(f"\n📦 Status Check:")
    print(f"  • Gemini SDK: {'✅ Available' if GEMINI_AVAILABLE else '❌ Not available'}")
    print(f"  • OpenAI SDK: {'✅ Available' if OPENAI_AVAILABLE else '❌ Not available'}")

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    print(f"  • GEMINI_API_KEY: {'✅ Set' if api_key else '❌ Not set'}")

    if not api_key:
        print(f"\n❌ ERROR: GEMINI_API_KEY not set in .env file!")
        print(f"💡 Please add it to your .env file:")
        print(f"   GEMINI_API_KEY=your_actual_api_key_here")
        sys.exit(1)

    # Show configuration
    config = get_llm_config()
    print(f"\n⚙️  Configuration:")
    print(f"  • Primary Provider: {config['primary_provider']}")
    print(f"  • Gemini Model: {config['gemini']['model']}")
    print(f"  • SDK Type: {config['gemini']['sdk_type']}")

    # Verify the model is set correctly
    if config['gemini']['model'] == MODEL_TO_TEST:
        print(f"\n✅ Model set correctly: {MODEL_TO_TEST}")
    else:
        print(f"\n⚠️  Model mismatch!")
        print(f"   Expected: {MODEL_TO_TEST}")
        print(f"   Actual: {config['gemini']['model']}")

    # Test prompts
    test_prompts = [
        {
            'name': 'Simple Math',
            'prompt': 'What is 123 + 456? Answer in one sentence.'
        },
        {
            'name': 'Creative Writing',
            'prompt': 'Write a haiku about artificial intelligence.'
        },
        {
            'name': 'Code Generation',
            'prompt': 'Write a simple Python function that adds two numbers.'
        }
    ]

    print(f"\n{'=' * 70}")
    print(f"🧪 Running {len(test_prompts)} Test Cases")
    print(f"{'=' * 70}")

    for i, test in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: {test['name']}")
        print(f"─" * 70)
        print(f"Prompt: {test['prompt']}")
        print(f"\n⏳ Calling {MODEL_TO_TEST}...")

        try:
            response = _call_llm(test['prompt'])

            print(f"\n✅ Success!")
            print(f"📄 Response:")
            print(f"─" * 70)
            # Show first 300 chars
            preview = response[:300] + "..." if len(response) > 300 else response
            print(preview)
            print(f"─" * 70)
            print(f"📊 Response length: {len(response)} characters")

        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Error: {error_msg}")

            # Check for common errors
            if "404" in error_msg or "not found" in error_msg.lower():
                print(f"\n💡 The model '{MODEL_TO_TEST}' may not be available.")
                print(f"   This could be because:")
                print(f"   1. The model name is incorrect")
                print(f"   2. The model is not available to your API key")
                print(f"   3. The model is in a different region")
                print(f"\n   Try these alternatives:")
                print(f"   • gemini-2.0-flash-exp")
                print(f"   • gemini-2.5-flash")
                print(f"   • gemini-1.5-flash-8b")

            elif "permission" in error_msg.lower() or "access" in error_msg.lower():
                print(f"\n💡 Your API key may not have access to '{MODEL_TO_TEST}'")
                print(f"   Try using a stable model like gemini-2.5-flash")

            elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
                print(f"\n💡 Rate limit or quota exceeded.")
                print(f"   Wait a few minutes and try again.")

        # Small delay between tests
        import time
        if i < len(test_prompts):
            print(f"\n⏸️  Waiting 2 seconds before next test...")
            time.sleep(2)

    print(f"\n{'=' * 70}")
    print(f"✅ Testing Complete")
    print(f"{'=' * 70}")

    print(f"\n💡 To use this model permanently, add to your .env file:")
    print(f"   GEMINI_MODEL_NAME={MODEL_TO_TEST}")

    print(f"\n🔗 Available models to try:")
    models = [
        "gemini-2.0-flash-exp",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro"
    ]
    for model in models:
        print(f"   • {model}")

except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print(f"💡 Make sure gemini_analyzer.py exists and is importable")
except Exception as e:
    print(f"\n❌ Unexpected Error: {e}")
    import traceback
    traceback.print_exc()

print()
