"""
Basic OpenAI API Test Script
Tests if your OpenAI API key is working correctly.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openai():
    """Test OpenAI API connection and quota."""
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        return False
    
    print(f"✓ API Key found: {api_key[:8]}...{api_key[-4:]}")
    
    # Try to import OpenAI
    try:
        from openai import OpenAI
        print("✓ OpenAI package installed")
    except ImportError:
        print("❌ OpenAI package not installed. Run: pip install openai")
        return False
    
    # Initialize client
    try:
        client = OpenAI(api_key=api_key)
        print("✓ OpenAI client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return False
    
    # Test API call
    model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    print(f"\nTesting model: {model}")
    print("-" * 40)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say 'Hello! OpenAI is working!' in one sentence."}
            ],
            max_tokens=50
        )
        
        result = response.choices[0].message.content
        print(f"✅ SUCCESS! Response: {result}")
        
        # Show usage
        if response.usage:
            print(f"\nToken usage:")
            print(f"  - Prompt tokens: {response.usage.prompt_tokens}")
            print(f"  - Completion tokens: {response.usage.completion_tokens}")
            print(f"  - Total tokens: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            print(f"❌ QUOTA EXCEEDED: Your OpenAI account has no credits.")
            print("   → Add billing at: https://platform.openai.com/account/billing")
        elif "401" in error_msg or "invalid" in error_msg.lower():
            print(f"❌ INVALID API KEY: Check your OPENAI_API_KEY in .env")
        else:
            print(f"❌ API Error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("OpenAI API Test")
    print("=" * 50)
    print()
    
    success = test_openai()
    
    print()
    print("=" * 50)
    print(f"Result: {'PASS ✅' if success else 'FAIL ❌'}")
    print("=" * 50)
