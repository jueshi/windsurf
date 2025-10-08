import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv


def test_basic_call():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set. Please add it to your .env file or environment.")
        sys.exit(1)

    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        print("Failed to initialize Gemini model.\n")
        print(str(e))
        sys.exit(2)

    try:
        resp = model.generate_content("Return the string OK")
        print("Model response:\n", resp.text)
    except Exception as e:
        print("Call failed:\n")
        print(str(e))
        sys.exit(3)


if __name__ == "__main__":
    print("Testing Gemini API connectivity...")
    test_basic_call()
    print("\nConnectivity test finished.")
