import os
import sys
try:
    from google import genai
    _USE_NEW = True
except ImportError:
    import google.generativeai as genai
    _USE_NEW = False
from dotenv import load_dotenv


def test_basic_call():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set. Please add it to your .env file or environment.")
        sys.exit(1)

    try:
        if _USE_NEW:
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(model="gemini-2.5-flash", contents="Return the string OK")
        else:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
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
