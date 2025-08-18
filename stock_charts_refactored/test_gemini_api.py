import os
from dotenv import load_dotenv
import google.generativeai as genai

def test_gemini_api():
    """Test if the Gemini API key is accessible and working."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    print(f"API key found: {bool(api_key)}")
    
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        return
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("Model initialized successfully")
        
        response = model.generate_content('Hello, can you respond with a short test message?')
        print(f"Response received: {response.text[:100]}...")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_gemini_api()
