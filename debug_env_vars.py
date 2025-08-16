"""
Debug environment variables and SEC EDGAR API access
"""
import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

def debug_env_vars():
    print("Python version:", sys.version)
    print("Current working directory:", os.getcwd())
    
    # Check if .env file exists
    env_path = Path(".env")
    print(f".env file exists in current directory: {env_path.exists()}")
    
    # Try to load environment variables
    print("\nAttempting to load environment variables...")
    load_dotenv(verbose=True)
    
    # Check for SEC_EDGAR_EMAIL
    email = os.getenv("SEC_EDGAR_EMAIL")
    print(f"SEC_EDGAR_EMAIL: {email}")
    
    # Check for other environment variables
    print(f"GEMINI_API_KEY: {'Set' if os.getenv('GEMINI_API_KEY') else 'Not set'}")
    print(f"SERPAPI_API_KEY: {'Set' if os.getenv('SERPAPI_API_KEY') else 'Not set'}")
    
    # Test SEC EDGAR API access
    print("\nTesting SEC EDGAR API access...")
    url = "https://www.sec.gov/files/company_tickers.json"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Referer': 'https://www.sec.gov/edgar/searchedgar/companysearch'
    }
    
    # Add email to headers if available
    if email:
        headers['From'] = email
        print(f"Added From header with email: {email}")
    else:
        print("WARNING: No email found for From header")
    
    print(f"Requesting: {url}")
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success! Retrieved {len(data)} company entries")
        else:
            print(f"Failed with status code: {response.status_code}")
            print(f"Response headers: {response.headers}")
            print(f"Response content (first 500 chars): {response.content[:500]}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting environment variables and SEC EDGAR API debug...")
    debug_env_vars()
    print("\nDebug complete")
