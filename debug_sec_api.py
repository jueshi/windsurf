"""
Debug SEC EDGAR API access with explicit environment variable handling
"""
import os
import sys
import requests

# Explicitly set the SEC email
SEC_EMAIL = "jueshi@gmail.com"
print(f"Using hardcoded SEC email: {SEC_EMAIL}")

# Test URL - company tickers JSON
url = "https://www.sec.gov/files/company_tickers.json"

# Set up headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
    'Referer': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    'From': SEC_EMAIL  # Add email directly
}

print(f"Requesting: {url}")
print(f"Headers: {headers}")

try:
    print("Sending request...")
    response = requests.get(url, headers=headers, timeout=30)
    print(f"Response received with status code: {response.status_code}")
    
    if response.status_code == 200:
        # Successfully got the data
        data = response.json()
        count = len(data)
        print(f"Successfully retrieved data with {count} entries")
        
        # Try to find Apple as a test
        for _, company in data.items():
            if company.get('ticker') == 'AAPL':
                print(f"Found AAPL: CIK = {company.get('cik_str')}")
                break
    else:
        print(f"Failed with status code: {response.status_code}")
        print(f"Response headers: {response.headers}")
        print(f"Response content: {response.content[:500]}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("Debug complete")
