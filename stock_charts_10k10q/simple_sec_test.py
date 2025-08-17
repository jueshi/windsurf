"""
Simple test script for SEC EDGAR URL retrieval
"""

import os
from dotenv import load_dotenv
from gemini_analyzer import _get_filing_url, _get_text_from_url

# Load environment variables
load_dotenv()

# Check environment variables
sec_email = os.getenv("SEC_EDGAR_EMAIL")
print(f"SEC_EDGAR_EMAIL: {'Set' if sec_email else 'Not set'}")

# Test with a single ticker
ticker = "AAPL"
filing_type = "10-K"

print(f"\nTesting {ticker} {filing_type} URL retrieval:")
try:
    url = _get_filing_url(ticker, filing_type)
    print(f"URL: {url}")
    
    if url:
        print("\nTesting text extraction:")
        text = _get_text_from_url(url)
        if text:
            print(f"Successfully extracted {len(text)} characters")
            print("\nFirst 300 characters:")
            print(text[:300])
        else:
            print("Failed to extract text")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
