"""
Quick test for SEC API cache and retry logic
"""
import os
import time
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if SEC email is set
sec_email = os.getenv("SEC_EDGAR_EMAIL")
print(f"SEC_EDGAR_EMAIL: {'Set' if sec_email else 'Not set'}")

try:
    # Import our SEC API wrapper
    from sec_api_wrapper import SECAPIWrapper
    
    # Create a real SEC API wrapper
    api = SECAPIWrapper(use_mock=False)
    
    # Test with a single ticker
    ticker = "AAPL"
    print(f"\nTesting with ticker: {ticker}")
    
    # Test CIK lookup with caching
    print("\nStep 1: Get company CIK")
    start_time = time.time()
    cik = api.get_company_cik(ticker)
    elapsed = time.time() - start_time
    
    if cik:
        print(f"Found CIK: {cik}")
        print(f"Time taken: {elapsed:.2f} seconds")
        
        # Test second call (should use cache)
        print("\nStep 2: Get company CIK again (should use cache)")
        start_time = time.time()
        cik2 = api.get_company_cik(ticker)
        elapsed = time.time() - start_time
        
        print(f"Found CIK: {cik2}")
        print(f"Time taken: {elapsed:.2f} seconds")
        
        # Test filing info retrieval
        print("\nStep 3: Get latest 10-K filing info")
        start_time = time.time()
        filing_info = api.get_latest_filing_info(cik, "10-K")
        elapsed = time.time() - start_time
        
        if filing_info:
            print(f"Found filing from {filing_info['filingDate']}")
            print(f"Time taken: {elapsed:.2f} seconds")
            print(f"Filing URL: {filing_info['detailUrl']}")
            
            # Test successful
            print("\nTest completed successfully!")
            sys.exit(0)
        else:
            print("Failed to get filing info")
    else:
        print(f"Failed to get CIK for {ticker}")
    
    # If we got here, something failed
    print("\nTest failed!")
    sys.exit(1)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
