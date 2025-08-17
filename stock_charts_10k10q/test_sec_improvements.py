"""
Test script for SEC API wrapper with improved caching and retry logic
"""
import os
import time
import logging
import sys

# Configure logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Import our modules
try:
    from sec_api_wrapper import SECAPIWrapper
    import sec_api_cache
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_caching():
    """Test the improved caching mechanism"""
    print("\n=== Testing SEC API Caching ===")
    
    # Create a real SEC API wrapper
    api = SECAPIWrapper(use_mock=False)
    
    # Test with a single ticker
    ticker = "AAPL"
    
    # First call - might hit the API
    print(f"\nFirst call for {ticker}:")
    start_time = time.time()
    try:
        cik = api.get_company_cik(ticker)
        elapsed = time.time() - start_time
        print(f"CIK: {cik}")
        print(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}")
    
    # Second call - should use cache
    print(f"\nSecond call for {ticker} (should use cache):")
    start_time = time.time()
    try:
        cik = api.get_company_cik(ticker)
        elapsed = time.time() - start_time
        print(f"CIK: {cik}")
        print(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}")

def test_filing_retrieval():
    """Test filing retrieval with caching"""
    print("\n=== Testing SEC Filing Retrieval ===")
    
    # Create a real SEC API wrapper
    api = SECAPIWrapper(use_mock=False)
    
    # Test with a single ticker and filing type
    ticker = "AAPL"
    form_type = "10-K"
    
    print(f"\nRetrieving {form_type} for {ticker}:")
    try:
        # Get CIK
        cik = api.get_company_cik(ticker)
        print(f"CIK: {cik}")
        
        # Get filing info
        start_time = time.time()
        filing_info = api.get_latest_filing_info(cik, form_type)
        elapsed = time.time() - start_time
        
        if filing_info:
            print(f"Filing found: {filing_info['accessionNumber']}")
            print(f"Filing date: {filing_info['filingDate']}")
            print(f"Time taken: {elapsed:.2f} seconds")
        else:
            print(f"No {form_type} filing found for {ticker}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    print("Starting SEC API improvement tests...")
    
    # Check environment variables
    sec_email = os.getenv("SEC_EDGAR_EMAIL")
    print(f"SEC_EDGAR_EMAIL: {'Set' if sec_email else 'Not set'}")
    
    if not sec_email:
        print("WARNING: SEC_EDGAR_EMAIL environment variable not set.")
        print("Set this variable in your .env file for proper SEC API access.")
    
    # Run tests
    try:
        test_caching()
        test_filing_retrieval()
        print("\nAll tests completed.")
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
