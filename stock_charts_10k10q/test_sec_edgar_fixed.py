"""
Test script to verify SEC EDGAR URL retrieval and text extraction functionality.
This script tests the fixed implementation to ensure it properly handles 403 Forbidden errors.
"""

import os
import sys
from dotenv import load_dotenv
from gemini_analyzer import _get_filing_url, _get_text_from_url

def test_sec_edgar_url_retrieval():
    """Test SEC EDGAR URL retrieval functionality."""
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    sec_email = os.getenv("SEC_EDGAR_EMAIL")
    if not sec_email:
        print("WARNING: SEC_EDGAR_EMAIL not set in environment variables. SEC API access may fail.")
    
    # Test tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    filing_types = ["10-K", "10-Q"]
    
    success_count = 0
    total_tests = len(test_tickers) * len(filing_types)
    
    print("\n=== Testing SEC EDGAR URL Retrieval ===\n")
    
    for ticker in test_tickers:
        for filing_type in filing_types:
            print(f"\nTesting {ticker} {filing_type}:")
            try:
                url = _get_filing_url(ticker, filing_type)
                if url:
                    print(f"SUCCESS: Retrieved URL: {url}")
                    success_count += 1
                else:
                    print(f"FAILED: Could not retrieve URL for {ticker} {filing_type}")
            except Exception as e:
                print(f"ERROR: Exception occurred: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\nURL Retrieval Success Rate: {success_count}/{total_tests} ({success_count/total_tests*100:.1f}%)")
    
    return success_count, total_tests

def test_text_extraction():
    """Test text extraction from SEC EDGAR URLs."""
    # First get a valid URL
    print("\n=== Testing Text Extraction ===\n")
    
    # Try with a known ticker that should work
    ticker = "AAPL"
    filing_type = "10-K"
    
    try:
        url = _get_filing_url(ticker, filing_type)
        if not url:
            print(f"Could not get URL for {ticker} {filing_type}, skipping text extraction test")
            return 0, 1
        
        print(f"Testing text extraction from URL: {url}")
        text = _get_text_from_url(url)
        
        if text:
            print(f"SUCCESS: Extracted {len(text)} characters of text")
            # Print a small sample of the text
            print("\nSample text (first 500 characters):")
            print(text[:500] + "...\n")
            return 1, 1
        else:
            print("FAILED: Could not extract text from URL")
            return 0, 1
    except Exception as e:
        print(f"ERROR: Exception occurred during text extraction: {e}")
        import traceback
        traceback.print_exc()
        return 0, 1

def main():
    """Main test function."""
    print("Starting SEC EDGAR API tests...")
    
    # Test URL retrieval
    url_success, url_total = test_sec_edgar_url_retrieval()
    
    # Test text extraction
    text_success, text_total = test_text_extraction()
    
    # Calculate overall success
    total_success = url_success + text_success
    total_tests = url_total + text_total
    
    print("\n=== Test Summary ===")
    print(f"URL Retrieval: {url_success}/{url_total} tests passed")
    print(f"Text Extraction: {text_success}/{text_total} tests passed")
    print(f"Overall: {total_success}/{total_tests} tests passed ({total_success/total_tests*100:.1f}%)")
    
    # Return non-zero exit code if any tests failed
    if total_success < total_tests:
        print("\nSome tests failed. Please check the output above for details.")
        return 1
    else:
        print("\nAll tests passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
