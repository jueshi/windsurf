"""
Basic test script for SEC API integration
"""

import os
import sys
import traceback

def main():
    try:
        print("Starting SEC integration test")
        
        # Test SEC API wrapper
        print("\nTesting SEC API wrapper...")
        import sec_api_wrapper
        
        # Test mock mode
        print("Setting mock mode to True")
        sec_api_wrapper.use_mock_sec_api(True)
        print(f"Mock mode is: {sec_api_wrapper.using_mock_api()}")
        
        # Test with a ticker
        ticker = "AAPL"
        print(f"\nTesting with ticker: {ticker}")
        
        api = sec_api_wrapper.sec_api
        print("Getting company CIK...")
        cik = api.get_company_cik(ticker)
        print(f"CIK for {ticker}: {cik}")
        
        print("\nTest completed successfully")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
