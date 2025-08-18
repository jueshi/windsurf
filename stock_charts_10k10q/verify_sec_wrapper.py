"""
Verification script for SEC API wrapper functionality
Tests both mock and real data modes
"""

import sys
import sec_api_wrapper

def main():
    """Main function to test SEC API wrapper"""
    print("\nSEC API Wrapper Verification Script")
    print("==================================\n")
    
    # Test with mock data
    print("Testing with MOCK data:")
    sec_api_wrapper.use_mock_sec_api(True)
    mock_status = sec_api_wrapper.using_mock_api()
    print(f"Using mock API: {mock_status}")
    
    if not mock_status:
        print("ERROR: Failed to enable mock API mode")
        return
        
    # Test a ticker with mock data
    ticker = "AAPL"
    test_ticker(ticker, "mock")
    
    # Switch to real data (if requested)
    if len(sys.argv) > 1 and sys.argv[1].lower() == "real":
        print("\n\nTesting with REAL data:")
        sec_api_wrapper.use_mock_sec_api(False)
        real_status = not sec_api_wrapper.using_mock_api()
        print(f"Using real API: {real_status}")
        
        if not real_status:
            print("ERROR: Failed to enable real API mode")
            return
            
        # Test a ticker with real data
        test_ticker(ticker, "real")
    else:
        print("\n\nSkipping real API test (add 'real' argument to test with real API)")
    
    print("\nVerification complete!")

def test_ticker(ticker, mode):
    """Test SEC API wrapper with a specific ticker"""
    print(f"\nTesting {ticker} in {mode} mode:")
    
    api = sec_api_wrapper.sec_api
    
    # Step 1: Get company CIK
    print("\nStep 1: Getting company CIK...")
    cik = api.get_company_cik(ticker)
    print(f"CIK for {ticker}: {cik}")
    
    if not cik:
        print(f"ERROR: Could not find CIK for {ticker}")
        return
    
    # Step 2: Get latest 10-K filing info
    print("\nStep 2: Getting latest 10-K filing info...")
    filing_info = api.get_latest_filing_info(cik, "10-K")
    
    if filing_info:
        print(f"Found 10-K filing from {filing_info.get('filingDate', 'unknown date')}")
        print(f"Filing URL: {filing_info.get('detailUrl', 'unknown URL')}")
    else:
        print(f"ERROR: Could not find 10-K filing for {ticker}")
        return
    
    # Step 3: Download filing
    print("\nStep 3: Downloading filing...")
    html_content = api.download_filing(filing_info)
    
    if html_content:
        print(f"Successfully downloaded {len(html_content)} bytes")
        print(f"First 100 characters: {html_content[:100]}...")
    else:
        print(f"ERROR: Failed to download filing for {ticker}")
        return
    
    print(f"\nSuccessfully tested {ticker} in {mode} mode")

if __name__ == "__main__":
    main()
