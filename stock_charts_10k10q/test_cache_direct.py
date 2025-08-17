"""
Direct test of SEC API cache module
"""
import os
import time
import sys
import json
from pathlib import Path

# Import the sec_api_cache module directly
try:
    import sec_api_cache
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_company_tickers_cache():
    """Test the company tickers cache functionality"""
    print("\n=== Testing Company Tickers Cache ===")
    
    # First call - might hit the API
    print("\nFirst call to get_company_tickers():")
    start_time = time.time()
    try:
        tickers = sec_api_cache.get_company_tickers()
        elapsed = time.time() - start_time
        print(f"Retrieved {len(tickers)} tickers")
        print(f"Time taken: {elapsed:.2f} seconds")
        
        # Show a few sample tickers
        sample_tickers = list(tickers.items())[:5]
        print(f"Sample tickers: {sample_tickers}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Second call - should use cache
    print("\nSecond call to get_company_tickers() (should use cache):")
    start_time = time.time()
    try:
        tickers = sec_api_cache.get_company_tickers()
        elapsed = time.time() - start_time
        print(f"Retrieved {len(tickers)} tickers")
        print(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_company_cik_lookup():
    """Test the company CIK lookup with caching"""
    print("\n=== Testing Company CIK Lookup ===")
    
    # Test tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    for ticker in test_tickers:
        print(f"\nTesting CIK lookup for {ticker}:")
        
        # First call - might hit the API
        print(f"First call for {ticker}:")
        start_time = time.time()
        try:
            cik = sec_api_cache.get_company_cik(ticker)
            elapsed = time.time() - start_time
            print(f"CIK: {cik}")
            print(f"Time taken: {elapsed:.2f} seconds")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Second call - should use cache
        print(f"Second call for {ticker} (should use cache):")
        start_time = time.time()
        try:
            cik = sec_api_cache.get_company_cik(ticker)
            elapsed = time.time() - start_time
            print(f"CIK: {cik}")
            print(f"Time taken: {elapsed:.2f} seconds")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

def check_cache_files():
    """Check if cache files exist and show their contents"""
    print("\n=== Checking Cache Files ===")
    
    cache_dir = Path("cache")
    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        return
    
    print(f"Cache directory exists: {cache_dir}")
    
    # Check company tickers cache
    tickers_cache = cache_dir / "company_tickers.json"
    if tickers_cache.exists():
        print(f"Company tickers cache exists: {tickers_cache}")
        try:
            with open(tickers_cache, 'r') as f:
                data = json.load(f)
                print(f"Cache contains {len(data)} entries")
        except Exception as e:
            print(f"Error reading cache file: {e}")
    else:
        print(f"Company tickers cache not found: {tickers_cache}")
    
    # Check CIK cache files
    cik_cache_dir = cache_dir / "company_tickers"
    if cik_cache_dir.exists():
        print(f"CIK cache directory exists: {cik_cache_dir}")
        cik_files = list(cik_cache_dir.glob("*.json"))
        print(f"Found {len(cik_files)} CIK cache files")
        
        # Show a few sample files
        for file in cik_files[:3]:
            print(f"  - {file.name}")
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    print(f"    Content: {data}")
            except Exception as e:
                print(f"    Error reading file: {e}")
    else:
        print(f"CIK cache directory not found: {cik_cache_dir}")

def main():
    """Run all tests"""
    print("Starting direct SEC API cache tests...")
    
    # Check environment variables
    sec_email = os.getenv("SEC_EDGAR_EMAIL")
    print(f"SEC_EDGAR_EMAIL: {'Set' if sec_email else 'Not set'}")
    
    if not sec_email:
        print("WARNING: SEC_EDGAR_EMAIL environment variable not set.")
        print("Set this variable in your .env file for proper SEC API access.")
    
    # Run tests
    try:
        check_cache_files()
        test_company_tickers_cache()
        test_company_cik_lookup()
        print("\nAll tests completed.")
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
