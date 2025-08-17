"""
Basic test for SEC EDGAR API access with improved caching and retry logic
"""
import os
import time
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# Try to import our SEC API modules
try:
    import sec_api_cache
except ImportError as e:
    print(f"Error importing sec_api_cache: {e}")
    sec_api_cache = None

def test_direct_sec_api():
    """Test direct SEC API access without our caching layer"""
    print("\n=== Testing Direct SEC API Access ===\n")
    
    # Load environment variables
    load_dotenv()
    
    # Get SEC email
    email = os.getenv("SEC_EDGAR_EMAIL")
    print(f"Using SEC email: {email}")
    
    # Basic test URL - company tickers JSON
    url = "https://www.sec.gov/files/company_tickers.json"
    
    # Set up headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Referer': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    }
    
    # Add email to headers
    if email:
        headers['From'] = email
    
    print(f"Requesting: {url}")
    
    try:
        start_time = time.time()
        response = requests.get(url, headers=headers, timeout=30)
        elapsed = time.time() - start_time
        print(f"Response status code: {response.status_code}")
        print(f"Time taken: {elapsed:.2f} seconds")
        
        if response.status_code == 200:
            # Successfully got the data
            data = response.json()
            count = len(data)
            print(f"Successfully retrieved data with {count} entries")
            
            # Try to find Apple as a test
            found = False
            for _, company in data.items():
                if company.get('ticker') == 'AAPL':
                    print(f"Found AAPL: CIK = {company.get('cik_str')}")
                    found = True
                    break
            
            if not found:
                print("Could not find AAPL in the data")
            
            return True
        else:
            print(f"Failed with status code: {response.status_code}")
            print(f"Response headers: {response.headers}")
            print(f"Response content: {response.content[:500]}")
            return False
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cached_company_tickers():
    """Test company tickers retrieval with caching"""
    print("\n=== Testing Cached Company Tickers ===\n")
    
    if not sec_api_cache:
        print("sec_api_cache module not available, skipping test")
        return False
    
    try:
        # First call - might hit the API
        print("First call to get_company_tickers():")
        start_time = time.time()
        tickers = sec_api_cache.get_company_tickers()
        elapsed = time.time() - start_time
        
        if tickers:
            count = len(tickers)
            print(f"Successfully retrieved {count} tickers")
            print(f"Time taken: {elapsed:.2f} seconds")
            
            # Second call - should use cache and be faster
            print("\nSecond call to get_company_tickers() (should use cache):")
            start_time = time.time()
            tickers = sec_api_cache.get_company_tickers()
            cached_elapsed = time.time() - start_time
            
            print(f"Successfully retrieved {len(tickers)} tickers")
            print(f"Time taken: {cached_elapsed:.2f} seconds")
            
            if cached_elapsed < elapsed:
                print(f"Cache improved performance by {(elapsed - cached_elapsed) / elapsed * 100:.1f}%")
            else:
                print("Warning: Cached call was not faster")
            
            return True
        else:
            print("Failed to retrieve company tickers")
            return False
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cached_cik_lookup():
    """Test CIK lookup with caching"""
    print("\n=== Testing Cached CIK Lookup ===\n")
    
    if not sec_api_cache:
        print("sec_api_cache module not available, skipping test")
        return False
    
    # Test with a few tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    results = {}
    
    for ticker in test_tickers:
        try:
            print(f"\nTesting {ticker}:")
            
            # First call - might hit the API
            print(f"First call for {ticker}:")
            start_time = time.time()
            cik = sec_api_cache.get_company_cik(ticker)
            elapsed = time.time() - start_time
            
            if cik:
                print(f"CIK: {cik}")
                print(f"Time taken: {elapsed:.2f} seconds")
                results[ticker] = cik
                
                # Second call - should use cache and be faster
                print(f"Second call for {ticker} (should use cache):")
                start_time = time.time()
                cached_cik = sec_api_cache.get_company_cik(ticker)
                cached_elapsed = time.time() - start_time
                
                print(f"CIK: {cached_cik}")
                print(f"Time taken: {cached_elapsed:.2f} seconds")
                
                if cached_elapsed < elapsed:
                    print(f"Cache improved performance by {(elapsed - cached_elapsed) / elapsed * 100:.1f}%")
                else:
                    print("Warning: Cached call was not faster")
                
                # Verify results match
                if cik != cached_cik:
                    print(f"ERROR: CIK mismatch: {cik} vs {cached_cik}")
            else:
                print(f"Failed to retrieve CIK for {ticker}")
        
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    return len(results) > 0

def check_cache_files():
    """Check if cache files exist and show their contents"""
    print("\n=== Checking Cache Files ===\n")
    
    cache_dir = Path("cache")
    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        return False
    
    print(f"Cache directory exists: {cache_dir}")
    
    # Check company tickers cache
    tickers_cache = cache_dir / "company_tickers.json"
    if tickers_cache.exists():
        print(f"Company tickers cache exists: {tickers_cache}")
        try:
            with open(tickers_cache, 'r') as f:
                data = json.load(f)
                print(f"Cache contains {len(data)} entries")
                return True
        except Exception as e:
            print(f"Error reading cache file: {e}")
    else:
        print(f"Company tickers cache not found: {tickers_cache}")
    
    return False

if __name__ == "__main__":
    print("Testing SEC EDGAR API access with improved caching...")
    
    # Run tests
    direct_result = test_direct_sec_api()
    cache_files = check_cache_files()
    cached_tickers = test_cached_company_tickers()
    cached_cik = test_cached_cik_lookup()
    
    # Print summary
    print("\n=== Test Summary ===\n")
    print(f"Direct SEC API access: {'PASSED' if direct_result else 'FAILED'}")
    print(f"Cache files check: {'PASSED' if cache_files else 'FAILED'}")
    print(f"Cached company tickers: {'PASSED' if cached_tickers else 'FAILED'}")
    print(f"Cached CIK lookup: {'PASSED' if cached_cik else 'FAILED'}")
    
    overall = all([direct_result, cached_tickers, cached_cik])
    print(f"\nOverall test result: {'PASSED' if overall else 'FAILED'}")
