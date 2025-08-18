import requests
import sys

def test_sec_api():
    """
    Simple test of SEC API connectivity
    """
    # Set up headers to avoid 403 errors
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    # Test URL - SEC company tickers JSON
    url = "https://www.sec.gov/files/company_tickers.json"
    
    print(f"Testing SEC API connection to: {url}")
    print(f"Using headers: {headers}")
    
    try:
        # Make the request
        response = requests.get(url, headers=headers)
        
        # Print response info
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {response.headers}")
        
        if response.status_code == 200:
            # Success - print a small sample of the data
            data = response.json()
            print(f"Successfully retrieved data. Sample of first 3 entries:")
            for i in range(3):
                if str(i) in data:
                    print(f"  {data[str(i)]}")
            return 0
        else:
            # Error
            print(f"Error response: {response.text[:500]}")
            return 1
            
    except Exception as e:
        print(f"Exception occurred: {e}")
        return 1

if __name__ == "__main__":
    # Force stdout to flush after each print
    import functools
    print = functools.partial(print, flush=True)
    
    print("SEC API Test Script")
    print("==================")
    sys.exit(test_sec_api())
