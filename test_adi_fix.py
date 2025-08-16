import os
import sys
from dotenv import load_dotenv
from gemini_analyzer import _download_sec_filing

def main():
    """Test downloading ADI's 10-K filing using the fixed _download_sec_filing function"""
    # Test ticker
    ticker = "ADI"
    
    print(f"=== Testing SEC EDGAR downloader with {ticker} ===")
    
    # Test direct download
    print("\nTesting direct download of 10-K filing:")
    success, file_path, content, url = _download_sec_filing(ticker, "10-K")
    
    if success and content:
        print(f"SUCCESS: Downloaded 10-K filing for {ticker}")
        print(f"File path: {file_path}")
        print(f"Content length: {len(content)} characters")
        print(f"URL: {url}")
        
        # Print a preview of the content
        preview_length = min(500, len(content))
        print(f"\nPreview of the first {preview_length} characters:")
        print("-" * 80)
        print(content[:preview_length])
        print("-" * 80)
        return 0
    else:
        print(f"FAILED: Could not download 10-K filing for {ticker}")
        print(f"Error: {content}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
