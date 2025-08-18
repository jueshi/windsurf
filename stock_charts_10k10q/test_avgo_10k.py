import os
import sys
from dotenv import load_dotenv
from sec_edgar_helper import download_filing

def main():
    """Test downloading and analyzing AVGO's 10-K filing"""
    # Load environment variables
    load_dotenv()
    
    # Download AVGO's 10-K filing
    ticker = "AVGO"
    filing_type = "10-K"
    
    print(f"Starting 10-K analysis for {ticker}...")
    success, file_path, content = download_filing(ticker, filing_type)
    
    if not success:
        print(f"Error: {content}")
        return 1
    
    print(f"Successfully downloaded {filing_type} for {ticker}")
    print(f"File path: {file_path}")
    print(f"Content length: {len(content)} characters")
    
    # Print a preview of the content
    preview_length = min(1000, len(content))
    print(f"\nPreview of the first {preview_length} characters:")
    print("-" * 80)
    print(content[:preview_length])
    print("-" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
