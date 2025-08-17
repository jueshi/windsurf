import os
import sys
from dotenv import load_dotenv
from gemini_analyzer import _download_sec_filing, analyze_10k_report

def main():
    """Test downloading and analyzing AVGO's 10-K filing"""
    # Load environment variables
    load_dotenv()
    
    # Test ticker
    ticker = "AVGO"
    
    print(f"=== Testing SEC EDGAR downloader with {ticker} ===")
    
    # Test direct download
    print("\n1. Testing direct download of 10-K filing:")
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
    else:
        print(f"FAILED: Could not download 10-K filing for {ticker}")
        print(f"Error: {content}")
    
    # Test the full analysis function (optional - comment out if you don't want to run the full analysis)
    print("\n2. Testing full 10-K analysis (this may take some time):")
    print("Skipping full analysis for now. Uncomment the code to run it.")
    
    # Uncomment the following lines to test the full analysis
    # analysis = analyze_10k_report(ticker)
    # if analysis:
    #     print(f"Analysis completed successfully. Length: {len(analysis)} characters")
    #     print("\nPreview of analysis:")
    #     print("-" * 80)
    #     print(analysis[:500])
    #     print("-" * 80)
    # else:
    #     print("Analysis failed.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
