"""
Test script to run 10-K analysis with enhanced debug output.
"""

import os
import sys
from dotenv import load_dotenv
from sec_edgar_downloader import Downloader
from gemini_analyzer import analyze_10k_report

def main():
    """Main function to test 10-K analysis."""
    # Load environment variables
    load_dotenv()
    sec_edgar_email = os.getenv('SEC_EDGAR_EMAIL')
    
    if not sec_edgar_email:
        print("SEC_EDGAR_EMAIL environment variable not set")
        return
    
    ticker = "AAPL"  # Apple Inc.
    year = 2023
    
    # Check for existing 10-K file
    base_path = os.path.join("sec-edgar-filings", ticker, "10-K")
    if os.path.exists(base_path):
        filing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        if filing_dirs:
            filing_dirs.sort(reverse=True)
            latest_filing = filing_dirs[0]
            filing_path = os.path.join(base_path, latest_filing, "full-submission.txt")
            if os.path.exists(filing_path):
                print(f"Found existing 10-K file: {filing_path}")
                # Run analysis with enhanced debug output
                analyze_10k_report(filing_path)
                return
    
    # If no existing file, download a new one
    print(f"Downloading 10-K for {ticker} ({year})...")
    try:
        # Create downloader
        dl = Downloader(sec_edgar_email)
        
        # Download 10-K
        dl.get("10-K", ticker, after=f"{year}-01-01", before=f"{year+1}-01-01", download_details=True)
        
        # Find the downloaded file
        filing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        if not filing_dirs:
            print("No filing directories found")
            return
        
        filing_dirs.sort(reverse=True)
        latest_filing = filing_dirs[0]
        filing_path = os.path.join(base_path, latest_filing, "full-submission.txt")
        
        if not os.path.exists(filing_path):
            print(f"Filing not found: {filing_path}")
            return
        
        print(f"Successfully downloaded 10-K for {ticker} ({year}): {filing_path}")
        
        # Run analysis with enhanced debug output
        analyze_10k_report(filing_path)
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
