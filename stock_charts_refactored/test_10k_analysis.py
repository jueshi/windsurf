import os
import sys
import re
from dotenv import load_dotenv
import edgar_downloader
import gemini_analyzer

def check_10k_structure(file_path):
    """
    Check the structure of the 10-K file to see if it contains the expected sections.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        print(f"File size: {len(content)} characters")
        
        # Check for common 10-K section headers
        sections = [
            ("Item 1. Business", re.search(r"Item\s+1\.\s+Business", content, re.IGNORECASE | re.DOTALL)),
            ("Item 1A. Risk Factors", re.search(r"Item\s+1A\.\s+Risk Factors", content, re.IGNORECASE | re.DOTALL)),
            ("Item 7. Management's Discussion", re.search(r"Item\s+7\.\s+Management's Discussion and Analysis", content, re.IGNORECASE | re.DOTALL))
        ]
        
        print("\nSection headers found:")
        for section_name, match in sections:
            if match:
                print(f"✓ {section_name} found at position {match.start()}")
                # Print a snippet of text around the match
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                snippet = content[start:end]
                print(f"  Context: ...{snippet}...\n")
            else:
                print(f"✗ {section_name} NOT found")
                
        return content
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def test_10k_analysis(ticker):
    """
    Test the 10-K report analysis functionality.
    
    Args:
        ticker (str): The stock ticker to analyze.
    """
    print(f"Testing 10-K analysis for {ticker}")
    
    # Load environment variables
    load_dotenv()
    email_address = os.getenv("SEC_EDGAR_EMAIL")
    if not email_address:
        print("Error: SEC_EDGAR_EMAIL not found in environment variables.")
        return
    
    print(f"Downloading latest 10-K report for {ticker}...")
    file_path = edgar_downloader.download_latest_10k(ticker, email_address)
    
    if file_path:
        print(f"10-K report downloaded to: {file_path}")
        
        # Check the structure of the 10-K file
        print("\n--- Checking 10-K File Structure ---")
        content = check_10k_structure(file_path)
        if not content:
            return
            
        print("\n--- Running Gemini Analysis ---")
        analysis_result = gemini_analyzer.analyze_10k_report(file_path)
        print("\n--- Analysis Result ---\n")
        print(analysis_result)
    else:
        print(f"Could not download 10-K report for {ticker}.")


if __name__ == "__main__":
    # Use command line argument for ticker or default to AAPL
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    test_10k_analysis(ticker)
