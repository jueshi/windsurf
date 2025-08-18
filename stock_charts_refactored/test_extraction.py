import os
import re
import sys
from dotenv import load_dotenv
import edgar_downloader

def extract_section_robust(text, start_pattern, end_pattern):
    """A more robust version of extract_section that handles SEC Edgar file formats better."""
    # Try to find the start pattern
    start_match = re.search(start_pattern, text, re.IGNORECASE | re.DOTALL)
    if not start_match:
        return None
        
    # Get the position right after the start pattern
    start_index = start_match.end()
    
    # Look for the end pattern, but limit the search to a reasonable chunk of text
    # This helps avoid matching end patterns from much later sections
    search_limit = min(len(text) - start_index, 100000)  # Limit to ~100K chars after start
    search_text = text[start_index:start_index + search_limit]
    
    end_match = re.search(end_pattern, search_text, re.IGNORECASE | re.DOTALL)
    if not end_match:
        # If no end pattern found, take a reasonable chunk of text (avoid taking the whole file)
        max_section_length = 50000  # Limit section to ~50K chars if no end marker found
        section_text = search_text[:min(len(search_text), max_section_length)]
    else:
        end_index = end_match.start()
        section_text = search_text[:end_index]
    
    # Clean up the extracted text
    # Remove HTML tags that might be present in the SEC filing
    section_text = re.sub(r'<[^>]+>', ' ', section_text)
    # Normalize whitespace
    section_text = re.sub(r'\\s+', ' ', section_text)
    # Remove any non-printable characters
    section_text = ''.join(char for char in section_text if char.isprintable() or char.isspace())
    
    return section_text.strip()

def test_extraction(ticker=None, file_path=None):
    """
    Test the extraction of 10-K sections using our robust extraction function.
    
    Args:
        ticker (str, optional): Stock ticker to download a 10-K for.
        file_path (str, optional): Direct path to an existing 10-K file.
    """
    if not file_path and ticker:
        # Download the 10-K if a ticker is provided
        load_dotenv()
        email_address = os.getenv("SEC_EDGAR_EMAIL")
        if not email_address:
            print("Error: SEC_EDGAR_EMAIL not found in environment variables.")
            return
            
        print(f"Downloading latest 10-K report for {ticker}...")
        file_path = edgar_downloader.download_latest_10k(ticker, email_address)
    
    if not file_path or not os.path.exists(file_path):
        # Try to find an existing 10-K file
        print("Looking for existing 10-K files...")
        sec_filings_dir = "sec-edgar-filings"
        if os.path.exists(sec_filings_dir):
            for ticker_dir in os.listdir(sec_filings_dir):
                ten_k_dir = os.path.join(sec_filings_dir, ticker_dir, "10-K")
                if os.path.exists(ten_k_dir):
                    filing_dirs = sorted(os.listdir(ten_k_dir))
                    if filing_dirs:
                        latest_filing = filing_dirs[-1]
                        file_path = os.path.join(ten_k_dir, latest_filing, "full-submission.txt")
                        print(f"Using existing 10-K file: {file_path}")
                        break
    
    if not file_path or not os.path.exists(file_path):
        print("Error: Could not find a 10-K file to analyze.")
        return
    
    # Read the 10-K file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            report_text = f.read()
        
        print(f"Successfully read 10-K file: {file_path}")
        print(f"File size: {len(report_text)} characters")
        
        # Define patterns to test
        business_patterns = [
            (r"Item\s+1\.\s+Business", r"Item\s+1A\."),
            (r"ITEM\s+1\.\s+BUSINESS", r"ITEM\s+1A\."),
            (r"Item\s+1\s+Business", r"Item\s+1A"),
            (r"ITEM\s+1\s+BUSINESS", r"ITEM\s+1A"),
            (r"BUSINESS", r"RISK\s+FACTORS")
        ]
        
        risk_patterns = [
            (r"Item\s+1A\.\s+Risk Factors", r"Item\s+1B\."),
            (r"ITEM\s+1A\.\s+RISK FACTORS", r"ITEM\s+1B\."),
            (r"Item\s+1A\s+Risk Factors", r"Item\s+1B"),
            (r"ITEM\s+1A\s+RISK FACTORS", r"ITEM\s+1B"),
            (r"RISK\s+FACTORS", r"UNRESOLVED STAFF COMMENTS")
        ]
        
        mda_patterns = [
            (r"Item\s+7\.\s+Management's Discussion and Analysis", r"Item\s+7A\."),
            (r"ITEM\s+7\.\s+MANAGEMENT'S DISCUSSION AND ANALYSIS", r"ITEM\s+7A\."),
            (r"Item\s+7\s+Management's Discussion and Analysis", r"Item\s+7A"),
            (r"ITEM\s+7\s+MANAGEMENT'S DISCUSSION AND ANALYSIS", r"ITEM\s+7A"),
            (r"MANAGEMENT'S DISCUSSION AND ANALYSIS", r"QUANTITATIVE AND QUALITATIVE DISCLOSURES")
        ]
        
        # Test business section extraction
        print("\n--- Testing Business Section Extraction ---")
        for i, (start_pattern, end_pattern) in enumerate(business_patterns):
            print(f"\nTrying pattern {i+1}: {start_pattern}")
            section_text = extract_section_robust(report_text, start_pattern, end_pattern)
            if section_text:
                print(f"✓ Extracted {len(section_text)} characters")
                print(f"Preview: {section_text[:200]}...")
            else:
                print("✗ No match found")
        
        # Test risk factors section extraction
        print("\n--- Testing Risk Factors Section Extraction ---")
        for i, (start_pattern, end_pattern) in enumerate(risk_patterns):
            print(f"\nTrying pattern {i+1}: {start_pattern}")
            section_text = extract_section_robust(report_text, start_pattern, end_pattern)
            if section_text:
                print(f"✓ Extracted {len(section_text)} characters")
                print(f"Preview: {section_text[:200]}...")
            else:
                print("✗ No match found")
        
        # Test MD&A section extraction
        print("\n--- Testing MD&A Section Extraction ---")
        for i, (start_pattern, end_pattern) in enumerate(mda_patterns):
            print(f"\nTrying pattern {i+1}: {start_pattern}")
            section_text = extract_section_robust(report_text, start_pattern, end_pattern)
            if section_text:
                print(f"✓ Extracted {len(section_text)} characters")
                print(f"Preview: {section_text[:200]}...")
            else:
                print("✗ No match found")
                
    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    # Use command line argument for ticker or file path
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.exists(arg):
            test_extraction(file_path=arg)
        else:
            test_extraction(ticker=arg)
    else:
        test_extraction()  # Try to find an existing 10-K file
