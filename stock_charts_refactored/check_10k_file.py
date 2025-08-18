import os
import re
import sys
import glob
from dotenv import load_dotenv
import edgar_downloader

def check_10k_file(ticker=None, file_path=None):
    """
    Check the structure of a 10-K file to diagnose section extraction issues.
    
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
        
    if not file_path:
        # Try to find an existing 10-K file
        print("Looking for existing 10-K files...")
        sec_filings_dir = "sec-edgar-filings"
        if os.path.exists(sec_filings_dir):
            # Find any 10-K directories
            ten_k_dirs = glob.glob(os.path.join(sec_filings_dir, "*", "10-K"))
            if ten_k_dirs:
                # Get the first ticker's latest 10-K
                ticker_dir = ten_k_dirs[0]
                ticker = os.path.basename(os.path.dirname(ticker_dir))
                print(f"Found 10-K directory for {ticker}")
                
                # Get the latest filing
                filing_dirs = sorted(os.listdir(ticker_dir))
                if filing_dirs:
                    latest_filing = filing_dirs[-1]
                    file_path = os.path.join(ticker_dir, latest_filing, "full-submission.txt")
                    print(f"Using existing 10-K file: {file_path}")
    
    if not file_path or not os.path.exists(file_path):
        print("Error: Could not find a 10-K file to analyze.")
        return
        
    # Analyze the file structure
    print(f"\nAnalyzing 10-K file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        file_size = len(content)
        print(f"File size: {file_size} characters")
        
        # Check for SGML tags that indicate the document structure
        sgml_tags = re.findall(r"<(DOCUMENT|TYPE|SEQUENCE|FILENAME|DESCRIPTION)>([^<]+)</\1>", content)
        if sgml_tags:
            print("\nSGML Document Tags:")
            for tag, value in sgml_tags[:10]:  # Show first 10 tags
                print(f"  {tag}: {value}")
            if len(sgml_tags) > 10:
                print(f"  ... and {len(sgml_tags) - 10} more tags")
                
        # Look for document separators
        doc_separators = re.findall(r"<DOCUMENT>", content)
        print(f"\nFound {len(doc_separators)} document separators")
        
        # Check for common 10-K section headers
        section_patterns = [
            ("Item 1. Business", r"Item\s+1\.[\s\n]+Business", 100),
            ("Item 1A. Risk Factors", r"Item\s+1A\.[\s\n]+Risk\s+Factors", 100),
            ("Item 7. MD&A", r"Item\s+7\.[\s\n]+Management'?s\s+Discussion\s+and\s+Analysis", 100)
        ]
        
        print("\nSearching for section headers:")
        for section_name, pattern, context_size in section_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE | re.DOTALL))
            if matches:
                print(f"✓ {section_name}: Found {len(matches)} matches")
                for i, match in enumerate(matches[:2]):  # Show first 2 matches
                    pos = match.start()
                    start = max(0, pos - context_size)
                    end = min(file_size, pos + context_size)
                    context = content[start:end].replace('\n', ' ')
                    print(f"  Match {i+1} at position {pos}:")
                    print(f"  Context: ...{context}...")
            else:
                print(f"✗ {section_name}: Not found")
                
        # Try alternative patterns
        print("\nTrying alternative patterns:")
        alt_patterns = [
            ("Business Section", r"BUSINESS", 50),
            ("Risk Factors", r"RISK\s+FACTORS", 50),
            ("MD&A", r"MANAGEMENT'?S\s+DISCUSSION", 50)
        ]
        
        for section_name, pattern, context_size in alt_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE | re.DOTALL))
            if matches:
                print(f"✓ {section_name}: Found {len(matches)} matches")
                for i, match in enumerate(matches[:2]):  # Show first 2 matches
                    pos = match.start()
                    start = max(0, pos - context_size)
                    end = min(file_size, pos + context_size)
                    context = content[start:end].replace('\n', ' ')
                    print(f"  Match {i+1} at position {pos}:")
                    print(f"  Context: ...{context}...")
            else:
                print(f"✗ {section_name}: Not found")
                
    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    # Use command line argument for ticker or file path
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.exists(arg):
            check_10k_file(file_path=arg)
        else:
            check_10k_file(ticker=arg)
    else:
        check_10k_file()  # Try to find an existing 10-K file
