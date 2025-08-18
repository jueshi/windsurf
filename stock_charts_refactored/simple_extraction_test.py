"""
Simple test script to validate 10-K section extraction with clear output.
"""

import os
import re
import html
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def extract_section_robust(text, start_pattern, end_pattern, max_chars=100000):
    """
    Extract a section from text using regex patterns, with improved robustness.
    Limits search to max_chars after start match to avoid overreaching.
    Cleans HTML tags and normalizes whitespace.
    """
    try:
        # Find the start of the section
        start_match = re.search(start_pattern, text, re.IGNORECASE)
        if not start_match:
            return ""
        
        start_pos = start_match.start()
        
        # Limit search window to avoid overreaching
        search_text = text[start_pos:start_pos + max_chars]
        
        # Find the end of the section within the limited window
        end_match = re.search(end_pattern, search_text, re.IGNORECASE)
        if not end_match:
            # If no end pattern found, take a reasonable chunk
            section_text = search_text[:50000]  # Take first 50K chars as fallback
        else:
            section_text = search_text[:end_match.start()]
        
        # Clean HTML tags
        section_text = re.sub(r'<[^>]+>', ' ', section_text)
        
        # Normalize whitespace
        section_text = re.sub(r'\\s+', ' ', section_text)
        
        # Unescape HTML entities
        section_text = html.unescape(section_text)
        
        # Remove non-printable characters
        section_text = ''.join(c for c in section_text if c.isprintable() or c in '\\n\\t')
        
        return section_text.strip()
    except Exception as e:
        logging.error(f"Error extracting section: {str(e)}")
        return ""

def test_extraction():
    """Test extraction on a sample 10-K file."""
    # Look for existing 10-K file
    ticker = "AAPL"  # Apple Inc.
    base_path = os.path.join("sec-edgar-filings", ticker, "10-K")
    
    if not os.path.exists(base_path):
        print(f"No 10-K files found for {ticker}")
        return
    
    # Find the most recent filing
    filing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not filing_dirs:
        print(f"No filing directories found for {ticker}")
        return
    
    # Sort by date
    filing_dirs.sort(reverse=True)
    latest_filing = filing_dirs[0]
    filing_path = os.path.join(base_path, latest_filing, "full-submission.txt")
    
    if not os.path.exists(filing_path):
        print(f"Filing not found: {filing_path}")
        return
    
    print(f"Testing extraction on: {filing_path}")
    
    # Read the file
    try:
        with open(filing_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        print(f"File size: {len(content)} characters")
        
        # Test Business section extraction
        print("\n--- TESTING BUSINESS SECTION EXTRACTION ---")
        business_patterns = [
            # Standard patterns
            (r"Item\s+1\.\s+Business", r"Item\s+1A\."),
            (r"ITEM\s+1\.\s+BUSINESS", r"ITEM\s+1A\."),
            # Common variations
            (r"PART\s+I\s+Item\s+1\.\s+Business", r"Item\s+1[Aa]\."),
            # HTML-formatted reports
            (r"<[^>]*>\s*Item\s+1\.\s*Business\s*</[^>]*>", r"<[^>]*>\s*Item\s+1A\."),
            # Broader patterns
            (r"Business", r"Risk Factors"),
            (r"BUSINESS", r"RISK\s+FACTORS")
        ]
        
        for i, (start_pattern, end_pattern) in enumerate(business_patterns):
            business_text = extract_section_robust(content, start_pattern, end_pattern)
            success = len(business_text) > 100
            print(f"Pattern {i+1}: {start_pattern[:30]}... - Success: {success}, Length: {len(business_text)}")
            if success:
                print(f"Preview: {business_text[:150]}...")
                break
        
        # Test MD&A section extraction
        print("\n--- TESTING MD&A SECTION EXTRACTION ---")
        mda_patterns = [
            # Standard patterns
            (r"Item\s+7\.\s+Management's Discussion and Analysis", r"Item\s+7A\."),
            (r"ITEM\s+7\.\s+MANAGEMENT'S DISCUSSION AND ANALYSIS", r"ITEM\s+7A\."),
            # Common variations
            (r"Item\s+7\.\s+Management's Discussion", r"Item\s+8\."),
            (r"ITEM\s+7\.\s+MANAGEMENT'S DISCUSSION", r"ITEM\s+8\."),
            # Financial condition variations
            (r"Management's Discussion and Analysis of Financial Condition", r"Financial Statements"),
            # HTML-formatted reports
            (r"<[^>]*>\s*Item\s+7\.\s*Management's Discussion\s*</[^>]*>", r"<[^>]*>\s*Item\s+8\."),
            # Broader patterns
            (r"MANAGEMENT'S DISCUSSION AND ANALYSIS", r"QUANTITATIVE AND QUALITATIVE DISCLOSURES"),
            (r"Management's Discussion and Analysis", r"Financial Statements")
        ]
        
        for i, (start_pattern, end_pattern) in enumerate(mda_patterns):
            mda_text = extract_section_robust(content, start_pattern, end_pattern)
            success = len(mda_text) > 100
            print(f"Pattern {i+1}: {start_pattern[:30]}... - Success: {success}, Length: {len(mda_text)}")
            if success:
                print(f"Preview: {mda_text[:150]}...")
                break
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_extraction()
