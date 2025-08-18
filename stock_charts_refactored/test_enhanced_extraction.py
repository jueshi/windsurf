"""
Test script to validate enhanced 10-K section extraction patterns.
This script will test the extraction of Business and MD&A sections from 10-K reports
using the improved extraction patterns.
"""

import os
import sys
import re
import logging
import html
from dotenv import load_dotenv
from sec_edgar_downloader import Downloader

# Copy of extract_section_robust function from gemini_analyzer.py
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
        section_text = re.sub(r'\s+', ' ', section_text)
        
        # Unescape HTML entities
        section_text = html.unescape(section_text)
        
        # Remove non-printable characters
        section_text = ''.join(c for c in section_text if c.isprintable() or c in '\n\t')
        
        return section_text.strip()
    except Exception as e:
        logging.error(f"Error extracting section: {str(e)}")
        return ""

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Load environment variables
load_dotenv()
sec_edgar_email = os.getenv('SEC_EDGAR_EMAIL')

def download_10k(ticker, year):
    """Download 10-K report for the specified ticker and year."""
    if not sec_edgar_email:
        logging.error("SEC_EDGAR_EMAIL environment variable not set")
        return None
    
    try:
        # Create downloader
        dl = Downloader(sec_edgar_email)
        
        # Download 10-K for the specified ticker and year
        dl.get("10-K", ticker, after=f"{year}-01-01", before=f"{year+1}-01-01", download_details=True)
        
        # Construct the path to the downloaded file
        base_path = os.path.join("sec-edgar-filings", ticker, "10-K")
        
        # Find the most recent filing directory
        if not os.path.exists(base_path):
            logging.error(f"Download directory not found: {base_path}")
            return None
        
        # Get the most recent filing
        filing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        if not filing_dirs:
            logging.error("No filing directories found")
            return None
        
        # Sort by date (assuming directory names are dates in format YYYY-MM-DD)
        filing_dirs.sort(reverse=True)
        latest_filing = filing_dirs[0]
        
        # Full path to the full-submission.txt file
        filing_path = os.path.join(base_path, latest_filing, "full-submission.txt")
        
        if not os.path.exists(filing_path):
            logging.error(f"Filing not found: {filing_path}")
            return None
        
        logging.info(f"Successfully downloaded 10-K for {ticker} ({year}): {filing_path}")
        return filing_path
    
    except Exception as e:
        logging.error(f"Error downloading 10-K: {str(e)}")
        return None

def test_extraction(file_path):
    """Test extraction of Business and MD&A sections from the 10-K file."""
    if not file_path or not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return
    
    try:
        # Read the 10-K file
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        logging.info(f"File size: {len(content)} characters")
        
        # Define section patterns from gemini_analyzer.py
        # Business section patterns
        business_patterns = [
            # Standard patterns
            (r"Item\s+1\.\s+Business", r"Item\s+1A\."),
            (r"ITEM\s+1\.\s+BUSINESS", r"ITEM\s+1A\."),
            (r"Item\s+1\s+Business", r"Item\s+1A"),
            (r"ITEM\s+1\s+BUSINESS", r"ITEM\s+1A"),
            
            # Common variations
            (r"Item\s+1\.\s+Business", r"Item\s+1[Aa]\."),
            (r"ITEM\s+1\.\s+BUSINESS", r"ITEM\s+1[Aa]\."),
            (r"PART\s+I\s+Item\s+1\.\s+Business", r"Item\s+1[Aa]\."),
            (r"PART\s+I\s+ITEM\s+1\.\s+BUSINESS", r"ITEM\s+1[Aa]\."),
            
            # HTML-formatted reports
            (r"<[^>]*>\s*Item\s+1\.\s*Business\s*</[^>]*>", r"<[^>]*>\s*Item\s+1A\."),
            (r"<[^>]*>\s*ITEM\s+1\.\s*BUSINESS\s*</[^>]*>", r"<[^>]*>\s*ITEM\s+1A\."),
            
            # Broader patterns
            (r"Business", r"Risk Factors"),
            (r"BUSINESS", r"RISK\s+FACTORS"),
            (r"Company Overview", r"Risk"),
            (r"COMPANY OVERVIEW", r"RISK")
        ]
        
        # MD&A section patterns
        mda_patterns = [
            # Standard patterns
            (r"Item\s+7\.\s+Management's Discussion and Analysis", r"Item\s+7A\."),
            (r"ITEM\s+7\.\s+MANAGEMENT'S DISCUSSION AND ANALYSIS", r"ITEM\s+7A\."),
            (r"Item\s+7\s+Management's Discussion and Analysis", r"Item\s+7A"),
            (r"ITEM\s+7\s+MANAGEMENT'S DISCUSSION AND ANALYSIS", r"ITEM\s+7A"),
            
            # Common variations with Item 8 as endpoint
            (r"Item\s+7\.\s+Management's Discussion", r"Item\s+8\."),
            (r"ITEM\s+7\.\s+MANAGEMENT'S DISCUSSION", r"ITEM\s+8\."),
            (r"Item\s+7\s+Management's Discussion", r"Item\s+8"),
            (r"ITEM\s+7\s+MANAGEMENT'S DISCUSSION", r"ITEM\s+8"),
            
            # Financial condition variations
            (r"Management's Discussion and Analysis of Financial Condition", r"Financial Statements"),
            (r"MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION", r"FINANCIAL STATEMENTS"),
            (r"Management's Discussion and Analysis of Financial Condition and Results of Operations", r"Financial Statements"),
            (r"MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS", r"FINANCIAL STATEMENTS"),
            
            # HTML-formatted reports
            (r"<[^>]*>\s*Item\s+7\.\s*Management's Discussion\s*</[^>]*>", r"<[^>]*>\s*Item\s+8\."),
            (r"<[^>]*>\s*ITEM\s+7\.\s*MANAGEMENT'S DISCUSSION\s*</[^>]*>", r"<[^>]*>\s*ITEM\s+8\."),
            
            # Part II variations
            (r"PART\s+II\s+Item\s+7\.\s+Management's Discussion", r"Item\s+8\."),
            (r"PART\s+II\s+ITEM\s+7\.\s+MANAGEMENT'S DISCUSSION", r"ITEM\s+8\."),
            
            # Broader patterns
            (r"MANAGEMENT'S DISCUSSION AND ANALYSIS", r"QUANTITATIVE AND QUALITATIVE DISCLOSURES"),
            (r"Management's Discussion and Analysis", r"Financial Statements"),
            (r"MANAGEMENT'S DISCUSSION", r"FINANCIAL STATEMENTS"),
            (r"MD&A", r"FINANCIAL STATEMENTS"),
            
            # Very broad patterns (last resort)
            (r"DISCUSSION AND ANALYSIS", r"FINANCIAL"),
            (r"Discussion and Analysis", r"Financial"),
            (r"MANAGEMENT DISCUSSION", r"FINANCIAL"),
            (r"Management Discussion", r"Financial")
        ]
        
        # Test Business section extraction
        logging.info("Testing Business section extraction:")
        for i, (start_pattern, end_pattern) in enumerate(business_patterns):
            business_text = extract_section_robust(content, start_pattern, end_pattern)
            success = len(business_text) > 100  # Arbitrary threshold for "success"
            status = "✓" if success else "✗"
            logging.info(f"{status} Pattern {i+1}: {start_pattern[:30]}... - Length: {len(business_text)} characters")
            if success:
                preview = business_text[:100].replace('\n', ' ').strip() + "..."
                logging.info(f"   Preview: {preview}")
                break  # Stop after finding a successful pattern
        
        # Test MD&A section extraction
        logging.info("\nTesting MD&A section extraction:")
        for i, (start_pattern, end_pattern) in enumerate(mda_patterns):
            mda_text = extract_section_robust(content, start_pattern, end_pattern)
            success = len(mda_text) > 100  # Arbitrary threshold for "success"
            status = "✓" if success else "✗"
            logging.info(f"{status} Pattern {i+1}: {start_pattern[:30]}... - Length: {len(mda_text)} characters")
            if success:
                preview = mda_text[:100].replace('\n', ' ').strip() + "..."
                logging.info(f"   Preview: {preview}")
                break  # Stop after finding a successful pattern
        
    except Exception as e:
        logging.error(f"Error testing extraction: {str(e)}")

def main():
    """Main function to download and test 10-K extraction."""
    # Check for existing 10-K file first
    ticker = "AAPL"  # Apple Inc.
    year = 2023
    
    # Look for existing 10-K file
    base_path = os.path.join("sec-edgar-filings", ticker, "10-K")
    if os.path.exists(base_path):
        filing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        if filing_dirs:
            filing_dirs.sort(reverse=True)
            latest_filing = filing_dirs[0]
            filing_path = os.path.join(base_path, latest_filing, "full-submission.txt")
            if os.path.exists(filing_path):
                logging.info(f"Found existing 10-K file: {filing_path}")
                test_extraction(filing_path)
                return
    
    # If no existing file, download a new one
    logging.info(f"Downloading 10-K for {ticker} ({year})...")
    file_path = download_10k(ticker, year)
    if file_path:
        test_extraction(file_path)
    else:
        logging.error("Failed to download 10-K file")

if __name__ == "__main__":
    main()
