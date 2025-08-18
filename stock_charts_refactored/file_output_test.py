"""
Test script that writes extraction results to a file instead of printing to console.
"""

import os
import re
import sys
from dotenv import load_dotenv

def extract_section_robust(text, start_pattern, end_pattern, max_chars=100000):
    """Extract a section from text using regex patterns."""
    try:
        # Find the start of the section
        start_match = re.search(start_pattern, text, re.IGNORECASE)
        if not start_match:
            return None
        
        start_pos = start_match.end()
        
        # Limit the search window to avoid overreaching
        search_text = text[start_pos:start_pos + max_chars]
        
        # Find the end of the section
        end_match = re.search(end_pattern, search_text, re.IGNORECASE)
        if not end_match:
            # If no end pattern found, return up to max_chars
            return search_text
        
        end_pos = end_match.start()
        
        # Extract the section
        section_text = search_text[:end_pos].strip()
        return section_text
    except Exception as e:
        with open("extraction_error.log", "a") as f:
            f.write(f"Error extracting section: {e}\n")
        return None

def main():
    """Main function to test file loading and extraction."""
    # Open output file
    with open("extraction_results.txt", "w") as output_file:
        # Check for existing 10-K file
        ticker = "AAPL"
        base_path = os.path.join("sec-edgar-filings", ticker, "10-K")
        
        output_file.write(f"Testing 10-K extraction for {ticker}\n")
        output_file.write("-" * 50 + "\n\n")
        
        if os.path.exists(base_path):
            filing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            if filing_dirs:
                filing_dirs.sort(reverse=True)
                latest_filing = filing_dirs[0]
                filing_path = os.path.join(base_path, latest_filing, "full-submission.txt")
                
                if os.path.exists(filing_path):
                    output_file.write(f"Found existing 10-K file: {filing_path}\n")
                    
                    try:
                        # Read the file
                        with open(filing_path, "r", encoding="utf-8") as f:
                            report_text = f.read()
                        
                        output_file.write(f"Successfully read file. Length: {len(report_text)} characters\n\n")
                        
                        # Define extraction patterns
                        business_patterns = [
                            (r"Item\s+1\.\s+Business", r"Item\s+1A\.\s+Risk\s+Factors"),
                            (r"ITEM\s+1\.\s+BUSINESS", r"ITEM\s+1A\.\s+RISK\s+FACTORS"),
                            (r"Item\s+1\s+Business", r"Item\s+1A\s+Risk\s+Factors"),
                            (r"ITEM\s+1\s+BUSINESS", r"ITEM\s+1A\s+RISK\s+FACTORS"),
                            (r"BUSINESS", r"RISK\s+FACTORS"),
                            (r"<[^>]*>\s*Item\s+1\.?\s*Business\s*</[^>]*>", r"<[^>]*>\s*Item\s+1A\.?\s*Risk\s+Factors\s*</[^>]*>"),
                        ]
                        
                        risk_patterns = [
                            (r"Item\s+1A\.\s+Risk\s+Factors", r"Item\s+1B\."),
                            (r"ITEM\s+1A\.\s+RISK\s+FACTORS", r"ITEM\s+1B\."),
                            (r"Item\s+1A\s+Risk\s+Factors", r"Item\s+1B"),
                            (r"ITEM\s+1A\s+RISK\s+FACTORS", r"ITEM\s+1B"),
                            (r"Item\s+1A\.\s+Risk\s+Factors", r"Item\s+2\."),
                            (r"ITEM\s+1A\.\s+RISK\s+FACTORS", r"ITEM\s+2\."),
                        ]
                        
                        mda_patterns = [
                            (r"Item\s+7\.\s+Management's Discussion and Analysis", r"Item\s+7A\."),
                            (r"ITEM\s+7\.\s+MANAGEMENT'S DISCUSSION AND ANALYSIS", r"ITEM\s+7A\."),
                            (r"Item\s+7\s+Management's Discussion and Analysis", r"Item\s+7A"),
                            (r"ITEM\s+7\s+MANAGEMENT'S DISCUSSION AND ANALYSIS", r"ITEM\s+7A"),
                            (r"Item\s+7\.\s+Management's Discussion", r"Item\s+8\."),
                            (r"ITEM\s+7\.\s+MANAGEMENT'S DISCUSSION", r"ITEM\s+8\."),
                            (r"Item\s+7\s+Management's Discussion", r"Item\s+8"),
                            (r"ITEM\s+7\s+MANAGEMENT'S DISCUSSION", r"ITEM\s+8"),
                            (r"Management's Discussion and Analysis of Financial Condition", r"Financial Statements"),
                            (r"MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION", r"FINANCIAL STATEMENTS"),
                        ]
                        
                        # Test Business section extraction
                        output_file.write("BUSINESS SECTION EXTRACTION\n")
                        output_file.write("-" * 30 + "\n")
                        
                        for i, (start_pattern, end_pattern) in enumerate(business_patterns):
                            output_file.write(f"Trying pattern {i+1}: {start_pattern[:30]}...\n")
                            business_text = extract_section_robust(report_text, start_pattern, end_pattern)
                            success = business_text and len(business_text) > 100
                            output_file.write(f"  Result: {'Success' if success else 'Failed'}, Length: {len(business_text) if business_text else 0} characters\n")
                            if success:
                                output_file.write(f"  Preview: {business_text[:100].replace(chr(10), ' ')}...\n")
                                output_file.write(f"  Pattern used: {start_pattern} to {end_pattern}\n")
                                break
                        
                        # Test Risk Factors section extraction
                        output_file.write("\nRISK FACTORS SECTION EXTRACTION\n")
                        output_file.write("-" * 30 + "\n")
                        
                        for i, (start_pattern, end_pattern) in enumerate(risk_patterns):
                            output_file.write(f"Trying pattern {i+1}: {start_pattern[:30]}...\n")
                            risk_text = extract_section_robust(report_text, start_pattern, end_pattern)
                            success = risk_text and len(risk_text) > 100
                            output_file.write(f"  Result: {'Success' if success else 'Failed'}, Length: {len(risk_text) if risk_text else 0} characters\n")
                            if success:
                                output_file.write(f"  Preview: {risk_text[:100].replace(chr(10), ' ')}...\n")
                                output_file.write(f"  Pattern used: {start_pattern} to {end_pattern}\n")
                                break
                        
                        # Test MD&A section extraction
                        output_file.write("\nMD&A SECTION EXTRACTION\n")
                        output_file.write("-" * 30 + "\n")
                        
                        for i, (start_pattern, end_pattern) in enumerate(mda_patterns):
                            output_file.write(f"Trying pattern {i+1}: {start_pattern[:30]}...\n")
                            mda_text = extract_section_robust(report_text, start_pattern, end_pattern)
                            success = mda_text and len(mda_text) > 100
                            output_file.write(f"  Result: {'Success' if success else 'Failed'}, Length: {len(mda_text) if mda_text else 0} characters\n")
                            if success:
                                output_file.write(f"  Preview: {mda_text[:100].replace(chr(10), ' ')}...\n")
                                output_file.write(f"  Pattern used: {start_pattern} to {end_pattern}\n")
                                break
                        
                        # Summary
                        output_file.write("\nEXTRACTION SUMMARY\n")
                        output_file.write("-" * 30 + "\n")
                        output_file.write(f"Business section: {'Extracted' if business_text and len(business_text) > 100 else 'Not found'}\n")
                        output_file.write(f"Risk Factors section: {'Extracted' if risk_text and len(risk_text) > 100 else 'Not found'}\n")
                        output_file.write(f"MD&A section: {'Extracted' if mda_text and len(mda_text) > 100 else 'Not found'}\n")
                        
                    except Exception as e:
                        output_file.write(f"Error: {str(e)}\n")
                else:
                    output_file.write(f"Filing not found: {filing_path}\n")
        else:
            output_file.write(f"Base path not found: {base_path}\n")
    
    print(f"Extraction results written to extraction_results.txt")

if __name__ == "__main__":
    main()
