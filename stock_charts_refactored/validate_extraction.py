"""
Validation script for 10-K section extraction.
This script tests the improved extraction patterns for Business, Risk Factors, and MD&A sections.
"""

import os
import re
import html
import sys
import logging
from pathlib import Path

# Implement the extract_section_robust function directly
def extract_section_robust(text, start_patterns, end_patterns):
    """A robust version of extract_section that handles SEC Edgar file formats better.
    
    Args:
        text (str): The text to extract from
        start_patterns (list): List of regex patterns to try for section start
        end_patterns (list): List of regex patterns to try for section end
        
    Returns:
        str: The extracted section text or None if not found
    """
    # Try each start pattern
    for start_pattern in start_patterns:
        start_match = re.search(start_pattern, text, re.IGNORECASE | re.DOTALL)
        if not start_match:
            continue
        
        # Get the position right after the start pattern
        start_index = start_match.end()
        
        # Look for the end pattern, but limit the search to a reasonable chunk of text
        # This helps avoid matching end patterns from much later sections
        search_limit = min(len(text) - start_index, 100000)  # Limit to ~100K chars after start
        search_text = text[start_index:start_index + search_limit]
        
        # Try each end pattern
        for end_pattern in end_patterns:
            try:
                end_match = re.search(end_pattern, search_text, re.IGNORECASE | re.DOTALL)
                if not end_match:
                    continue
                
                end_index = end_match.start()
                section_text = search_text[:end_index]
                
                # Clean up the extracted text
                # Remove HTML tags that might be present in the SEC filing
                section_text = re.sub(r'<[^>]+>', ' ', section_text)
                # Decode HTML entities
                section_text = html.unescape(section_text)
                # Normalize whitespace
                section_text = re.sub(r'\s+', ' ', section_text)
                # Remove any non-printable characters
                section_text = ''.join(char for char in section_text if char.isprintable() or char.isspace())
                
                # Check if the extracted text is substantial enough
                if len(section_text.strip()) > 100:
                    return section_text.strip()
            except Exception as e:
                print(f"Error with end pattern: {e}")
    
    # If we've tried all patterns and none worked, try direct position extraction for MD&A
    # This is a last resort based on our analysis of the 10-K structure
    if any("item 7" in p.lower() or "management" in p.lower() for p in start_patterns):
        try:
            # Search for Item 7 and Item 7A markers
            item7_match = re.search(r'<span[^>]*>Item 7\.&#160;&#160;&#160;&#160;Management&#8217;s Discussion', text)
            item7a_match = re.search(r'<span[^>]*>Item 7A\.&#160;&#160;&#160;&#160;Quantitative', text)
            
            if item7_match and item7a_match:
                mda_text = text[item7_match.start():item7a_match.start()]
                # Clean up the extracted text
                mda_text = re.sub(r'<[^>]+>', ' ', mda_text)
                mda_text = html.unescape(mda_text)
                mda_text = re.sub(r'\s+', ' ', mda_text)
                mda_text = ''.join(char for char in mda_text if char.isprintable() or char.isspace())
                
                if len(mda_text.strip()) > 100:
                    print("Used direct position extraction for MD&A section")
                    return mda_text.strip()
        except Exception as e:
            print(f"Error with direct position extraction: {e}")
    
    # If all else fails, return None
    return None

def load_10k_report(file_path):
    """Load a 10-K report file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        return None

def write_section_to_file(section_name, section_text, output_dir):
    """Write extracted section to a file for review."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, f"{section_name.replace(' ', '_')}.txt")
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(section_text)
        logging.info(f"Wrote {section_name} section to {output_path}")
    except Exception as e:
        logging.error(f"Error writing {section_name} section to file: {e}")

def setup_logging():
    """Set up logging to both console and file"""
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extraction_validation")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set up logging
    log_file = os.path.join(output_dir, "extraction_debug.log")
    
    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging to: {log_file}")
    return log_file

def main():
    # Set up logging
    log_file = setup_logging()
    
    # Define the test file path - use the AAPL 10-K file we found
    file_path = "c:\\Users\\juesh\\OneDrive\\Documents\\windsurf\\stock_charts_refactored\\sec-edgar-filings\\AAPL\\10-K\\0000320193-24-000123\\full-submission.txt"
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return
    
    logging.info(f"Using 10-K file: {file_path}")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extraction_validation")
    
    # Load the report
    logging.info(f"Loading 10-K report from {file_path}...")
    report_text = load_10k_report(file_path)
    if not report_text:
        return
    logging.info(f"Report loaded. Length: {len(report_text)} characters")
    
    # Define the extraction patterns (copied from gemini_analyzer.py)
    # Business section patterns
    business_start_patterns = [
        # HTML formatted patterns
        r"<span[^>]*>Item\s+1\.?\s*(?:&#160;)*\s*Business</span>",
        r"<span[^>]*>ITEM\s+1\.?\s*(?:&#160;)*\s*BUSINESS</span>",
        # Regular patterns
        r"Item\s+1\.\s*Business",
        r"ITEM\s+1\.\s*BUSINESS",
        r"Item\s+1\s+Business",
        r"ITEM\s+1\s+BUSINESS",
        # Common variations
        r"PART\s+I\s+Item\s+1\.\s+Business",
        r"PART\s+I\s+ITEM\s+1\.\s+BUSINESS",
        # Broader patterns
        r"Business",
        r"BUSINESS",
        r"Company Overview",
        r"COMPANY OVERVIEW"
    ]
    
    business_end_patterns = [
        # HTML formatted patterns
        r"<span[^>]*>Item\s+1A\.?\s*(?:&#160;)*\s*Risk\s+Factors</span>",
        r"<span[^>]*>ITEM\s+1A\.?\s*(?:&#160;)*\s*RISK\s+FACTORS</span>",
        # Regular patterns
        r"Item\s+1A\.\s*Risk\s+Factors",
        r"ITEM\s+1A\.\s*RISK\s+FACTORS",
        r"Item\s+1A\s+Risk\s+Factors",
        r"ITEM\s+1A\s+RISK\s+FACTORS",
        # Broader patterns
        r"Risk Factors",
        r"RISK\s+FACTORS"
    ]
    
    # Risk Factors section patterns
    risk_start_patterns = [
        # HTML formatted patterns
        r"<span[^>]*>Item\s+1A\.?\s*(?:&#160;)*\s*Risk\s+Factors</span>",
        r"<span[^>]*>ITEM\s+1A\.?\s*(?:&#160;)*\s*RISK\s+FACTORS</span>",
        # Regular patterns
        r"Item\s+1A\.\s*Risk\s+Factors",
        r"ITEM\s+1A\.\s*RISK\s+FACTORS",
        r"Item\s+1A\s+Risk\s+Factors",
        r"ITEM\s+1A\s+RISK\s+FACTORS",
        # Broader patterns
        r"Risk Factors",
        r"RISK\s+FACTORS"
    ]
    
    risk_end_patterns = [
        # HTML formatted patterns
        r"<span[^>]*>Item\s+1B\.?\s*(?:&#160;)*\s*</span>",
        r"<span[^>]*>ITEM\s+1B\.?\s*(?:&#160;)*\s*</span>",
        r"<span[^>]*>Item\s+2\.?\s*(?:&#160;)*\s*</span>",
        r"<span[^>]*>ITEM\s+2\.?\s*(?:&#160;)*\s*</span>",
        # Regular patterns
        r"Item\s+1B\.",
        r"ITEM\s+1B\.",
        r"Item\s+2\.",
        r"ITEM\s+2\.",
        # Broader patterns
        r"UNRESOLVED STAFF COMMENTS",
        r"Unresolved Staff Comments"
    ]
    
    # MD&A section patterns
    mda_start_patterns = [
        # Exact match from analysis
        r"<span style=\"color:#000000;font-family:'Helvetica',sans-serif;font-size:9pt;font-weight:700;line-height:120%\">Item 7\.&#160;&#160;&#160;&#160;Management&#8217;s Discussion and Analysis",
        # More generic patterns based on analysis
        r"<span[^>]*>Item 7\.&#160;&#160;&#160;&#160;Management&#8217;s Discussion and Analysis",
        r"<span[^>]*>Item\s+7\.?\s*(?:&#160;)*\s*Management(?:&#8217;|')?s\s+Discussion\s+and\s+Analysis",
        r"<span[^>]*>ITEM\s+7\.?\s*(?:&#160;)*\s*MANAGEMENT(?:&#8217;|')?S\s+DISCUSSION\s+AND\s+ANALYSIS",
        # Anchor to position from analysis
        r"Item 7\.&#160;&#160;&#160;&#160;Management&#8217;s Discussion and Analysis",
        # Regular patterns
        r"Item\s+7\.\s*Management(?:&#8217;|')?s\s+Discussion\s+and\s+Analysis",
        r"ITEM\s+7\.\s*MANAGEMENT(?:&#8217;|')?S\s+DISCUSSION\s+AND\s+ANALYSIS",
        r"Item\s+7\s+Management(?:&#8217;|')?s\s+Discussion\s+and\s+Analysis",
        r"ITEM\s+7\s+MANAGEMENT(?:&#8217;|')?S\s+DISCUSSION\s+AND\s+ANALYSIS",
        # Common variations
        r"Item\s+7\.\s*Management(?:&#8217;|')?s\s+Discussion",
        r"ITEM\s+7\.\s*MANAGEMENT(?:&#8217;|')?S\s+DISCUSSION",
        # Financial condition variations
        r"Management(?:&#8217;|')?s\s+Discussion\s+and\s+Analysis\s+of\s+Financial\s+Condition",
        r"MANAGEMENT(?:&#8217;|')?S\s+DISCUSSION\s+AND\s+ANALYSIS\s+OF\s+FINANCIAL\s+CONDITION"
    ]
    
    mda_end_patterns = [
        # Exact match from analysis
        r"<span style=\"color:#000000;font-family:'Helvetica',sans-serif;font-size:9pt;font-weight:700;line-height:120%\">Item 7A\.&#160;&#160;&#160;&#160;Quantitative and Qualitative Disclosures About Market Risk</span>",
        # More generic patterns based on analysis
        r"<span[^>]*>Item 7A\.&#160;&#160;&#160;&#160;Quantitative and Qualitative Disclosures About Market Risk</span>",
        r"<span[^>]*>Item\s+7A\.?\s*(?:&#160;)*\s*Quantitative\s+and\s+Qualitative\s+Disclosures\s+About\s+Market\s+Risk</span>",
        r"<span[^>]*>ITEM\s+7A\.?\s*(?:&#160;)*\s*QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES\s+ABOUT\s+MARKET\s+RISK</span>",
        # If 7A doesn't exist, try Item 8
        r"<span[^>]*>Item\s+8\.?\s*(?:&#160;)*\s*Financial\s+Statements\s+and\s+Supplementary\s+Data</span>",
        r"<span[^>]*>ITEM\s+8\.?\s*(?:&#160;)*\s*FINANCIAL\s+STATEMENTS\s+AND\s+SUPPLEMENTARY\s+DATA</span>",
        # Regular patterns
        r"Item\s+7A\.\s*Quantitative",
        r"ITEM\s+7A\.\s*QUANTITATIVE",
        r"Item\s+8\.\s*Financial",
        r"ITEM\s+8\.\s*FINANCIAL",
        # Broader patterns
        r"Quantitative\s+and\s+Qualitative\s+Disclosures",
        r"QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES",
        r"Financial\s+Statements",
        r"FINANCIAL\s+STATEMENTS"
    ]
    
    # Extract each section
    logging.info("\n--- VALIDATING BUSINESS SECTION EXTRACTION ---")
    business_text = extract_section_robust(report_text, business_start_patterns, business_end_patterns)
    logging.info(f"Business section extracted: {business_text is not None}")
    if business_text:
        logging.info(f"Business section length: {len(business_text)}")
        logging.info(f"Preview: {business_text[:150].replace(chr(10), ' ')}...")
        write_section_to_file("Business", business_text, output_dir)
    
    logging.info("\n--- VALIDATING RISK FACTORS SECTION EXTRACTION ---")
    risk_text = extract_section_robust(report_text, risk_start_patterns, risk_end_patterns)
    logging.info(f"Risk Factors section extracted: {risk_text is not None}")
    if risk_text:
        logging.info(f"Risk Factors section length: {len(risk_text)}")
        logging.info(f"Preview: {risk_text[:150].replace(chr(10), ' ')}...")
        write_section_to_file("Risk_Factors", risk_text, output_dir)
    
    logging.info("\n--- VALIDATING MD&A SECTION EXTRACTION ---")
    
    # Debug MD&A pattern matching
    logging.info("Searching for MD&A start patterns...")
    for i, pattern in enumerate(mda_start_patterns):
        match = re.search(pattern, report_text, re.IGNORECASE | re.DOTALL)
        if match:
            start_pos = match.start()
            end_pos = match.end()
            context_before = report_text[max(0, start_pos-50):start_pos]
            context_after = report_text[end_pos:end_pos+50]
            logging.info(f"Found MD&A start pattern #{i}: '{pattern[:30]}...' at position {start_pos}")
            logging.info(f"Context before: '{context_before}'")
            logging.info(f"Context after: '{context_after}'")
            
            # Try to find end pattern from this start position
            search_limit = min(len(report_text) - end_pos, 100000)
            search_text = report_text[end_pos:end_pos + search_limit]
            
            for j, end_pattern in enumerate(mda_end_patterns):
                end_match = re.search(end_pattern, search_text, re.IGNORECASE | re.DOTALL)
                if end_match:
                    end_start_pos = end_match.start()
                    end_end_pos = end_match.end()
                    logging.info(f"  Found MD&A end pattern #{j}: '{end_pattern[:30]}...' at relative position {end_start_pos}")
                    logging.info(f"  MD&A section would be {end_start_pos} characters long")
                    break
        else:
            logging.info(f"No match for MD&A start pattern #{i}: '{pattern[:30]}...'")
    
    # Try direct position extraction for MD&A
    item7_match = re.search(r'<span[^>]*>Item 7\.&#160;&#160;&#160;&#160;Management&#8217;s Discussion', report_text)
    item7a_match = re.search(r'<span[^>]*>Item 7A\.&#160;&#160;&#160;&#160;Quantitative', report_text)
    
    if item7_match and item7a_match:
        logging.info(f"Direct position extraction possible: Item 7 at {item7_match.start()}, Item 7A at {item7a_match.start()}")
        logging.info(f"Section would be {item7a_match.start() - item7_match.start()} characters long")
    else:
        if not item7_match:
            logging.info("Could not find Item 7 marker for direct position extraction")
        if not item7a_match:
            logging.info("Could not find Item 7A marker for direct position extraction")
    
    # Now try the actual extraction
    mda_text = extract_section_robust(report_text, mda_start_patterns, mda_end_patterns)
    logging.info(f"MD&A section extracted: {mda_text is not None}")
    if mda_text:
        logging.info(f"MD&A section length: {len(mda_text)}")
        logging.info(f"Preview: {mda_text[:150].replace(chr(10), ' ')}...")
        write_section_to_file("MD&A", mda_text, output_dir)
    
    # Summary of results
    logging.info("\n--- EXTRACTION VALIDATION SUMMARY ---")
    logging.info(f"Business section: {'✓ Extracted' if business_text else '✗ Failed'}")
    logging.info(f"Risk Factors section: {'✓ Extracted' if risk_text else '✗ Failed'}")
    logging.info(f"MD&A section: {'✓ Extracted' if mda_text else '✗ Failed'}")
    logging.info(f"\nExtracted sections saved to: {output_dir}")
    logging.info(f"Full validation log saved to: {log_file}")

if __name__ == "__main__":
    main()
