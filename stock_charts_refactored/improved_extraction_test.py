"""
Test script with improved extraction patterns for HTML-formatted 10-K reports.
"""

import os
import re
import html

def clean_html(text):
    """Remove HTML tags and clean up the text."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode HTML entities
    text = html.unescape(text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_section_robust(text, start_pattern, end_pattern, max_chars=100000, clean=True):
    """Extract a section from text using regex patterns with improved HTML handling."""
    try:
        # Find the start of the section
        start_match = re.search(start_pattern, text, re.IGNORECASE | re.DOTALL)
        if not start_match:
            return None
        
        start_pos = start_match.end()
        
        # Limit the search window to avoid overreaching
        search_text = text[start_pos:start_pos + max_chars]
        
        # Find the end of the section
        end_match = re.search(end_pattern, search_text, re.IGNORECASE | re.DOTALL)
        if not end_match:
            # If no end pattern found, return up to max_chars
            section_text = search_text
        else:
            end_pos = end_match.start()
            section_text = search_text[:end_pos]
        
        # Clean the extracted text if requested
        if clean:
            section_text = clean_html(section_text)
        
        return section_text.strip()
    except Exception as e:
        print(f"Error extracting section: {e}")
        return None

def main():
    """Test improved extraction patterns for HTML-formatted 10-K reports."""
    # Check for existing 10-K file
    ticker = "AAPL"
    base_path = os.path.join("sec-edgar-filings", ticker, "10-K")
    
    if os.path.exists(base_path):
        filing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        if filing_dirs:
            filing_dirs.sort(reverse=True)
            latest_filing = filing_dirs[0]
            filing_path = os.path.join(base_path, latest_filing, "full-submission.txt")
            
            if os.path.exists(filing_path):
                print(f"Found existing 10-K file: {filing_path}")
                
                try:
                    # Read the file
                    with open(filing_path, "r", encoding="utf-8") as f:
                        report_text = f.read()
                    
                    print(f"Successfully read file. Length: {len(report_text)} characters")
                    
                    # Write results to file
                    with open("improved_extraction_results.txt", "w", encoding="utf-8") as out_file:
                        out_file.write(f"Improved 10-K Extraction Results for {ticker}\n")
                        out_file.write("=" * 50 + "\n\n")
                        
                        # Define improved extraction patterns based on HTML structure
                        business_patterns = [
                            # HTML formatted patterns
                            (r'<span[^>]*>Item\s+1\.?\s*(?:&#160;)*\s*Business</span>', 
                             r'<span[^>]*>Item\s+1A\.?\s*(?:&#160;)*\s*Risk\s+Factors</span>'),
                            (r'<span[^>]*>ITEM\s+1\.?\s*(?:&#160;)*\s*BUSINESS</span>', 
                             r'<span[^>]*>ITEM\s+1A\.?\s*(?:&#160;)*\s*RISK\s+FACTORS</span>'),
                            # Regular patterns
                            (r"Item\s+1\.\s*Business", r"Item\s+1A\.\s*Risk\s+Factors"),
                            (r"ITEM\s+1\.\s*BUSINESS", r"ITEM\s+1A\.\s*RISK\s+FACTORS"),
                            # Broader patterns
                            (r"BUSINESS", r"RISK\s+FACTORS"),
                        ]
                        
                        risk_patterns = [
                            # HTML formatted patterns
                            (r'<span[^>]*>Item\s+1A\.?\s*(?:&#160;)*\s*Risk\s+Factors</span>', 
                             r'<span[^>]*>Item\s+1B\.?\s*(?:&#160;)*\s*</span>'),
                            (r'<span[^>]*>ITEM\s+1A\.?\s*(?:&#160;)*\s*RISK\s+FACTORS</span>', 
                             r'<span[^>]*>ITEM\s+1B\.?\s*(?:&#160;)*\s*</span>'),
                            # If 1B doesn't exist, try Item 2
                            (r'<span[^>]*>Item\s+1A\.?\s*(?:&#160;)*\s*Risk\s+Factors</span>', 
                             r'<span[^>]*>Item\s+2\.?\s*(?:&#160;)*\s*</span>'),
                            # Regular patterns
                            (r"Item\s+1A\.\s*Risk\s+Factors", r"Item\s+1B\."),
                            (r"ITEM\s+1A\.\s*RISK\s+FACTORS", r"ITEM\s+1B\."),
                            # If 1B doesn't exist, try Item 2
                            (r"Item\s+1A\.\s*Risk\s+Factors", r"Item\s+2\."),
                            (r"ITEM\s+1A\.\s*RISK\s+FACTORS", r"ITEM\s+2\."),
                            # Very broad pattern
                            (r"RISK\s+FACTORS", r"UNRESOLVED\s+STAFF\s+COMMENTS"),
                        ]
                        
                        mda_patterns = [
                            # HTML formatted patterns
                            (r'<span[^>]*>Item\s+7\.?\s*(?:&#160;)*\s*Management(?:&#8217;|\')?s\s+Discussion\s+and\s+Analysis</span>', 
                             r'<span[^>]*>Item\s+7A\.?\s*(?:&#160;)*\s*</span>'),
                            (r'<span[^>]*>ITEM\s+7\.?\s*(?:&#160;)*\s*MANAGEMENT(?:&#8217;|\')?S\s+DISCUSSION\s+AND\s+ANALYSIS</span>', 
                             r'<span[^>]*>ITEM\s+7A\.?\s*(?:&#160;)*\s*</span>'),
                            # If 7A doesn't exist, try Item 8
                            (r'<span[^>]*>Item\s+7\.?\s*(?:&#160;)*\s*Management(?:&#8217;|\')?s\s+Discussion</span>', 
                             r'<span[^>]*>Item\s+8\.?\s*(?:&#160;)*\s*</span>'),
                            (r'<span[^>]*>ITEM\s+7\.?\s*(?:&#160;)*\s*MANAGEMENT(?:&#8217;|\')?S\s+DISCUSSION</span>', 
                             r'<span[^>]*>ITEM\s+8\.?\s*(?:&#160;)*\s*</span>'),
                            # Regular patterns
                            (r"Item\s+7\.\s*Management(?:&#8217;|\')?s\s+Discussion\s+and\s+Analysis", r"Item\s+7A\."),
                            (r"ITEM\s+7\.\s*MANAGEMENT(?:&#8217;|\')?S\s+DISCUSSION\s+AND\s+ANALYSIS", r"ITEM\s+7A\."),
                            # If 7A doesn't exist, try Item 8
                            (r"Item\s+7\.\s*Management(?:&#8217;|\')?s\s+Discussion", r"Item\s+8\."),
                            (r"ITEM\s+7\.\s*MANAGEMENT(?:&#8217;|\')?S\s+DISCUSSION", r"ITEM\s+8\."),
                            # Very broad patterns
                            (r"Management(?:&#8217;|\')?s\s+Discussion\s+and\s+Analysis\s+of\s+Financial\s+Condition", 
                             r"Quantitative\s+and\s+Qualitative\s+Disclosures"),
                            (r"MANAGEMENT(?:&#8217;|\')?S\s+DISCUSSION\s+AND\s+ANALYSIS\s+OF\s+FINANCIAL\s+CONDITION", 
                             r"QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES"),
                        ]
                        
                        # Test Business section extraction
                        out_file.write("BUSINESS SECTION EXTRACTION\n")
                        out_file.write("-" * 30 + "\n")
                        
                        business_text = None
                        for i, (start_pattern, end_pattern) in enumerate(business_patterns):
                            out_file.write(f"Trying pattern {i+1}: {start_pattern[:50]}...\n")
                            result = extract_section_robust(report_text, start_pattern, end_pattern)
                            success = result and len(result) > 100
                            out_file.write(f"  Result: {'Success' if success else 'Failed'}, Length: {len(result) if result else 0} characters\n")
                            if success:
                                business_text = result
                                out_file.write(f"  Preview: {result[:200]}...\n")
                                out_file.write(f"  Pattern used: {start_pattern[:50]}... to {end_pattern[:50]}...\n")
                                break
                        
                        # Test Risk Factors section extraction
                        out_file.write("\nRISK FACTORS SECTION EXTRACTION\n")
                        out_file.write("-" * 30 + "\n")
                        
                        risk_text = None
                        for i, (start_pattern, end_pattern) in enumerate(risk_patterns):
                            out_file.write(f"Trying pattern {i+1}: {start_pattern[:50]}...\n")
                            result = extract_section_robust(report_text, start_pattern, end_pattern)
                            success = result and len(result) > 100
                            out_file.write(f"  Result: {'Success' if success else 'Failed'}, Length: {len(result) if result else 0} characters\n")
                            if success:
                                risk_text = result
                                out_file.write(f"  Preview: {result[:200]}...\n")
                                out_file.write(f"  Pattern used: {start_pattern[:50]}... to {end_pattern[:50]}...\n")
                                break
                        
                        # Test MD&A section extraction
                        out_file.write("\nMD&A SECTION EXTRACTION\n")
                        out_file.write("-" * 30 + "\n")
                        
                        mda_text = None
                        for i, (start_pattern, end_pattern) in enumerate(mda_patterns):
                            out_file.write(f"Trying pattern {i+1}: {start_pattern[:50]}...\n")
                            result = extract_section_robust(report_text, start_pattern, end_pattern)
                            success = result and len(result) > 100
                            out_file.write(f"  Result: {'Success' if success else 'Failed'}, Length: {len(result) if result else 0} characters\n")
                            if success:
                                mda_text = result
                                out_file.write(f"  Preview: {result[:200]}...\n")
                                out_file.write(f"  Pattern used: {start_pattern[:50]}... to {end_pattern[:50]}...\n")
                                break
                        
                        # Summary
                        out_file.write("\nEXTRACTION SUMMARY\n")
                        out_file.write("-" * 30 + "\n")
                        out_file.write(f"Business section: {'Extracted' if business_text else 'Not found'}\n")
                        out_file.write(f"Risk Factors section: {'Extracted' if risk_text else 'Not found'}\n")
                        out_file.write(f"MD&A section: {'Extracted' if mda_text else 'Not found'}\n")
                    
                    print(f"Extraction results written to improved_extraction_results.txt")
                    
                except Exception as e:
                    print(f"Error: {str(e)}")
            else:
                print(f"Filing not found: {filing_path}")
    else:
        print(f"Base path not found: {base_path}")

if __name__ == "__main__":
    main()
