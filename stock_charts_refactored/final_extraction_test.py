"""
Final test script with improved extraction patterns for HTML-formatted 10-K reports.
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

def extract_section_robust(text, start_patterns, end_patterns, max_chars=100000, clean=True):
    """Extract a section from text using multiple regex patterns with improved HTML handling."""
    # Try each start pattern
    for start_pattern in start_patterns:
        start_match = re.search(start_pattern, text, re.IGNORECASE | re.DOTALL)
        if not start_match:
            continue
        
        start_pos = start_match.end()
        
        # Limit the search window to avoid overreaching
        search_text = text[start_pos:start_pos + max_chars]
        
        # Try each end pattern
        for end_pattern in end_patterns:
            try:
                end_match = re.search(end_pattern, search_text, re.IGNORECASE | re.DOTALL)
                if not end_match:
                    continue
                
                end_pos = end_match.start()
                section_text = search_text[:end_pos]
                
                # Clean the extracted text if requested
                if clean:
                    section_text = clean_html(section_text)
                
                # Check if the extracted text is substantial enough
                if len(section_text) > 100:
                    return {
                        'text': section_text.strip(),
                        'start_pattern': start_pattern,
                        'end_pattern': end_pattern
                    }
            except Exception as e:
                print(f"Error with end pattern {end_pattern}: {e}")
    
    # If we've tried all patterns and none worked, return None
    return None

def main():
    """Test final improved extraction patterns for HTML-formatted 10-K reports."""
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
                    with open("final_extraction_results.txt", "w", encoding="utf-8") as out_file:
                        out_file.write(f"Final 10-K Extraction Results for {ticker}\n")
                        out_file.write("=" * 50 + "\n\n")
                        
                        # Define improved extraction patterns based on HTML structure
                        business_start_patterns = [
                            r'<span[^>]*>Item\s+1\.?\s*(?:&#160;)*\s*Business</span>',
                            r'<span[^>]*>ITEM\s+1\.?\s*(?:&#160;)*\s*BUSINESS</span>',
                            r"Item\s+1\.\s*Business",
                            r"ITEM\s+1\.\s*BUSINESS"
                        ]
                        
                        business_end_patterns = [
                            r'<span[^>]*>Item\s+1A\.?\s*(?:&#160;)*\s*Risk\s+Factors</span>',
                            r'<span[^>]*>ITEM\s+1A\.?\s*(?:&#160;)*\s*RISK\s+FACTORS</span>',
                            r"Item\s+1A\.\s*Risk\s+Factors",
                            r"ITEM\s+1A\.\s*RISK\s+FACTORS"
                        ]
                        
                        risk_start_patterns = [
                            r'<span[^>]*>Item\s+1A\.?\s*(?:&#160;)*\s*Risk\s+Factors</span>',
                            r'<span[^>]*>ITEM\s+1A\.?\s*(?:&#160;)*\s*RISK\s+FACTORS</span>',
                            r"Item\s+1A\.\s*Risk\s+Factors",
                            r"ITEM\s+1A\.\s*RISK\s+FACTORS"
                        ]
                        
                        risk_end_patterns = [
                            r'<span[^>]*>Item\s+1B\.?\s*(?:&#160;)*\s*</span>',
                            r'<span[^>]*>ITEM\s+1B\.?\s*(?:&#160;)*\s*</span>',
                            r'<span[^>]*>Item\s+2\.?\s*(?:&#160;)*\s*</span>',
                            r'<span[^>]*>ITEM\s+2\.?\s*(?:&#160;)*\s*</span>',
                            r"Item\s+1B\.",
                            r"ITEM\s+1B\.",
                            r"Item\s+2\.",
                            r"ITEM\s+2\."
                        ]
                        
                        # Based on our analysis, we need to target the specific HTML structure for MD&A
                        mda_start_patterns = [
                            # Exact match from analysis
                            r'<span style="color:#000000;font-family:\'Helvetica\',sans-serif;font-size:9pt;font-weight:700;line-height:120%">Item 7\.&#160;&#160;&#160;&#160;Management&#8217;s Discussion and Analysis',
                            # More generic patterns based on analysis
                            r'<span[^>]*>Item 7\.&#160;&#160;&#160;&#160;Management&#8217;s Discussion and Analysis',
                            r'<span[^>]*>Item\s+7\.?\s*(?:&#160;)*\s*Management(?:&#8217;|\')?s\s+Discussion\s+and\s+Analysis',
                            r'<span[^>]*>ITEM\s+7\.?\s*(?:&#160;)*\s*MANAGEMENT(?:&#8217;|\')?S\s+DISCUSSION\s+AND\s+ANALYSIS',
                            # Anchor to position 350717 from analysis
                            r'Item 7\.&#160;&#160;&#160;&#160;Management&#8217;s Discussion and Analysis',
                            # Regular patterns as fallback
                            r"Item\s+7\.\s*Management(?:&#8217;|\')?s\s+Discussion\s+and\s+Analysis",
                            r"ITEM\s+7\.\s*MANAGEMENT(?:&#8217;|\')?S\s+DISCUSSION\s+AND\s+ANALYSIS"
                        ]
                        
                        mda_end_patterns = [
                            # Exact match from analysis
                            r'<span style="color:#000000;font-family:\'Helvetica\',sans-serif;font-size:9pt;font-weight:700;line-height:120%">Item 7A\.&#160;&#160;&#160;&#160;Quantitative and Qualitative Disclosures About Market Risk</span>',
                            # More generic patterns based on analysis
                            r'<span[^>]*>Item 7A\.&#160;&#160;&#160;&#160;Quantitative and Qualitative Disclosures About Market Risk</span>',
                            r'<span[^>]*>Item\s+7A\.?\s*(?:&#160;)*\s*Quantitative\s+and\s+Qualitative\s+Disclosures\s+About\s+Market\s+Risk</span>',
                            r'<span[^>]*>ITEM\s+7A\.?\s*(?:&#160;)*\s*QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES\s+ABOUT\s+MARKET\s+RISK</span>',
                            # If 7A doesn't exist, try Item 8
                            r'<span[^>]*>Item\s+8\.?\s*(?:&#160;)*\s*Financial\s+Statements\s+and\s+Supplementary\s+Data</span>',
                            r'<span[^>]*>ITEM\s+8\.?\s*(?:&#160;)*\s*FINANCIAL\s+STATEMENTS\s+AND\s+SUPPLEMENTARY\s+DATA</span>',
                            # Regular patterns as fallback
                            r"Item\s+7A\.\s*Quantitative",
                            r"ITEM\s+7A\.\s*QUANTITATIVE",
                            r"Item\s+8\.\s*Financial",
                            r"ITEM\s+8\.\s*FINANCIAL"
                        ]
                        
                        # Test Business section extraction
                        out_file.write("BUSINESS SECTION EXTRACTION\n")
                        out_file.write("-" * 30 + "\n")
                        
                        business_result = extract_section_robust(report_text, business_start_patterns, business_end_patterns)
                        if business_result:
                            out_file.write(f"Success! Extracted {len(business_result['text'])} characters\n")
                            out_file.write(f"Start pattern: {business_result['start_pattern'][:50]}...\n")
                            out_file.write(f"End pattern: {business_result['end_pattern'][:50]}...\n")
                            out_file.write(f"Preview: {business_result['text'][:200]}...\n\n")
                        else:
                            out_file.write("Failed to extract Business section\n\n")
                        
                        # Test Risk Factors section extraction
                        out_file.write("RISK FACTORS SECTION EXTRACTION\n")
                        out_file.write("-" * 30 + "\n")
                        
                        risk_result = extract_section_robust(report_text, risk_start_patterns, risk_end_patterns)
                        if risk_result:
                            out_file.write(f"Success! Extracted {len(risk_result['text'])} characters\n")
                            out_file.write(f"Start pattern: {risk_result['start_pattern'][:50]}...\n")
                            out_file.write(f"End pattern: {risk_result['end_pattern'][:50]}...\n")
                            out_file.write(f"Preview: {risk_result['text'][:200]}...\n\n")
                        else:
                            out_file.write("Failed to extract Risk Factors section\n\n")
                        
                        # Test MD&A section extraction
                        out_file.write("MD&A SECTION EXTRACTION\n")
                        out_file.write("-" * 30 + "\n")
                        
                        mda_result = extract_section_robust(report_text, mda_start_patterns, mda_end_patterns)
                        if mda_result:
                            out_file.write(f"Success! Extracted {len(mda_result['text'])} characters\n")
                            out_file.write(f"Start pattern: {mda_result['start_pattern'][:50]}...\n")
                            out_file.write(f"End pattern: {mda_result['end_pattern'][:50]}...\n")
                            out_file.write(f"Preview: {mda_result['text'][:200]}...\n\n")
                        else:
                            out_file.write("Failed to extract MD&A section\n\n")
                        
                        # Try direct position extraction for MD&A as a last resort
                        if not mda_result:
                            out_file.write("ATTEMPTING DIRECT POSITION EXTRACTION FOR MD&A\n")
                            out_file.write("-" * 30 + "\n")
                            
                            # From our analysis, we know MD&A starts around position 350717
                            # and Item 7A starts around position 464869
                            try:
                                mda_text = report_text[350717:464869]
                                mda_text = clean_html(mda_text)
                                if len(mda_text) > 100:
                                    out_file.write(f"Success! Extracted {len(mda_text)} characters using direct position\n")
                                    out_file.write(f"Preview: {mda_text[:200]}...\n\n")
                                else:
                                    out_file.write("Direct position extraction failed - text too short\n\n")
                            except Exception as e:
                                out_file.write(f"Direct position extraction error: {e}\n\n")
                        
                        # Summary
                        out_file.write("EXTRACTION SUMMARY\n")
                        out_file.write("-" * 30 + "\n")
                        out_file.write(f"Business section: {'Extracted' if business_result else 'Not found'}\n")
                        out_file.write(f"Risk Factors section: {'Extracted' if risk_result else 'Not found'}\n")
                        out_file.write(f"MD&A section: {'Extracted' if mda_result else 'Not found'}\n")
                    
                    print(f"Extraction results written to final_extraction_results.txt")
                    
                except Exception as e:
                    print(f"Error: {str(e)}")
            else:
                print(f"Filing not found: {filing_path}")
    else:
        print(f"Base path not found: {base_path}")

if __name__ == "__main__":
    main()
