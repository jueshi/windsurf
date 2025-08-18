"""
Simple test script to check 10-K file loading and extraction.
"""

import os
import re

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
        print(f"Error extracting section: {e}")
        return None

def main():
    """Main function to test file loading and extraction."""
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
                    
                    # Try a simple extraction
                    business_pattern = r"Item\s+1\.\s+Business"
                    end_pattern = r"Item\s+1A\.\s+Risk\s+Factors"
                    
                    business_section = extract_section_robust(report_text, business_pattern, end_pattern)
                    
                    if business_section:
                        print(f"Successfully extracted Business section. Length: {len(business_section)} characters")
                        print(f"Preview: {business_section[:100].replace(chr(10), ' ')}...")
                    else:
                        print("Failed to extract Business section")
                    
                except Exception as e:
                    print(f"Error: {str(e)}")
            else:
                print(f"Filing not found: {filing_path}")
    else:
        print(f"Base path not found: {base_path}")

if __name__ == "__main__":
    main()
