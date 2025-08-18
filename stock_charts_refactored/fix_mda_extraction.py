"""
Script to fix MD&A section extraction from 10-K reports.
This script uses more targeted approaches to extract the MD&A section.
"""

import os
import re
import html
import sys
from pathlib import Path

def load_10k_report(file_path):
    """Load a 10-K report file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def write_to_file(content, file_path):
    """Write content to a file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Wrote content to {file_path}")
        return True
    except Exception as e:
        print(f"Error writing to file: {e}")
        return False

def clean_extracted_text(text):
    """Clean up extracted text by removing HTML tags and normalizing whitespace."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode HTML entities
    text = html.unescape(text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove any non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char.isspace())
    return text.strip()

def extract_mda_section_approach1(report_text):
    """Extract MD&A section using regex patterns."""
    print("\nApproach 1: Using regex patterns")
    
    # Define MD&A start patterns
    start_patterns = [
        r'Item\s+7\.?\s*Management(?:\'|&#8217;)s\s+Discussion',
        r'ITEM\s+7\.?\s*MANAGEMENT(?:\'|&#8217;)s\s+DISCUSSION',
        r'<span[^>]*>Item\s+7\.?\s*(?:&#160;)*\s*Management(?:\'|&#8217;)s\s+Discussion',
        r'<span[^>]*>ITEM\s+7\.?\s*(?:&#160;)*\s*MANAGEMENT(?:\'|&#8217;)s\s+DISCUSSION',
        r'Item 7\.&#160;&#160;&#160;&#160;Management&#8217;s Discussion',
        r'ITEM 7\.&#160;&#160;&#160;&#160;MANAGEMENT&#8217;S DISCUSSION'
    ]
    
    # Define MD&A end patterns
    end_patterns = [
        r'Item\s+7A\.?\s*Quantitative\s+and\s+Qualitative\s+Disclosures',
        r'ITEM\s+7A\.?\s*QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES',
        r'<span[^>]*>Item\s+7A\.?\s*(?:&#160;)*\s*Quantitative\s+and\s+Qualitative\s+Disclosures',
        r'<span[^>]*>ITEM\s+7A\.?\s*(?:&#160;)*\s*QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES',
        r'Item 7A\.&#160;&#160;&#160;&#160;Quantitative and Qualitative Disclosures',
        r'ITEM 7A\.&#160;&#160;&#160;&#160;QUANTITATIVE AND QUALITATIVE DISCLOSURES'
    ]
    
    # Try each start pattern
    for i, start_pattern in enumerate(start_patterns):
        start_match = re.search(start_pattern, report_text, re.IGNORECASE | re.DOTALL)
        if not start_match:
            print(f"  No match for start pattern #{i+1}: {start_pattern[:30]}...")
            continue
        
        print(f"  Found start pattern #{i+1}: {start_pattern[:30]}... at position {start_match.start()}")
        
        # Get the position right after the start pattern
        start_index = start_match.end()
        
        # Look for the end pattern, but limit the search to a reasonable chunk of text
        search_limit = min(len(report_text) - start_index, 200000)  # Increased to 200K chars
        search_text = report_text[start_index:start_index + search_limit]
        
        # Try each end pattern
        for j, end_pattern in enumerate(end_patterns):
            end_match = re.search(end_pattern, search_text, re.IGNORECASE | re.DOTALL)
            if not end_match:
                print(f"    No match for end pattern #{j+1}: {end_pattern[:30]}...")
                continue
            
            print(f"    Found end pattern #{j+1}: {end_pattern[:30]}... at relative position {end_match.start()}")
            
            end_index = end_match.start()
            section_text = search_text[:end_index]
            
            # Clean up the extracted text
            cleaned_text = clean_extracted_text(section_text)
            
            print(f"    Extracted MD&A section: {len(cleaned_text)} characters")
            return cleaned_text
    
    print("  No successful extraction with approach 1")
    return None

def extract_mda_section_approach2(report_text):
    """Extract MD&A section using direct position extraction."""
    print("\nApproach 2: Using direct position extraction")
    
    # Search for Item 7 and Item 7A markers
    item7_match = re.search(r'<span[^>]*>Item 7\.&#160;&#160;&#160;&#160;Management&#8217;s Discussion', report_text)
    item7a_match = re.search(r'<span[^>]*>Item 7A\.&#160;&#160;&#160;&#160;Quantitative', report_text)
    
    if not item7_match:
        print("  Could not find Item 7 marker")
        return None
    
    if not item7a_match:
        print("  Could not find Item 7A marker")
        return None
    
    print(f"  Found Item 7 marker at position {item7_match.start()}")
    print(f"  Found Item 7A marker at position {item7a_match.start()}")
    
    # Extract the section between Item 7 and Item 7A
    section_text = report_text[item7_match.start():item7a_match.start()]
    
    # Clean up the extracted text
    cleaned_text = clean_extracted_text(section_text)
    
    print(f"  Extracted MD&A section: {len(cleaned_text)} characters")
    return cleaned_text

def extract_mda_section_approach3(report_text):
    """Extract MD&A section using broader search and context analysis."""
    print("\nApproach 3: Using broader search and context analysis")
    
    # Look for Management's Discussion and Analysis in various formats
    patterns = [
        r'Management(?:\'|&#8217;)s\s+Discussion\s+and\s+Analysis\s+of\s+Financial\s+Condition',
        r'MANAGEMENT(?:\'|&#8217;)S\s+DISCUSSION\s+AND\s+ANALYSIS\s+OF\s+FINANCIAL\s+CONDITION',
        r'Item\s+7\.\s*Management(?:\'|&#8217;)s\s+Discussion',
        r'ITEM\s+7\.\s*MANAGEMENT(?:\'|&#8217;)S\s+DISCUSSION'
    ]
    
    # Try to find the start of the MD&A section
    start_index = None
    for i, pattern in enumerate(patterns):
        matches = list(re.finditer(pattern, report_text, re.IGNORECASE | re.DOTALL))
        if matches:
            # Use the first match as the start of the MD&A section
            start_match = matches[0]
            start_index = start_match.start()
            print(f"  Found MD&A start with pattern #{i+1}: {pattern[:30]}... at position {start_index}")
            break
    
    if start_index is None:
        print("  Could not find MD&A start")
        return None
    
    # Look for the end of the MD&A section (Item 7A or Item 8)
    end_patterns = [
        r'Item\s+7A\.\s*Quantitative',
        r'ITEM\s+7A\.\s*QUANTITATIVE',
        r'Item\s+8\.\s*Financial',
        r'ITEM\s+8\.\s*FINANCIAL'
    ]
    
    # Limit the search to a reasonable chunk of text after the start
    search_limit = min(len(report_text) - start_index, 300000)  # Increased to 300K chars
    search_text = report_text[start_index:start_index + search_limit]
    
    # Try to find the end of the MD&A section
    end_index = None
    for i, pattern in enumerate(end_patterns):
        end_match = re.search(pattern, search_text, re.IGNORECASE | re.DOTALL)
        if end_match:
            end_index = end_match.start()
            print(f"  Found MD&A end with pattern #{i+1}: {pattern[:30]}... at relative position {end_index}")
            break
    
    if end_index is None:
        print("  Could not find MD&A end")
        return None
    
    # Extract the section between start and end
    section_text = search_text[:end_index]
    
    # Clean up the extracted text
    cleaned_text = clean_extracted_text(section_text)
    
    print(f"  Extracted MD&A section: {len(cleaned_text)} characters")
    return cleaned_text

def extract_mda_section_approach4(report_text):
    """Extract MD&A section using a combination of approaches."""
    print("\nApproach 4: Using a combination of approaches")
    
    # First, try to find the Item 7 section using a very specific pattern
    item7_pattern = r'Item 7\.&#160;&#160;&#160;&#160;Management&#8217;s Discussion and Analysis'
    item7_match = re.search(item7_pattern, report_text)
    
    if not item7_match:
        print(f"  Could not find exact Item 7 pattern: {item7_pattern}")
        # Try a more general pattern
        item7_pattern = r'Item\s+7\.?\s*Management(?:\'|&#8217;)s\s+Discussion'
        item7_match = re.search(item7_pattern, report_text, re.IGNORECASE)
    
    if not item7_match:
        print("  Could not find Item 7 marker with any pattern")
        return None
    
    print(f"  Found Item 7 marker at position {item7_match.start()}")
    
    # Now look for Item 7A or Item 8 (end of MD&A section)
    end_patterns = [
        r'Item 7A\.&#160;&#160;&#160;&#160;Quantitative and Qualitative Disclosures',
        r'Item\s+7A\.?\s*Quantitative',
        r'Item\s+8\.?\s*Financial'
    ]
    
    # Start searching from the Item 7 position
    start_index = item7_match.start()
    
    # Limit the search to a reasonable chunk of text after the start
    search_limit = min(len(report_text) - start_index, 300000)  # 300K chars
    search_text = report_text[start_index:start_index + search_limit]
    
    # Try to find the end of the MD&A section
    end_index = None
    for i, pattern in enumerate(end_patterns):
        end_match = re.search(pattern, search_text, re.IGNORECASE)
        if end_match:
            end_index = end_match.start()
            print(f"  Found MD&A end with pattern #{i+1}: {pattern[:30]}... at relative position {end_index}")
            break
    
    if end_index is None:
        print("  Could not find MD&A end")
        return None
    
    # Extract the section between start and end
    section_text = search_text[:end_index]
    
    # Clean up the extracted text
    cleaned_text = clean_extracted_text(section_text)
    
    print(f"  Extracted MD&A section: {len(cleaned_text)} characters")
    return cleaned_text

def main():
    # Define the test file path - use the AAPL 10-K file
    file_path = "c:\\Users\\juesh\\OneDrive\\Documents\\windsurf\\stock_charts_refactored\\sec-edgar-filings\\AAPL\\10-K\\0000320193-24-000123\\full-submission.txt"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print(f"Using 10-K file: {file_path}")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extraction_validation")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the report
    print(f"Loading 10-K report...")
    report_text = load_10k_report(file_path)
    if not report_text:
        return
    print(f"Report loaded. Length: {len(report_text)} characters")
    
    # Try different approaches to extract the MD&A section
    mda_text1 = extract_mda_section_approach1(report_text)
    mda_text2 = extract_mda_section_approach2(report_text)
    mda_text3 = extract_mda_section_approach3(report_text)
    mda_text4 = extract_mda_section_approach4(report_text)
    
    # Write the extracted sections to files for comparison
    if mda_text1:
        write_to_file(mda_text1, os.path.join(output_dir, "mda_approach1.txt"))
    if mda_text2:
        write_to_file(mda_text2, os.path.join(output_dir, "mda_approach2.txt"))
    if mda_text3:
        write_to_file(mda_text3, os.path.join(output_dir, "mda_approach3.txt"))
    if mda_text4:
        write_to_file(mda_text4, os.path.join(output_dir, "mda_approach4.txt"))
    
    # Summary of results
    print("\n--- EXTRACTION SUMMARY ---")
    print(f"Approach 1: {'✓ Extracted' if mda_text1 else '✗ Failed'} {len(mda_text1) if mda_text1 else 0} characters")
    print(f"Approach 2: {'✓ Extracted' if mda_text2 else '✗ Failed'} {len(mda_text2) if mda_text2 else 0} characters")
    print(f"Approach 3: {'✓ Extracted' if mda_text3 else '✗ Failed'} {len(mda_text3) if mda_text3 else 0} characters")
    print(f"Approach 4: {'✓ Extracted' if mda_text4 else '✗ Failed'} {len(mda_text4) if mda_text4 else 0} characters")
    
    # Determine the best approach
    best_approach = None
    best_length = 0
    
    if mda_text1 and len(mda_text1) > best_length:
        best_approach = 1
        best_length = len(mda_text1)
    if mda_text2 and len(mda_text2) > best_length:
        best_approach = 2
        best_length = len(mda_text2)
    if mda_text3 and len(mda_text3) > best_length:
        best_approach = 3
        best_length = len(mda_text3)
    if mda_text4 and len(mda_text4) > best_length:
        best_approach = 4
        best_length = len(mda_text4)
    
    if best_approach:
        print(f"\nBest approach: {best_approach} with {best_length} characters")
        # Copy the best result to MD&A.txt
        best_text = locals()[f"mda_text{best_approach}"]
        write_to_file(best_text, os.path.join(output_dir, "MD&A.txt"))
        print(f"Copied best result to MD&A.txt")
    else:
        print("\nNo successful extraction")

if __name__ == "__main__":
    main()
