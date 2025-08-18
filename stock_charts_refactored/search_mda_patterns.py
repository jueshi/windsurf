"""
Script to search for MD&A section patterns in 10-K file
"""

import os
import re
import sys

def load_10k_report(file_path):
    """Load a 10-K report file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def search_mda_patterns(report_text):
    """Search for MD&A section patterns in the report."""
    # Search for Item 7 and Item 7A markers with context
    print("Searching for MD&A section markers...")
    
    # Common patterns for MD&A section
    patterns = [
        r'Item\s+7\.?\s*Management(?:\'|&#8217;)s\s+Discussion',
        r'ITEM\s+7\.?\s*MANAGEMENT(?:\'|&#8217;)s\s+DISCUSSION',
        r'<span[^>]*>Item\s+7\.?\s*(?:&#160;)*\s*Management(?:\'|&#8217;)s\s+Discussion',
        r'<span[^>]*>ITEM\s+7\.?\s*(?:&#160;)*\s*MANAGEMENT(?:\'|&#8217;)s\s+DISCUSSION',
        r'<div[^>]*>Item\s+7\.?\s*(?:&#160;)*\s*Management(?:\'|&#8217;)s\s+Discussion',
        r'<div[^>]*>ITEM\s+7\.?\s*(?:&#160;)*\s*MANAGEMENT(?:\'|&#8217;)s\s+DISCUSSION',
        r'Item 7\.&#160;&#160;&#160;&#160;Management&#8217;s Discussion',
        r'ITEM 7\.&#160;&#160;&#160;&#160;MANAGEMENT&#8217;S DISCUSSION'
    ]
    
    # Search for each pattern and print context
    for i, pattern in enumerate(patterns):
        print(f"\nSearching for pattern {i+1}: {pattern}")
        matches = list(re.finditer(pattern, report_text, re.IGNORECASE | re.DOTALL))
        print(f"Found {len(matches)} matches")
        
        for j, match in enumerate(matches[:3]):  # Limit to first 3 matches to avoid overwhelming output
            start_pos = match.start()
            end_pos = match.end()
            
            # Get context around the match
            context_before = report_text[max(0, start_pos-100):start_pos]
            context_after = report_text[end_pos:min(len(report_text), end_pos+100)]
            
            print(f"\nMatch {j+1} at position {start_pos}:")
            print(f"Context before: '{context_before}'")
            print(f"Matched text: '{match.group(0)}'")
            print(f"Context after: '{context_after}'")
    
    # Now search for Item 7A markers (end of MD&A section)
    print("\n\nSearching for Item 7A markers (end of MD&A section)...")
    end_patterns = [
        r'Item\s+7A\.?\s*Quantitative\s+and\s+Qualitative\s+Disclosures',
        r'ITEM\s+7A\.?\s*QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES',
        r'<span[^>]*>Item\s+7A\.?\s*(?:&#160;)*\s*Quantitative\s+and\s+Qualitative\s+Disclosures',
        r'<span[^>]*>ITEM\s+7A\.?\s*(?:&#160;)*\s*QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES',
        r'<div[^>]*>Item\s+7A\.?\s*(?:&#160;)*\s*Quantitative\s+and\s+Qualitative\s+Disclosures',
        r'<div[^>]*>ITEM\s+7A\.?\s*(?:&#160;)*\s*QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES',
        r'Item 7A\.&#160;&#160;&#160;&#160;Quantitative and Qualitative Disclosures',
        r'ITEM 7A\.&#160;&#160;&#160;&#160;QUANTITATIVE AND QUALITATIVE DISCLOSURES'
    ]
    
    for i, pattern in enumerate(end_patterns):
        print(f"\nSearching for end pattern {i+1}: {pattern}")
        matches = list(re.finditer(pattern, report_text, re.IGNORECASE | re.DOTALL))
        print(f"Found {len(matches)} matches")
        
        for j, match in enumerate(matches[:3]):  # Limit to first 3 matches
            start_pos = match.start()
            end_pos = match.end()
            
            # Get context around the match
            context_before = report_text[max(0, start_pos-100):start_pos]
            context_after = report_text[end_pos:min(len(report_text), end_pos+100)]
            
            print(f"\nMatch {j+1} at position {start_pos}:")
            print(f"Context before: '{context_before}'")
            print(f"Matched text: '{match.group(0)}'")
            print(f"Context after: '{context_after}'")
    
    # Save a small sample of the report for examination
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extraction_validation")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save a 10K character sample from the middle of the file for examination
    middle_pos = len(report_text) // 2
    sample_start = max(0, middle_pos - 5000)
    sample_end = min(len(report_text), middle_pos + 5000)
    sample_text = report_text[sample_start:sample_end]
    
    sample_path = os.path.join(output_dir, "10k_sample.txt")
    with open(sample_path, 'w', encoding='utf-8') as file:
        file.write(sample_text)
    print(f"\nSaved 10K character sample to {sample_path}")

def main():
    # Define the test file path - use the AAPL 10-K file
    file_path = "c:\\Users\\juesh\\OneDrive\\Documents\\windsurf\\stock_charts_refactored\\sec-edgar-filings\\AAPL\\10-K\\0000320193-24-000123\\full-submission.txt"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print(f"Using 10-K file: {file_path}")
    
    # Load the report
    print(f"Loading 10-K report...")
    report_text = load_10k_report(file_path)
    if not report_text:
        return
    print(f"Report loaded. Length: {len(report_text)} characters")
    
    # Search for MD&A patterns
    search_mda_patterns(report_text)
    
    # Write results to a file for easier viewing
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extraction_validation")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, "mda_pattern_search_results.txt")
    print(f"\nTo capture the full output, run this command:")
    print(f"python search_mda_patterns.py > {output_path}")

if __name__ == "__main__":
    main()
