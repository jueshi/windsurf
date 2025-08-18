"""
Script to specifically analyze the MD&A section structure in 10-K reports.
"""

import os
import re

def find_potential_mda_markers(text):
    """Find potential MD&A section markers in the text."""
    # Look for variations of "Management's Discussion and Analysis"
    potential_markers = []
    
    # Common MD&A markers with surrounding context
    patterns = [
        r'(?:Item|ITEM)\s+7[^<>{}\n]*(?:Management|MANAGEMENT)[^<>{}\n]*(?:Discussion|DISCUSSION)[^<>{}\n]*(?:Analysis|ANALYSIS)',
        r'(?:Management|MANAGEMENT)[^<>{}\n]*(?:Discussion|DISCUSSION)[^<>{}\n]*(?:Analysis|ANALYSIS)',
        r'(?:MD&A|MD & A|M D & A)',
    ]
    
    with open("mda_analysis_results.txt", "w", encoding="utf-8") as out_file:
        out_file.write("MD&A SECTION ANALYSIS\n")
        out_file.write("=" * 50 + "\n\n")
        
        for pattern in patterns:
            out_file.write(f"Searching for pattern: {pattern}\n")
            out_file.write("-" * 50 + "\n")
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            count = 0
            
            for match in matches:
                count += 1
                match_text = match.group(0)
                start_pos = match.start()
                end_pos = match.end()
                
                # Get surrounding context (100 chars before and after)
                context_start = max(0, start_pos - 100)
                context_end = min(len(text), end_pos + 100)
                context = text[context_start:context_end]
                
                # Replace newlines for better readability
                context = context.replace('\n', ' [NEWLINE] ')
                
                out_file.write(f"Match {count}:\n")
                out_file.write(f"Position: {start_pos}-{end_pos}\n")
                out_file.write(f"Matched text: {match_text}\n")
                out_file.write(f"Context: ...{context}...\n\n")
                
                potential_markers.append({
                    'pattern': pattern,
                    'match': match_text,
                    'position': (start_pos, end_pos),
                    'context': context
                })
            
            out_file.write(f"Total matches for this pattern: {count}\n\n")
        
        # Look for HTML structure around potential MD&A sections
        out_file.write("\nHTML STRUCTURE ANALYSIS\n")
        out_file.write("=" * 50 + "\n\n")
        
        # Find HTML tags that might contain section headers
        html_patterns = [
            r'<[^>]*>(?:Item|ITEM)\s+7[^<]*<[^>]*>',
            r'<[^>]*>(?:Management|MANAGEMENT)[^<]*(?:Discussion|DISCUSSION)[^<]*(?:Analysis|ANALYSIS)[^<]*<[^>]*>',
        ]
        
        for pattern in html_patterns:
            out_file.write(f"Searching for HTML pattern: {pattern}\n")
            out_file.write("-" * 50 + "\n")
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            count = 0
            
            for match in matches:
                count += 1
                match_text = match.group(0)
                start_pos = match.start()
                end_pos = match.end()
                
                # Get surrounding context (100 chars before and after)
                context_start = max(0, start_pos - 100)
                context_end = min(len(text), end_pos + 100)
                context = text[context_start:context_end]
                
                out_file.write(f"Match {count}:\n")
                out_file.write(f"Position: {start_pos}-{end_pos}\n")
                out_file.write(f"Matched text: {match_text}\n")
                out_file.write(f"Context: ...{context}...\n\n")
            
            out_file.write(f"Total matches for this pattern: {count}\n\n")
        
        # Search for section that might follow MD&A (Item 7A or Item 8)
        next_section_patterns = [
            r'<[^>]*>(?:Item|ITEM)\s+7A[^<]*<[^>]*>',
            r'<[^>]*>(?:Item|ITEM)\s+8[^<]*<[^>]*>',
            r'(?:Item|ITEM)\s+7A[^<>{}\n]*',
            r'(?:Item|ITEM)\s+8[^<>{}\n]*',
        ]
        
        out_file.write("\nNEXT SECTION MARKER ANALYSIS\n")
        out_file.write("=" * 50 + "\n\n")
        
        for pattern in next_section_patterns:
            out_file.write(f"Searching for next section pattern: {pattern}\n")
            out_file.write("-" * 50 + "\n")
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            count = 0
            
            for match in matches:
                count += 1
                match_text = match.group(0)
                start_pos = match.start()
                end_pos = match.end()
                
                # Get surrounding context (100 chars before and after)
                context_start = max(0, start_pos - 100)
                context_end = min(len(text), end_pos + 100)
                context = text[context_start:context_end]
                
                out_file.write(f"Match {count}:\n")
                out_file.write(f"Position: {start_pos}-{end_pos}\n")
                out_file.write(f"Matched text: {match_text}\n")
                out_file.write(f"Context: ...{context}...\n\n")
            
            out_file.write(f"Total matches for this pattern: {count}\n\n")
    
    return potential_markers

def main():
    """Analyze the MD&A section structure in 10-K reports."""
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
                    
                    # Analyze MD&A section structure
                    potential_markers = find_potential_mda_markers(report_text)
                    print(f"Analysis complete. Found {len(potential_markers)} potential MD&A markers.")
                    print("Results written to mda_analysis_results.txt")
                    
                except Exception as e:
                    print(f"Error: {str(e)}")
            else:
                print(f"Filing not found: {filing_path}")
    else:
        print(f"Base path not found: {base_path}")

if __name__ == "__main__":
    main()
