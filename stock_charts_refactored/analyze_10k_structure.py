"""
Script to analyze the structure of a 10-K file to find section markers.
"""

import os
import re

def main():
    """Analyze 10-K file structure to find section markers."""
    ticker = "AAPL"
    base_path = os.path.join("sec-edgar-filings", ticker, "10-K")
    
    if not os.path.exists(base_path):
        print(f"Base path not found: {base_path}")
        return
    
    filing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not filing_dirs:
        print("No filing directories found")
        return
    
    filing_dirs.sort(reverse=True)
    latest_filing = filing_dirs[0]
    filing_path = os.path.join(base_path, latest_filing, "full-submission.txt")
    
    if not os.path.exists(filing_path):
        print(f"Filing not found: {filing_path}")
        return
    
    print(f"Analyzing 10-K file: {filing_path}")
    
    try:
        with open(filing_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Write the analysis to a file
        with open("10k_structure_analysis.txt", "w", encoding="utf-8") as out_file:
            out_file.write(f"10-K Structure Analysis for {ticker}\n")
            out_file.write("=" * 50 + "\n\n")
            
            # Search for common section markers with context
            section_keywords = [
                "BUSINESS", 
                "RISK FACTOR", 
                "MANAGEMENT'S DISCUSSION",
                "MANAGEMENT DISCUSSION",
                "Item 1", 
                "Item 1A", 
                "Item 7",
                "ITEM 1",
                "ITEM 1A",
                "ITEM 7"
            ]
            
            for keyword in section_keywords:
                out_file.write(f"\nSearching for: {keyword}\n")
                out_file.write("-" * 40 + "\n")
                
                # Find all occurrences with context
                matches = list(re.finditer(keyword, content, re.IGNORECASE))
                out_file.write(f"Found {len(matches)} matches\n\n")
                
                # Show first 5 matches with context
                for i, match in enumerate(matches[:5]):
                    pos = match.start()
                    # Get context (100 chars before and after)
                    start_ctx = max(0, pos - 100)
                    end_ctx = min(len(content), pos + 100)
                    
                    # Extract the context and format for readability
                    context = content[start_ctx:end_ctx]
                    context = context.replace('\n', ' ').replace('\r', '')
                    
                    # Highlight the match
                    match_text = match.group(0)
                    highlight_context = context.replace(match_text, f"**{match_text}**")
                    
                    out_file.write(f"Match {i+1} at position {pos}:\n")
                    out_file.write(f"...{highlight_context}...\n\n")
            
            # Look for document structure
            out_file.write("\n\nDocument Structure Analysis\n")
            out_file.write("=" * 30 + "\n\n")
            
            # Find SGML tags
            sgml_tags = re.findall(r"<(DOCUMENT|TYPE|SEQUENCE|FILENAME|DESCRIPTION)>([^<]+)</\1>", content)
            if sgml_tags:
                out_file.write("SGML Document Tags:\n")
                for tag, value in sgml_tags[:20]:
                    out_file.write(f"  {tag}: {value}\n")
            
            # Find HTML structure if present
            html_tags = re.findall(r"<(html|body|div|h1|h2|h3|h4|table|tr|td)[^>]*>", content, re.IGNORECASE)
            if html_tags:
                tag_counts = {}
                for tag in html_tags:
                    tag = tag.lower()
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                
                out_file.write("\nHTML Structure:\n")
                for tag, count in tag_counts.items():
                    out_file.write(f"  {tag}: {count} occurrences\n")
            
            # Check for specific section formats
            out_file.write("\nSection Format Analysis:\n")
            
            # Check for numeric section markers (e.g., "ITEM 1.", "ITEM 1A.")
            numeric_markers = re.findall(r"(ITEM|Item)\s+(\d+[A-Z]?)\.?\s+([A-Z][A-Za-z\s']+)", content)
            if numeric_markers:
                out_file.write("Numeric Section Markers:\n")
                seen = set()
                for prefix, num, title in numeric_markers[:20]:
                    marker = f"{prefix} {num}. {title}"
                    if marker not in seen:
                        out_file.write(f"  {marker}\n")
                        seen.add(marker)
            
            # Check for all-caps section titles
            caps_titles = re.findall(r"([A-Z][A-Z\s']{10,50})", content)
            if caps_titles:
                out_file.write("\nAll-Caps Section Titles:\n")
                seen = set()
                for title in caps_titles[:20]:
                    title = title.strip()
                    if len(title) > 10 and title not in seen:
                        out_file.write(f"  {title}\n")
                        seen.add(title)
        
        print(f"Analysis complete. Results written to 10k_structure_analysis.txt")
        
    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    main()
