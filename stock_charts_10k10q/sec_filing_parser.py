"""
SEC Filing Parser - Extract key sections from 10-K and 10-Q filings

Uses edgartools to parse SEC filings and extract only the relevant sections
for analysis, reducing the text size from millions of characters to a manageable
amount for LLM processing.

Key sections extracted:
- 10-K: Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A), Item 7A (Market Risk)
- 10-Q: Item 1 (Financial Statements), Item 2 (MD&A), Item 1A (Risk Factors)
"""

import os
import logging
from typing import Optional, Dict, Tuple
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maximum characters per section to prevent token overflow
MAX_SECTION_CHARS = 50000  # ~12,500 tokens per section
MAX_TOTAL_CHARS = 200000   # ~50,000 tokens total (safe for Gemini)


def _extract_sections_from_html(filing, form_type: str) -> Dict[str, str]:
    """
    Extract sections from filing HTML using regex patterns.
    Fallback method when structured extraction fails.
    """
    import re
    from bs4 import BeautifulSoup
    
    sections = {}
    
    try:
        # Get the HTML content
        html_content = filing.html()
        if not html_content:
            logger.warning("Could not get HTML content from filing")
            return sections
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script, style, and XBRL elements
        for element in soup(["script", "style", "ix:header", "ix:hidden"]):
            element.decompose()
        
        # Remove XBRL inline tags but keep their text content
        for ix_tag in soup.find_all(lambda tag: tag.name and tag.name.startswith('ix:')):
            ix_tag.unwrap()
        
        # Get text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up XBRL artifacts that might remain
        import re
        # Remove lines that look like XBRL data (short lines with colons, dates, numbers only)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Skip XBRL-like lines (very short, mostly numbers/dates, or namespace-like)
            if len(line) < 50:
                # Skip if it looks like XBRL data
                if re.match(r'^[\d\-:\.]+$', line):  # Just numbers, dates
                    continue
                if re.match(r'^[a-z]+:[A-Za-z]+', line):  # Namespace prefixes
                    continue
                if re.match(r'^(iso4217|xbrli|utr):', line):  # Common XBRL prefixes
                    continue
                if re.match(r'^\d{10}$', line):  # CIK numbers
                    continue
                if re.match(r'^(true|false)$', line.lower()):  # Boolean values
                    continue
                if re.match(r'^P\d+[YMD]$', line):  # Duration values like P1Y
                    continue
                if re.match(r'^http://', line):  # URLs
                    continue
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        logger.info(f"HTML text length: {len(text):,} chars")
        
        # Define section patterns for 10-K
        if form_type == "10-K":
            patterns = {
                "business": [
                    r"(?i)ITEM\s*1[.\s]*BUSINESS(.*?)(?=ITEM\s*1A|ITEM\s*2|$)",
                    r"(?i)PART\s*I.*?ITEM\s*1[.\s]*BUSINESS(.*?)(?=ITEM\s*1A|$)",
                ],
                "risk_factors": [
                    r"(?i)ITEM\s*1A[.\s]*RISK\s*FACTORS(.*?)(?=ITEM\s*1B|ITEM\s*2|$)",
                ],
                "mda": [
                    r"(?i)ITEM\s*7[.\s]*MANAGEMENT.?S\s*DISCUSSION(.*?)(?=ITEM\s*7A|ITEM\s*8|$)",
                    r"(?i)MD&A(.*?)(?=ITEM\s*7A|ITEM\s*8|$)",
                ],
                "market_risk": [
                    r"(?i)ITEM\s*7A[.\s]*QUANTITATIVE(.*?)(?=ITEM\s*8|$)",
                ],
            }
        else:  # 10-Q
            patterns = {
                "mda": [
                    r"(?i)ITEM\s*2[.\s]*MANAGEMENT.?S\s*DISCUSSION(.*?)(?=ITEM\s*3|ITEM\s*4|$)",
                ],
                "risk_factors": [
                    r"(?i)ITEM\s*1A[.\s]*RISK\s*FACTORS(.*?)(?=ITEM\s*2|ITEM\s*6|$)",
                ],
            }
        
        # Try to extract each section
        for section_name, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    if len(content) > 500:  # Only if we got meaningful content
                        max_chars = MAX_SECTION_CHARS if section_name != "market_risk" else MAX_SECTION_CHARS // 2
                        sections[section_name] = _truncate_text(content, max_chars, section_name)
                        logger.info(f"Extracted {section_name} via regex: {len(sections[section_name]):,} chars")
                        break
        
        # If we still don't have enough content, just truncate the whole thing
        total_chars = sum(len(s) for s in sections.values())
        if total_chars < 10000:
            logger.info("Regex extraction got minimal content, using truncated full text")
            sections["full_text"] = _truncate_text(text, MAX_TOTAL_CHARS, "Full Filing")
        
    except Exception as e:
        logger.error(f"HTML extraction failed: {e}")
    
    return sections


def _truncate_text(text: str, max_chars: int, section_name: str = "") -> str:
    """Truncate text to max_chars with a note about truncation"""
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    # Try to end at a sentence boundary
    last_period = truncated.rfind('.')
    if last_period > max_chars * 0.8:  # Only if we're not losing too much
        truncated = truncated[:last_period + 1]
    
    logger.info(f"Truncated {section_name} from {len(text):,} to {len(truncated):,} chars")
    return truncated + f"\n\n[... {section_name} truncated for length ...]"


def extract_10k_sections(ticker: str) -> Tuple[bool, str, Dict[str, str]]:
    """
    Extract key sections from the latest 10-K filing for a ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "JOBY")
        
    Returns:
        Tuple of (success, error_message, sections_dict)
        sections_dict contains: business, risk_factors, mda, market_risk
    """
    try:
        from edgar import Company, set_identity
        
        # Set identity for SEC API (required)
        email = os.getenv("SEC_EDGAR_EMAIL", "user@example.com")
        set_identity(f"StockAnalyzer {email}")
        
        logger.info(f"Fetching 10-K for {ticker} using edgartools...")
        
        # Get the company and its filings
        company = Company(ticker)
        filings = company.get_filings(form="10-K")
        
        if not filings or len(filings) == 0:
            return False, f"No 10-K filings found for {ticker}", {}
        
        # Get the latest 10-K
        latest_10k = filings.latest()
        if not latest_10k:
            return False, f"Could not retrieve latest 10-K for {ticker}", {}
        
        logger.info(f"Found 10-K filed on {latest_10k.filing_date}")
        
        sections = {}
        filing_date = str(latest_10k.filing_date)
        accession = latest_10k.accession_number
        
        # Try to get the filing object for structured extraction
        try:
            tenk = latest_10k.obj()
            
            # Extract Item 1 - Business Description
            try:
                item1 = tenk["Item 1"]
                if item1:
                    sections["business"] = _truncate_text(str(item1), MAX_SECTION_CHARS, "Business Description")
                    logger.info(f"Extracted Item 1 (Business): {len(sections['business']):,} chars")
            except Exception as e:
                logger.warning(f"Could not extract Item 1: {e}")
            
            # Extract Item 1A - Risk Factors
            try:
                item1a = tenk["Item 1A"]
                if item1a:
                    sections["risk_factors"] = _truncate_text(str(item1a), MAX_SECTION_CHARS, "Risk Factors")
                    logger.info(f"Extracted Item 1A (Risk Factors): {len(sections['risk_factors']):,} chars")
            except Exception as e:
                logger.warning(f"Could not extract Item 1A: {e}")
            
            # Extract Item 7 - Management's Discussion and Analysis (MD&A)
            try:
                item7 = tenk["Item 7"]
                if item7:
                    sections["mda"] = _truncate_text(str(item7), MAX_SECTION_CHARS, "MD&A")
                    logger.info(f"Extracted Item 7 (MD&A): {len(sections['mda']):,} chars")
            except Exception as e:
                logger.warning(f"Could not extract Item 7: {e}")
            
            # Extract Item 7A - Market Risk
            try:
                item7a = tenk["Item 7A"]
                if item7a:
                    sections["market_risk"] = _truncate_text(str(item7a), MAX_SECTION_CHARS // 2, "Market Risk")
                    logger.info(f"Extracted Item 7A (Market Risk): {len(sections['market_risk']):,} chars")
            except Exception as e:
                logger.warning(f"Could not extract Item 7A: {e}")
                
        except Exception as e:
            logger.warning(f"Structured extraction failed: {e}")
        
        # If structured extraction failed, try HTML-based extraction
        total_chars = sum(len(s) for s in sections.values() if isinstance(s, str))
        
        if total_chars == 0:
            logger.info("Structured extraction failed, trying HTML-based extraction...")
            sections = _extract_sections_from_html(latest_10k, "10-K")
        
        # Calculate total size
        total_chars = sum(len(s) for s in sections.values() if isinstance(s, str))
        logger.info(f"Total extracted text: {total_chars:,} chars")
        
        if total_chars == 0:
            return False, f"Could not extract any sections from 10-K for {ticker}", {}
        
        # Add metadata
        sections["_metadata"] = {
            "ticker": ticker,
            "filing_date": filing_date,
            "form": "10-K",
            "accession_number": accession,
            "total_chars": total_chars
        }
        
        return True, "", sections
        
    except ImportError as e:
        return False, f"edgartools not installed: {e}", {}
    except Exception as e:
        logger.error(f"Error extracting 10-K sections: {e}")
        return False, f"Error extracting 10-K: {str(e)}", {}


def extract_10q_sections(ticker: str) -> Tuple[bool, str, Dict[str, str]]:
    """
    Extract key sections from the latest 10-Q filing for a ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "JOBY")
        
    Returns:
        Tuple of (success, error_message, sections_dict)
        sections_dict contains: financial_statements, mda, risk_factors
    """
    try:
        from edgar import Company, set_identity
        
        # Set identity for SEC API (required)
        email = os.getenv("SEC_EDGAR_EMAIL", "user@example.com")
        set_identity(f"StockAnalyzer {email}")
        
        logger.info(f"Fetching 10-Q for {ticker} using edgartools...")
        
        # Get the company and its filings
        company = Company(ticker)
        filings = company.get_filings(form="10-Q")
        
        if not filings or len(filings) == 0:
            return False, f"No 10-Q filings found for {ticker}", {}
        
        # Get the latest 10-Q
        latest_10q = filings.latest()
        if not latest_10q:
            return False, f"Could not retrieve latest 10-Q for {ticker}", {}
        
        logger.info(f"Found 10-Q filed on {latest_10q.filing_date}")
        
        sections = {}
        filing_date = str(latest_10q.filing_date)
        accession = latest_10q.accession_number
        
        # Try to get the filing object for structured extraction
        try:
            tenq = latest_10q.obj()
            
            # Extract Part I, Item 2 - MD&A (most important for quarterly)
            try:
                item2 = tenq["Item 2"]
                if item2:
                    sections["mda"] = _truncate_text(str(item2), MAX_SECTION_CHARS, "MD&A")
                    logger.info(f"Extracted Item 2 (MD&A): {len(sections['mda']):,} chars")
            except Exception as e:
                logger.warning(f"Could not extract Item 2: {e}")
            
            # Extract Part II, Item 1A - Risk Factors (if updated)
            try:
                item1a = tenq["Item 1A"]
                if item1a:
                    sections["risk_factors"] = _truncate_text(str(item1a), MAX_SECTION_CHARS // 2, "Risk Factors")
                    logger.info(f"Extracted Item 1A (Risk Factors): {len(sections['risk_factors']):,} chars")
            except Exception as e:
                logger.warning(f"Could not extract Item 1A: {e}")
                
        except Exception as e:
            logger.warning(f"Structured extraction failed: {e}")
        
        # If structured extraction failed, try HTML-based extraction
        total_chars = sum(len(s) for s in sections.values() if isinstance(s, str))
        
        if total_chars == 0:
            logger.info("Structured extraction failed, trying HTML-based extraction...")
            sections = _extract_sections_from_html(latest_10q, "10-Q")
        
        # Calculate total size
        total_chars = sum(len(s) for s in sections.values() if isinstance(s, str))
        logger.info(f"Total extracted text: {total_chars:,} chars")
        
        if total_chars == 0:
            return False, f"Could not extract any sections from 10-Q for {ticker}", {}
        
        # Add metadata
        sections["_metadata"] = {
            "ticker": ticker,
            "filing_date": str(latest_10q.filing_date),
            "form": "10-Q",
            "accession_number": latest_10q.accession_number,
            "total_chars": total_chars
        }
        
        return True, "", sections
        
    except ImportError as e:
        return False, f"edgartools not installed: {e}", {}
    except Exception as e:
        logger.error(f"Error extracting 10-Q sections: {e}")
        return False, f"Error extracting 10-Q: {str(e)}", {}


def format_sections_for_analysis(sections: Dict[str, str], form_type: str = "10-K") -> str:
    """
    Format extracted sections into a single text block for LLM analysis.
    
    Args:
        sections: Dictionary of section name -> section text
        form_type: "10-K" or "10-Q"
        
    Returns:
        Formatted text ready for LLM analysis
    """
    metadata = sections.get("_metadata", {})
    ticker = metadata.get("ticker", "Unknown")
    filing_date = metadata.get("filing_date", "Unknown")
    
    output = []
    output.append(f"{'='*80}")
    output.append(f"{ticker} {form_type} FILING ANALYSIS")
    output.append(f"Filing Date: {filing_date}")
    output.append(f"{'='*80}\n")
    
    if form_type == "10-K":
        section_order = [
            ("business", "ITEM 1: BUSINESS DESCRIPTION"),
            ("risk_factors", "ITEM 1A: RISK FACTORS"),
            ("mda", "ITEM 7: MANAGEMENT'S DISCUSSION AND ANALYSIS"),
            ("market_risk", "ITEM 7A: MARKET RISK DISCLOSURES"),
            ("full_text", "FILING CONTENT"),  # Fallback when section extraction fails
        ]
    else:  # 10-Q
        section_order = [
            ("mda", "ITEM 2: MANAGEMENT'S DISCUSSION AND ANALYSIS"),
            ("risk_factors", "ITEM 1A: RISK FACTORS (UPDATES)"),
            ("full_text", "FILING CONTENT"),  # Fallback when section extraction fails
        ]
    
    for key, title in section_order:
        content = sections.get(key, "")
        if content and isinstance(content, str):
            output.append(f"\n{'='*60}")
            output.append(f"{title}")
            output.append(f"{'='*60}\n")
            output.append(content)
    
    result = "\n".join(output)
    
    # Final safety truncation
    if len(result) > MAX_TOTAL_CHARS:
        result = result[:MAX_TOTAL_CHARS]
        result += f"\n\n[... Content truncated at {MAX_TOTAL_CHARS:,} characters ...]"
        logger.warning(f"Final output truncated to {MAX_TOTAL_CHARS:,} chars")
    
    return result


def get_filing_for_analysis(ticker: str, form_type: str = "10-K") -> Tuple[bool, str, str]:
    """
    Main entry point: Get a filing's key sections formatted for LLM analysis.
    
    Args:
        ticker: Stock ticker symbol
        form_type: "10-K" or "10-Q"
        
    Returns:
        Tuple of (success, error_message, formatted_text)
    """
    ticker = ticker.upper()
    form_type = form_type.upper()
    
    if form_type == "10-K":
        success, error, sections = extract_10k_sections(ticker)
    elif form_type == "10-Q":
        success, error, sections = extract_10q_sections(ticker)
    else:
        return False, f"Unsupported form type: {form_type}", ""
    
    if not success:
        return False, error, ""
    
    formatted_text = format_sections_for_analysis(sections, form_type)
    
    logger.info(f"Successfully prepared {form_type} for {ticker}: {len(formatted_text):,} chars")
    
    return True, "", formatted_text


# Test function
if __name__ == "__main__":
    import sys
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    form_type = sys.argv[2] if len(sys.argv) > 2 else "10-K"
    
    print(f"\nTesting {form_type} extraction for {ticker}...")
    success, error, text = get_filing_for_analysis(ticker, form_type)
    
    if success:
        print(f"\n✓ Success! Extracted {len(text):,} characters")
        print(f"\nFirst 2000 characters:\n{'-'*60}")
        print(text[:2000])
        print(f"\n{'-'*60}")
        print(f"... and {len(text) - 2000:,} more characters")
    else:
        print(f"\n✗ Failed: {error}")
