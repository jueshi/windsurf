"""
SEC EDGAR API wrapper for fetching company filings.
Provides access to 10-K, 10-Q, and other SEC filings.
"""
import os
import re
import time
import requests
import pandas as pd
import logging
from typing import Optional, Dict, List, Any, Tuple
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from pathlib import Path

# Load .env from parent directory if not found in current
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)
load_dotenv()  # Also try current directory


class SECAPIWrapper:
    """
    Wrapper for SEC EDGAR API with caching and rate limiting.
    SEC requires proper User-Agent headers with contact email.
    """
    
    # Cache for CIK lookups (rarely changes)
    _cik_cache: Dict[str, str] = {}
    _last_request_time: float = 0
    _min_request_interval: float = 0.1  # SEC rate limit: 10 requests/second
    
    def __init__(self):
        self._init_headers()
    
    def _init_headers(self):
        """Initialize or reinitialize headers with current env settings."""
        self.email = os.getenv("SEC_EDGAR_EMAIL")
        print(f"SEC_EDGAR_EMAIL loaded: {self.email}")  # Use print for immediate output
        logging.info(f"SEC_EDGAR_EMAIL loaded: {self.email}")
        if not self.email:
            logging.warning("SEC_EDGAR_EMAIL not set. SEC API access may fail.")

        self.headers = {
            'User-Agent': 'StockToolbox/1.0',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Host': 'data.sec.gov'
        }
        if self.email:
            self.headers['From'] = self.email
            # SEC requires User-Agent to contain email or company name
            self.headers['User-Agent'] = f"StockToolbox/1.0 ({self.email})"
        
        print(f"User-Agent set to: {self.headers['User-Agent']}")
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting for SEC API requests."""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def get_company_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) for a ticker symbol.
        Results are cached since CIKs rarely change.
        """
        ticker = ticker.upper()
        
        # Check cache first
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]
        
        url = "https://www.sec.gov/files/company_tickers.json"

        try:
            self._rate_limit()
            
            # Use separate headers for www.sec.gov vs data.sec.gov
            headers = self.headers.copy()
            headers.pop('Host', None)

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            companies = response.json()

            for _, company in companies.items():
                if company['ticker'] == ticker:
                    cik = str(company['cik_str']).zfill(10)
                    self._cik_cache[ticker] = cik
                    return cik
            return None
        except requests.exceptions.Timeout:
            logging.error(f"Timeout fetching CIK for {ticker}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error fetching CIK for {ticker}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error fetching CIK for {ticker}: {e}")
            return None

    def get_latest_filing_info(self, cik: str, form_type: str = "10-K") -> Optional[Dict[str, Any]]:
        """
        Get the latest filing info for a given form type (10-K, 10-Q, etc.).
        
        Args:
            cik: The company's CIK number (10 digits, zero-padded)
            form_type: The SEC form type (default: "10-K")
            
        Returns:
            Dict with filing info or None if not found
        """
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"

        try:
            self._rate_limit()
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            recent = data.get('filings', {}).get('recent', {})
            form_types = recent.get('form', [])
            accession_numbers = recent.get('accessionNumber', [])
            filing_dates = recent.get('filingDate', [])
            primary_docs = recent.get('primaryDocument', [])  # Get primary document name

            for i, form in enumerate(form_types):
                if form == form_type:
                    accession_number = accession_numbers[i]
                    filing_date = filing_dates[i]
                    primary_doc = primary_docs[i] if i < len(primary_docs) else None
                    
                    # Construct detailed URL
                    acc_no_dash = accession_number.replace('-', '')
                    detail_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dash}/{accession_number}-index.html"

                    return {
                        "accessionNumber": accession_number,
                        "filingDate": filing_date,
                        "form": form,
                        "detailUrl": detail_url,
                        "primaryDocument": primary_doc,
                        "cik": cik
                    }
            return None
        except requests.exceptions.Timeout:
            logging.error(f"Timeout fetching filing info for CIK {cik}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error fetching filing info for CIK {cik}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error fetching filing info for CIK {cik}: {e}")
            return None

    def get_filing_content(self, cik: str, accession_number: str, primary_doc: Optional[str] = None) -> Optional[str]:
        """
        Get the HTML content of a filing.
        
        Args:
            cik: The company's CIK number
            accession_number: The filing's accession number
            primary_doc: Optional primary document name (if known)
            
        Returns:
            HTML content as string or None if not found
        """
        acc_no_dash = accession_number.replace('-', '')
        # CIK in archive URLs should NOT have leading zeros
        cik_no_zeros = cik.lstrip('0')
        
        headers = self.headers.copy()
        headers.pop('Host', None)
        
        logging.info(f"Fetching filing content for CIK {cik_no_zeros}, accession {accession_number}")
        
        # If we know the primary document, try it first
        if primary_doc:
            try:
                self._rate_limit()
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{acc_no_dash}/{primary_doc}"
                logging.info(f"Trying primary doc URL: {doc_url}")
                response = requests.get(doc_url, headers=headers, timeout=20)
                if response.status_code == 200:
                    return response.text
            except Exception as e:
                logging.warning(f"Failed to fetch primary doc {primary_doc}: {e}")
        
        # Fallback: parse index.json to find the main document
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{acc_no_dash}/index.json"
        logging.info(f"Trying index URL: {index_url}")
        logging.info(f"Using headers: {headers}")

        try:
            self._rate_limit()
            response = requests.get(index_url, headers=headers, timeout=10)
            logging.info(f"Index response status: {response.status_code}")
            
            if response.status_code == 403:
                logging.error("SEC returned 403 Forbidden. Check User-Agent header and email configuration.")
                logging.error(f"Current User-Agent: {headers.get('User-Agent')}")
            
            if response.status_code == 200:
                files = response.json().get('directory', {}).get('item', [])
                
                # Priority order for finding main document
                candidates = []
                for file in files:
                    name = file.get('name', '')
                    size_str = file.get('size', '0')
                    # Size can be string or empty, convert safely
                    try:
                        size = int(size_str) if size_str else 0
                    except (ValueError, TypeError):
                        size = 0
                    
                    if name.endswith('.htm') or name.endswith('.html'):
                        # Score based on likelihood of being main document
                        score = 0
                        if '10k' in name.lower() or '10q' in name.lower():
                            score += 10
                        if accession_number.replace('-', '') in name:
                            score += 5
                        if '-' not in name:
                            score += 3
                        # Larger files are more likely to be the main doc
                        score += min(size // 100000, 5)
                        
                        candidates.append((name, score))
                
                # Sort by score descending
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                for name, _ in candidates[:3]:  # Try top 3 candidates
                    try:
                        self._rate_limit()
                        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{acc_no_dash}/{name}"
                        logging.info(f"Trying document URL: {doc_url}")
                        doc_response = requests.get(doc_url, headers=headers, timeout=20)
                        if doc_response.status_code == 200 and len(doc_response.text) > 1000:
                            logging.info(f"Successfully fetched document: {name}")
                            return doc_response.text
                    except Exception as e:
                        logging.warning(f"Failed to fetch {name}: {e}")
                        continue
            
            logging.warning(f"No suitable document found for CIK {cik_no_zeros}, accession {accession_number}")
            return None
        except Exception as e:
            logging.error(f"Error fetching filing content: {e}")
            return None

    def extract_tables(self, html_content: str) -> List[pd.DataFrame]:
        """
        Extract tables from HTML content.
        
        Args:
            html_content: HTML string containing tables
            
        Returns:
            List of DataFrames, one per table found
        """
        if not html_content:
            return []
            
        try:
            dfs = pd.read_html(html_content)
            # Filter out tiny tables (likely navigation/formatting)
            return [df for df in dfs if df.shape[0] > 1 and df.shape[1] > 1]
        except ValueError:
            # No tables found
            return []
        except Exception as e:
            logging.error(f"Error extracting tables: {e}")
            return []

    def extract_sections(self, html_content: str, form_type: str = "10-K") -> List[Dict[str, Any]]:
        """
        Extract key sections from SEC filing HTML content.
        
        Args:
            html_content: HTML string of the filing
            form_type: Type of filing (10-K or 10-Q)
            
        Returns:
            List of dicts with section info: {id, title, anchor}
        """
        if not html_content:
            return []
        
        # Key sections for 10-K filings
        sections_10k = [
            ("item1", "Item 1", "Business"),
            ("item1a", "Item 1A", "Risk Factors"),
            ("item1b", "Item 1B", "Unresolved Staff Comments"),
            ("item1c", "Item 1C", "Cybersecurity"),
            ("item2", "Item 2", "Properties"),
            ("item3", "Item 3", "Legal Proceedings"),
            ("item4", "Item 4", "Mine Safety Disclosures"),
            ("item5", "Item 5", "Market for Registrant's Common Equity"),
            ("item6", "Item 6", "Reserved"),
            ("item7", "Item 7", "Management's Discussion and Analysis (MD&A)"),
            ("item7a", "Item 7A", "Quantitative and Qualitative Disclosures About Market Risk"),
            ("item8", "Item 8", "Financial Statements and Supplementary Data"),
            ("item9", "Item 9", "Changes in and Disagreements With Accountants"),
            ("item9a", "Item 9A", "Controls and Procedures"),
            ("item9b", "Item 9B", "Other Information"),
            ("item9c", "Item 9C", "Disclosure Regarding Foreign Jurisdictions"),
            ("item10", "Item 10", "Directors, Executive Officers and Corporate Governance"),
            ("item11", "Item 11", "Executive Compensation"),
            ("item12", "Item 12", "Security Ownership"),
            ("item13", "Item 13", "Certain Relationships and Related Transactions"),
            ("item14", "Item 14", "Principal Accountant Fees and Services"),
            ("item15", "Item 15", "Exhibits and Financial Statement Schedules"),
            ("item16", "Item 16", "Form 10-K Summary"),
        ]
        
        # Key sections for 10-Q filings
        sections_10q = [
            ("part1item1", "Part I, Item 1", "Financial Statements"),
            ("part1item2", "Part I, Item 2", "Management's Discussion and Analysis (MD&A)"),
            ("part1item3", "Part I, Item 3", "Quantitative and Qualitative Disclosures About Market Risk"),
            ("part1item4", "Part I, Item 4", "Controls and Procedures"),
            ("part2item1", "Part II, Item 1", "Legal Proceedings"),
            ("part2item1a", "Part II, Item 1A", "Risk Factors"),
            ("part2item2", "Part II, Item 2", "Unregistered Sales of Equity Securities"),
            ("part2item3", "Part II, Item 3", "Defaults Upon Senior Securities"),
            ("part2item4", "Part II, Item 4", "Mine Safety Disclosures"),
            ("part2item5", "Part II, Item 5", "Other Information"),
            ("part2item6", "Part II, Item 6", "Exhibits"),
        ]
        
        sections = sections_10k if form_type == "10-K" else sections_10q
        found_sections = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text()
            
            for section_id, item_label, description in sections:
                # Build regex patterns to find section headers
                # Pattern matches variations like "Item 1.", "ITEM 1", "Item 1 -", "Item 1:", etc.
                item_num = item_label.replace("Part I, ", "").replace("Part II, ", "")
                
                # Extract just the item number for flexible matching (e.g., "1A" from "Item 1A")
                item_number_only = item_num.replace("Item ", "")
                
                # Multiple patterns to catch different formatting styles
                patterns = [
                    # Full label with description
                    rf'(?i)\b{re.escape(item_label)}[\.\s\-:]+\s*{re.escape(description[:15])}',
                    rf'(?i)\b{re.escape(item_num)}[\.\s\-:]+\s*{re.escape(description[:15])}',
                    # Just the item label
                    rf'(?i)\b{re.escape(item_label)}[\.\s\-:]+',
                    rf'(?i)\b{re.escape(item_num)}[\.\s\-:]+\s*[A-Z]',
                    # Item with number only (e.g., "ITEM 1A" or "Item 1A.")
                    rf'(?i)\bITEM\s*{re.escape(item_number_only)}[\.\s\-:]+',
                    # Part patterns for 10-Q
                    rf'(?i)PART\s+[I]+[,\s]+ITEM\s*{re.escape(item_number_only)}',
                    rf'(?i)PART\s+II[,\s]+ITEM\s*{re.escape(item_number_only)}',
                ]
                
                found = False
                for pattern in patterns:
                    if re.search(pattern, text_content):
                        found = True
                        break
                
                if found:
                    found_sections.append({
                        "id": section_id,
                        "label": item_label,
                        "description": description,
                        "found": True
                    })
            
            return found_sections
            
        except Exception as e:
            logging.error(f"Error extracting sections: {e}")
            return []

    def get_section_content(self, html_content: str, section_id: str, form_type: str = "10-K") -> str:
        """
        Extract content for a specific section from the filing.
        
        Args:
            html_content: HTML string of the filing
            section_id: The section identifier (e.g., 'item1a')
            form_type: Type of filing
            
        Returns:
            HTML content of the section (truncated for display)
        """
        if not html_content:
            return "<p>No content available.</p>"
        
        # Section mappings
        section_map_10k = {
            "item1": ("Item 1", "Business"),
            "item1a": ("Item 1A", "Risk Factors"),
            "item1b": ("Item 1B", "Unresolved Staff Comments"),
            "item1c": ("Item 1C", "Cybersecurity"),
            "item2": ("Item 2", "Properties"),
            "item3": ("Item 3", "Legal Proceedings"),
            "item4": ("Item 4", "Mine Safety Disclosures"),
            "item5": ("Item 5", "Market for Registrant"),
            "item6": ("Item 6", "Reserved"),
            "item7": ("Item 7", "Management's Discussion"),
            "item7a": ("Item 7A", "Quantitative and Qualitative"),
            "item8": ("Item 8", "Financial Statements"),
            "item9": ("Item 9", "Changes in and Disagreements"),
            "item9a": ("Item 9A", "Controls and Procedures"),
            "item9b": ("Item 9B", "Other Information"),
            "item9c": ("Item 9C", "Disclosure Regarding Foreign"),
            "item10": ("Item 10", "Directors"),
            "item11": ("Item 11", "Executive Compensation"),
            "item12": ("Item 12", "Security Ownership"),
            "item13": ("Item 13", "Certain Relationships"),
            "item14": ("Item 14", "Principal Accountant"),
            "item15": ("Item 15", "Exhibits"),
            "item16": ("Item 16", "Form 10-K Summary"),
        }
        
        section_map_10q = {
            "part1item1": ("Part I, Item 1", "Financial Statements"),
            "part1item2": ("Part I, Item 2", "Management's Discussion"),
            "part1item3": ("Part I, Item 3", "Quantitative"),
            "part1item4": ("Part I, Item 4", "Controls"),
            "part2item1": ("Part II, Item 1", "Legal Proceedings"),
            "part2item1a": ("Part II, Item 1A", "Risk Factors"),
            "part2item2": ("Part II, Item 2", "Unregistered Sales"),
            "part2item3": ("Part II, Item 3", "Defaults"),
            "part2item4": ("Part II, Item 4", "Mine Safety"),
            "part2item5": ("Part II, Item 5", "Other Information"),
            "part2item6": ("Part II, Item 6", "Exhibits"),
        }
        
        section_map = section_map_10k if form_type == "10-K" else section_map_10q
        
        if section_id not in section_map:
            return "<p>Section not found.</p>"
        
        item_label, description = section_map[section_id]
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(separator='\n')
            
            # Find the start of this section
            item_num = item_label.replace("Part I, ", "").replace("Part II, ", "")
            start_pattern = rf'(?i)({re.escape(item_label)}|{re.escape(item_num)})[\.\s\-:]+\s*{re.escape(description[:15])}'
            
            start_match = re.search(start_pattern, text)
            if not start_match:
                # Try simpler pattern
                start_pattern = rf'(?i){re.escape(item_label)}[\.\s\-:]+'
                start_match = re.search(start_pattern, text)
            
            if not start_match:
                return f"<p>Could not locate {item_label} in the document.</p>"
            
            start_pos = start_match.start()
            
            # Find the next section to determine end
            # Get list of all items after current one
            all_items = list(section_map.keys())
            current_idx = all_items.index(section_id)
            
            end_pos = len(text)
            for next_id in all_items[current_idx + 1:]:
                next_label, next_desc = section_map[next_id]
                next_num = next_label.replace("Part I, ", "").replace("Part II, ", "")
                end_pattern = rf'(?i)({re.escape(next_label)}|{re.escape(next_num)})[\.\s\-:]+\s*{re.escape(next_desc[:10])}'
                end_match = re.search(end_pattern, text[start_pos + 100:])
                if end_match:
                    end_pos = start_pos + 100 + end_match.start()
                    break
            
            # Extract section text
            section_text = text[start_pos:end_pos]
            
            # Truncate if too long (for display)
            max_chars = 15000
            if len(section_text) > max_chars:
                section_text = section_text[:max_chars] + "\n\n... [Content truncated for display] ..."
            
            # Convert to simple HTML with paragraphs
            paragraphs = section_text.split('\n\n')
            html_output = ""
            for p in paragraphs:
                p = p.strip()
                if p:
                    html_output += f"<p>{p}</p>\n"
            
            return html_output if html_output else "<p>Section content could not be extracted.</p>"
            
        except Exception as e:
            logging.error(f"Error extracting section content: {e}")
            return f"<p>Error extracting section: {str(e)}</p>"


sec_api = SECAPIWrapper()
