"""
Standalone test for SEC EDGAR URL retrieval without importing from main module
"""
import os
import requests
import json
from dotenv import load_dotenv
from bs4 import BeautifulSoup

def get_sec_filing_url(ticker, filing_type):
    """
    Standalone function to retrieve SEC filing URL
    """
    print(f"Starting SEC filing URL retrieval for {ticker} {filing_type}...")
    
    # Load environment variables
    load_dotenv()
    email = os.getenv("SEC_EDGAR_EMAIL")
    print(f"Using SEC email: {email}")
    
    ticker = ticker.upper()
    filing_type = filing_type.upper()
    
    # Step 1: Get CIK number for the ticker
    cik_url = "https://www.sec.gov/files/company_tickers.json"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Referer': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    }
    
    # Add email to headers
    if email:
        headers['From'] = email
    else:
        print("WARNING: SEC_EDGAR_EMAIL not set in environment variables")
    
    try:
        print("Requesting CIK data...")
        response = requests.get(cik_url, headers=headers, timeout=30)
        print(f"CIK request status: {response.status_code}")
        response.raise_for_status()
        companies = response.json()
        
        # Find the CIK for the ticker
        cik = None
        for _, company in companies.items():
            if company['ticker'] == ticker:
                cik = str(company['cik_str']).zfill(10)
                break
        
        if not cik:
            print(f"Could not find CIK for ticker {ticker}")
            return None
        
        print(f"Found CIK for {ticker}: {cik}")
        
        # Step 2: Get the company's submissions feed
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        
        # SEC API requires specific headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Referer': 'https://www.sec.gov/edgar/browse/',
            'Host': 'data.sec.gov'
        }
        
        # Add email to headers
        if email:
            headers['From'] = email
        
        print(f"Requesting submissions data from: {submissions_url}")
        response = requests.get(submissions_url, headers=headers, timeout=30)
        print(f"Submissions request status: {response.status_code}")
        response.raise_for_status()
        
        submissions = response.json()
        
        # Step 3: Find the latest filing of the requested type
        recent_filings = submissions.get('filings', {}).get('recent', {})
        if not recent_filings:
            print(f"No recent filings found for {ticker}")
            return None
        
        form_types = recent_filings.get('form', [])
        accession_numbers = recent_filings.get('accessionNumber', [])
        
        # Find the latest filing of the requested type
        for i, form in enumerate(form_types):
            if filing_type in form:
                accession_number = accession_numbers[i].replace('-', '')
                
                # Construct the index URL for this filing
                index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{accession_numbers[i]}-index.htm"
                print(f"Found {filing_type} filing: {index_url}")
                return index_url
        
        print(f"No {filing_type} filings found for {ticker}")
        return None
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if hasattr(e, 'response') and e.response.status_code == 403:
            print("403 Forbidden error. This may be due to rate limiting or missing/invalid email header.")
            if hasattr(e, 'response'):
                print(f"Response headers: {e.response.headers}")
                print(f"Response content: {e.response.content[:500]}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_text_from_url(url):
    """
    Standalone function to extract text from a URL
    """
    print(f"Extracting text from URL: {url}")
    
    # Load environment variables
    load_dotenv()
    email = os.getenv("SEC_EDGAR_EMAIL")
    
    try:
        # Check if the URL is an SEC EDGAR index page
        if "-index.htm" in url:
            print("This is an index page, finding the actual document...")
            
            # First, get the index page to find the actual document link
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml',
                'Accept-Language': 'en-US,en;q=0.9',
                'Connection': 'keep-alive',
                'Referer': 'https://www.sec.gov/edgar/browse/'
            }
            
            # Add email to headers
            if email:
                headers['From'] = email
            
            response = requests.get(url, headers=headers, timeout=30)
            print(f"Index page request status: {response.status_code}")
            response.raise_for_status()
            
            # Parse the index page to find the actual document link
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for document links in the table
            document_links = []
            
            # First try to find the table with filing documents
            filing_table = None
            for table in soup.find_all('table'):
                if table.find('th', text=lambda t: t and 'Document' in t):
                    filing_table = table
                    break
            
            if filing_table:
                # Extract links from the filing table
                for row in filing_table.find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        a_tag = cells[2].find('a') if len(cells) > 2 else cells[0].find('a')
                        if a_tag and a_tag.get('href'):
                            href = a_tag.get('href')
                            if '.htm' in href and 'index' not in href:
                                # Found a potential document link
                                if href.startswith('/'):
                                    document_links.append(f"https://www.sec.gov{href}")
                                elif href.startswith('http'):
                                    document_links.append(href)
                                else:
                                    # Relative link
                                    base_url = url[:url.rfind('/')]
                                    document_links.append(f"{base_url}/{href}")
            
            if not document_links:
                # If no links found in table, try all links
                for a_tag in soup.find_all('a'):
                    href = a_tag.get('href')
                    if href and ('.htm' in href) and ('index' not in href):
                        # Found a potential document link
                        if href.startswith('/'):
                            document_links.append(f"https://www.sec.gov{href}")
                        elif href.startswith('http'):
                            document_links.append(href)
                        else:
                            # Relative link
                            base_url = url[:url.rfind('/')]
                            document_links.append(f"{base_url}/{href}")
            
            if document_links:
                print(f"Found {len(document_links)} document links")
                
                # Try to find the main document (10-K or 10-Q)
                main_doc_keywords = ['10-k', '10-q', '10k', '10q', 'form10-k', 'form10-q', 'form10k', 'form10q']
                
                # First priority: Look for the main document
                for doc_url in document_links:
                    doc_lower = doc_url.lower()
                    if any(keyword in doc_lower for keyword in main_doc_keywords):
                        url = doc_url
                        print(f"Selected main document URL: {url}")
                        break
                else:
                    # Second priority: Look for htm files that might be the main document
                    for doc_url in document_links:
                        if doc_url.lower().endswith('.htm') or doc_url.lower().endswith('.html'):
                            url = doc_url
                            print(f"Selected HTML document URL: {url}")
                            break
                    else:
                        # Last resort: use the first document
                        url = document_links[0]
                        print(f"Using first document URL: {url}")
            else:
                print("No document links found in the index page")
                return None
        
        # Now fetch the actual document
        print(f"Fetching document from: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Referer': 'https://www.sec.gov/edgar/browse/'
        }
        
        # Add email to headers
        if email:
            headers['From'] = email
        
        response = requests.get(url, headers=headers, timeout=30)
        print(f"Document request status: {response.status_code}")
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove scripts, styles, and other non-content elements
        for element in soup(["script", "style", "meta", "link", "noscript"]):
            element.decompose()
        
        # Extract text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up the text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        
        print(f"Successfully extracted {len(text)} characters of text")
        
        # Return a sample of the text
        return text[:1000] + "..."
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if hasattr(e, 'response') and e.response.status_code == 403:
            print("403 Forbidden error. This may be due to rate limiting or missing/invalid email header.")
            if hasattr(e, 'response'):
                print(f"Response headers: {e.response.headers}")
                print(f"Response content: {e.response.content[:500]}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Main test function
    """
    print("=== Testing SEC EDGAR URL Retrieval ===")
    
    # Test with a known ticker
    ticker = "AAPL"
    filing_type = "10-K"
    
    # Get the filing URL
    url = get_sec_filing_url(ticker, filing_type)
    if url:
        print(f"\nSuccessfully retrieved URL: {url}")
        
        # Extract text from the URL
        print("\n=== Testing Text Extraction ===")
        text = get_text_from_url(url)
        if text:
            print("\nSample text:")
            print(text)
            print("\nAll tests passed!")
        else:
            print("Text extraction failed")
    else:
        print("URL retrieval failed")

if __name__ == "__main__":
    main()
