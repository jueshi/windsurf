import os
import sys
import requests
import traceback
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Simple standalone test script for SEC EDGAR URL retrieval

def get_sec_filing_url(ticker, filing_type):
    """Get SEC filing URL for a ticker"""
    print(f"Getting {filing_type} filing URL for {ticker}...")
    ticker = ticker.upper()
    filing_type = filing_type.upper()
    
    try:
        # Step 1: Get CIK number for the ticker
        print("Step 1: Getting CIK number...")
        cik_url = "https://www.sec.gov/files/company_tickers.json"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        
        response = requests.get(cik_url, headers=headers)
        response.raise_for_status()
        companies = response.json()
        
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
        print("Step 2: Getting submissions feed...")
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        
        # SEC API requires specific headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        }
        
        # Add email to headers as required by SEC
        load_dotenv()
        email = os.getenv("SEC_EDGAR_EMAIL")
        if email:
            headers['From'] = email
        
        response = requests.get(submissions_url, headers=headers)
        response.raise_for_status()
        submissions = response.json()
        
        # Step 3: Find the latest filing of the requested type
        print("Step 3: Finding latest filing...")
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
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None


def get_text_from_url(url):
    """Fetch and extract text from a URL"""
    print(f"Fetching text from URL: {url}")
    
    try:
        # Check if the URL is an SEC EDGAR index page
        if "-index.htm" in url:
            # First, get the index page to find the actual document link
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            
            print("Fetching index page to find document link...")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse the index page to find the actual document link
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for document links in the table
            document_links = []
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
                print(f"Found document links: {document_links}")
                # Try to find the main document (10-K or 10-Q)
                for doc_url in document_links:
                    if '10-k' in doc_url.lower() or '10-q' in doc_url.lower() or '10k' in doc_url.lower() or '10q' in doc_url.lower():
                        url = doc_url
                        print(f"Selected main document URL: {url}")
                        break
                else:
                    # If no specific 10-K/10-Q found, use the first document
                    url = document_links[0]
                    print(f"Using first document URL: {url}")
            else:
                print("No document links found in the index page.")
        
        # Now fetch the actual document
        print(f"Fetching document from: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
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
        return text
        
    except requests.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during text extraction: {e}")
        traceback.print_exc()
        return None


def main():
    """Main function to test SEC EDGAR URL retrieval"""
    # Load environment variables
    load_dotenv()
    
    # Check if SEC_EDGAR_EMAIL is set
    sec_email = os.getenv("SEC_EDGAR_EMAIL")
    if not sec_email:
        print("SEC_EDGAR_EMAIL environment variable is not set. This is required for SEC API access.")
        return
    
    print(f"Using SEC_EDGAR_EMAIL: {sec_email}")
    
    # Test with a single ticker
    ticker = "AAPL"
    filing_type = "10-K"
    
    # Get the filing URL
    filing_url = get_sec_filing_url(ticker, filing_type)
    
    if filing_url:
        print(f"\nSuccessfully retrieved URL: {filing_url}")
        
        # Test text extraction
        text = get_text_from_url(filing_url)
        
        if text:
            print("\nText extraction successful!")
            print(f"Text length: {len(text)} characters")
            print("\nPreview of extracted text:")
            print("-" * 80)
            print(text[:500] + "..." if len(text) > 500 else text)
            print("-" * 80)
        else:
            print("\nText extraction failed.")
    else:
        print(f"\nFailed to retrieve {filing_type} URL for {ticker}")


if __name__ == "__main__":
    main()
