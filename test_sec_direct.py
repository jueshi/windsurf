"""
Direct test of SEC EDGAR URL retrieval with detailed logging
"""
import os
import requests
import json
from dotenv import load_dotenv
from bs4 import BeautifulSoup

def test_sec_api():
    """Test SEC EDGAR API directly"""
    # Load environment variables
    load_dotenv()
    
    # Get SEC email
    email = os.getenv("SEC_EDGAR_EMAIL")
    print(f"Using SEC email: {email}")
    
    # Test ticker
    ticker = "AAPL"
    print(f"Testing with ticker: {ticker}")
    
    # Step 1: Get CIK number for the ticker
    cik_url = "https://www.sec.gov/files/company_tickers.json"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Referer': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    }
    
    # Add email to headers
    if email:
        headers['From'] = email
    
    print("Requesting CIK data...")
    try:
        response = requests.get(cik_url, headers=headers, timeout=30)
        response.raise_for_status()
        print(f"CIK request status: {response.status_code}")
        companies = response.json()
        
        # Find the CIK for the ticker
        cik = None
        for _, company in companies.items():
            if company['ticker'] == ticker:
                cik = str(company['cik_str']).zfill(10)
                break
        
        if not cik:
            print(f"Could not find CIK for ticker {ticker}")
            return
        
        print(f"Found CIK for {ticker}: {cik}")
        
        # Step 2: Get the company's submissions feed
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        
        # SEC API requires specific headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.sec.gov/edgar/browse/',
            'Host': 'data.sec.gov'
        }
        
        # Add email to headers
        if email:
            headers['From'] = email
        
        print(f"Requesting submissions data from: {submissions_url}")
        print(f"Headers: {headers}")
        
        response = requests.get(submissions_url, headers=headers, timeout=30)
        response.raise_for_status()
        print(f"Submissions request status: {response.status_code}")
        
        submissions = response.json()
        
        # Step 3: Find the latest 10-K filing
        recent_filings = submissions.get('filings', {}).get('recent', {})
        if not recent_filings:
            print("No recent filings found")
            return
        
        form_types = recent_filings.get('form', [])
        accession_numbers = recent_filings.get('accessionNumber', [])
        
        # Find the latest 10-K filing
        filing_type = "10-K"
        for i, form in enumerate(form_types):
            if filing_type in form:
                accession_number = accession_numbers[i].replace('-', '')
                
                # Construct the index URL for this filing
                index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{accession_numbers[i]}-index.htm"
                print(f"Found {filing_type} filing: {index_url}")
                
                # Test accessing the index page
                print(f"Accessing index page: {index_url}")
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
                
                response = requests.get(index_url, headers=headers, timeout=30)
                response.raise_for_status()
                print(f"Index page request status: {response.status_code}")
                
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
                                        base_url = index_url[:index_url.rfind('/')]
                                        document_links.append(f"{base_url}/{href}")
                
                if document_links:
                    print(f"Found {len(document_links)} document links:")
                    for i, link in enumerate(document_links[:5]):  # Show first 5 links
                        print(f"{i+1}. {link}")
                    
                    # Try to access the first document
                    if document_links:
                        doc_url = document_links[0]
                        print(f"\nAccessing document: {doc_url}")
                        
                        response = requests.get(doc_url, headers=headers, timeout=30)
                        response.raise_for_status()
                        print(f"Document request status: {response.status_code}")
                        print(f"Document size: {len(response.content)} bytes")
                        
                        # Extract some text from the document
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Remove scripts, styles, and other non-content elements
                        for element in soup(["script", "style", "meta", "link", "noscript"]):
                            element.decompose()
                        
                        # Extract text
                        text = soup.get_text(separator='\n', strip=True)
                        
                        # Print a sample of the text
                        print("\nSample text (first 500 characters):")
                        print(text[:500] + "...")
                        
                        return True
                else:
                    print("No document links found in the index page")
                
                break
        else:
            print(f"No {filing_type} filings found")
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if hasattr(e, 'response') and e.response.status_code == 403:
            print("403 Forbidden error. This may be due to rate limiting or missing/invalid email header.")
            print(f"Response headers: {e.response.headers}")
            print(f"Response content: {e.response.content[:500]}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sec_api()
