"""
Simple test script for SEC EDGAR URL retrieval and text extraction
"""
import os
import sys
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

def test_sec_edgar():
    """Test SEC EDGAR URL retrieval and text extraction"""
    # Load environment variables
    load_dotenv()
    
    # Get SEC email
    email = os.getenv("SEC_EDGAR_EMAIL")
    print(f"Using SEC email: {email}")
    
    # Test ticker
    ticker = "AAPL"
    print(f"Testing with ticker: {ticker}")
    
    # Step 1: Get CIK number for the ticker
    print("\nStep 1: Getting CIK number...")
    cik_url = "https://www.sec.gov/files/company_tickers.json"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Referer': 'https://www.sec.gov/edgar/searchedgar/companysearch'
    }
    
    # Add email to headers
    if email:
        headers['From'] = email
    
    try:
        print(f"Requesting: {cik_url}")
        response = requests.get(cik_url, headers=headers, timeout=30)
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Failed to get CIK data. Status code: {response.status_code}")
            return False
        
        companies = response.json()
        
        # Find the CIK for the ticker
        cik = None
        for _, company in companies.items():
            if company['ticker'] == ticker:
                cik = str(company['cik_str']).zfill(10)
                break
        
        if not cik:
            print(f"Could not find CIK for ticker {ticker}")
            return False
        
        print(f"Found CIK for {ticker}: {cik}")
        
        # Step 2: Get the company's submissions feed
        print("\nStep 2: Getting submissions data...")
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
        
        print(f"Requesting: {submissions_url}")
        response = requests.get(submissions_url, headers=headers, timeout=30)
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Failed to get submissions data. Status code: {response.status_code}")
            if response.status_code == 403:
                print("403 Forbidden error. This may be due to rate limiting or missing/invalid email header.")
                print(f"Headers used: {headers}")
            return False
        
        submissions = response.json()
        
        # Step 3: Find the latest 10-K filing
        print("\nStep 3: Finding latest 10-K filing...")
        filing_type = "10-K"
        recent_filings = submissions.get('filings', {}).get('recent', {})
        
        if not recent_filings:
            print("No recent filings found")
            return False
        
        form_types = recent_filings.get('form', [])
        accession_numbers = recent_filings.get('accessionNumber', [])
        
        # Find the latest 10-K filing
        found = False
        for i, form in enumerate(form_types):
            if filing_type in form:
                accession_number = accession_numbers[i].replace('-', '')
                
                # Construct the index URL for this filing
                index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{accession_numbers[i]}-index.htm"
                print(f"Found {filing_type} filing: {index_url}")
                found = True
                
                # Step 4: Access the index page
                print("\nStep 4: Accessing index page...")
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
                
                print(f"Requesting: {index_url}")
                response = requests.get(index_url, headers=headers, timeout=30)
                print(f"Response status: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"Failed to access index page. Status code: {response.status_code}")
                    return False
                
                print("Successfully accessed index page!")
                
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
                    print("Found filing table, extracting document links...")
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
                
                if not document_links:
                    print("No links found in table, searching all links...")
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
                                base_url = index_url[:index_url.rfind('/')]
                                document_links.append(f"{base_url}/{href}")
                
                if document_links:
                    print(f"Found {len(document_links)} document links")
                    
                    # Try to find the main document (10-K or 10-Q)
                    main_doc_keywords = ['10-k', '10-q', '10k', '10q', 'form10-k', 'form10-q', 'form10k', 'form10q']
                    
                    doc_url = None
                    # First priority: Look for the main document
                    for link in document_links:
                        link_lower = link.lower()
                        if any(keyword in link_lower for keyword in main_doc_keywords):
                            doc_url = link
                            print(f"Selected main document URL: {doc_url}")
                            break
                    
                    if not doc_url:
                        # Second priority: Look for htm files that might be the main document
                        for link in document_links:
                            if link.lower().endswith('.htm') or link.lower().endswith('.html'):
                                doc_url = link
                                print(f"Selected HTML document URL: {doc_url}")
                                break
                    
                    if not doc_url:
                        # Last resort: use the first document
                        doc_url = document_links[0]
                        print(f"Using first document URL: {doc_url}")
                    
                    # Step 5: Access the document
                    print("\nStep 5: Accessing document...")
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Connection': 'keep-alive',
                        'Referer': index_url
                    }
                    
                    # Add email to headers
                    if email:
                        headers['From'] = email
                    
                    print(f"Requesting: {doc_url}")
                    response = requests.get(doc_url, headers=headers, timeout=30)
                    print(f"Response status: {response.status_code}")
                    
                    if response.status_code != 200:
                        print(f"Failed to access document. Status code: {response.status_code}")
                        return False
                    
                    print("Successfully accessed document!")
                    print(f"Content length: {len(response.content)} bytes")
                    
                    # Extract text from the document
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
                    print("\nSample text (first 500 characters):")
                    print(text[:500] + "...")
                    
                    print("\nAll tests passed successfully!")
                    print(f"SEC EDGAR URL retrieval and text extraction is working correctly for {ticker} {filing_type}")
                    return True
                else:
                    print("No document links found in the index page")
                    return False
                
                break
        
        if not found:
            print(f"No {filing_type} filings found for {ticker}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting SEC EDGAR test...")
    result = test_sec_edgar()
    print(f"\nTest {'passed' if result else 'failed'}")
    sys.exit(0 if result else 1)
