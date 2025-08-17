"""
Minimal test script for SEC EDGAR API access
"""
import os
import sys
import requests
from dotenv import load_dotenv

def main():
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
            return 1
        
        companies = response.json()
        
        # Find the CIK for the ticker
        cik = None
        for _, company in companies.items():
            if company['ticker'] == ticker:
                cik = str(company['cik_str']).zfill(10)
                break
        
        if not cik:
            print(f"Could not find CIK for ticker {ticker}")
            return 1
        
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
            return 1
        
        submissions = response.json()
        
        # Step 3: Find the latest 10-K filing
        print("\nStep 3: Finding latest 10-K filing...")
        filing_type = "10-K"
        recent_filings = submissions.get('filings', {}).get('recent', {})
        
        if not recent_filings:
            print("No recent filings found")
            return 1
        
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
                    return 1
                
                print("Successfully accessed index page!")
                print(f"Content length: {len(response.content)} bytes")
                
                # Print success message
                print("\nAll tests passed successfully!")
                print(f"SEC EDGAR URL retrieval is working correctly for {ticker} {filing_type}")
                return 0
        
        if not found:
            print(f"No {filing_type} filings found for {ticker}")
            return 1
            
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
