"""
SEC EDGAR API test with detailed logging
"""
import os
import sys
import requests
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sec_api_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_sec_api():
    """Test SEC EDGAR API access with detailed logging"""
    # Load environment variables
    load_dotenv()
    
    # Get SEC email
    email = os.getenv("SEC_EDGAR_EMAIL")
    logger.info(f"Using SEC email: {email}")
    
    # Test ticker
    ticker = "AAPL"
    logger.info(f"Testing with ticker: {ticker}")
    
    # Step 1: Get CIK number for the ticker
    logger.info("Step 1: Getting CIK number...")
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
        logger.info(f"Requesting: {cik_url}")
        response = requests.get(cik_url, headers=headers, timeout=30)
        logger.info(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Failed to get CIK data. Status code: {response.status_code}")
            return False
        
        companies = response.json()
        
        # Find the CIK for the ticker
        cik = None
        for _, company in companies.items():
            if company['ticker'] == ticker:
                cik = str(company['cik_str']).zfill(10)
                break
        
        if not cik:
            logger.error(f"Could not find CIK for ticker {ticker}")
            return False
        
        logger.info(f"Found CIK for {ticker}: {cik}")
        
        # Step 2: Get the company's submissions feed
        logger.info("Step 2: Getting submissions data...")
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
        
        logger.info(f"Requesting: {submissions_url}")
        response = requests.get(submissions_url, headers=headers, timeout=30)
        logger.info(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Failed to get submissions data. Status code: {response.status_code}")
            if response.status_code == 403:
                logger.error("403 Forbidden error. This may be due to rate limiting or missing/invalid email header.")
                logger.error(f"Headers used: {headers}")
            return False
        
        submissions = response.json()
        
        # Step 3: Find the latest 10-K filing
        logger.info("Step 3: Finding latest 10-K filing...")
        filing_type = "10-K"
        recent_filings = submissions.get('filings', {}).get('recent', {})
        
        if not recent_filings:
            logger.error("No recent filings found")
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
                logger.info(f"Found {filing_type} filing: {index_url}")
                found = True
                
                # Step 4: Access the index page
                logger.info("Step 4: Accessing index page...")
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
                
                logger.info(f"Requesting: {index_url}")
                response = requests.get(index_url, headers=headers, timeout=30)
                logger.info(f"Response status: {response.status_code}")
                
                if response.status_code != 200:
                    logger.error(f"Failed to access index page. Status code: {response.status_code}")
                    return False
                
                logger.info("Successfully accessed index page!")
                logger.info(f"Content length: {len(response.content)} bytes")
                
                # Print success message
                logger.info("All tests passed successfully!")
                logger.info(f"SEC EDGAR URL retrieval is working correctly for {ticker} {filing_type}")
                return True
        
        if not found:
            logger.error(f"No {filing_type} filings found for {ticker}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("Starting SEC EDGAR API test with logging...")
    print("Check sec_api_test.log for detailed output")
    result = test_sec_api()
    print(f"Test {'passed' if result else 'failed'}")
    sys.exit(0 if result else 1)
