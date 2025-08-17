import os
import tempfile
from dotenv import load_dotenv
from sec_edgar_downloader import Downloader

def download_filing(ticker, filing_type, limit=1):
    """
    Downloads SEC filings using sec_edgar_downloader package.
    
    Args:
        ticker (str): The stock ticker symbol.
        filing_type (str): The filing type (e.g., '10-K', '10-Q').
        limit (int): Maximum number of filings to download.
        
    Returns:
        tuple: (success, file_path, text_content) where:
            - success (bool): True if download was successful
            - file_path (str): Path to the downloaded file
            - text_content (str): Content of the downloaded file
    """
    # Load environment variables
    load_dotenv()
    sec_email = os.getenv("SEC_EDGAR_EMAIL")
    
    if not sec_email:
        print("ERROR: SEC_EDGAR_EMAIL environment variable is not set.")
        return False, None, "SEC_EDGAR_EMAIL environment variable is not set."
    
    # Initialize downloader with company name and email
    dl = Downloader("Stone & Associates Inc", sec_email)
    
    try:
        # Download the filing
        print(f"Downloading {filing_type} filing for {ticker}...")
        result = dl.get(filing_type, ticker, limit=limit)
        print(f"Download result: {result}")
        
        # Get the path to the downloaded files
        download_folder = dl.get_download_folder()
        filing_path = os.path.join(download_folder, "sec-edgar-filings", ticker, filing_type)
        
        if not os.path.exists(filing_path):
            return False, None, f"Download completed but no files found in {filing_path}"
        
        # Find the most recent filing (assuming it's in a subfolder)
        latest_folder = None
        latest_time = 0
        
        for item in os.listdir(filing_path):
            item_path = os.path.join(filing_path, item)
            if os.path.isdir(item_path):
                folder_time = os.path.getmtime(item_path)
                if folder_time > latest_time:
                    latest_time = folder_time
                    latest_folder = item_path
        
        if not latest_folder:
            return False, None, "No filing folders found"
        
        # Find the filing document (usually a .txt file)
        filing_document = None
        for root, _, files in os.walk(latest_folder):
            for file in files:
                if file.endswith('.txt'):
                    filing_document = os.path.join(root, file)
                    break
            if filing_document:
                break
        
        if not filing_document:
            # Try looking for HTML files if no text file is found
            for root, _, files in os.walk(latest_folder):
                for file in files:
                    if file.endswith('.htm') or file.endswith('.html'):
                        filing_document = os.path.join(root, file)
                        break
                if filing_document:
                    break
        
        if not filing_document:
            return False, None, f"No filing document found in {latest_folder}"
        
        # Read the content of the filing
        try:
            with open(filing_document, 'r', encoding='utf-8') as f:
                content = f.read()
            return True, filing_document, content
        except Exception as e:
            return False, filing_document, f"Error reading file: {e}"
            
    except Exception as e:
        return False, None, f"Error downloading {filing_type} for {ticker}: {e}"
