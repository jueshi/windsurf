import os
import sys
from dotenv import load_dotenv
from sec_edgar_downloader import Downloader

def main():
    """Test downloading ADI's 10-K filing directly using sec_edgar_downloader"""
    # Load environment variables
    load_dotenv()
    sec_email = os.getenv("SEC_EDGAR_EMAIL")
    print(f"SEC_EDGAR_EMAIL: {'Set' if sec_email else 'Not set'}")
    
    if not sec_email:
        print("ERROR: SEC_EDGAR_EMAIL environment variable is not set.")
        print("Please set it in your .env file.")
        return 1
    
    # Initialize downloader with company name and email
    print("Initializing SEC EDGAR downloader...")
    dl = Downloader("Stone & Associates Inc", sec_email)
    
    # Download ADI's latest 10-K filing
    ticker = "ADI"
    filing_type = "10-K"
    print(f"Downloading latest {filing_type} filing for {ticker}...")
    
    try:
        # Download the filing
        result = dl.get(filing_type, ticker, limit=1)
        print(f"Download result: {result}")
        
        # Try to find the downloaded files
        print("\nSearching for downloaded files...")
        
        # Try multiple possible locations for the sec-edgar-filings directory
        possible_paths = [
            # Current working directory
            os.path.join(os.getcwd(), "sec-edgar-filings"),
            # Parent directory of current working directory
            os.path.join(os.path.dirname(os.getcwd()), "sec-edgar-filings"),
            # User's Documents directory
            os.path.join(os.path.expanduser("~"), "OneDrive", "Documents", "windsurf", "sec-edgar-filings"),
            # Absolute path from error message
            os.path.join("C:", "Users", "juesh", "OneDrive", "Documents", "windsurf", "sec-edgar-filings"),
            # Two levels up from current directory
            os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "sec-edgar-filings")
        ]
        
        # Check each possible path
        for base_path in possible_paths:
            print(f"Checking: {base_path}")
            if os.path.exists(base_path):
                print(f"Found SEC filings base directory: {base_path}")
                
                # Check for ticker directory
                ticker_path = os.path.join(base_path, ticker)
                if os.path.exists(ticker_path):
                    print(f"Found ticker directory: {ticker_path}")
                    
                    # Check for filing type directory
                    filing_path = os.path.join(ticker_path, filing_type)
                    if os.path.exists(filing_path):
                        print(f"Found filing type directory: {filing_path}")
                        
                        # List contents of filing type directory
                        print(f"\nListing contents of {filing_path}:")
                        for item in os.listdir(filing_path):
                            print(f" - {item}")
                            
                            # If it's a directory, check its contents
                            item_path = os.path.join(filing_path, item)
                            if os.path.isdir(item_path):
                                print(f"   Contents of {item}:")
                                for subitem in os.listdir(item_path):
                                    print(f"   - {subitem}")
                                    
                                    # If it's a text file, show a preview
                                    subitem_path = os.path.join(item_path, subitem)
                                    if subitem.endswith('.txt') and os.path.isfile(subitem_path):
                                        try:
                                            with open(subitem_path, 'r', encoding='utf-8') as f:
                                                content = f.read(500)  # Read first 500 characters
                                                print("\nPreview of downloaded content:")
                                                print("-" * 80)
                                                print(content + "...")
                                                print("-" * 80)
                                        except Exception as e:
                                            print(f"Error reading file: {e}")
        
        return 0
    except Exception as e:
        print(f"Error downloading {filing_type} for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
