import os
import sys
from dotenv import load_dotenv
from sec_edgar_downloader import Downloader

def main():
    """Test downloading AVGO's 10-K filing using sec_edgar_downloader"""
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
    
    # Download AVGO's latest 10-K filing
    ticker = "AVGO"
    filing_type = "10-K"
    print(f"Downloading latest {filing_type} filing for {ticker}...")
    
    try:
        result = dl.get(filing_type, ticker, limit=1)
        print(f"Download result: {result}")
        
        # Get the path to the downloaded files
        download_folder = dl.get_download_folder()
        print(f"Files downloaded to: {download_folder}")
        
        # List downloaded files
        filing_path = os.path.join(download_folder, "sec-edgar-filings", ticker, filing_type)
        if os.path.exists(filing_path):
            print(f"\nListing files in {filing_path}:")
            for item in os.listdir(filing_path):
                print(f" - {item}")
            
            # Find the most recent filing folder
            latest_folder = None
            latest_time = 0
            for item in os.listdir(filing_path):
                item_path = os.path.join(filing_path, item)
                if os.path.isdir(item_path):
                    folder_time = os.path.getmtime(item_path)
                    if folder_time > latest_time:
                        latest_time = folder_time
                        latest_folder = item_path
            
            if latest_folder:
                print(f"\nMost recent filing folder: {os.path.basename(latest_folder)}")
                print(f"Listing files in {latest_folder}:")
                for item in os.listdir(latest_folder):
                    print(f" - {item}")
                
                # Find and display a text file if available
                for root, _, files in os.walk(latest_folder):
                    for file in files:
                        if file.endswith('.txt'):
                            file_path = os.path.join(root, file)
                            print(f"\nFound text file: {file_path}")
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read(1000)  # Read first 1000 characters
                                    print("\nPreview of downloaded content:")
                                    print("-" * 80)
                                    print(content + "...")
                                    print("-" * 80)
                            except Exception as e:
                                print(f"Error reading file: {e}")
                            break
            else:
                print("No filing folders found.")
        else:
            print(f"No files found in {filing_path}")
        
        return 0
    except Exception as e:
        print(f"Error downloading {filing_type} for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
