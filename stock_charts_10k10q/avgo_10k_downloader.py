import os
import sys
import traceback
from dotenv import load_dotenv
from sec_edgar_downloader import Downloader
import time

# Check environment variables
load_dotenv()
sec_email = os.getenv("SEC_EDGAR_EMAIL")
print(f"SEC_EDGAR_EMAIL: {'Set' if sec_email else 'Not set'}")

if not sec_email:
    print("ERROR: SEC_EDGAR_EMAIL environment variable is not set.")
    print("Please set it in your .env file.")
    sys.exit(1)

# Initialize the downloader with your company name and email
# SEC requires the email for User-Agent header
dl = Downloader("Stone & Associates Inc", sec_email)

# Download AVGO's latest 10-K filing
print(f"Downloading latest 10-K filing for AVGO...")
try:
    result = dl.get("10-K", "AVGO", limit=1)
    print(f"Download result: {result}")
except Exception as e:
    print(f"Error downloading AVGO 10-K: {e}")
    traceback.print_exc()
    sys.exit(1)

# Get the path to the downloaded files
download_folder = dl.get_download_folder()
print(f"Files downloaded to: {download_folder}")

# List downloaded files
print("\nDownloaded files:")
avgo_path = os.path.join(download_folder, "sec-edgar-filings", "AVGO", "10-K")
if os.path.exists(avgo_path):
    for root, dirs, files in os.walk(avgo_path):
        for file in files:
            if file.endswith('.txt') or file.endswith('.html') or file.endswith('.htm'):
                file_path = os.path.join(root, file)
                print(f" - {file_path}")
                
                # Print first few lines of the first text file found
                if file.endswith('.txt'):
                    print("\nPreview of downloaded content:")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read(1000)  # Read first 1000 characters
                            print(content + "...\n")
                    except Exception as e:
                        print(f"Error reading file: {e}")
                    break
else:
    print(f"No files found in {avgo_path}")
