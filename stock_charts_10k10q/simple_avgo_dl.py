import os
from dotenv import load_dotenv
from sec_edgar_downloader import Downloader

# Load environment variables
load_dotenv()
sec_email = os.getenv("SEC_EDGAR_EMAIL")
print(f"SEC_EDGAR_EMAIL: {'Set' if sec_email else 'Not set'}")

# Initialize downloader with company name and email
dl = Downloader("Stone & Associates Inc", sec_email)

# Download AVGO's latest 10-K filing
print(f"Downloading latest 10-K filing for AVGO...")
result = dl.get("10-K", "AVGO", limit=1)
print(f"Download result: {result}")

# Print download location
print(f"Files downloaded to: {dl.get_download_folder()}")
