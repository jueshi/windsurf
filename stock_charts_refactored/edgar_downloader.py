import os
from sec_edgar_downloader import Downloader

def download_latest_10k(ticker, email_address):
    """
    Downloads the latest 10-K filing for a given ticker.

    Args:
        ticker (str): The stock ticker symbol.
        email_address (str): Your email address to be used as the user agent.

    Returns:
        str: The path to the downloaded 10-K filing, or None if it fails.
    """
    try:
        # Initialize the downloader
        dl = Downloader("MyCompanyName", email_address)

        # Get the latest 10-K filing
        # The library saves the filing to a directory structure like:
        # sec-edgar-filings/{ticker}/10-K/{accession_number}/full-submission.txt
        dl.get("10-K", ticker, limit=1)

        # Find the path to the downloaded file
        filings_dir = os.path.join("sec-edgar-filings", ticker, "10-K")
        if not os.path.exists(filings_dir):
            return None

        # Get the latest filing by sorting the directories by name
        latest_filing_dir = sorted(os.listdir(filings_dir))[-1]
        filing_path = os.path.join(filings_dir, latest_filing_dir, "full-submission.txt")

        if os.path.exists(filing_path):
            return filing_path
        else:
            return None

    except Exception as e:
        print(f"An error occurred while downloading the 10-K filing: {e}")
        return None
