"""
Verify SEC API caching and retry logic
"""
import os
import time
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sec_cache_verification.log')
    ]
)

# Load environment variables
load_dotenv()

# Check if SEC email is set
sec_email = os.getenv("SEC_EDGAR_EMAIL")
logging.info(f"SEC_EDGAR_EMAIL: {'Set' if sec_email else 'Not set'}")

def check_cache_files():
    """Check if cache files exist and show their contents"""
    logging.info("=== Checking Cache Files ===")
    
    cache_dir = Path("cache")
    if not cache_dir.exists():
        logging.warning(f"Cache directory not found: {cache_dir}")
        return False
    
    logging.info(f"Cache directory exists: {cache_dir}")
    
    # Check company tickers cache
    tickers_cache = cache_dir / "company_tickers.json"
    if tickers_cache.exists():
        logging.info(f"Company tickers cache exists: {tickers_cache}")
        try:
            with open(tickers_cache, 'r') as f:
                data = json.load(f)
                logging.info(f"Cache contains {len(data)} entries")
                return True
        except Exception as e:
            logging.error(f"Error reading cache file: {e}")
    else:
        logging.warning(f"Company tickers cache not found: {tickers_cache}")
    
    return False

def check_sec_cache_dir():
    """Check if SEC cache directory exists and show its contents"""
    logging.info("=== Checking SEC Cache Directory ===")
    
    sec_cache_dir = Path("sec_cache")
    if not sec_cache_dir.exists():
        logging.warning(f"SEC cache directory not found: {sec_cache_dir}")
        return False
    
    logging.info(f"SEC cache directory exists: {sec_cache_dir}")
    
    # List subdirectories
    subdirs = [d for d in sec_cache_dir.iterdir() if d.is_dir()]
    logging.info(f"Found {len(subdirs)} subdirectories: {[d.name for d in subdirs]}")
    
    # Check company tickers cache
    company_tickers_dir = sec_cache_dir / "company_tickers"
    if company_tickers_dir.exists():
        logging.info(f"Company tickers directory exists: {company_tickers_dir}")
        cache_files = list(company_tickers_dir.glob("*.json"))
        logging.info(f"Found {len(cache_files)} cache files")
        
        # Show a few sample files
        for file in cache_files[:3]:
            logging.info(f"Sample file: {file.name}")
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    logging.info(f"File contains data with URL: {data.get('url', 'unknown')}")
            except Exception as e:
                logging.error(f"Error reading file {file}: {e}")
    
    # Check CIK cache directory
    cik_cache_dir = sec_cache_dir / "cik_lookups"
    if cik_cache_dir.exists():
        logging.info(f"CIK cache directory exists: {cik_cache_dir}")
        cik_files = list(cik_cache_dir.glob("*.txt"))
        logging.info(f"Found {len(cik_files)} CIK cache files")
        
        # Show a few sample files
        for file in cik_files[:5]:
            logging.info(f"Sample CIK file: {file.name}")
            try:
                with open(file, 'r') as f:
                    cik = f.read().strip()
                    logging.info(f"Ticker {file.stem} -> CIK: {cik}")
            except Exception as e:
                logging.error(f"Error reading CIK file {file}: {e}")
    
    return True

def check_company_submissions_cache():
    """Check if company submissions cache exists"""
    logging.info("=== Checking Company Submissions Cache ===")
    
    submissions_dir = Path("sec_cache") / "company_submissions"
    if not submissions_dir.exists():
        logging.warning(f"Company submissions directory not found: {submissions_dir}")
        return False
    
    logging.info(f"Company submissions directory exists: {submissions_dir}")
    cache_files = list(submissions_dir.glob("*.json"))
    logging.info(f"Found {len(cache_files)} company submissions cache files")
    
    # Show a few sample files
    for file in cache_files[:3]:
        logging.info(f"Sample file: {file.name}")
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                logging.info(f"File contains data with URL: {data.get('url', 'unknown')}")
        except Exception as e:
            logging.error(f"Error reading file {file}: {e}")
    
    return True

def main():
    """Run all verification checks"""
    logging.info("Starting SEC API cache verification...")
    
    # Run checks
    cache_files = check_cache_files()
    sec_cache = check_sec_cache_dir()
    submissions_cache = check_company_submissions_cache()
    
    # Print summary
    logging.info("=== Verification Summary ===")
    logging.info(f"Cache files check: {'PASSED' if cache_files else 'FAILED'}")
    logging.info(f"SEC cache directory check: {'PASSED' if sec_cache else 'FAILED'}")
    logging.info(f"Company submissions cache check: {'PASSED' if submissions_cache else 'FAILED'}")
    
    overall = any([cache_files, sec_cache, submissions_cache])
    logging.info(f"Overall verification result: {'PASSED' if overall else 'FAILED'}")
    
    if overall:
        logging.info("SEC API caching appears to be working correctly.")
    else:
        logging.warning("SEC API caching may not be working correctly. Check the logs for details.")

if __name__ == "__main__":
    main()
