#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SEC API Cache module for handling SEC EDGAR API requests with caching and rate limiting

This is the central module for all SEC API interactions. Other modules should import
from here rather than implementing their own SEC API calls.
"""

import os
import json
import time
import random
import shutil
import requests
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# Custom Exceptions for SEC API
# =============================================================================

class SECAPIError(Exception):
    """Base exception for SEC API errors"""
    pass

class SECRateLimitError(SECAPIError):
    """Raised when rate limited by SEC (403 or 429 status)"""
    pass

class SECNotFoundError(SECAPIError):
    """Raised when ticker or filing not found"""
    pass

class SECNetworkError(SECAPIError):
    """Raised on network failures (connection, timeout, SSL)"""
    pass

class SECParseError(SECAPIError):
    """Raised when response cannot be parsed"""
    pass

# User-friendly error messages
ERROR_MESSAGES = {
    SECRateLimitError: "SEC is temporarily limiting requests. Please wait a few minutes and try again.",
    SECNotFoundError: "Could not find the requested data in SEC database. Please verify the ticker symbol.",
    SECNetworkError: "Network error connecting to SEC. Please check your internet connection.",
    SECParseError: "Error parsing SEC response. The filing format may have changed.",
}

def get_user_friendly_message(exception):
    """Get user-friendly message for an exception"""
    for exc_type, message in ERROR_MESSAGES.items():
        if isinstance(exception, exc_type):
            return message
    return str(exception)

# =============================================================================
# Configuration
# =============================================================================

# Configure cache directory (can be overridden via environment variable)
CACHE_DIR = Path(os.getenv("SEC_CACHE_DIR", "sec_cache"))
CACHE_DIR.mkdir(exist_ok=True)

# Cache expiration (in days)
CACHE_EXPIRATION = {
    "company_tickers": 7,  # Company tickers cache expires after 7 days
    "company_submissions": 1,  # Company submissions cache expires after 1 day
    "filing_document": 30,  # Filing documents cache for 30 days
}

# Rate limiting configuration - SEC is very strict, use conservative values
MIN_DELAY = int(os.getenv("SEC_MIN_DELAY", "12"))  # SEC recommends 10 seconds, use 12 for safety
MAX_RETRIES = 5  # Maximum retry attempts
BASE_BACKOFF = 15  # Base backoff time in seconds (increased from 10)
MAX_BACKOFF = 300  # Maximum backoff time (5 minutes)
RATE_LIMIT_BACKOFF = 60  # Wait 60 seconds after a 403 error

# Track last request time (shared across all SEC requests)
last_request_time = 0
_rate_limit_lock = None  # Will be initialized on first use

def _get_rate_limit_lock():
    """Get or create the rate limit lock for thread safety"""
    global _rate_limit_lock
    if _rate_limit_lock is None:
        import threading
        _rate_limit_lock = threading.Lock()
    return _rate_limit_lock

# Global cache for company tickers to avoid repeated lookups
company_tickers_cache = None
company_tickers_last_update = None

# =============================================================================
# Standardized Headers (single source of truth)
# =============================================================================

# Standard headers for all SEC requests
# NOTE: SEC blocks browser-like User-Agents with 403!
# Must use simple User-Agent with app name and email
SEC_HEADERS_TEMPLATE = {
    "User-Agent": "StockAnalyzer user@example.com",  # Will be overridden by get_headers()
    "Accept": "application/json, text/html, */*",
    "Accept-Encoding": "gzip, deflate",
}

# =============================================================================
# Cache Management Functions
# =============================================================================

def clear_all_cache(include_memory=True):
    """
    Clear all SEC API cache (both file and optionally in-memory)
    
    Args:
        include_memory (bool): Also clear in-memory cache
        
    Returns:
        bool: True if successful
    """
    global company_tickers_cache, company_tickers_last_update
    
    # Clear in-memory cache
    if include_memory:
        company_tickers_cache = None
        company_tickers_last_update = None
        print("Cleared in-memory SEC cache")
    
    # Clear file cache
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print(f"Removed cache directory: {CACHE_DIR}")
    
    CACHE_DIR.mkdir(exist_ok=True)
    print(f"Recreated cache directory: {CACHE_DIR}")
    
    return True

def configure_cache(cache_dir=None, min_delay=None):
    """
    Configure cache settings
    
    Args:
        cache_dir (str): Path to cache directory
        min_delay (int): Minimum delay between requests in seconds
    """
    global CACHE_DIR, MIN_DELAY
    
    if cache_dir:
        CACHE_DIR = Path(cache_dir)
        CACHE_DIR.mkdir(exist_ok=True)
        print(f"SEC cache directory set to: {CACHE_DIR}")
    
    if min_delay is not None:
        MIN_DELAY = min_delay
        print(f"SEC minimum delay set to: {MIN_DELAY}s")

def get_headers():
    """
    Get headers for SEC EDGAR API requests.
    
    SEC requires a User-Agent with company/app name and email.
    Browser-like headers cause 403 errors!
    """
    # Try to get email from environment variable
    email = os.getenv("SEC_EDGAR_EMAIL", "jueshi@gmail.com")
    
    # SEC-compliant headers - simple User-Agent with email
    # DO NOT use browser-like headers - SEC blocks them with 403!
    headers = {
        "User-Agent": f"StockAnalyzer {email}",
        "Accept": "application/json, text/html, */*",
        "Accept-Encoding": "gzip, deflate",
    }
    
    return headers

def get_cache_path(url, cache_type):
    """
    Get cache file path for a URL
    
    Args:
        url (str): URL to cache
        cache_type (str): Type of cache (company_tickers, company_submissions, filing_document)
        
    Returns:
        Path: Path object for cache file
    """
    # Create hash of URL for filename
    url_hash = hashlib.md5(url.encode()).hexdigest()
    
    # Create subdirectory for cache type
    cache_subdir = CACHE_DIR / cache_type
    cache_subdir.mkdir(exist_ok=True)
    
    return cache_subdir / f"{url_hash}.json"

def is_cache_valid(cache_path, cache_type):
    """
    Check if cache is valid (exists and not expired)
    
    Args:
        cache_path (Path): Path to cache file
        cache_type (str): Type of cache
        
    Returns:
        bool: True if cache is valid, False otherwise
    """
    if not cache_path.exists():
        return False
    
    # Check if cache has expired
    expiration_days = CACHE_EXPIRATION.get(cache_type, 1)  # Default to 1 day
    
    # Get file modification time
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    
    # Check if file is older than expiration
    return datetime.now() - mtime < timedelta(days=expiration_days)

def save_to_cache(url, data, cache_type):
    """
    Save data to cache
    
    Args:
        url (str): URL being cached
        data: Data to cache (will be JSON serialized)
        cache_type (str): Type of cache
    """
    cache_path = get_cache_path(url, cache_type)
    
    # Save data with metadata
    cache_data = {
        "url": url,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f)
    
    print(f"Cached data for {url} at {cache_path}")

def load_from_cache(url, cache_type):
    """
    Load data from cache
    
    Args:
        url (str): URL to load from cache
        cache_type (str): Type of cache
        
    Returns:
        Data from cache or None if not found/expired
    """
    cache_path = get_cache_path(url, cache_type)
    
    if not is_cache_valid(cache_path, cache_type):
        return None
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        print(f"Loaded from cache: {url}")
        return cache_data["data"]
    except Exception as e:
        print(f"Error loading from cache: {e}")
        return None

def make_sec_request(url, cache_type, max_retries=MAX_RETRIES, force_refresh=False):
    """
    Make a request to SEC API with caching and rate limiting
    
    Args:
        url (str): URL to request
        cache_type (str): Type of cache
        max_retries (int): Maximum number of retries
        force_refresh (bool): Force refresh cache
        
    Returns:
        Response data or None if all retries fail
    """
    global last_request_time  # Need global to modify the module-level variable
    
    # Check cache first (unless force refresh)
    if not force_refresh:
        cached_data = load_from_cache(url, cache_type)
        if cached_data is not None:
            return cached_data
    
    # Get headers for request
    headers = get_headers()
    print(f"Requesting URL: {url}")
    
    # Use lock for thread-safe rate limiting
    lock = _get_rate_limit_lock()
    
    # Try request with exponential backoff
    for attempt in range(max_retries):
        try:
            # Thread-safe rate limiting
            with lock:
                current_time = time.time()
                time_since_last_request = current_time - last_request_time
                
                if time_since_last_request < MIN_DELAY:
                    wait_time = MIN_DELAY - time_since_last_request + random.uniform(2, 5)
                    print(f"Waiting {wait_time:.2f} seconds to respect SEC rate limits...")
                    time.sleep(wait_time)
                
                # Update last request time before releasing lock
                last_request_time = time.time()
            
            # Calculate backoff with exponential increase and jitter
            if attempt > 0:
                # Exponential backoff with jitter - more aggressive for 403 errors
                max_jitter = min(BASE_BACKOFF * (2 ** attempt), MAX_BACKOFF)
                delay = max_jitter * 0.5 + max_jitter * 0.5 * random.random()
                print(f"Retry attempt {attempt+1}/{max_retries}. Waiting {delay:.2f} seconds...")
                time.sleep(delay)
            
            # Make request with a longer timeout
            response = requests.get(url, headers=headers, timeout=120)
            
            if response.status_code == 200:
                # Parse response based on content type
                if response.headers.get('Content-Type', '').startswith('application/json'):
                    data = response.json()
                else:
                    data = response.text
                
                # Cache successful response
                save_to_cache(url, data, cache_type)
                
                return data
            elif response.status_code == 403:
                print(f"Rate limit exceeded (403). Waiting {RATE_LIMIT_BACKOFF}s before retry...")
                # Force a much longer wait time for rate limit errors
                time.sleep(RATE_LIMIT_BACKOFF)
                # Update last_request_time to prevent immediate retry from other threads
                with lock:
                    last_request_time = time.time()
            elif response.status_code == 429:
                print(f"Too many requests (429). Waiting {RATE_LIMIT_BACKOFF * 2}s before retry...")
                # Force an even longer wait time for explicit rate limiting
                time.sleep(RATE_LIMIT_BACKOFF * 2)
                with lock:
                    last_request_time = time.time()
            else:
                print(f"Error: Status code {response.status_code}")
                print(f"Response: {response.text[:500]}...")
                # For non-rate-limit errors, we might still retry but with shorter backoff
                time.sleep(MIN_DELAY)
        
        except Exception as e:
            print(f"Request error: {str(e)}")
            # Continue to retry
    
    print(f"Failed after {max_retries} attempts")
    return None

def get_company_tickers(force_refresh=False):
    """
    Get all company tickers from SEC with in-memory caching
    
    Args:
        force_refresh (bool): Force refresh the cache
        
    Returns:
        dict: Company tickers data or None if request fails
    """
    global company_tickers_cache, company_tickers_last_update
    
    # Check if we have a valid in-memory cache
    current_time = datetime.now()
    cache_valid = (
        company_tickers_cache is not None and 
        company_tickers_last_update is not None and
        (current_time - company_tickers_last_update).days < CACHE_EXPIRATION["company_tickers"] and
        not force_refresh
    )
    
    if cache_valid:
        print("Using in-memory company tickers cache")
        return company_tickers_cache
    
    # If not in memory, try to get from file cache or make request
    url = "https://www.sec.gov/files/company_tickers.json"
    data = make_sec_request(url, "company_tickers", force_refresh=force_refresh)
    
    # Update in-memory cache if request was successful
    if data is not None:
        company_tickers_cache = data
        company_tickers_last_update = current_time
    
    return data

def get_company_cik(ticker):
    """
    Get company CIK number from ticker
    
    Args:
        ticker (str): Company ticker symbol
        
    Returns:
        str: CIK number (10 digits with leading zeros) or None if not found
    """
    print(f"Looking up CIK for {ticker}...")
    
    # Create cache directory for CIK lookups
    cik_cache_dir = CACHE_DIR / "cik_lookups"
    cik_cache_dir.mkdir(exist_ok=True)
    
    # Check if we have this ticker in a dedicated cache file
    ticker_cache_path = cik_cache_dir / f"{ticker.upper()}.txt"
    if ticker_cache_path.exists():
        try:
            with open(ticker_cache_path, 'r') as f:
                cached_cik = f.read().strip()
                if cached_cik:
                    print(f"Found CIK in dedicated cache: {cached_cik} for {ticker}")
                    return cached_cik
        except Exception as e:
            print(f"Error reading ticker cache: {e}")
    
    # Get all company tickers
    companies = get_company_tickers()
    
    if not companies:
        print("Failed to get company tickers")
        return None
    
    # Find the company by ticker
    for _, company in companies.items():
        if company["ticker"].upper() == ticker.upper():
            # Format CIK with leading zeros to 10 digits
            cik = str(company["cik_str"]).zfill(10)
            print(f"Found CIK: {cik} for {company['title']}")
            
            # Save to dedicated cache file for faster future lookups
            try:
                with open(ticker_cache_path, 'w') as f:
                    f.write(cik)
                print(f"Saved CIK to dedicated cache: {ticker_cache_path}")
            except Exception as e:
                print(f"Error saving to ticker cache: {e}")
                
            return cik
    
    print(f"Could not find CIK for ticker: {ticker}")
    return None

def get_company_submissions(cik):
    """
    Get company submissions from SEC
    
    Args:
        cik (str): Company CIK number (10 digits with leading zeros)
        
    Returns:
        dict: Company submissions data or None if request fails
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    return make_sec_request(url, "company_submissions")

def get_latest_filing_info(cik, form_type="10-K"):
    """
    Get the latest filing info for a company
    
    Args:
        cik (str): Company CIK number (10 digits with leading zeros)
        form_type (str): Form type to search for (10-K, 10-Q, etc.)
        
    Returns:
        dict: Filing information or None if not found
    """
    print(f"Finding latest {form_type} filing for CIK {cik}...")
    
    # Get company submissions
    submissions = get_company_submissions(cik)
    
    if not submissions:
        print("Failed to get company submissions")
        return None
    
    # Get recent filings
    recent_filings = submissions.get("filings", {}).get("recent", {})
    
    if not recent_filings:
        print("No recent filings found")
        return None
    
    # Get form types and filing dates
    form_types = recent_filings.get("form", [])
    filing_dates = recent_filings.get("filingDate", [])
    accession_numbers = recent_filings.get("accessionNumber", [])
    primary_documents = recent_filings.get("primaryDocument", [])
    
    if not form_types or not filing_dates or not accession_numbers:
        print("Missing filing data")
        return None
    
    # Find the latest filing of the specified form type
    latest_filing = None
    latest_date = None
    
    for i, form in enumerate(form_types):
        if form == form_type:
            filing_date = filing_dates[i]
            
            # Update if this is the first or a more recent filing
            if latest_date is None or filing_date > latest_date:
                latest_date = filing_date
                accession_number = accession_numbers[i].replace("-", "")
                
                latest_filing = {
                    "form": form,
                    "filingDate": filing_date,
                    "accessionNumber": accession_number,
                    "primaryDocument": primary_documents[i] if i < len(primary_documents) else "",
                    "detailUrl": f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_number}/{primary_documents[i]}" if i < len(primary_documents) else ""
                }
    
    if latest_filing:
        print(f"Found latest {form_type} filing from {latest_filing['filingDate']}")
        return latest_filing
    else:
        print(f"No {form_type} filings found")
        return None

def download_filing(filing_info):
    """
    Download the filing document
    
    Args:
        filing_info (dict): Filing information from get_latest_filing_info
        
    Returns:
        str: HTML content of the filing or None if download fails
    """
    if not filing_info or "detailUrl" not in filing_info:
        print("Invalid filing info")
        return None
    
    url = filing_info["detailUrl"]
    print(f"Downloading filing from {url}...")
    
    # Download filing with caching
    html_content = make_sec_request(url, "filing_document")
    
    if html_content:
        print(f"Successfully downloaded filing")
        return html_content
    else:
        print("Failed to download filing")
        return None

# Test function
def test_sec_api(ticker="AAPL", form_type="10-K"):
    """
    Test SEC API functions
    
    Args:
        ticker (str): Company ticker symbol
        form_type (str): Form type to search for
    """
    print(f"\n{'='*80}")
    print(f"Testing SEC API with {ticker} ({form_type})")
    print(f"{'='*80}\n")
    
    # Step 1: Get company CIK
    print("\nStep 1: Getting company CIK...")
    cik = get_company_cik(ticker)
    if not cik:
        print("Test failed: Could not get CIK")
        return False
    
    # Step 2: Get latest filing info
    print("\nStep 2: Getting latest filing info...")
    filing_info = get_latest_filing_info(cik, form_type)
    if not filing_info:
        print("Test failed: Could not get filing info")
        return False
    
    # Step 3: Download filing
    print("\nStep 3: Downloading filing...")
    html_content = download_filing(filing_info)
    if not html_content:
        print("Test failed: Could not download filing")
        return False
    
    print("\nAll tests passed successfully!")
    return True

if __name__ == "__main__":
    import sys
    
    ticker = "AAPL"
    form_type = "10-K"
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    if len(sys.argv) > 2:
        form_type = sys.argv[2]
    
    test_sec_api(ticker, form_type)
