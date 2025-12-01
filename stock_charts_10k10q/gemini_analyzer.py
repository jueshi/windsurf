import os
import re
import json
import time
import requests
import logging
import google.generativeai as genai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pathlib import Path
from sec_edgar_downloader import Downloader
from lxml import html
from serpapi import GoogleSearch
from functools import wraps
from datetime import datetime, timedelta
from threading import Lock

# Helper to list accessible Gemini models that support generateContent
def _list_supported_gemini_models() -> list:
    """Return the Gemini model names accessible to the current API key."""
    try:
        models = genai.list_models()
    except Exception as err:
        logging.warning("Unable to list Gemini models: %s", err)
        return []

    supported = []
    for model in models:
        methods = getattr(model, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            supported.append(model.name)
    return supported


# Helper to format Gemini API errors with actionable guidance
def _format_gemini_error(e: Exception) -> str:
    """
    Return a user-friendly error message for common Gemini API failures.

    Specifically detects SERVICE_DISABLED (API not enabled) and provides
    activation guidance.
    """
    msg = str(e)
    # Common signals for API not enabled
    service_disabled_tokens = [
        "SERVICE_DISABLED",
        "Generative Language API has not been used",
        "API has not been used in project",
        "it is disabled"
    ]
    if any(tok in msg for tok in service_disabled_tokens):
        # Try to surface activation URL if present in the error payload
        activation_hint = "https://console.developers.google.com/apis/api/generativelanguage.googleapis.com/overview"
        # If a project number is embedded in the message, add it as a hint
        project_hint = ""
        import re as _re
        m = _re.search(r"project[s]?\/?(\d{6,})", msg)
        if m:
            project_hint = f"?project={m.group(1)}"
        return (
            "Gemini API is disabled for your Google Cloud project.\n\n"
            "How to fix:\n"
            "1) Open the Generative Language API page in Google Cloud Console:\n"
            f"   {activation_hint}{project_hint}\n"
            "2) Click Enable.\n"
            "3) Ensure Billing is enabled on the same project.\n"
            "4) Make sure the API key you are using (GEMINI_API_KEY) belongs to this project.\n"
            "5) Wait a few minutes for propagation, then retry.\n\n"
            f"Original error: {msg}"
        )
    unsupported_tokens = [
        "404",
        "not found",
        "NOT_FOUND",
        "not supported",
        "Unsupported",
    ]
    if any(tok in msg for tok in unsupported_tokens):
        available = _list_supported_gemini_models()
        if available:
            formatted = "\n".join(f"- {name}" for name in available)
            return (
                "Requested Gemini model is unavailable for this API key/version.\n\n"
                "Try one of the accessible models that support generateContent:\n"
                f"{formatted}\n\n"
                f"Original error: {msg}"
            )
        return (
            "Requested Gemini model is unavailable for this API key/version, and listing "
            "alternatives failed. Run `genai.list_models()` after configuring your API key "
            "to inspect supported models.\n\n"
            f"Original error: {msg}"
        )

    # Default: return the raw error message
    return f"An error occurred while communicating with the Gemini API: {msg}"

# Rate limiting decorator
def rate_limited(max_per_minute):
    """
    Decorator to limit the number of API calls per minute.
    
    Args:
        max_per_minute (int): Maximum number of API calls allowed per minute
    """
    interval = 60.0 / max_per_minute
    last_called = [0.0]  # Using list to make it mutable in the nested function
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = max(0, interval - elapsed)
            
            if wait_time > 0:
                logging.info(f"Rate limiting: Waiting {wait_time:.2f} seconds before next API call")
                time.sleep(wait_time)
            
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

class GeminiAnalyzer:
    # Class variable to track last API call time
    _last_api_call = 0
    MIN_API_INTERVAL = 2.0  # Minimum seconds between API calls (adjust as needed)
    
    @classmethod
    def _rate_limited_generate_content(cls, model, prompt):
        """Helper method to enforce rate limiting on API calls"""
        current_time = time.time()
        time_since_last_call = current_time - cls._last_api_call
        
        # Calculate how long to wait before making the next call
        wait_time = max(0, cls.MIN_API_INTERVAL - time_since_last_call)
        
        if wait_time > 0:
            logging.info(f"Rate limiting: Waiting {wait_time:.2f} seconds before next API call")
            time.sleep(wait_time)
        
        # Make the API call
        response = model.generate_content(prompt)
        
        # Update the last call time
        cls._last_api_call = time.time()
        
        return response

import random

class GeminiRateLimiter:
    _instance = None
    _lock = Lock()
    _last_call_time = 0
    MIN_INTERVAL = 10.0  # Increased: Minimum seconds between API calls for free tier
    MAX_RETRIES = 5      # Increased: Maximum number of retries for rate limit errors
    BASE_DELAY = 10.0    # Increased: Base delay in seconds for exponential backoff
    
    # Patterns that indicate rate limiting or quota exceeded
    RATE_LIMIT_PATTERNS = [
        "RATE_LIMIT_EXCEEDED",
        "429",
        "quota",
        "exceeded your current quota",
        "rate limit",
        "too many requests",
        "Resource has been exhausted",
    ]
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GeminiRateLimiter, cls).__new__(cls)
        return cls._instance
    
    def _is_rate_limit_error(self, error_msg: str) -> bool:
        """Check if an error message indicates a rate limit issue"""
        error_lower = error_msg.lower()
        return any(pattern.lower() in error_lower for pattern in self.RATE_LIMIT_PATTERNS)
    
    def _extract_retry_delay(self, error_msg: str) -> float:
        """Extract retry delay from error message if present"""
        import re
        # Look for patterns like "retry in 33.344336875s" or "retry_delay { seconds: 33 }"
        patterns = [
            r'retry in (\d+\.?\d*)s',
            r'retry_delay\s*\{\s*seconds:\s*(\d+)',
            r'Please retry in (\d+\.?\d*)',
        ]
        for pattern in patterns:
            match = re.search(pattern, error_msg, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None
    
    def wait_for_rate_limit(self):
        with self._lock:
            current_time = time.time()
            time_since_last_call = current_time - self._last_call_time
            wait_time = max(0, self.MIN_INTERVAL - time_since_last_call)
            
            if wait_time > 0:
                logging.info(f"Rate limiting: Waiting {wait_time:.2f} seconds before next API call")
                time.sleep(wait_time)
            
            self._last_call_time = time.time()
    
    def make_api_call_with_retry(self, api_call_func, *args, **kwargs):
        """
        Makes an API call with retry logic for rate limit errors.
        
        Args:
            api_call_func: The function that makes the API call
            *args: Positional arguments to pass to the API call function
            **kwargs: Keyword arguments to pass to the API call function
            
        Returns:
            The result of the API call
            
        Raises:
            Exception: If all retry attempts are exhausted
        """
        last_exception = None
        
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                # Apply rate limiting
                self.wait_for_rate_limit()
                
                # Make the API call
                return api_call_func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                error_msg = str(e)
                
                if self._is_rate_limit_error(error_msg) and attempt < self.MAX_RETRIES:
                    # Try to extract retry delay from error message
                    suggested_delay = self._extract_retry_delay(error_msg)
                    
                    if suggested_delay:
                        # Use the suggested delay plus a small buffer
                        delay = suggested_delay + 5.0
                        logging.warning(f"Rate limit/quota exceeded (attempt {attempt + 1}/{self.MAX_RETRIES}). "
                                     f"API suggested {suggested_delay:.1f}s, waiting {delay:.1f}s...")
                    else:
                        # Calculate exponential backoff with jitter
                        delay = self.BASE_DELAY * (2 ** attempt) * (0.5 + random.random())
                        logging.warning(f"Rate limit/quota exceeded (attempt {attempt + 1}/{self.MAX_RETRIES}). "
                                     f"Retrying in {delay:.2f} seconds...")
                    
                    time.sleep(delay)
                    continue
                elif attempt >= self.MAX_RETRIES:
                    logging.error(f"Max retries ({self.MAX_RETRIES}) exceeded. Last error: {error_msg}")
                else:
                    # For non-rate-limit errors, re-raise immediately
                    raise
        
        # If we get here, all retries were exhausted
        raise last_exception or Exception("Unknown error in API call with retry")

# ----- Gemini model selection helpers -----
def _get_gemini_model_candidates() -> list:
    """Build a list of Gemini model candidates using env override and sensible fallbacks."""
    env_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    candidates = [env_name, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-8b"]
    # De-duplicate while preserving order
    seen = set()
    ordered = []
    for n in candidates:
        if n and n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered

def _init_gemini_model_with_fallback() -> genai.GenerativeModel:
    """Try to initialize a GenerativeModel across known candidates; fall back on 404/unsupported."""
    last_err = None
    for name in _get_gemini_model_candidates():
        try:
            return genai.GenerativeModel(name)
        except Exception as e:
            last_err = e
            msg = str(e)
            if any(tok in msg for tok in ["404", "not found", "NOT_FOUND", "not supported", "Unsupported"]):
                continue
            # Other errors (auth, service disabled) should bubble for better guidance
            raise
    # If none worked, raise the last error for formatting upstream
    raise last_err or Exception("No supported Gemini model found")

def analyze_ticker(ticker, company_info):
    """
    Analyzes a stock ticker using Google Gemini API with rate limiting.

    Args:
        ticker (str): The stock ticker symbol.
        company_info (dict): A dictionary containing fundamental data about the company.

    Returns:
        str: The business analysis from Gemini API.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    # Initialize rate limiter
    rate_limiter = GeminiRateLimiter()
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    try:
        model = _init_gemini_model_with_fallback()
    except Exception as e:
        # If API is disabled or unauthorized, return actionable guidance
        return _format_gemini_error(e)


    prompt = f"""
    对以下公司进行详细的商业分析，公司股票代码为 '{ticker}'。
    这是该公司的一些基本数据：
    - **公司名称:** {company_info.get('longName', 'N/A')}
    - **行业板块:** {company_info.get('sector', 'N/A')}
    - **具体行业:** {company_info.get('industry', 'N/A')}
    - **市值:** {company_info.get('marketCap', 'N/A')}
    - **市盈率（过去12个月）:** {company_info.get('trailingPE', 'N/A')}
    - **远期市盈率:** {company_info.get('forwardPE', 'N/A')}
    - **股息率:** {company_info.get('dividendYield', 'N/A')}
    - **贝塔系数:** {company_info.get('beta', 'N/A')}
    - **52周最高价:** {company_info.get('fiftyTwoWeekHigh', 'N/A')}
    - **52周最低价:** {company_info.get('fiftyTwoWeekLow', 'N/A')}
    - **业务摘要:** {company_info.get('longBusinessSummary', 'N/A')}

    请提供一份结构良好、详细的中文商业分析，涵盖以下方面：
    1.  **商业模式:** 描述公司的主要商业模式及其收入来源。
    2.  **竞争格局:** 主要竞争对手是谁？这家公司的竞争优势是什么？
    3.  **财务状况:** 根据所提供的指标，简要评估公司的财务状况。
    4.  **增长前景:** 这家公司潜在的增长动力是什么？
    5.  **潜在风险:** 与这家公司相关的主要风险是什么？

    请用Chinese提供结构良好且详细的分析。Followed by an English version of the response in a separate paragraph as well.
    """

    try:
        # Use make_api_call_with_retry to handle 429 quota errors automatically
        response = rate_limiter.make_api_call_with_retry(
            model.generate_content, prompt
        )
        return response.text
    except Exception as e:
        return _format_gemini_error(e)

def _get_filing_url(ticker, filing_type):
    """
    Retrieves the latest SEC filing URL for a given ticker using SEC EDGAR API.
    Includes improved error handling for 403 Forbidden errors.
    """
    print(f"\n=== Starting _get_filing_url for {ticker} {filing_type} ===")
    ticker = ticker.upper()
    filing_type = filing_type.upper()
    
    # First try the SEC EDGAR API approach
    try:
        print(f"Retrieving {filing_type} filing for {ticker} using SEC EDGAR API...")
        
        # Step 1: Get CIK number for the ticker
        cik_url = f"https://www.sec.gov/files/company_tickers.json"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Referer': 'https://www.sec.gov/edgar/searchedgar/companysearch'
        }
        
        # Add email to headers as required by SEC
        load_dotenv()
        email = os.getenv("SEC_EDGAR_EMAIL")
        if email:
            headers['From'] = email
        else:
            print("Warning: SEC_EDGAR_EMAIL not set in environment variables. SEC API access may fail.")
        
        try:
            response = requests.get(cik_url, headers=headers, timeout=30)
            response.raise_for_status()
            companies = response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"403 Forbidden error accessing SEC API. This may be due to rate limiting or missing/invalid email header.")
                print(f"Using email: {email if email else 'Not set'}")
                # Try the SerpAPI approach as fallback
                return _get_filing_url_via_serpapi(ticker, filing_type)
            else:
                raise
        
        cik = None
        for _, company in companies.items():
            if company['ticker'] == ticker:
                # Format CIK with leading zeros to 10 digits
                cik = str(company['cik_str']).zfill(10)
                break
        
        if not cik:
            print(f"Could not find CIK for ticker {ticker}")
            # Try the SerpAPI approach as fallback
            return _get_filing_url_via_serpapi(ticker, filing_type)
        
        print(f"Found CIK for {ticker}: {cik}")
        
        # Step 2: Get the company's submissions feed
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
        
        # Add email to headers as required by SEC
        if email:
            headers['From'] = email
        
        try:
            response = requests.get(submissions_url, headers=headers, timeout=30)
            response.raise_for_status()
            submissions = response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"403 Forbidden error accessing SEC submissions API. This may be due to rate limiting or missing/invalid email header.")
                # Try the SerpAPI approach as fallback
                return _get_filing_url_via_serpapi(ticker, filing_type)
            else:
                raise
        
        # Step 3: Find the latest filing of the requested type
        recent_filings = submissions.get('filings', {}).get('recent', {})
        if not recent_filings:
            print(f"No recent filings found for {ticker}")
            return _get_filing_url_via_serpapi(ticker, filing_type)
        
        form_types = recent_filings.get('form', [])
        accession_numbers = recent_filings.get('accessionNumber', [])
        
        # Find the latest filing of the requested type
        for i, form in enumerate(form_types):
            if filing_type in form:
                accession_number = accession_numbers[i].replace('-', '')
                
                # Construct the index URL for this filing
                index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{accession_numbers[i]}-index.htm"
                print(f"Found {filing_type} filing: {index_url}")
                return index_url
        
        print(f"No {filing_type} filings found for {ticker}")
        
    except requests.exceptions.RequestException as e:
        print(f"Request error retrieving SEC filing via API: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"Error retrieving SEC filing via API: {e}")
        import traceback
        traceback.print_exc()
    
    # If we get here, try the SerpAPI approach
    return _get_filing_url_via_serpapi(ticker, filing_type)

def _get_filing_url_via_serpapi(ticker, filing_type):
    """
    Searches for the latest SEC filing URL for a given ticker using SerpApi as a fallback.
    """
    # Load API key from environment variables
    load_dotenv()
    api_key = os.getenv("SERPAPI_API_KEY")
    
    if not api_key:
        print("Warning: SERPAPI_API_KEY not found in environment variables.")
        # Return a direct link to the SEC company search page
        ticker = ticker.upper()
        return f"https://www.sec.gov/edgar/browse/?CIK={ticker}&owner=exclude"
    
    # Prepare search query
    query = f"{ticker} {filing_type} filing site:sec.gov"
    print(f"Searching with query via SerpAPI: {query}")
    
    try:
        # Setup SerpApi search parameters
        params = {
            "q": query,
            "api_key": api_key,
            "num": "5",  # Number of results
        }
        
        # Execute search
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Extract organic results
        if "organic_results" in results:
            for result in results["organic_results"]:
                url = result.get("link")
                if url and (".htm" in url) and ("Archives/edgar" in url):
                    print(f"Found potential filing URL via SerpAPI: {url}")
                    return url
        
        print("No suitable filing URL found in search results.")
    except Exception as e:
        print(f"An error occurred during SerpAPI search: {e}")
    
    # Final fallback - return a direct link to the SEC company search page
def _get_text_from_url(url):
    """
    Extracts plain text from a URL, handling SEC EDGAR index pages.
    Includes improved error handling for 403 Forbidden errors.
    """
    try:
        print(f"\n=== Starting text extraction from URL: {url} ===")
        
        # Check if the URL is an SEC EDGAR index page
        if "-index.htm" in url:
            print("This is an index page, finding the actual document...")
            
            # First, get the index page to find the actual document link
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml',
                'Accept-Language': 'en-US,en;q=0.9',
                'Connection': 'keep-alive',
                'Referer': 'https://www.sec.gov/edgar/browse/'
            }
            
            # Add email to headers as required by SEC
            load_dotenv()
            email = os.getenv("SEC_EDGAR_EMAIL")
            if email:
                headers['From'] = email
                print(f"Using email header: {email}")
            else:
                print("WARNING: SEC_EDGAR_EMAIL not set in environment variables. SEC API access may fail.")
            
            try:
                print(f"Requesting index page: {url}")
                response = requests.get(url, headers=headers, timeout=30)
                print(f"Index page response status: {response.status_code}")
                response.raise_for_status()
                
                # Parse the index page to find the actual document link
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for document links in the table
                document_links = []
                
                # First try to find the table with filing documents
                filing_table = None
                for table in soup.find_all('table'):
                    if table.find('th', text=lambda t: t and 'Document' in t):
                        filing_table = table
                        break
                
                if filing_table:
                    print("Found filing table, extracting document links...")
                    # Extract links from the filing table
                    for row in filing_table.find_all('tr'):
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            a_tag = cells[2].find('a') if len(cells) > 2 else cells[0].find('a')
                            if a_tag and a_tag.get('href'):
                                href = a_tag.get('href')
                                if '.htm' in href and 'index' not in href:
                                    # Found a potential document link
                                    if href.startswith('/'):
                                        document_links.append(f"https://www.sec.gov{href}")
                                    elif href.startswith('http'):
                                        document_links.append(href)
                                    else:
                                        # Relative link
                                        base_url = url[:url.rfind('/')]
                                        document_links.append(f"{base_url}/{href}")
                
                if not document_links:
                    print("No links found in table, searching all links...")
                    # If no links found in table, try all links
                    for a_tag in soup.find_all('a'):
                        href = a_tag.get('href')
                        if href and ('.htm' in href) and ('index' not in href):
                            # Found a potential document link
                            if href.startswith('/'):
                                document_links.append(f"https://www.sec.gov{href}")
                            elif href.startswith('http'):
                                document_links.append(href)
                            else:
                                # Relative link
                                base_url = url[:url.rfind('/')]
                                document_links.append(f"{base_url}/{href}")
                
                if document_links:
                    print(f"Found {len(document_links)} document links")
                    
                    # Try to find the main document (10-K or 10-Q)
                    main_doc_keywords = ['10-k', '10-q', '10k', '10q', 'form10-k', 'form10-q', 'form10k', 'form10q']
                    
                    # First priority: Look for the main document
                    for doc_url in document_links:
                        doc_lower = doc_url.lower()
                        if any(keyword in doc_lower for keyword in main_doc_keywords):
                            url = doc_url
                            print(f"Selected main document URL: {url}")
                            break
                    else:
                        # Second priority: Look for htm files that might be the main document
                        for doc_url in document_links:
                            if doc_url.lower().endswith('.htm') or doc_url.lower().endswith('.html'):
                                url = doc_url
                                print(f"Selected HTML document URL: {url}")
                                break
                        else:
                            # Last resort: use the first document
                            url = document_links[0]
                            print(f"Using first document URL: {url}")
                else:
                    print("No document links found in the index page")
                    return None
                    
            except requests.exceptions.HTTPError as e:
                print(f"HTTP Error accessing index page: {e}")
                if hasattr(e, 'response') and e.response.status_code == 403:
                    print("403 Forbidden error. This may be due to rate limiting or missing/invalid email header.")
                    print(f"Headers used: {headers}")
                return None
            except Exception as e:
                print(f"Error accessing index page: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        # Now fetch the actual document
        print(f"Fetching document from: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Referer': 'https://www.sec.gov/edgar/browse/'
        }
        
        # Add email to headers as required by SEC
        load_dotenv()
        email = os.getenv("SEC_EDGAR_EMAIL")
        if email:
            headers['From'] = email
            print(f"Using email header: {email}")
        
        try:
            print(f"Requesting document: {url}")
            response = requests.get(url, headers=headers, timeout=30)
            print(f"Document response status: {response.status_code}")
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove scripts, styles, and other non-content elements
            for element in soup(["script", "style", "meta", "link", "noscript"]):
                element.decompose()
            
            # Extract text
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up the text
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = '\n'.join(lines)
            
            print(f"Successfully extracted {len(text)} characters of text")
            return text
            
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error accessing document: {e}")
            if hasattr(e, 'response') and e.response.status_code == 403:
                print("403 Forbidden error. This may be due to rate limiting or missing/invalid email header.")
                print(f"Headers used: {headers}")
            return None
        except Exception as e:
            print(f"Error accessing document: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    except requests.exceptions.RequestException as e:
        print(f"Request error fetching URL {url}: {e}")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        import traceback
        traceback.print_exc()
        return None

def _download_sec_filing(ticker, filing_type):
    """
    Download SEC filing using sec_edgar_downloader package.
    Returns: (success, file_path, content, url)
    """
    print(f"Downloading {filing_type} filing for {ticker} using sec_edgar_downloader...")
    
    # Special case for ADI ticker - use the known path directly
    if ticker == "ADI" and filing_type == "10-K":
        known_path = os.path.join("C:", "Users", "juesh", "OneDrive", "Documents", "windsurf", 
                                "sec-edgar-filings", "ADI", "10-K", "0000006281-24-000204")
        if os.path.exists(known_path):
            print(f"Using known path for ADI 10-K: {known_path}")
            # Find the filing document (HTML or text file)
            filing_document = None
            for root, _, files in os.walk(known_path):
                for file in files:
                    if file.endswith('.txt'):
                        filing_document = os.path.join(root, file)
                        break
                if filing_document:
                    break
            
            if filing_document:
                try:
                    with open(filing_document, 'r', encoding='utf-8') as f:
                        content = f.read()
                    sec_url = f"https://www.sec.gov/edgar/browse/?CIK={ticker}&owner=exclude"
                    return True, filing_document, content, sec_url
                except Exception as e:
                    print(f"Error reading ADI filing: {e}")
                    return False, None, f"Error reading ADI filing: {e}", None

    # Load environment variables
    load_dotenv()
    sec_email = os.getenv("SEC_EDGAR_EMAIL")
    if not sec_email:
        return False, None, "SEC_EDGAR_EMAIL environment variable not set", None
    
    # Initialize downloader
    dl = Downloader("Stone & Associates Inc", sec_email)
    
    try:
        # Download the filing
        print(f"Downloading {filing_type} filing for {ticker} using sec_edgar_downloader...")
        result = dl.get(filing_type, ticker, limit=1)
        print(f"Download result: {result}")
        
        if result == 0:
            return False, None, f"No {filing_type} filings found for {ticker}", None
        
        # Get the download folder from the Downloader object
        # The download_folder attribute returns a WindowsPath object
        try:
            # Access the download_folder attribute (not a method)
            base_path = dl.download_folder
            print(f"Download folder from Downloader: {base_path}")
            
            # Convert WindowsPath to string if needed
            base_path_str = str(base_path)
            
            # Construct the path to the sec-edgar-filings directory
            sec_filings_path = os.path.join(base_path_str, "sec-edgar-filings")
            print(f"SEC filings path: {sec_filings_path}")
            
            # Construct the path to the ticker's filings
            filing_path = os.path.join(sec_filings_path, ticker, filing_type)
            print(f"Filing path: {filing_path}")
            
            # Check if the path exists
            if not os.path.exists(filing_path):
                print(f"Filing path does not exist: {filing_path}")
                
                # Try alternative paths
                alternative_paths = [
                    # Current working directory
                    os.path.join(os.getcwd(), "sec-edgar-filings", ticker, filing_type),
                    # Absolute path from error message
                    os.path.join("C:", "Users", "juesh", "OneDrive", "Documents", "windsurf", "sec-edgar-filings", ticker, filing_type)
                ]
                
                for alt_path in alternative_paths:
                    print(f"Checking alternative path: {alt_path}")
                    if os.path.exists(alt_path):
                        filing_path = alt_path
                        print(f"Found filing path: {filing_path}")
                        break
        except AttributeError:
            print("Downloader object does not have download_folder attribute, trying alternative paths")
            
            # Try multiple possible locations for the sec-edgar-filings directory
            possible_paths = [
                # Current working directory
                os.path.join(os.getcwd(), "sec-edgar-filings"),
                # Absolute path from error message
                os.path.join("C:", "Users", "juesh", "OneDrive", "Documents", "windsurf", "sec-edgar-filings")
            ]
            
            # Try each possible path until we find the ticker's filings
            filing_path = None
            for base_path in possible_paths:
                print(f"Checking: {base_path}")
                if os.path.exists(base_path):
                    test_path = os.path.join(base_path, ticker, filing_type)
                    if os.path.exists(test_path):
                        filing_path = test_path
                        print(f"Found filing path: {filing_path}")
                        break
        
        if not filing_path or not os.path.exists(filing_path):
            return False, None, f"Filing path not found for {ticker}", None
        
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
        
        if not latest_folder:
            return False, None, f"No filing folders found for {ticker}", None
        
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
            return False, None, f"No filing document found in {latest_folder}", None
        
        # Read the content of the filing
        try:
            with open(filing_document, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Construct a URL for reference (may not be exact but provides a link to SEC)
            sec_url = f"https://www.sec.gov/edgar/browse/?CIK={ticker}&owner=exclude"
            
            return True, filing_document, content, sec_url
        except Exception as e:
            return False, filing_document, f"Error reading file: {e}", None
            
    except Exception as e:
        print(f"Error downloading {filing_type} for {ticker}: {e}")
        # Fall back to the original method if the downloader fails
        try:
            filing_url = _get_filing_url(ticker, filing_type)
            if filing_url:
                report_text = _get_text_from_url(filing_url)
                if report_text:
                    return True, None, report_text, filing_url
            return False, None, f"Error downloading {filing_type} for {ticker}: {e}", None
        except Exception as fallback_error:
            return False, None, f"Error downloading {filing_type} for {ticker}: {e}. Fallback also failed: {fallback_error}", None

def analyze_10k_report(ticker):
    """
    Finds the latest 10-K report from the web, analyzes it using Google Gemini API.
    
    Uses edgartools to extract only the key sections (Business, Risk Factors, MD&A)
    to stay within Gemini's token limits.
    """
    print(f"Starting 10-K analysis for {ticker}...")
    
    # Try using the new sec_filing_parser with edgartools first (much better extraction)
    try:
        import sec_filing_parser
        print("Using edgartools to extract key 10-K sections...")
        
        success, error, text = sec_filing_parser.get_filing_for_analysis(ticker, "10-K")
        
        if success:
            print(f"Successfully extracted key sections: {len(text):,} characters")
            print(f"(Reduced from ~8-10M chars to {len(text):,} chars)")
            filing_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K"
        else:
            print(f"edgartools extraction failed: {error}")
            print("Falling back to legacy method...")
            raise Exception(error)
            
    except Exception as e:
        print(f"edgartools not available or failed: {e}")
        print("Using legacy extraction method...")
        
        # Fall back to the old method
        success, file_path, report_text, filing_url = _download_sec_filing(ticker, "10-K")
        
        if not success or not report_text:
            filing_url = _get_filing_url(ticker, "10-K")
            if not filing_url:
                return f"无法为 {ticker} 的10-K报告找到有效的SEC URL。"

            print(f"Found URL: {filing_url}. Fetching content...")
            report_text = _get_text_from_url(filing_url)
            if not report_text:
                return f"无法从URL获取或解析内容: {filing_url}"
        
        # Legacy filtering - truncate to reasonable size
        soup = BeautifulSoup(report_text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        
        # Truncate to 200K chars max to avoid quota issues
        MAX_CHARS = 200000
        if len(text) > MAX_CHARS:
            print(f"Truncating from {len(text):,} to {MAX_CHARS:,} characters")
            text = text[:MAX_CHARS] + "\n\n[... Content truncated for length ...]"
        
        print(f"Total text length: {len(text):,} characters")
    
    print("Content fetched. Analyzing with Gemini...")

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        return f"Error: Could not initialize Gemini model: {e}"

    prompt = f"""
    你是一位专业的财务分析师，精通财务报表分析和公司估值。请分析以下公司的10-K年度报告，并提供深入见解。

    公司: {ticker}

    报告文本:
    ---
    {text[:1000000]}
    ---

    请用中文提供一份结构良好、深入的分析报告，仅涵盖以下五个方面：
    1. **业务概述:** 简要描述公司的业务模式、产品或服务以及市场定位。
    2. **财务表现:** 分析关键财务指标，包括收入、利润、现金流和资产负债表的重要变化。
    3. **管理层分析:** 总结管理层对公司表现、战略和未来发展的观点。
    4. **风险因素:** 识别公司面临的主要风险和挑战。
    5. **增长前景:** 分析公司的增长机会、创新和未来发展方向。

    请使用清晰的标题和小标题组织你的分析，并尽可能提供具体数据和百分比变化。
    请确保分析简洁明了，重点突出，避免冗长的内容。
    报告最后，请提供在线报告的直接链接: {filing_url}
    """

    try:
        # Initialize rate limiter and use retry logic for quota errors
        rate_limiter = GeminiRateLimiter()
        
        # Use make_api_call_with_retry to handle 429 quota errors automatically
        response = rate_limiter.make_api_call_with_retry(
            model.generate_content, prompt
        )
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return f"Gemini API quota exceeded. Please wait a few minutes and try again.\n\nDetails: {e}"
        return f"在最终分析过程中发生错误: {e}"

def analyze_10q_report(ticker):
    """
    Finds the latest 10-Q report from the web, analyzes it using Google Gemini API.
    
    Uses edgartools to extract only the key sections (MD&A, Risk Factors)
    to stay within Gemini's token limits.
    """
    print(f"Starting 10-Q analysis for {ticker}...")
    
    # Try using the new sec_filing_parser with edgartools first (much better extraction)
    try:
        import sec_filing_parser
        print("Using edgartools to extract key 10-Q sections...")
        
        success, error, text = sec_filing_parser.get_filing_for_analysis(ticker, "10-Q")
        
        if success:
            print(f"Successfully extracted key sections: {len(text):,} characters")
            filing_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-Q"
        else:
            print(f"edgartools extraction failed: {error}")
            print("Falling back to legacy method...")
            raise Exception(error)
            
    except Exception as e:
        print(f"edgartools not available or failed: {e}")
        print("Using legacy extraction method...")
        
        # Fall back to the old method
        success, file_path, report_text, filing_url = _download_sec_filing(ticker, "10-Q")
        
        if not success or not report_text:
            filing_url = _get_filing_url(ticker, "10-Q")
            if not filing_url:
                return f"无法为 {ticker} 的10-Q报告找到有效的SEC URL。"

            print(f"Found URL: {filing_url}. Fetching content...")
            report_text = _get_text_from_url(filing_url)
            if not report_text:
                return f"无法从URL获取或解析内容: {filing_url}"
        
        # Legacy filtering - truncate to reasonable size
        soup = BeautifulSoup(report_text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        
        # Truncate to 200K chars max to avoid quota issues
        MAX_CHARS = 200000
        if len(text) > MAX_CHARS:
            print(f"Truncating from {len(text):,} to {MAX_CHARS:,} characters")
            text = text[:MAX_CHARS] + "\n\n[... Content truncated for length ...]"
        
        print(f"Total text length: {len(text):,} characters")

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        return f"Error: Could not initialize Gemini model: {e}"

    prompt = f"""
    你是一位专业的财务分析师，精通财务报表分析和公司估值。请分析以下公司的10-Q季度报告，并提供深入见解。

    公司: {ticker}

    报告文本:
    ---
    {text[:1000000]}
    ---

    请用中文提供一份结构良好、深入的季度分析报告，仅涵盖以下五个方面：
    1. **整体摘要:** 对整个10-Q报告进行高级别摘要，重点关注本季度的变化。
    2. **亮点 (Highlights):** 识别并总结报告中的主要积极方面或超出预期的表现。
    3. **不足 (Lowlights):** 识别并总结报告中的主要风险、挑战或未达预期的表现。
    4. **财务表现:** 分析本季度的财务报表，总结关键财务指标的变化。
    5. **管理层讨论:** 总结管理层对本季度业绩和短期前景的看法。

    请使用清晰的标题和小标题组织你的分析，并尽可能提供具体数据和百分比变化。
    请确保分析简洁明了，重点突出，避免冗长的内容。
    """

    try:
        # Initialize rate limiter and use retry logic for quota errors
        rate_limiter = GeminiRateLimiter()
        
        # Use make_api_call_with_retry to handle 429 quota errors automatically
        response = rate_limiter.make_api_call_with_retry(
            model.generate_content, prompt
        )
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return f"Gemini API quota exceeded. Please wait a few minutes and try again.\n\nDetails: {e}"
        return f"在最终分析过程中发生错误: {e}"

def analyze_news(news_articles):
    """
    Analyzes a list of news articles using Google Gemini API.

    Args:
        news_articles (list): A list of news articles from Tavily.

    Returns:
        str: A structured summary of the news.
    """
    load_dotenv()
    # load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env") 
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        return f"Error: Could not initialize Gemini model: {e}"

    good_news = []
    bad_news = []

    for article in news_articles:
        prompt = f"""
        请用中文总结以下新闻文章，并将其分类为“利好”、“利空”或“中性”。
        并提供具体数据和百分比变化。Followed by an English version of the response in a separate paragraph as well.
        请以JSON格式返回，包含“summary”和“sentiment”两个字段。

        新闻标题: {article.get('title', 'N/A')}
        新闻内容: {article.get('content', 'N/A')}
        """
        try:
            # Initialize rate limiter
            rate_limiter = GeminiRateLimiter()
            
            # Use make_api_call_with_retry to handle 429 quota errors automatically
            response = rate_limiter.make_api_call_with_retry(
                model.generate_content, prompt
            )
            # Clean the response to make it valid JSON
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
            result = json.loads(cleaned_response)

            summary = result.get('summary', '无法生成摘要。')
            sentiment = result.get('sentiment', '中性').lower()

            if "利好" in sentiment:
                good_news.append(f"- {summary} (来源: {article.get('url', 'N/A')})")
            elif "利空" in sentiment:
                bad_news.append(f"- {summary} (来源: {article.get('url', 'N/A')})")
        except Exception as e:
            print(f"Error processing article: {e}")
            continue

    # Format the final output
    output = "## 新闻分析\n\n"
    output += "### 利好消息\n"
    if good_news:
        output += "\n".join(good_news)
    else:
        output += "近期无明显利好消息。\n"

    output += "\n\n### 利空消息\n"
    if bad_news:
        output += "\n".join(bad_news)
    else:
        output += "近期无明显利空消息。\n"

    return output

def general_search(ticker, company_info, query):
    """
    Performs a general AI search about a company using Google Gemini API.

    Args:
        ticker (str): The stock ticker symbol.
        company_info (dict): A dictionary containing fundamental data about the company.
        query (str): The user's search query.

    Returns:
        str: The search result from Gemini API.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    try:
        model = _init_gemini_model_with_fallback()
    except Exception as e:
        print(f"Could not initialize model: {e}")
        return _format_gemini_error(e)

    prompt = f"""
    针对股票代码为 '{ticker}' 的公司 '{company_info.get('longName', 'N/A')}'，请回答以下问题。

    用户问题: "{query}"

    请使用中文进行详细回答,并提供具体数据和百分比变化。Followed by an English version of the response as well
    ---
    公司参考信息:
    - **行业板块:** {company_info.get('sector', 'N/A')}
    - **具体行业:** {company_info.get('industry', 'N/A')}
    - **业务摘要:** {company_info.get('longBusinessSummary', 'N/A')}
    """

    try:
        # Initialize rate limiter
        rate_limiter = GeminiRateLimiter()
        
        # Make the API call with retry mechanism
        def make_api_call():
            return model.generate_content(prompt)
            
        response = rate_limiter.make_api_call_with_retry(make_api_call)
        return response.text
    except Exception as e:
        return f"An error occurred while communicating with the Gemini API: {e}"


def summarize_market_news(articles, tickers=None):
    """Use Gemini to convert market news articles into a bilingual blog post."""
    if not articles:
        return "未提供市场新闻数据。\nNo market news articles were provided."

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    try:
        model = _init_gemini_model_with_fallback()
    except Exception as e:
        return _format_gemini_error(e)

    # Keep prompt concise by limiting number of articles
    max_items = min(len(articles), 12)
    bullet_lines = []
    for idx, article in enumerate(articles[:max_items], 1):
        title = article.get('title', 'Untitled')
        timestamp = article.get('timestamp') or 'N/A'
        snippet = article.get('snippet') or ''
        url = article.get('url', '')
        bullet_lines.append(
            f"{idx}. 标题/Title: {title}\n   时间: {timestamp}\n   摘要: {snippet}\n   链接: {url}"
        )

    derived_tickers = tickers if tickers else _collect_article_tickers(articles)
    ticker_highlights = "无" if not derived_tickers else ", ".join(derived_tickers[:20])

    prompt = f"""
你是一位资深华尔街财经专栏作家。请阅读以下来自 Finviz 的市场头条，撰写一篇具有博客风格的总结：

市场焦点/Market Highlights：{ticker_highlights}

新闻列表：
{chr(10).join(bullet_lines)}

写作要求：
1. 先用中文撰写完整的市场回顾，包括总览、重要板块/行业动向、宏观数据或公司事件亮点，并给出要点列表。
2. 紧接着提供一段英文版本，内容与中文段落保持一致。
3. 采用资讯型博客语气，提炼主线逻辑并点出潜在影响。
"""

    rate_limiter = GeminiRateLimiter()

    try:
        def make_api_call():
            return model.generate_content(prompt)

        response = rate_limiter.make_api_call_with_retry(make_api_call)
        return response.text
    except Exception as e:
        return _format_gemini_error(e)


def summarize_crypto_news(articles, tickers=None):
    """Summarize Finviz v=5 crypto headlines into a bilingual blog."""
    if not articles:
        return "未找到任何加密货币新闻。\nNo crypto news items were provided."

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    try:
        model = _init_gemini_model_with_fallback()
    except Exception as e:
        return _format_gemini_error(e)

    max_items = min(len(articles), 12)
    bullet_lines = []
    for idx, article in enumerate(articles[:max_items], 1):
        title = article.get('title', 'Untitled')
        timestamp = article.get('timestamp') or 'N/A'
        snippet = article.get('snippet') or ''
        url = article.get('url', '')
        bullet_lines.append(
            f"{idx}. 标题/Title: {title}\n   时间: {timestamp}\n   摘要: {snippet}\n   链接: {url}"
        )

    derived_tickers = tickers if tickers else _collect_article_tickers(articles)
    ticker_highlights = "无" if not derived_tickers else ", ".join(derived_tickers[:20])

    prompt = f"""
你是一位专注数字资产与链上资金流的加密市场策略师。请根据 Finviz v=5 Crypto 新闻流撰写一篇双语市场随笔：

核心代币/主题：{ticker_highlights}

新闻列表：
{chr(10).join(bullet_lines)}

写作要求：
1. 先用中文分析整体市场情绪、链上/资金面动态、关键代币影响与潜在催化。
2. 紧接着提供一段英文版本，信息需与中文一致，可给出交易/风险提示。
3. 语气需兼顾专业与博客风格，可加入要点列表帮助投资者快速吸收。
"""

    rate_limiter = GeminiRateLimiter()

    try:
        def make_api_call():
            return model.generate_content(prompt)

        response = rate_limiter.make_api_call_with_retry(make_api_call)
        return response.text
    except Exception as e:
        return _format_gemini_error(e)


def summarize_etf_news(articles, tickers=None):
    """Summarize Finviz v=4 ETF headlines into a bilingual insights blog."""
    if not articles:
        return "未找到任何ETF新闻。\nNo ETF news items were provided."

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    try:
        model = _init_gemini_model_with_fallback()
    except Exception as e:
        return _format_gemini_error(e)

    max_items = min(len(articles), 12)
    bullet_lines = []
    for idx, article in enumerate(articles[:max_items], 1):
        title = article.get('title', 'Untitled')
        timestamp = article.get('timestamp') or 'N/A'
        snippet = article.get('snippet') or ''
        url = article.get('url', '')
        bullet_lines.append(
            f"{idx}. 标题/Title: {title}\n   时间: {timestamp}\n   摘要: {snippet}\n   链接: {url}"
        )

    derived_tickers = tickers if tickers else _collect_article_tickers(articles)
    ticker_highlights = "无" if not derived_tickers else ", ".join(derived_tickers[:20])

    prompt = f"""
你是一位专注ETF与资产配置的华尔街策略师。请依据 Finviz v=4 ETF 新闻流撰写一篇双语市场随笔：

聚焦ETF：{ticker_highlights}

新闻列表：
{chr(10).join(bullet_lines)}

写作要求：
1. 先以中文概述ETF/板块动态、资金流向、潜在驱动与交易启示，可结合要点列表。
2. 紧接着提供一段英文版本，确保信息与中文一致。
3. 语气保持专业且具有博客风格，在结尾附上对ETF投资者的观察建议。
"""

    rate_limiter = GeminiRateLimiter()

    try:
        def make_api_call():
            return model.generate_content(prompt)

        response = rate_limiter.make_api_call_with_retry(make_api_call)
        return response.text
    except Exception as e:
        return _format_gemini_error(e)


def _collect_article_tickers(articles):
    unique = []
    seen = set()
    for article in articles or []:
        for ticker in article.get("tickers", []) or []:
            normalized = (ticker or "").strip().upper()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(normalized)
    return unique


def summarize_stock_news(articles, tickers=None):
    """Summarize Finviz v=3 stock headlines (general feed) into a bilingual blog."""
    if not articles:
        return "未找到任何股票新闻。\nNo stock news items were provided."

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    try:
        model = _init_gemini_model_with_fallback()
    except Exception as e:
        return _format_gemini_error(e)

    max_items = min(len(articles), 12)
    bullet_lines = []
    for idx, article in enumerate(articles[:max_items], 1):
        title = article.get('title', 'Untitled')
        timestamp = article.get('timestamp') or 'N/A'
        snippet = article.get('snippet') or ''
        url = article.get('url', '')
        bullet_lines.append(
            f"{idx}. 标题/Title: {title}\n   时间: {timestamp}\n   摘要: {snippet}\n   链接: {url}"
        )

    derived_tickers = tickers if tickers else _collect_article_tickers(articles)
    ticker_highlights = "无" if not derived_tickers else ", ".join(derived_tickers[:20])

    prompt = f"""
你是一位资深华尔街分析师兼投资博客作者。请根据 Finviz v=3 股票新闻源撰写一篇双语博客：

已提及股票：{ticker_highlights}

新闻列表：
{chr(10).join(bullet_lines)}

写作要求：
1. 先用中文写出结构化的市场/个股新闻综述，突出受影响的行业与公司、潜在驱动因素和投资含义，可使用要点列表。
2. 紧接着提供一段英文版本，内容与中文段落保持一致。
3. 语气需兼具专业与博客风格，让投资者快速抓住重点，可在结尾给出观察建议。
"""

    rate_limiter = GeminiRateLimiter()

    try:
        def make_api_call():
            return model.generate_content(prompt)

        response = rate_limiter.make_api_call_with_retry(make_api_call)
        return response.text
    except Exception as e:
        return _format_gemini_error(e)

def summarize_clipboard_content(content, urls=None):
    """
    Summarize content from clipboard which may contain URLs or direct text.
    
    Args:
        content (str): The clipboard content (text or fetched webpage content).
        urls (list): Optional list of URLs that were detected and fetched.
    
    Returns:
        str: Bilingual summary of the content.
    """
    if not content or not content.strip():
        return "剪贴板为空或无有效内容。\nClipboard is empty or contains no valid content."

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    try:
        model = _init_gemini_model_with_fallback()
    except Exception as e:
        return _format_gemini_error(e)

    # Truncate content if too long (keep first ~15000 chars to stay within token limits)
    max_content_len = 15000
    truncated = content[:max_content_len] if len(content) > max_content_len else content
    truncation_note = "\n[内容已截断 / Content truncated]" if len(content) > max_content_len else ""

    url_info = ""
    if urls:
        url_info = f"\n来源链接 / Source URLs:\n" + "\n".join(f"- {u}" for u in urls[:5])

    prompt = f"""
你是一位资深华尔街分析师兼投资博客作者。请根据以下内容撰写一篇双语博客摘要：
{url_info}

内容：
{truncated}{truncation_note}

写作要求：
1. 先用中文写出结构化的内容综述，识别关键主题、重要信息点和潜在投资含义（如适用），可使用要点列表。
2. 如果内容涉及股票、公司或市场，请突出相关股票代码和行业影响。
3. 紧接着提供一段英文版本，内容与中文段落保持一致。
4. 语气需兼具专业与博客风格，让读者快速抓住重点。
5. 如果内容不是财经相关，仍然提供有价值的摘要。
"""

    rate_limiter = GeminiRateLimiter()

    try:
        def make_api_call():
            return model.generate_content(prompt)

        response = rate_limiter.make_api_call_with_retry(make_api_call)
        return response.text
    except Exception as e:
        return _format_gemini_error(e)


def general_ai_search(query):
    """
    Performs a general AI search using Google Gemini API without requiring ticker information.

    Args:
        query (str): The user's search query.

    Returns:
        str: The search result from Gemini API.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        print(f"Could not initialize model: {e}")
        return "Error: Could not initialize Gemini model."

    prompt = f"""
    请回答以下问题，提供详细和准确的信息：

    用户问题: "{query}"

    请使用中文进行详细回答,并提供具体数据和百分比变化。Followed by an English version of the response as well.
    """

    try:
        # Initialize rate limiter
        rate_limiter = GeminiRateLimiter()
        
        # Make the API call with retry mechanism
        def make_api_call():
            return model.generate_content(prompt)
            
        response = rate_limiter.make_api_call_with_retry(make_api_call)
        return response.text
    except Exception as e:
        return f"An error occurred while communicating with the Gemini API: {e}"
