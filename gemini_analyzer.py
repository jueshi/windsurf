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

def analyze_ticker(ticker, company_info):
    """
    Analyzes a stock ticker using Google Gemini API.

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

    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        print(f"Could not initialize model: {e}")
        print("Available models:")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
        return "Error: Could not initialize Gemini model."


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

    请用中文提供结构良好且详细的分析。
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while communicating with the Gemini API: {e}"

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
    """
    print(f"Starting 10-K analysis for {ticker}...")
    
    # Try the new downloader method first
    success, file_path, report_text, filing_url = _download_sec_filing(ticker, "10-K")
    
    if not success or not report_text:
        # If the new method fails, try the original method
        filing_url = _get_filing_url(ticker, "10-K")
        if not filing_url:
            return f"无法为 {ticker} 的10-K报告找到有效的SEC URL。"

        print(f"Found URL: {filing_url}. Fetching content...")
        report_text = _get_text_from_url(filing_url)
        if not report_text:
            return f"无法从URL获取或解析内容: {filing_url}"
    
    print(f"Successfully retrieved 10-K report for {ticker}")
    if file_path:
        print(f"Local file path: {file_path}")
    if filing_url:
        print(f"Filing URL: {filing_url}")

    # 用BeautifulSoup解析HTML
    soup = BeautifulSoup(report_text, 'html.parser')

    # 提取页面中的全部文本（去掉标签、脚本）
    # full_text = soup.get_text(separator='\n', strip=True)
    
    # Save HTML content to a file
    html_path = f"{ticker}_10Q.html"
    try:
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"HTML content saved to {html_path}")
    except Exception as e:
        print(f"Could not save HTML content: {e}")
        print("Continuing with text analysis...")
    
    # Filter content to focus on relevant sections for the analysis
    print("Filtering content for relevant sections...")
    
    # Define keywords for each section we're interested in
    relevant_sections = {
        'summary': ['summary', 'overview', 'highlights', '概述', '摘要'],
        'highlights': ['highlights', 'achievements', 'growth', 'increase', 'positive', '亮点', '增长'],
        'lowlights': ['challenges', 'risks', 'decrease', 'negative', 'decline', '风险', '挑战', '下降'],
        'financial': ['financial', 'revenue', 'income', 'earnings', 'profit', 'loss', 'balance', 'cash flow', '财务', '收入', '利润'],
        'management': ['management discussion', 'MD&A', 'outlook', 'guidance', 'future', '管理层', '展望']
    }
    
    
    # Extract relevant sections
    relevant_sections_text = {}
    for section, keywords in relevant_sections.items():
        relevant_sections_text[section] = []
        for keyword in keywords:
            for match in soup.find_all(text=lambda t: t and keyword.lower() in t.lower()):
                if match.parent.name not in ['style', 'script', '[document]']:
                    relevant_sections_text[section].append(match.parent.get_text(separator='\n', strip=True))
    
    # Combine the relevant sections
    relevant_text = []
    for section_content in relevant_sections_text.values():
        relevant_text.extend(section_content)
    
    # Remove duplicates
    relevant_text = list(set(relevant_text))
    
    # Remove empty strings
    relevant_text = [text for text in relevant_text if text.strip()]
    
    # If we didn't find enough relevant content, include some key sections by position
    if len(relevant_text) < 20:
        # Include beginning (often contains business overview)
        relevant_text.extend(paragraphs[:5])
        
        # Include some middle parts (often risk factors)
        middle_start = len(paragraphs) // 3
        relevant_text.extend(paragraphs[middle_start:middle_start+5])
        
        # Include some end parts (often outlook)
        relevant_text.extend(paragraphs[-5:])
    
    # Combine the relevant text
    text = '\n\n'.join(relevant_text)
    
    # Save the filtered text
    try:
        with open(f"{ticker}_10K_filtered.txt", 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Filtered text saved to {ticker}_10K_filtered.txt")
    except Exception as e:
        print(f"Could not save filtered text: {e}")
    
    # Print a sample of the filtered text
    print("Sample of filtered text:")
    print(text[:500] + "...")
    print(f"Total filtered text length: {len(text)} characters")

    # report_text = report_text[:200000]
    print("Content fetched. Analyzing with Gemini...")

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
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
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"在最终分析过程中发生错误: {e}"

def analyze_10q_report(ticker):
    """
    Finds the latest 10-Q report from the web, analyzes it using Google Gemini API.
    """
    print(f"Starting 10-Q analysis for {ticker}...")
    
    # Try the new downloader method first
    success, file_path, report_text, filing_url = _download_sec_filing(ticker, "10-Q")
    
    if not success or not report_text:
        # If the new method fails, try the original method
        filing_url = _get_filing_url(ticker, "10-Q")
        if not filing_url:
            return f"无法为 {ticker} 的10-Q报告找到有效的SEC URL。"

        print(f"Found URL: {filing_url}. Fetching content...")
        report_text = _get_text_from_url(filing_url)
        if not report_text:
            return f"无法从URL获取或解析内容: {filing_url}"
    
    print(f"Successfully retrieved 10-Q report for {ticker}")
    if file_path:
        print(f"Local file path: {file_path}")
    if filing_url:
        print(f"Filing URL: {filing_url}")
        
    # 用BeautifulSoup解析HTML
    soup = BeautifulSoup(report_text, 'html.parser')

    # 提取页面中的全部文本（去掉标签、脚本）
    # full_text = soup.get_text(separator='\n', strip=True)
    
    # Save HTML content to a file
    html_path = f"{ticker}_10Q.html"
    try:
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"HTML content saved to {html_path}")
    except Exception as e:
        print(f"Could not save HTML content: {e}")
        print("Continuing with text analysis...")
    
    # Filter content to focus on relevant sections for the analysis
    print("Filtering content for relevant sections...")
    
    # Define keywords for each section we're interested in
    relevant_sections = {
        'summary': ['summary', 'overview', 'highlights', '概述', '摘要'],
        'highlights': ['highlights', 'achievements', 'growth', 'increase', 'positive', '亮点', '增长'],
        'lowlights': ['challenges', 'risks', 'decrease', 'negative', 'decline', '风险', '挑战', '下降'],
        'financial': ['financial', 'revenue', 'income', 'earnings', 'profit', 'loss', 'balance', 'cash flow', '财务', '收入', '利润'],
        'management': ['management discussion', 'MD&A', 'outlook', 'guidance', 'future', '管理层', '展望']
    }
    
    
    # Extract relevant sections
    relevant_sections_text = {}
    for section, keywords in relevant_sections.items():
        relevant_sections_text[section] = []
        for keyword in keywords:
            for match in soup.find_all(text=lambda t: t and keyword.lower() in t.lower()):
                if match.parent.name not in ['style', 'script', '[document]']:
                    relevant_sections_text[section].append(match.parent.get_text(separator='\n', strip=True))
    
    # Combine the relevant sections
    relevant_text = []
    for section_content in relevant_sections_text.values():
        relevant_text.extend(section_content)
    
    # Remove duplicates
    relevant_text = list(set(relevant_text))
    
    # Remove empty strings
    relevant_text = [text for text in relevant_text if text.strip()]

    # If we didn't find enough relevant content, include some key sections by position
    if len(relevant_text) < 20:
        # Include beginning (often contains summary)
        relevant_text.extend(paragraphs[:5])
        
        # Include some middle parts (often financial data)
        middle_start = len(paragraphs) // 3
        relevant_text.extend(paragraphs[middle_start:middle_start+5])
        
        # Include some end parts (often outlook)
        relevant_text.extend(paragraphs[-5:])
    
    # Combine the relevant text
    text = '\n\n'.join(relevant_text)
    
    # Save the filtered text
    try:
        with open(f"{ticker}_10Q_filtered.txt", 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Filtered text saved to {ticker}_10Q_filtered.txt")
    except Exception as e:
        print(f"Could not save filtered text: {e}")
    
    # Print a sample of the filtered text
    print("Sample of filtered text:")
    print(text[:500] + "...")
    print(f"Total filtered text length: {len(text)} characters")


    import pandas as pd
    # 使用pandas直接提取所有HTML中的表格，返回DataFrame列表
    tables = pd.read_html(report_text)

    # 打印所有表格，或选择其中一个处理
    for i, table in enumerate(tables):
        print(f"Table {i}:")
        print(table)
        print()

    # 如果只是想保存特定的第一个表格为CSV
    tables[0].to_csv('extracted_table.csv', index=False)

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
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
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
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
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        return f"Error: Could not initialize Gemini model: {e}"

    good_news = []
    bad_news = []

    for article in news_articles:
        prompt = f"""
        请用中文总结以下新闻文章，并将其分类为“利好”、“利空”或“中性”。
        请以JSON格式返回，包含“summary”和“sentiment”两个字段。

        新闻标题: {article.get('title', 'N/A')}
        新闻内容: {article.get('content', 'N/A')}
        """
        try:
            response = model.generate_content(prompt)
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
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        print(f"Could not initialize model: {e}")
        return "Error: Could not initialize Gemini model."

    prompt = f"""
    针对股票代码为 '{ticker}' 的公司 '{company_info.get('longName', 'N/A')}'，请回答以下问题。

    用户问题: "{query}"

    请使用中文进行详细回答。
    ---
    公司参考信息:
    - **行业板块:** {company_info.get('sector', 'N/A')}
    - **具体行业:** {company_info.get('industry', 'N/A')}
    - **业务摘要:** {company_info.get('longBusinessSummary', 'N/A')}
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while communicating with the Gemini API: {e}"
