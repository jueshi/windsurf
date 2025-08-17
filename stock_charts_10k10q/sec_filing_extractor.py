import requests
import pandas as pd
import json
import sys
import os
import re
import traceback
import argparse
import random
from bs4 import BeautifulSoup
from datetime import datetime
import time
from dotenv import load_dotenv

# Configure basic logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_headers():
    """
    Get headers for SEC EDGAR API requests to avoid 403 errors
    """
    print("Setting up request headers...", flush=True)
    
    # Try to get email from environment variable
    import os
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file if present
    
    email = os.getenv("SEC_EDGAR_EMAIL", "jueshi@gmail.com")
    print(f"Using email: {email}", flush=True)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.67",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "From": email  # Add email header for SEC API
    }
    
    return headers

def make_sec_request(url, max_retries=5, initial_delay=1):
    """
    Make a request to SEC API with exponential backoff retry logic
    
    Args:
        url (str): URL to request
        max_retries (int): Maximum number of retries
        initial_delay (int): Initial delay in seconds
        
    Returns:
        requests.Response or None: Response object or None if all retries fail
    """
    headers = get_headers()
    print(f"Requesting URL: {url}", flush=True)
    
    for attempt in range(max_retries):
        try:
            # Add jitter to avoid synchronized requests
            if attempt > 0:
                delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Retry attempt {attempt+1}/{max_retries}. Waiting {delay:.2f} seconds...", flush=True)
                time.sleep(delay)
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response
            elif response.status_code == 403:
                print(f"Rate limit exceeded (403). Retrying after delay...", flush=True)
                # Continue to retry
            else:
                print(f"Error: Status code {response.status_code}", flush=True)
                print(f"Response: {response.text[:500]}...", flush=True)
                # For non-rate-limit errors, we might still retry
        
        except Exception as e:
            print(f"Request error: {str(e)}", flush=True)
            # Continue to retry
    
    print(f"Failed after {max_retries} attempts", flush=True)
    return None

def get_company_cik(ticker):
    """
    Get company CIK number from ticker
    """
    print(f"Looking up CIK for {ticker}...", flush=True)
    try:
        # SEC provides a JSON file with all CIK to ticker mappings
        print("Sending request to SEC for company tickers...", flush=True)
        response = make_sec_request("https://www.sec.gov/files/company_tickers.json")
        
        if not response or response.status_code != 200:
            print(f"Error fetching CIK data", flush=True)
            return None
            
        companies = response.json()
        
        # Find the company by ticker
        for _, company in companies.items():
            if company["ticker"].upper() == ticker.upper():
                # Format CIK with leading zeros to 10 digits
                cik = str(company["cik_str"]).zfill(10)
                print(f"Found CIK: {cik} for {company['title']}")
                return cik
                
        print(f"Could not find CIK for ticker {ticker}")
        return None
        
    except Exception as e:
        print(f"Error looking up CIK: {e}")
        traceback.print_exc()
        return None

def get_latest_filing_info(cik, form_type="10-K"):
    """
    Get the latest filing info for a company
    
    Args:
        cik (str): Company CIK number (10 digits with leading zeros)
        form_type (str): Form type to search for (10-K, 10-Q, etc.)
        
    Returns:
        dict: Filing information or None if not found
    """
    print(f"Finding latest {form_type} filing for CIK {cik}...", flush=True)
    try:
        # Get the company's submissions feed
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        print(f"Requesting data from: {url}", flush=True)
        
        # Use our rate-limited request function
        response = make_sec_request(url)
        
        if not response or response.status_code != 200:
            print(f"Error fetching company submissions", flush=True)
            return None
            
        data = response.json()
        
        # Get recent filings
        recent_filings = data.get("filings", {}).get("recent", {})
        
        if not recent_filings:
            print("No recent filings found")
            return None
            
        # Extract filing data
        form_types = recent_filings.get("form", [])
        accession_numbers = recent_filings.get("accessionNumber", [])
        filing_dates = recent_filings.get("filingDate", [])
        primary_docs = recent_filings.get("primaryDocument", [])
        
        # Find the latest filing of the specified type
        for i in range(len(form_types)):
            if form_types[i] == form_type:
                # Get the accession number without dashes for URL construction
                acc_no = accession_numbers[i].replace("-", "")
                
                # Get the CIK without leading zeros for URL construction
                cik_no_zeros = cik.lstrip("0")
                
                # Construct URLs
                filing_detail_url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{acc_no}/{primary_docs[i]}"
                
                filing_info = {
                    "cik": cik,
                    "accessionNumber": accession_numbers[i],
                    "filingDate": filing_dates[i],
                    "formType": form_types[i],
                    "primaryDocument": primary_docs[i],
                    "detailUrl": filing_detail_url
                }
                
                print(f"Found {form_type} filing from {filing_dates[i]}")
                print(f"Filing URL: {filing_detail_url}")
                return filing_info
                
        print(f"No {form_type} filings found")
        return None
        
    except Exception as e:
        print(f"Error finding filing: {e}")
        traceback.print_exc()
        return None

def download_filing(filing_info):
    """
    Download the filing document
    
    Args:
        filing_info (dict): Filing information from get_latest_filing_info
        
    Returns:
        str: HTML content of the filing or None if download fails
    """
    print(f"Downloading filing from {filing_info['detailUrl']}...", flush=True)
    try:
        # Use our rate-limited request function with longer initial delay
        response = make_sec_request(filing_info['detailUrl'], initial_delay=2)
        
        if not response or response.status_code != 200:
            print(f"Error downloading filing", flush=True)
            return None
            
        html_content = response.text
        print(f"Successfully downloaded {len(html_content)} bytes")
        return html_content
        
    except Exception as e:
        print(f"Error downloading filing: {e}")
        traceback.print_exc()
        return None

def extract_tables(html_content):
    """
    Extract tables from HTML content
    
    Args:
        html_content (str): HTML content of the filing
        
    Returns:
        list: List of pandas DataFrames containing tables
    """
    print("Extracting tables from HTML content...", flush=True)
    try:
        # Use pandas to extract tables
        print(f"HTML content size: {len(html_content)} bytes", flush=True)
        tables = pd.read_html(html_content)
        print(f"Found {len(tables)} tables", flush=True)
        return tables
        
    except Exception as e:
        print(f"Error extracting tables: {e}")
        traceback.print_exc()
        return []

def identify_financial_tables(tables):
    """
    Identify financial tables (balance sheet, income statement, cash flow)
    
    Args:
        tables (list): List of pandas DataFrames
        
    Returns:
        dict: Dictionary with identified financial tables
    """
    print("Identifying financial tables...")
    financial_tables = {
        "balance_sheet": None,
        "income_statement": None,
        "cash_flow": None
    }
    
    # Keywords to identify each type of financial statement
    bs_keywords = ["balance sheet", "assets", "liabilities", "stockholders equity", "shareholders equity"]
    is_keywords = ["income statement", "statement of operations", "revenues", "net income", "earnings per share"]
    cf_keywords = ["cash flow", "statement of cash flows", "operating activities", "investing activities", "financing activities"]
    
    # Check each table
    for i, table in enumerate(tables):
        # Convert table to string for keyword search
        table_str = str(table).lower()
        
        # Check for balance sheet
        if financial_tables["balance_sheet"] is None and any(keyword in table_str for keyword in bs_keywords):
            financial_tables["balance_sheet"] = table
            print(f"Found balance sheet (Table {i})")
            
        # Check for income statement
        if financial_tables["income_statement"] is None and any(keyword in table_str for keyword in is_keywords):
            financial_tables["income_statement"] = table
            print(f"Found income statement (Table {i})")
            
        # Check for cash flow statement
        if financial_tables["cash_flow"] is None and any(keyword in table_str for keyword in cf_keywords):
            financial_tables["cash_flow"] = table
            print(f"Found cash flow statement (Table {i})")
    
    return financial_tables

def clean_financial_table(table):
    """
    Clean and format a financial table
    
    Args:
        table (DataFrame): Table to clean
        
    Returns:
        DataFrame: Cleaned table
    """
    try:
        # Remove empty rows and columns
        table = table.dropna(how='all').dropna(axis=1, how='all')
        
        # Try to set the first column as index if it contains text
        if table.shape[1] > 1:
            first_col = table.iloc[:, 0]
            if first_col.dtype == 'object':
                table = table.set_index(table.columns[0])
                
        # Convert numeric columns to float
        for col in table.columns:
            try:
                # Remove any non-numeric characters (except decimal point and negative sign)
                table[col] = table[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                table[col] = pd.to_numeric(table[col], errors='coerce')
            except:
                pass
                
        return table
        
    except Exception as e:
        print(f"Error cleaning table: {e}")
        return table

def save_tables_to_excel(financial_tables, all_tables, ticker, output_dir="."):
    """
    Save tables to Excel files
    
    Args:
        financial_tables (dict): Dictionary with identified financial tables
        all_tables (list): List of all tables
        ticker (str): Company ticker
        output_dir (str): Output directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save financial tables
        if any(table is not None for table in financial_tables.values()):
            financial_file = os.path.join(output_dir, f"{ticker}_Financial_Statements.xlsx")
            with pd.ExcelWriter(financial_file) as writer:
                for name, table in financial_tables.items():
                    if table is not None:
                        # Clean the table
                        clean_table = clean_financial_table(table)
                        # Save to Excel
                        sheet_name = name.replace("_", " ").title()
                        clean_table.to_excel(writer, sheet_name=sheet_name)
            print(f"Saved financial statements to {financial_file}")
        
        # Save all tables
        all_tables_file = os.path.join(output_dir, f"{ticker}_All_Tables.xlsx")
        with pd.ExcelWriter(all_tables_file) as writer:
            for i, table in enumerate(all_tables[:30]):  # Limit to 30 tables
                table.to_excel(writer, sheet_name=f"Table_{i}")
        print(f"Saved all tables to {all_tables_file}")
        
        return True
        
    except Exception as e:
        print(f"Error saving tables: {e}")
        traceback.print_exc()
        return False

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Extract tables from SEC filings')
    parser.add_argument('ticker', nargs='?', default='AAPL', help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--form', '-f', default='10-K', choices=['10-K', '10-Q'], help='Form type (default: 10-K)')
    parser.add_argument('--output', '-o', default='.', help='Output directory (default: current directory)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    ticker = args.ticker.upper()
    form_type = args.form
    output_dir = args.output
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print(f"\n{'='*50}", flush=True)
    print(f"SEC Filing Extractor - Starting for {ticker} ({form_type})", flush=True)
    print(f"{'='*50}\n", flush=True)
    
    try:
        # Get company CIK
        print("Step 1: Getting company CIK...", flush=True)
        cik = get_company_cik(ticker)
        if not cik:
            print(f"Could not find CIK for {ticker}", flush=True)
            return 1
        
        # Get latest filing info
        print(f"\nStep 2: Getting latest {form_type} filing info...", flush=True)
        filing_info = get_latest_filing_info(cik, form_type)
        if not filing_info:
            print(f"Could not find {form_type} filing for {ticker}", flush=True)
            return 1
        
        # Download filing
        print("\nStep 3: Downloading filing...", flush=True)
        html_content = download_filing(filing_info)
        if not html_content:
            print("Failed to download filing", flush=True)
            return 1
        
        # Extract tables
        print("\nStep 4: Extracting tables...", flush=True)
        tables = extract_tables(html_content)
        if not tables:
            print("No tables found in filing", flush=True)
            return 1
        
        # Identify financial tables
        print("\nStep 5: Identifying financial tables...", flush=True)
        financial_tables = identify_financial_tables(tables)
        
        # Save tables to Excel
        print("\nStep 6: Saving tables to Excel...", flush=True)
        save_tables_to_excel(financial_tables, tables, ticker, output_dir)
        
        print(f"\n{'='*50}", flush=True)
        print(f"Successfully extracted and saved tables for {ticker}", flush=True)
        print(f"{'='*50}\n", flush=True)
        return 0
        
    except Exception as e:
        print(f"\nError: {e}", flush=True)
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
