import requests
import pandas as pd
import json
import sys
import os
import re
import traceback
from bs4 import BeautifulSoup
from datetime import datetime

def get_headers():
    """
    Get headers for SEC EDGAR API requests
    """
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive"
    }

def get_company_cik(ticker):
    """
    Get company CIK number from ticker
    """
    print(f"Looking up CIK for {ticker}...")
    try:
        # SEC provides a JSON file with all CIK to ticker mappings
        response = requests.get(
            "https://www.sec.gov/files/company_tickers.json", 
            headers=get_headers()
        )
        
        if response.status_code != 200:
            print(f"Error fetching CIK data: {response.status_code}")
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

def get_company_facts(cik):
    """
    Get company facts from SEC EDGAR API
    """
    print(f"Fetching company facts for CIK {cik}...")
    try:
        response = requests.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
            headers=get_headers()
        )
        
        if response.status_code != 200:
            print(f"Error fetching company facts: {response.status_code}")
            print(f"Response: {response.text[:500]}...")
            return None
            
        return response.json()
        
    except Exception as e:
        print(f"Error fetching company facts: {e}")
        traceback.print_exc()
        return None

def get_company_submissions(cik):
    """
    Get company submissions from SEC EDGAR API
    """
    print(f"Fetching company submissions for CIK {cik}...")
    try:
        response = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=get_headers()
        )
        
        if response.status_code != 200:
            print(f"Error fetching company submissions: {response.status_code}")
            print(f"Response: {response.text[:500]}...")
            return None
            
        return response.json()
        
    except Exception as e:
        print(f"Error fetching company submissions: {e}")
        traceback.print_exc()
        return None

def get_latest_10k_filing(cik):
    """
    Get the latest 10-K filing for a company
    """
    print(f"Finding latest 10-K filing for CIK {cik}...")
    try:
        submissions = get_company_submissions(cik)
        
        if not submissions:
            return None
            
        # Look for 10-K filings
        recent_filings = submissions.get("filings", {}).get("recent", {})
        
        if not recent_filings:
            print("No recent filings found")
            return None
            
        form_types = recent_filings.get("form", [])
        filing_dates = recent_filings.get("filingDate", [])
        accession_numbers = recent_filings.get("accessionNumber", [])
        primary_documents = recent_filings.get("primaryDocument", [])
        
        # Find the latest 10-K filing
        for i in range(len(form_types)):
            if form_types[i] == "10-K":
                filing_info = {
                    "form": form_types[i],
                    "filingDate": filing_dates[i],
                    "accessionNumber": accession_numbers[i],
                    "primaryDocument": primary_documents[i]
                }
                
                # Format accession number for URL (remove dashes)
                acc_no = filing_info["accessionNumber"].replace("-", "")
                
                # Create URLs for the filing
                filing_info["htmlUrl"] = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{acc_no}/{filing_info['primaryDocument']}"
                
                print(f"Found 10-K filing from {filing_info['filingDate']}")
                return filing_info
                
        print("No 10-K filings found")
        return None
        
    except Exception as e:
        print(f"Error finding 10-K filing: {e}")
        traceback.print_exc()
        return None

def extract_financial_tables(html_content):
    """
    Extract financial tables from HTML content
    """
    print("Extracting financial tables from HTML...")
    try:
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all tables
        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables in the document")
        
        # Extract tables
        financial_tables = []
        for i, table in enumerate(tables):
            try:
                # Convert to pandas DataFrame
                df = pd.read_html(str(table))[0]
                
                # Check if this looks like a financial table
                if df.shape[0] > 3 and df.shape[1] > 2:
                    # Look for financial keywords in the first column
                    first_col = df.iloc[:, 0].astype(str).str.lower()
                    financial_keywords = ['revenue', 'income', 'asset', 'liability', 'cash', 'equity', 
                                         'earnings', 'sales', 'cost', 'expense', 'profit', 'loss',
                                         'total', 'net', 'gross', 'operating', 'tax', 'dividend']
                    
                    # Check if any financial keywords are in the first column
                    if any(first_col.str.contains(keyword).any() for keyword in financial_keywords):
                        financial_tables.append({
                            'index': i,
                            'table': df,
                            'rows': df.shape[0],
                            'columns': df.shape[1]
                        })
            except Exception as e:
                print(f"Error extracting table {i}: {e}")
                continue
                
        print(f"Found {len(financial_tables)} potential financial tables")
        return financial_tables
        
    except Exception as e:
        print(f"Error extracting financial tables: {e}")
        traceback.print_exc()
        return []

def identify_financial_statements(tables):
    """
    Identify balance sheet, income statement, and cash flow statement from tables
    """
    print("Identifying financial statements...")
    
    financial_statements = {
        'balance_sheet': None,
        'income_statement': None,
        'cash_flow_statement': None
    }
    
    # Keywords to identify each statement
    bs_keywords = ['balance sheet', 'assets', 'liabilities', 'equity', 'current assets', 
                  'total assets', 'current liabilities', 'total liabilities']
    is_keywords = ['income statement', 'statement of operations', 'revenue', 'sales', 
                  'gross profit', 'operating income', 'net income', 'earnings per share']
    cf_keywords = ['cash flow', 'statement of cash flows', 'operating activities', 
                  'investing activities', 'financing activities', 'net cash']
    
    for table_info in tables:
        table = table_info['table']
        
        # Convert table to string for keyword search
        table_str = str(table).lower()
        
        # Check for balance sheet
        if financial_statements['balance_sheet'] is None and any(keyword in table_str for keyword in bs_keywords):
            financial_statements['balance_sheet'] = table
            print("Found balance sheet")
            
        # Check for income statement
        if financial_statements['income_statement'] is None and any(keyword in table_str for keyword in is_keywords):
            financial_statements['income_statement'] = table
            print("Found income statement")
            
        # Check for cash flow statement
        if financial_statements['cash_flow_statement'] is None and any(keyword in table_str for keyword in cf_keywords):
            financial_statements['cash_flow_statement'] = table
            print("Found cash flow statement")
            
    return financial_statements

def clean_financial_table(df):
    """
    Clean and format financial table
    """
    try:
        # Remove empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Try to set the first column as index if it contains text
        if df.shape[1] > 1:
            first_col = df.iloc[:, 0]
            if first_col.dtype == 'object':
                df = df.set_index(df.columns[0])
                
        # Convert numeric columns to float
        for col in df.columns:
            try:
                # Remove any non-numeric characters (except decimal point and negative sign)
                df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
                
        return df
        
    except Exception as e:
        print(f"Error cleaning financial table: {e}")
        return df

def main():
    # Get ticker from command line or use default
    ticker = "AAPL"  # Default ticker
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    
    try:
        # Get company CIK
        cik = get_company_cik(ticker)
        if not cik:
            print(f"Could not find CIK for {ticker}")
            return 1
            
        # Get latest 10-K filing
        filing_info = get_latest_10k_filing(cik)
        if not filing_info:
            print(f"Could not find 10-K filing for {ticker}")
            return 1
            
        # Download the filing HTML
        print(f"Downloading 10-K filing from {filing_info['htmlUrl']}...")
        response = requests.get(filing_info['htmlUrl'], headers=get_headers())
        
        if response.status_code != 200:
            print(f"Error downloading filing: {response.status_code}")
            print(f"Response: {response.text[:500]}...")
            return 1
            
        html_content = response.text
        print(f"Downloaded {len(html_content)} bytes of HTML content")
        
        # Extract financial tables
        tables = extract_financial_tables(html_content)
        
        if not tables:
            print("No financial tables found")
            return 1
            
        # Identify financial statements
        financial_statements = identify_financial_statements(tables)
        
        # Clean and format financial statements
        for statement_name, df in financial_statements.items():
            if df is not None:
                financial_statements[statement_name] = clean_financial_table(df)
        
        # Save to Excel
        output_file = f"{ticker}_Financial_Statements.xlsx"
        with pd.ExcelWriter(output_file) as writer:
            # Save identified financial statements
            for statement_name, df in financial_statements.items():
                if df is not None:
                    sheet_name = statement_name.replace('_', ' ').title()
                    df.to_excel(writer, sheet_name=sheet_name)
            
            # Save all potential financial tables
            for i, table_info in enumerate(tables[:10]):  # Save up to 10 tables
                sheet_name = f"Table_{i}"
                table_info['table'].to_excel(writer, sheet_name=sheet_name)
                
            # Save filing info
            pd.DataFrame([filing_info]).to_excel(writer, sheet_name="Filing Info")
        
        print(f"Financial statements saved to {output_file}")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
