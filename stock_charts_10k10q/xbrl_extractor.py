from sec_api import QueryApi
import pandas as pd
import requests
import json
import sys
import traceback
import os
from dotenv import load_dotenv

def extract_xbrl_data(ticker="AAPL"):
    """
    Extract XBRL data for a given ticker
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Dictionary containing financial statements or None if extraction fails
    """
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("SEC_API_KEY")
        
        if not api_key:
            print("Error: SEC_API_KEY not found in .env file")
            return None
        
        print(f"Using SEC API key: {api_key[:5]}...")
        
        # Initialize API client
        query_api = QueryApi(api_key=api_key)
        
        # Query specific company's 10-K filings
        print(f"Searching for {ticker} 10-K filings...")
        query = {
            "query": {"query_string": {"query": f"ticker:{ticker} AND formType:10-K"}},
            "from": "0",
            "size": "1",
            "sort": [{"filedAt": "desc"}]
        }
        
        filings = query_api.get_filings(query)
        
        if not filings or not filings.get('filings') or len(filings['filings']) == 0:
            print(f"No 10-K filings found for {ticker}")
            return None
        
        # Get the latest 10-K filing details
        filing = filings['filings'][0]
        filing_url = filing['linkToFilingDetails']
        accession_no = filing['accessionNo']
        company_name = filing['companyNameLong']
        company_cik = filing['cik']
        filing_date = filing['filedAt']
        
        print(f"Found filing: {filing_url}")
        print(f"Company: {company_name}")
        print(f"CIK: {company_cik}")
        print(f"Filing Date: {filing_date}")
        
        # Extract XBRL data
        print("Extracting XBRL data to JSON...")
        xbrl_json = query_api.xbrl_to_json(filing_url)
        print("Successfully extracted XBRL data")
        
        # Extract financial statements
        print("Extracting financial statements...")
        balance_sheet = query_api.get_balance_sheet(xbrl_json)
        income_statement = query_api.get_income_statement(xbrl_json)
        cash_flow = query_api.get_cash_flow_statement(xbrl_json)
        
        return {
            'balance_sheet': balance_sheet,
            'income_statement': income_statement,
            'cash_flow': cash_flow,
            'company_info': {
                'name': company_name,
                'ticker': ticker,
                'cik': company_cik,
                'filing_date': filing_date
            }
        }
        
    except Exception as e:
        print(f"Error extracting XBRL data: {e}")
        traceback.print_exc()
        return None

def save_to_excel(financial_data, output_file=None):
    """
    Save financial data to Excel
    
    Args:
        financial_data (dict): Dictionary containing financial statements
        output_file (str): Output file name (default: ticker_financials.xlsx)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not financial_data:
            print("No financial data to save")
            return False
        
        ticker = financial_data['company_info']['ticker']
        if not output_file:
            output_file = f"{ticker}_financials.xlsx"
        
        print(f"Saving financial data to {output_file}...")
        
        # Convert to DataFrames
        df_bs = pd.DataFrame(financial_data['balance_sheet'])
        df_is = pd.DataFrame(financial_data['income_statement'])
        df_cf = pd.DataFrame(financial_data['cash_flow'])
        
        # Save to Excel
        with pd.ExcelWriter(output_file) as writer:
            df_bs.to_excel(writer, sheet_name='Balance Sheet')
            df_is.to_excel(writer, sheet_name='Income Statement')
            df_cf.to_excel(writer, sheet_name='Cash Flow')
            
            # Add company info sheet
            pd.DataFrame([financial_data['company_info']]).to_excel(writer, sheet_name='Company Info')
        
        print(f"Financial data saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error saving to Excel: {e}")
        traceback.print_exc()
        return False

def main():
    # Get ticker from command line or use default
    ticker = "AAPL"  # Default ticker
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    
    # Extract XBRL data
    financial_data = extract_xbrl_data(ticker)
    
    if financial_data:
        # Save to Excel
        save_to_excel(financial_data)
        return 0
    else:
        print("Failed to extract financial data")
        return 1

if __name__ == "__main__":
    sys.exit(main())
