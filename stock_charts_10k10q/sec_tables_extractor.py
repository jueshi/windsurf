from sec_api import QueryApi
import pandas as pd
import requests
import io
import json
import sys
import traceback

from dotenv import load_dotenv
import os

def main():
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("SEC_API_KEY")
        
        if not api_key:
            print("Error: SEC_API_KEY not found in .env file")
            return 1
        
        # Initialize API client
        queryApi = QueryApi(api_key=api_key)
        
        # Set request headers to simulate browser access
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        # Query specific company's 10-K filings
        ticker = "AAPL"  # Default ticker
        if len(sys.argv) > 1:
            ticker = sys.argv[1]
            
        print(f"Searching for {ticker} 10-K filings...")
        
        query = {
            "query": { "query_string": { "query": f"ticker:{ticker} AND formType:10-K" }},
            "from": "0",
            "size": "1",
            "sort": [{ "filedAt": "desc" }]
        }
        
        filings = queryApi.get_filings(query)
        
        if not filings or not filings.get('filings') or len(filings['filings']) == 0:
            print(f"No 10-K filings found for {ticker}")
            return 1
        
        # Get the latest 10-K filing details
        filing_url = filings['filings'][0]['linkToFilingDetails']
        accession_no = filings['filings'][0]['accessionNo']
        company_name = filings['filings'][0]['companyNameLong']
        company_cik = filings['filings'][0]['cik']
        filing_date = filings['filings'][0]['filedAt']
        
        print(f"Found filing: {filing_url}")
        print(f"Company CIK: {company_cik}")
        print(f"Filing Date: {filing_date}")
        
        # Extract XBRL data
        xbrl_json = None
        try:
            print("Extracting XBRL data to JSON...")
            xbrl_json = queryApi.xbrl_to_json(filing_url)
            print("Successfully extracted XBRL data")
        except Exception as e:
            print(f"Error extracting XBRL data: {e}")
            
            # Try alternative method to get XBRL data
            try:
                print("Trying alternative method to get XBRL data...")
                # Get filing details
                filing_details = queryApi.get_filing_details(accession_no)
                
                # Find XBRL file URL
                xbrl_url = None
                for document in filing_details.get('documentFormatFiles', []):
                    if document.get('documentType', '').lower() == 'xbrl instance':
                        xbrl_url = document.get('documentUrl')
                        break
                
                if xbrl_url:
                    print(f"Found XBRL URL: {xbrl_url}")
                    # Use requests to get XBRL content directly
                    response = requests.get(xbrl_url, headers=headers)
                    if response.status_code == 200:
                        # Use SEC API to parse XBRL content
                        xbrl_json = queryApi.xbrl_to_json(response.text)
                        print("Successfully extracted XBRL data using alternative method")
                    else:
                        print(f"Failed to download XBRL file. Status code: {response.status_code}")
                else:
                    print("Could not find XBRL file URL")
            except Exception as inner_e:
                print(f"Error with alternative XBRL extraction: {inner_e}")
                traceback.print_exc()
        
        # Extract financial statements from XBRL
        if xbrl_json:
            try:
                print("Extracting balance sheet...")
                balance_sheet = queryApi.get_balance_sheet(xbrl_json)
                print("Successfully extracted balance sheet")
                
                print("Extracting income statement...")
                income_statement = queryApi.get_income_statement(xbrl_json)
                print("Successfully extracted income statement")
                
                print("Extracting cash flow statement...")
                cash_flow = queryApi.get_cash_flow_statement(xbrl_json)
                print("Successfully extracted cash flow statement")
                
                # Convert to DataFrames for analysis
                df_bs = pd.DataFrame(balance_sheet)
                df_is = pd.DataFrame(income_statement)
                df_cf = pd.DataFrame(cash_flow)
                
                # Export to Excel
                output_file = f"{ticker}_Financial_Statements.xlsx"
                with pd.ExcelWriter(output_file) as writer:
                    df_bs.to_excel(writer, sheet_name='Balance Sheet')
                    df_is.to_excel(writer, sheet_name='Income Statement')
                    df_cf.to_excel(writer, sheet_name='Cash Flow Statement')
                
                print(f"Financial statements exported to {output_file}")
                return 0
            except Exception as e:
                print(f"Error extracting financial statements from XBRL: {e}")
                traceback.print_exc()
        
        # If XBRL extraction failed or no financial statements were found, try HTML tables
        print("Downloading filing HTML content...")
        try:
            response = requests.get(filing_url, headers=headers)
            
            if response.status_code != 200:
                print(f"Failed to download filing. Status code: {response.status_code}")
                print(f"Response: {response.text[:500]}...")  # Print first 500 chars of response
                return 1
            
            html_content = response.text
            print(f"Downloaded {len(html_content)} bytes of HTML content")
            
            # Extract tables from HTML
            print("Extracting tables from HTML...")
            tables = pd.read_html(html_content)
            print(f"Found {len(tables)} tables in the filing")
            
            # Save tables to Excel
            output_file = f"{ticker}_10K_Tables.xlsx"
            with pd.ExcelWriter(output_file) as writer:
                # Save up to 30 tables
                for i, table in enumerate(tables[:30]):
                    sheet_name = f"Table_{i}"
                    table.to_excel(writer, sheet_name=sheet_name)
            
            print(f"Saved {min(len(tables), 30)} tables to {output_file}")
            return 0
            
        except Exception as e:
            print(f"Error extracting tables: {e}")
            traceback.print_exc()
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
