"""
SEC Filing Downloader - DEPRECATED

This module is deprecated. Please use sec_api_wrapper instead:

    from sec_api_wrapper import sec_api
    
    cik = sec_api.get_company_cik(ticker)
    filing_info = sec_api.get_latest_filing_info(cik, form_type)
    html_content = sec_api.download_filing(filing_info)

This module will be removed in a future version.
"""

import warnings
import requests
import sys
import os
import time
import argparse

# Emit deprecation warning when module is imported
warnings.warn(
    "sec_filing_downloader is deprecated. Use sec_api_wrapper instead. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

def get_headers():
    """
    Get headers for SEC EDGAR API requests to avoid 403 errors
    """
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
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
            print(f"Response: {response.text[:500]}...")
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
        return None

def get_latest_filing_info(cik, form_type="10-K"):
    """
    Get the latest filing info for a company
    """
    print(f"Finding latest {form_type} filing for CIK {cik}...")
    try:
        # Get the company's submissions feed
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        response = requests.get(url, headers=get_headers())
        
        if response.status_code != 200:
            print(f"Error fetching company submissions: {response.status_code}")
            print(f"Response: {response.text[:500]}...")
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
        return None

def download_filing(filing_info, output_file):
    """
    Download the filing document and save to file
    """
    print(f"Downloading filing from {filing_info['detailUrl']}...")
    try:
        # Add a delay to avoid hitting SEC rate limits
        time.sleep(1)
        
        response = requests.get(filing_info['detailUrl'], headers=get_headers())
        
        if response.status_code != 200:
            print(f"Error downloading filing: {response.status_code}")
            print(f"Response: {response.text[:500]}...")
            return False
            
        html_content = response.text
        print(f"Successfully downloaded {len(html_content)} bytes")
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Saved filing to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error downloading filing: {e}")
        return False

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Download SEC filings')
    parser.add_argument('ticker', help='Stock ticker symbol')
    parser.add_argument('--form', '-f', default='10-K', choices=['10-K', '10-Q'], help='Form type (default: 10-K)')
    parser.add_argument('--output', '-o', help='Output file (default: ticker_form.html)')
    
    args = parser.parse_args()
    ticker = args.ticker.upper()
    form_type = args.form
    
    if args.output:
        output_file = args.output
    else:
        output_file = f"{ticker}_{form_type.replace('-', '')}.html"
    
    print(f"\n{'='*50}")
    print(f"SEC Filing Downloader - Starting for {ticker} ({form_type})")
    print(f"{'='*50}\n")
    
    try:
        # Get company CIK
        cik = get_company_cik(ticker)
        if not cik:
            print(f"Could not find CIK for {ticker}")
            return 1
        
        # Get latest filing info
        filing_info = get_latest_filing_info(cik, form_type)
        if not filing_info:
            print(f"Could not find {form_type} filing for {ticker}")
            return 1
        
        # Download filing
        success = download_filing(filing_info, output_file)
        if not success:
            print("Failed to download filing")
            return 1
        
        print(f"\n{'='*50}")
        print(f"Successfully downloaded {form_type} filing for {ticker}")
        print(f"{'='*50}\n")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
