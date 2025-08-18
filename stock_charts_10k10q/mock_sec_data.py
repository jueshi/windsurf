#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mock SEC data provider for testing SEC filing extraction functionality
without hitting the actual SEC API
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Create mock data directory
MOCK_DATA_DIR = Path("mock_sec_data")
MOCK_DATA_DIR.mkdir(exist_ok=True)

# Sample company data
SAMPLE_COMPANIES = {
    "AAPL": {
        "cik_str": "0000320193",
        "title": "Apple Inc.",
        "ticker": "AAPL"
    },
    "MSFT": {
        "cik_str": "0000789019",
        "title": "Microsoft Corporation",
        "ticker": "MSFT"
    },
    "GOOGL": {
        "cik_str": "0001652044",
        "title": "Alphabet Inc.",
        "ticker": "GOOGL"
    },
    "AMZN": {
        "cik_str": "0001018724",
        "title": "Amazon.com, Inc.",
        "ticker": "AMZN"
    },
    "META": {
        "cik_str": "0001326801",
        "title": "Meta Platforms, Inc.",
        "ticker": "META"
    }
}

# Sample filing data
def generate_sample_filing_data(cik, ticker, form_type="10-K"):
    """Generate sample filing data for a company"""
    cik_no_zeros = cik.lstrip("0")
    accession_number = f"000032019323000001"
    primary_doc = f"{form_type.lower().replace('-', '')}.htm"
    
    # Generate filing dates (most recent first)
    today = datetime.now()
    filing_dates = []
    for i in range(5):
        filing_date = (today - timedelta(days=90*i)).strftime("%Y-%m-%d")
        filing_dates.append(filing_date)
    
    # Generate mock submissions data
    submissions = {
        "cik": cik,
        "entityType": "operating",
        "sic": "3571",
        "sicDescription": "ELECTRONIC COMPUTERS",
        "insiderTransactionForOwnerExists": 1,
        "insiderTransactionForIssuerExists": 1,
        "name": SAMPLE_COMPANIES[ticker]["title"],
        "tickers": [ticker],
        "exchanges": ["Nasdaq"],
        "ein": "942404110",
        "description": "APPLE COMPUTER INC",
        "website": f"https://www.{ticker.lower()}.com",
        "category": "Large accelerated filer",
        "fiscalYearEnd": "0930",
        "stateOfIncorporation": "CA",
        "phone": "408-996-1010",
        "flags": "Large accelerated filer",
        "formerNames": [],
        "filings": {
            "recent": {
                "accessionNumber": [
                    f"{accession_number.replace('1', str(i))}" for i in range(1, 6)
                ],
                "filingDate": filing_dates,
                "reportDate": filing_dates,
                "acceptanceDateTime": [
                    f"{date}T16:01:09.000Z" for date in filing_dates
                ],
                "act": ["34"] * 5,
                "form": [form_type] * 5,
                "fileNumber": [f"001-36743"] * 5,
                "filmNumber": [f"23100{i}" for i in range(1, 6)],
                "items": ["", "", "", "", ""],
                "size": [1000000] * 5,
                "isXBRL": [1] * 5,
                "isInlineXBRL": [1] * 5,
                "primaryDocument": [primary_doc] * 5,
                "primaryDocDescription": [f"{form_type} filing"] * 5
            },
            "files": []
        }
    }
    
    return submissions

# Sample HTML content for a filing
def generate_sample_html_content(ticker, form_type="10-K"):
    """Generate sample HTML content for a filing"""
    company_name = SAMPLE_COMPANIES[ticker]["title"]
    today = datetime.now()
    filing_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    
    # Create sample tables for financial statements
    income_statement = pd.DataFrame({
        "Item": ["Revenue", "Cost of Revenue", "Gross Profit", "Operating Expenses", "Operating Income", "Net Income"],
        "2023": ["$394.33B", "$224.11B", "$170.22B", "$54.32B", "$115.90B", "$96.99B"],
        "2022": ["$365.82B", "$208.58B", "$157.24B", "$50.83B", "$106.41B", "$94.68B"],
        "2021": ["$274.52B", "$169.56B", "$104.96B", "$43.89B", "$61.07B", "$57.41B"]
    })
    
    balance_sheet = pd.DataFrame({
        "Item": ["Cash and Cash Equivalents", "Short-term Investments", "Total Current Assets", 
                "Property, Plant and Equipment", "Total Assets", "Total Current Liabilities", 
                "Long-term Debt", "Total Liabilities", "Total Stockholders' Equity"],
        "2023": ["$29.97B", "$31.58B", "$143.55B", "$42.56B", "$352.83B", "$145.13B", "$110.22B", "$290.40B", "$62.43B"],
        "2022": ["$34.94B", "$26.81B", "$134.84B", "$39.44B", "$336.31B", "$153.98B", "$109.11B", "$302.08B", "$34.23B"]
    })
    
    cash_flow = pd.DataFrame({
        "Item": ["Net Cash from Operating Activities", "Net Cash used in Investing Activities", 
               "Net Cash used in Financing Activities", "Net Increase in Cash"],
        "2023": ["$113.76B", "-$10.55B", "-$110.30B", "-$4.97B"],
        "2022": ["$122.15B", "-$22.73B", "-$110.07B", "-$9.64B"],
        "2021": ["$104.04B", "-$14.55B", "-$93.35B", "-$3.85B"]
    })
    
    # Convert tables to HTML
    income_html = income_statement.to_html(index=False, classes="financial-table income-statement")
    balance_html = balance_sheet.to_html(index=False, classes="financial-table balance-sheet")
    cash_flow_html = cash_flow.to_html(index=False, classes="financial-table cash-flow")
    
    # Create full HTML document
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{company_name} {form_type} Filing</title>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
    </head>
    <body>
        <div class="header">
            <h1>{company_name}</h1>
            <h2>{form_type} Filing</h2>
            <p>For the fiscal year ended September 30, 2023</p>
            <p>Filed on {filing_date}</p>
        </div>
        
        <div class="section">
            <h3>PART I - FINANCIAL INFORMATION</h3>
            <h4>Item 1. Financial Statements</h4>
            
            <h5>CONSOLIDATED STATEMENTS OF OPERATIONS</h5>
            <p>(In millions, except per share amounts)</p>
            {income_html}
            
            <h5>CONSOLIDATED BALANCE SHEETS</h5>
            <p>(In millions, except per share amounts)</p>
            {balance_html}
            
            <h5>CONSOLIDATED STATEMENTS OF CASH FLOWS</h5>
            <p>(In millions)</p>
            {cash_flow_html}
        </div>
        
        <div class="section">
            <h3>PART II - OTHER INFORMATION</h3>
            <p>This is sample text for other information section...</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

class MockSECAPI:
    """Mock SEC API for testing"""
    
    def __init__(self):
        """Initialize mock SEC API"""
        self._initialize_mock_data()
    
    def _initialize_mock_data(self):
        """Initialize mock data files"""
        # Create company tickers JSON
        company_tickers = {}
        for i, (ticker, company) in enumerate(SAMPLE_COMPANIES.items(), 1):
            company_tickers[str(i)] = company
        
        # Save company tickers JSON
        company_tickers_path = MOCK_DATA_DIR / "company_tickers.json"
        with open(company_tickers_path, 'w', encoding='utf-8') as f:
            json.dump(company_tickers, f, indent=2)
        
        # Create submissions JSON for each company
        for ticker, company in SAMPLE_COMPANIES.items():
            cik = str(company["cik_str"]).zfill(10)
            
            # Create 10-K submissions
            submissions_10k = generate_sample_filing_data(cik, ticker, "10-K")
            submissions_10k_path = MOCK_DATA_DIR / f"{ticker}_10K_submissions.json"
            with open(submissions_10k_path, 'w', encoding='utf-8') as f:
                json.dump(submissions_10k, f, indent=2)
            
            # Create 10-Q submissions
            submissions_10q = generate_sample_filing_data(cik, ticker, "10-Q")
            submissions_10q_path = MOCK_DATA_DIR / f"{ticker}_10Q_submissions.json"
            with open(submissions_10q_path, 'w', encoding='utf-8') as f:
                json.dump(submissions_10q, f, indent=2)
            
            # Create HTML content for 10-K
            html_10k = generate_sample_html_content(ticker, "10-K")
            html_10k_path = MOCK_DATA_DIR / f"{ticker}_10K.html"
            with open(html_10k_path, 'w', encoding='utf-8') as f:
                f.write(html_10k)
            
            # Create HTML content for 10-Q
            html_10q = generate_sample_html_content(ticker, "10-Q")
            html_10q_path = MOCK_DATA_DIR / f"{ticker}_10Q.html"
            with open(html_10q_path, 'w', encoding='utf-8') as f:
                f.write(html_10q)
    
    def get_company_tickers(self):
        """Get company tickers JSON"""
        company_tickers_path = MOCK_DATA_DIR / "company_tickers.json"
        with open(company_tickers_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_company_cik(self, ticker):
        """Get company CIK from ticker"""
        if ticker not in SAMPLE_COMPANIES:
            return None
        
        return str(SAMPLE_COMPANIES[ticker]["cik_str"]).zfill(10)
    
    def get_company_submissions(self, cik, form_type="10-K"):
        """Get company submissions"""
        # Find ticker from CIK
        ticker = None
        for t, company in SAMPLE_COMPANIES.items():
            if str(company["cik_str"]).zfill(10) == cik:
                ticker = t
                break
        
        if not ticker:
            return None
        
        # Load submissions file
        submissions_path = MOCK_DATA_DIR / f"{ticker}_{form_type.replace('-', '')}_submissions.json"
        if not submissions_path.exists():
            return None
        
        with open(submissions_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_latest_filing_info(self, cik, form_type="10-K"):
        """Get latest filing info"""
        # Find ticker from CIK
        ticker = None
        for t, company in SAMPLE_COMPANIES.items():
            if str(company["cik_str"]).zfill(10) == cik:
                ticker = t
                break
        
        if not ticker:
            return None
        
        # Get submissions
        submissions = self.get_company_submissions(cik, form_type)
        if not submissions:
            return None
        
        # Get first (latest) filing
        recent = submissions.get("filings", {}).get("recent", {})
        if not recent:
            return None
        
        accession_number = recent.get("accessionNumber", [])[0] if recent.get("accessionNumber") else None
        filing_date = recent.get("filingDate", [])[0] if recent.get("filingDate") else None
        primary_document = recent.get("primaryDocument", [])[0] if recent.get("primaryDocument") else None
        
        if not accession_number or not filing_date or not primary_document:
            return None
        
        # Create filing info
        filing_info = {
            "form": form_type,
            "filingDate": filing_date,
            "accessionNumber": accession_number,
            "primaryDocument": primary_document,
            "detailUrl": f"mock://{ticker}_{form_type.replace('-', '')}.html"
        }
        
        return filing_info
    
    def download_filing(self, filing_info):
        """Download filing document"""
        if not filing_info or "detailUrl" not in filing_info:
            return None
        
        # Parse mock URL to get ticker and form type
        url = filing_info["detailUrl"]
        if not url.startswith("mock://"):
            return None
        
        file_name = url.replace("mock://", "")
        file_path = MOCK_DATA_DIR / file_name
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

# Test function
def test_mock_sec_api():
    """Test mock SEC API"""
    print("Testing Mock SEC API...")
    
    mock_api = MockSECAPI()
    
    # Test get_company_tickers
    print("\nTesting get_company_tickers...")
    tickers = mock_api.get_company_tickers()
    print(f"Found {len(tickers)} companies")
    
    # Test get_company_cik
    print("\nTesting get_company_cik...")
    for ticker in SAMPLE_COMPANIES:
        cik = mock_api.get_company_cik(ticker)
        print(f"{ticker}: CIK = {cik}")
    
    # Test get_latest_filing_info and download_filing
    print("\nTesting get_latest_filing_info and download_filing...")
    for ticker in SAMPLE_COMPANIES:
        cik = mock_api.get_company_cik(ticker)
        
        for form_type in ["10-K", "10-Q"]:
            print(f"\n{ticker} {form_type}:")
            filing_info = mock_api.get_latest_filing_info(cik, form_type)
            
            if filing_info:
                print(f"  Filing date: {filing_info['filingDate']}")
                print(f"  Accession number: {filing_info['accessionNumber']}")
                print(f"  Primary document: {filing_info['primaryDocument']}")
                print(f"  Detail URL: {filing_info['detailUrl']}")
                
                html_content = mock_api.download_filing(filing_info)
                if html_content:
                    print(f"  Downloaded {len(html_content)} bytes of HTML")
                else:
                    print("  Failed to download filing")
            else:
                print(f"  No filing info found")
    
    print("\nMock SEC API test completed")

if __name__ == "__main__":
    test_mock_sec_api()
