
import os
import requests
import pandas as pd
import logging
from dotenv import load_dotenv

load_dotenv()

class SECAPIWrapper:
    def __init__(self):
        self.email = os.getenv("SEC_EDGAR_EMAIL")
        if not self.email:
            logging.warning("SEC_EDGAR_EMAIL not set. SEC API access may fail.")

        self.headers = {
            'User-Agent': 'StockToolbox/1.0',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Host': 'data.sec.gov'
        }
        if self.email:
            self.headers['From'] = self.email
            # SEC requires User-Agent to contain email or company name
            self.headers['User-Agent'] = f"StockToolbox/1.0 ({self.email})"

    def get_company_cik(self, ticker):
        ticker = ticker.upper()
        url = "https://www.sec.gov/files/company_tickers.json"

        try:
            # Use separate headers for www.sec.gov vs data.sec.gov
            headers = self.headers.copy()
            headers.pop('Host', None)

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            companies = response.json()

            for _, company in companies.items():
                if company['ticker'] == ticker:
                    return str(company['cik_str']).zfill(10)
            return None
        except Exception as e:
            logging.error(f"Error fetching CIK for {ticker}: {e}")
            return None

    def get_latest_filing_info(self, cik, form_type="10-K"):
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            recent = data.get('filings', {}).get('recent', {})
            form_types = recent.get('form', [])
            accession_numbers = recent.get('accessionNumber', [])
            filing_dates = recent.get('filingDate', [])

            for i, form in enumerate(form_types):
                if form == form_type:
                    accession_number = accession_numbers[i]
                    filing_date = filing_dates[i]
                    # Construct detailed URL
                    # https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number_no_dash}/{accession_number}-index.htm
                    acc_no_dash = accession_number.replace('-', '')
                    detail_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dash}/{accession_number}-index.html"

                    return {
                        "accessionNumber": accession_number,
                        "filingDate": filing_date,
                        "form": form,
                        "detailUrl": detail_url,
                        "cik": cik
                    }
            return None
        except Exception as e:
            logging.error(f"Error fetching filing info for {cik}: {e}")
            return None

    def get_filing_content(self, cik, accession_number):
        # Construct the URL for the actual document.
        # This is tricky because the primary document name varies.
        # We need to parse the index page or try common names.
        # For simplicity/robustness, we'll use the 'index.json' directory listing trick or X-XBRL logic
        # But commonly, the primary document is at:
        # https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dash}/{primary_doc}.htm

        acc_no_dash = accession_number.replace('-', '')
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dash}/index.json"

        try:
            headers = self.headers.copy()
            headers.pop('Host', None)

            response = requests.get(index_url, headers=headers, timeout=10)
            if response.status_code == 200:
                files = response.json().get('directory', {}).get('item', [])
                for file in files:
                    name = file.get('name', '')
                    # Look for the main htm file. usually matches accession number or ticker-date
                    if name.endswith('.htm') and (accession_number in name or '-' not in name or '10k' in name or '10q' in name):
                         doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dash}/{name}"
                         # Fetch content
                         doc_response = requests.get(doc_url, headers=headers, timeout=20)
                         doc_response.raise_for_status()
                         return doc_response.text

            return None
        except Exception as e:
            logging.error(f"Error fetching filing content: {e}")
            return None

    def extract_tables(self, html_content):
        try:
            dfs = pd.read_html(html_content)
            return dfs
        except Exception as e:
            logging.error(f"Error extracting tables: {e}")
            return []

sec_api = SECAPIWrapper()
