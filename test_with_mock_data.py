#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for SEC filing extraction using mock data
"""

import os
import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import SEC API wrapper and set to use mock data
from sec_api_wrapper import use_mock_sec_api
sec_api = use_mock_sec_api(use_mock=True)

# Import SEC filing extractor functions
from sec_filing_extractor import extract_tables, identify_financial_tables, save_tables_to_excel

def test_sec_filing_extraction(ticker="AAPL", form_type="10-K"):
    """
    Test SEC filing extraction using mock data
    
    Args:
        ticker (str): Company ticker symbol
        form_type (str): Form type to extract (10-K or 10-Q)
    """
    output_dir = Path(f"test_output/{ticker}_{form_type}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file
    log_file = output_dir / "extraction_log.txt"
    
    def log(message):
        """Write message to log file and print to console"""
        print(message)
        with open(log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
            f.flush()
    
    # Clear previous log
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("")
    
    log(f"\n{'='*80}")
    log(f"Testing SEC Filing Extraction for {ticker} ({form_type})")
    log(f"Using MOCK SEC data")
    log(f"{'='*80}\n")
    
    start_time = datetime.now()
    
    try:
        # Step 1: Get company CIK
        log("\nStep 1: Getting company CIK...")
        cik = sec_api.get_company_cik(ticker)
        log(f"CIK: {cik}")
        
        if not cik:
            log("Error: Could not get CIK. Test failed.")
            return False
        
        # Step 2: Get latest filing info
        log("\nStep 2: Getting latest filing info...")
        filing_info = sec_api.get_latest_filing_info(cik, form_type)
        
        if not filing_info:
            log("Error: Could not get filing info. Test failed.")
            return False
        
        log(f"Filing date: {filing_info.get('filingDate')}")
        log(f"Accession number: {filing_info.get('accessionNumber')}")
        log(f"Detail URL: {filing_info.get('detailUrl')}")
        
        # Step 3: Download filing
        log("\nStep 3: Downloading filing...")
        html_content = sec_api.download_filing(filing_info)
        
        if not html_content:
            log("Error: Could not download filing. Test failed.")
            return False
        
        log(f"Downloaded {len(html_content)} bytes of HTML content")
        
        # Save HTML content for inspection
        html_file = output_dir / f"{ticker}_{form_type}_filing.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        log(f"Saved HTML content to {html_file}")
        
        # Step 4: Extract tables
        log("\nStep 4: Extracting tables...")
        tables = extract_tables(html_content)
        
        if not tables:
            log("Error: Could not extract tables. Test failed.")
            return False
        
        log(f"Extracted {len(tables)} tables")
        
        # Save all extracted tables for inspection
        all_tables_file = output_dir / f"{ticker}_{form_type}_all_tables.xlsx"
        with pd.ExcelWriter(all_tables_file, engine="openpyxl") as writer:
            for i, table in enumerate(tables):
                sheet_name = f"Table_{i+1}"
                table.to_excel(writer, sheet_name=sheet_name, index=False)
        log(f"Saved all tables to {all_tables_file}")
        
        # Step 5: Identify financial tables
        log("\nStep 5: Identifying financial tables...")
        financial_tables = identify_financial_tables(tables)
        
        log("Financial tables identified:")
        for name, table in financial_tables.items():
            if table is not None:
                log(f"  {name}: Found")
            else:
                log(f"  {name}: Not found")
        
        # Step 6: Save tables to Excel
        log("\nStep 6: Saving tables to Excel...")
        excel_file = output_dir / f"{ticker}_{form_type}_financial_tables.xlsx"
        success = save_tables_to_excel(financial_tables, tables, ticker, str(output_dir))
        
        if not success:
            log("Error: Could not save tables to Excel. Test failed.")
            return False
        
        log(f"Saved financial tables to {output_dir}")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        log(f"\n{'='*80}")
        log(f"Test completed successfully in {duration}")
        log(f"Output saved to: {os.path.abspath(output_dir)}")
        log(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        import traceback
        log(f"\nError: {str(e)}")
        log(traceback.format_exc())
        log("Test failed.")
        return False

if __name__ == "__main__":
    ticker = "AAPL"
    form_type = "10-K"
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    if len(sys.argv) > 2:
        form_type = sys.argv[2]
    
    test_sec_filing_extraction(ticker, form_type)
