#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for SEC filing extractor with output to a text file.
"""

import os
import sys
import time
import traceback
from datetime import datetime

# Import the SEC filing extractor module
import sec_filing_extractor

def write_output(message, output_file):
    """Write a message to both stdout and the output file."""
    print(message)
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
        f.flush()  # Ensure it's written immediately

def main():
    """Main function to test the SEC filing extractor."""
    # Create output file (not a .log file to avoid gitignore)
    output_file = "sec_test_output.txt"
    
    # Clear previous output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("")
    
    ticker = "AAPL"
    form_type = "10-K"
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    if len(sys.argv) > 2:
        form_type = sys.argv[2]
    
    output_dir = os.path.join("sec_filings", ticker)
    os.makedirs(output_dir, exist_ok=True)
    
    separator = f"\n{'='*80}"
    write_output(separator, output_file)
    write_output(f"Testing SEC Filing Extractor with {ticker} ({form_type})", output_file)
    write_output(f"Start time: {datetime.now()}", output_file)
    write_output(separator, output_file)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Get company CIK
        write_output("\nStep 1: Getting company CIK...", output_file)
        cik = sec_filing_extractor.get_company_cik(ticker)
        write_output(f"CIK result: {cik}", output_file)
        if not cik:
            write_output("Error: Could not get CIK. Test failed.", output_file)
            return
        
        # Step 2: Get latest filing info
        write_output("\nStep 2: Getting latest filing info...", output_file)
        filing_info = sec_filing_extractor.get_latest_filing_info(cik, form_type)
        write_output(f"Filing info result: {filing_info}", output_file)
        if not filing_info:
            write_output("Error: Could not get filing info. Test failed.", output_file)
            return
        
        # Step 3: Download filing
        write_output("\nStep 3: Downloading filing...", output_file)
        html_content = sec_filing_extractor.download_filing(filing_info)
        if html_content:
            write_output(f"Downloaded HTML content: {len(html_content)} bytes", output_file)
            # Save a sample of the HTML content
            sample_html = html_content[:1000] + "..." if len(html_content) > 1000 else html_content
            write_output(f"Sample HTML content:\n{sample_html}", output_file)
        else:
            write_output("Error: Could not download filing. Test failed.", output_file)
            return
        
        # Step 4: Extract tables
        write_output("\nStep 4: Extracting tables...", output_file)
        tables = sec_filing_extractor.extract_tables(html_content)
        if tables:
            write_output(f"Extracted {len(tables)} tables", output_file)
            # Print info about the first few tables
            for i, table in enumerate(tables[:3]):
                write_output(f"Table {i}: Shape {table.shape}", output_file)
                write_output(f"Sample data:\n{table.head(2)}", output_file)
        else:
            write_output("Error: Could not extract tables. Test failed.", output_file)
            return
        
        # Step 5: Identify financial tables
        write_output("\nStep 5: Identifying financial tables...", output_file)
        financial_tables = sec_filing_extractor.identify_financial_tables(tables)
        write_output(f"Financial tables identified: {list(financial_tables.keys())}", output_file)
        for name, table in financial_tables.items():
            if table is not None:
                write_output(f"{name}: Found", output_file)
            else:
                write_output(f"{name}: Not found", output_file)
        
        # Step 6: Save tables to Excel
        write_output("\nStep 6: Saving tables to Excel...", output_file)
        success = sec_filing_extractor.save_tables_to_excel(financial_tables, tables, ticker, output_dir)
        if success:
            write_output(f"Successfully saved tables to {output_dir}", output_file)
        else:
            write_output("Error: Could not save tables to Excel. Test failed.", output_file)
            return
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        write_output(separator, output_file)
        write_output(f"Test completed successfully in {duration}", output_file)
        write_output(f"Output saved to: {os.path.abspath(output_dir)}", output_file)
        write_output(separator, output_file)
        
    except Exception as e:
        write_output(f"\nError: {str(e)}", output_file)
        write_output(traceback.format_exc(), output_file)
        write_output("Test failed.", output_file)

if __name__ == "__main__":
    main()
