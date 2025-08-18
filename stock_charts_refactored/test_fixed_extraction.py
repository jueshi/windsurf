"""
Test script to validate the fixed extraction logic in gemini_analyzer_fixed.py
"""

import os
import sys
import logging
from pathlib import Path

# Import the fixed analyzer module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import gemini_analyzer_fixed as analyzer

def setup_logging():
    """Set up logging to both console and file"""
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extraction_validation")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set up logging
    log_file = os.path.join(output_dir, "fixed_extraction_test.log")
    
    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging to: {log_file}")
    return output_dir

def test_extraction():
    """Test the extraction of all three sections using the fixed logic"""
    # Set up logging and output directory
    output_dir = setup_logging()
    
    # Define the test file path - use the AAPL 10-K file
    file_path = "c:\\Users\\juesh\\OneDrive\\Documents\\windsurf\\stock_charts_refactored\\sec-edgar-filings\\AAPL\\10-K\\0000320193-24-000123\\full-submission.txt"
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return
    
    logging.info(f"Using 10-K file: {file_path}")
    
    # Load the report
    logging.info(f"Loading 10-K report...")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            report_text = file.read()
        logging.info(f"Report loaded. Length: {len(report_text)} characters")
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        return
    
    # Extract sections using the fixed extraction logic
    logging.info("Extracting Business section...")
    business_section = analyzer.extract_section_robust(
        report_text, 
        analyzer.business_start_patterns, 
        analyzer.business_end_patterns, 
        section_type='business'
    )
    logging.info(f"Business section extracted: {len(business_section) if business_section else 0} characters")
    
    logging.info("Extracting Risk Factors section...")
    risk_factors_section = analyzer.extract_section_robust(
        report_text, 
        analyzer.risk_start_patterns, 
        analyzer.risk_end_patterns, 
        section_type='risk'
    )
    logging.info(f"Risk Factors section extracted: {len(risk_factors_section) if risk_factors_section else 0} characters")
    
    logging.info("Extracting MD&A section...")
    mda_section = analyzer.extract_section_robust(
        report_text, 
        analyzer.mda_start_patterns, 
        analyzer.mda_end_patterns, 
        section_type='mda'
    )
    logging.info(f"MD&A section extracted: {len(mda_section) if mda_section else 0} characters")
    
    # Write the extracted sections to files for review
    if business_section:
        business_file = os.path.join(output_dir, "Business_fixed.txt")
        with open(business_file, 'w', encoding='utf-8') as file:
            file.write(business_section)
        logging.info(f"Business section written to: {business_file}")
    
    if risk_factors_section:
        risk_file = os.path.join(output_dir, "Risk_Factors_fixed.txt")
        with open(risk_file, 'w', encoding='utf-8') as file:
            file.write(risk_factors_section)
        logging.info(f"Risk Factors section written to: {risk_file}")
    
    if mda_section:
        mda_file = os.path.join(output_dir, "MD&A_fixed.txt")
        with open(mda_file, 'w', encoding='utf-8') as file:
            file.write(mda_section)
        logging.info(f"MD&A section written to: {mda_file}")
    
    # Summary of results
    logging.info("\n--- EXTRACTION SUMMARY ---")
    logging.info(f"Business section: {'✓ Extracted' if business_section else '✗ Failed'} {len(business_section) if business_section else 0} characters")
    logging.info(f"Risk Factors section: {'✓ Extracted' if risk_factors_section else '✗ Failed'} {len(risk_factors_section) if risk_factors_section else 0} characters")
    logging.info(f"MD&A section: {'✓ Extracted' if mda_section else '✗ Failed'} {len(mda_section) if mda_section else 0} characters")

if __name__ == "__main__":
    test_extraction()
