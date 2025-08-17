"""
Test script for SEC API wrapper with improved caching and retry logic
"""
import os
import time
import logging
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import our modules
import sec_api_wrapper
import sec_api_cache

def test_company_cik_lookup():
    """Test company CIK lookup with caching"""
    logging.info("Testing company CIK lookup with caching...")
    
    # Create a real SEC API wrapper
    api = sec_api_wrapper.SECAPIWrapper(use_mock=False)
    
    # Test tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # First run - should hit the API for some tickers
    start_time = time.time()
    first_run_results = {}
    for ticker in test_tickers:
        try:
            cik = api.get_company_cik(ticker)
            first_run_results[ticker] = cik
            logging.info(f"First run: {ticker} -> CIK: {cik}")
        except Exception as e:
            logging.error(f"Error getting CIK for {ticker}: {e}")
    
    first_run_time = time.time() - start_time
    logging.info(f"First run completed in {first_run_time:.2f} seconds")
    
    # Second run - should use cache for all tickers
    start_time = time.time()
    second_run_results = {}
    for ticker in test_tickers:
        try:
            cik = api.get_company_cik(ticker)
            second_run_results[ticker] = cik
            logging.info(f"Second run: {ticker} -> CIK: {cik}")
        except Exception as e:
            logging.error(f"Error getting CIK for {ticker}: {e}")
    
    second_run_time = time.time() - start_time
    logging.info(f"Second run completed in {second_run_time:.2f} seconds")
    
    # Verify results match
    all_match = True
    for ticker in test_tickers:
        if ticker in first_run_results and ticker in second_run_results:
            if first_run_results[ticker] != second_run_results[ticker]:
                logging.error(f"Mismatch for {ticker}: {first_run_results[ticker]} vs {second_run_results[ticker]}")
                all_match = False
    
    if all_match:
        logging.info("All CIK lookups match between runs")
    
    # Check if second run was faster (should be, due to caching)
    if second_run_time < first_run_time:
        logging.info(f"Caching improved performance: {first_run_time:.2f}s -> {second_run_time:.2f}s")
        logging.info(f"Speed improvement: {(first_run_time - second_run_time) / first_run_time * 100:.1f}%")
    else:
        logging.warning(f"Caching did not improve performance: {first_run_time:.2f}s -> {second_run_time:.2f}s")
    
    return all_match

def test_filing_retrieval():
    """Test filing retrieval with caching and retry logic"""
    logging.info("Testing filing retrieval with caching and retry logic...")
    
    # Create a real SEC API wrapper
    api = sec_api_wrapper.SECAPIWrapper(use_mock=False)
    
    # Test cases: (ticker, form_type)
    test_cases = [
        ('AAPL', '10-K'),
        ('MSFT', '10-Q'),
        ('AMZN', '10-K')
    ]
    
    # First run - should hit the API
    start_time = time.time()
    first_run_results = {}
    for ticker, form_type in test_cases:
        try:
            cik = api.get_company_cik(ticker)
            filing_info = api.get_latest_filing_info(cik, form_type)
            if filing_info:
                first_run_results[(ticker, form_type)] = filing_info['accessionNumber']
                logging.info(f"First run: {ticker} {form_type} -> Accession: {filing_info['accessionNumber']}")
            else:
                logging.warning(f"No {form_type} filing found for {ticker}")
        except Exception as e:
            logging.error(f"Error getting filing for {ticker} {form_type}: {e}")
    
    first_run_time = time.time() - start_time
    logging.info(f"First run completed in {first_run_time:.2f} seconds")
    
    # Second run - should use cache
    start_time = time.time()
    second_run_results = {}
    for ticker, form_type in test_cases:
        try:
            cik = api.get_company_cik(ticker)
            filing_info = api.get_latest_filing_info(cik, form_type)
            if filing_info:
                second_run_results[(ticker, form_type)] = filing_info['accessionNumber']
                logging.info(f"Second run: {ticker} {form_type} -> Accession: {filing_info['accessionNumber']}")
            else:
                logging.warning(f"No {form_type} filing found for {ticker}")
        except Exception as e:
            logging.error(f"Error getting filing for {ticker} {form_type}: {e}")
    
    second_run_time = time.time() - start_time
    logging.info(f"Second run completed in {second_run_time:.2f} seconds")
    
    # Verify results match
    all_match = True
    for case in test_cases:
        if case in first_run_results and case in second_run_results:
            if first_run_results[case] != second_run_results[case]:
                logging.error(f"Mismatch for {case}: {first_run_results[case]} vs {second_run_results[case]}")
                all_match = False
    
    if all_match:
        logging.info("All filing lookups match between runs")
    
    # Check if second run was faster (should be, due to caching)
    if second_run_time < first_run_time:
        logging.info(f"Caching improved performance: {first_run_time:.2f}s -> {second_run_time:.2f}s")
        logging.info(f"Speed improvement: {(first_run_time - second_run_time) / first_run_time * 100:.1f}%")
    else:
        logging.warning(f"Caching did not improve performance: {first_run_time:.2f}s -> {second_run_time:.2f}s")
    
    return all_match

def test_rate_limit_handling():
    """Test rate limit handling with exponential backoff"""
    logging.info("Testing rate limit handling with exponential backoff...")
    
    # Force clear the cache to ensure we hit the API
    sec_api_cache.clear_cache()
    
    # Create a real SEC API wrapper
    api = sec_api_wrapper.SECAPIWrapper(use_mock=False)
    
    # Generate a list of tickers to trigger rate limiting
    # Using 15 tickers should trigger rate limiting (10 requests per second limit)
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'PG', 'JNJ', 'WMT', 'UNH', 'BAC', 'HD']
    
    # Run rapid requests to trigger rate limiting
    start_time = time.time()
    success_count = 0
    failure_count = 0
    
    for ticker in test_tickers:
        try:
            cik = api.get_company_cik(ticker)
            if cik:
                success_count += 1
                logging.info(f"Successfully retrieved CIK for {ticker}: {cik}")
            else:
                failure_count += 1
                logging.warning(f"Failed to retrieve CIK for {ticker}")
        except Exception as e:
            failure_count += 1
            logging.error(f"Error getting CIK for {ticker}: {e}")
    
    total_time = time.time() - start_time
    logging.info(f"Rate limit test completed in {total_time:.2f} seconds")
    logging.info(f"Success: {success_count}, Failures: {failure_count}")
    
    # If we have more successes than failures, our retry logic is working
    return success_count > failure_count

def run_all_tests():
    """Run all SEC API wrapper tests"""
    logging.info("Starting SEC API wrapper tests...")
    
    results = {
        "cik_lookup": test_company_cik_lookup(),
        "filing_retrieval": test_filing_retrieval(),
        "rate_limit_handling": test_rate_limit_handling()
    }
    
    logging.info("Test results summary:")
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logging.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logging.info("All tests passed!")
    else:
        logging.warning("Some tests failed. Check logs for details.")
    
    return all_passed

if __name__ == "__main__":
    try:
        print("Starting SEC API wrapper tests...")
        run_all_tests()
    except Exception as e:
        import traceback
        print(f"Error running tests: {e}")
        traceback.print_exc()
