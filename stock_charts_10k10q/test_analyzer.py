"""
Simple test script for gemini_analyzer.py
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if GEMINI_API_KEY is set
api_key = os.getenv("GEMINI_API_KEY")
print(f"GEMINI_API_KEY: {'Set' if api_key else 'Not set'}")

# Check if SEC_EDGAR_EMAIL is set
sec_email = os.getenv("SEC_EDGAR_EMAIL")
print(f"SEC_EDGAR_EMAIL: {'Set' if sec_email else 'Not set'}")

# Import the analyze_10k_report function
try:
    from gemini_analyzer import analyze_10k_report
    print("Successfully imported analyze_10k_report function")
except Exception as e:
    print(f"Error importing analyze_10k_report: {e}")
    exit(1)

# Test with a ticker
ticker = "AAPL"
print(f"Testing analyze_10k_report with ticker: {ticker}")

try:
    # Call the function with a timeout
    import threading
    import time
    
    result = [None]
    error = [None]
    
    def run_analysis():
        try:
            result[0] = analyze_10k_report(ticker)
        except Exception as e:
            error[0] = e
    
    # Start the analysis in a separate thread
    thread = threading.Thread(target=run_analysis)
    thread.start()
    
    # Wait for up to 30 seconds
    timeout = 30
    start_time = time.time()
    while thread.is_alive() and time.time() - start_time < timeout:
        print(".", end="", flush=True)
        time.sleep(1)
    
    if thread.is_alive():
        print("\nAnalysis is taking too long, but it's running in the background.")
        print("Check for output files in the current directory.")
    else:
        if error[0]:
            print(f"\nError during analysis: {error[0]}")
        else:
            print(f"\nAnalysis completed. Result length: {len(result[0]) if result[0] else 0}")
            
except Exception as e:
    print(f"Error running analysis: {e}")
