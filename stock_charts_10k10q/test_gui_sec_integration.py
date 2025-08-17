"""
Test script for GUI integration with SEC API wrapper and mock data provider.
This script tests the GUI's ability to handle both real and mock SEC data.
"""

import os
import sys
import time
import logging
import traceback
import tkinter as tk
from tkinter import ttk, messagebox

# Check for tkcalendar package
try:
    from tkcalendar import DateEntry
    print("tkcalendar package imported successfully")
except ImportError:
    print("ERROR: tkcalendar package not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tkcalendar"])
    from tkcalendar import DateEntry
    print("tkcalendar package installed and imported successfully")

from gui import StockDataGUI
from data_manager import StockDataManager
import sec_api_wrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_gui_sec_integration.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_gui_sec_integration():
    """Test the GUI integration with SEC API wrapper and mock data provider."""
    print("Starting GUI SEC integration test")
    logger.info("Starting GUI SEC integration test")
    
    try:
        # Create the root window
        root = tk.Tk()
        root.title("SEC Filing Extraction Test")
        root.geometry("1200x800")
        print("Created root window")
        
        # Create the data manager
        manager = StockDataManager()
        print("Created data manager")
        
        # Create the GUI
        print("Creating GUI...")
        gui = StockDataGUI(root, manager)
        print("GUI created successfully")
        
        # Add test tickers to the manager
        test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        for ticker in test_tickers:
            manager.add_ticker(ticker)
        
        # Log test information
        print(f"Test GUI created with test tickers: {test_tickers}")
        logger.info("Test GUI created with test tickers: %s", test_tickers)
        logger.info("Testing SEC API wrapper integration")
        
        # Test switching between real and mock SEC API
        print("Testing switching to mock SEC API")
        sec_api_wrapper.use_mock_sec_api(True)
        print(f"Using mock API: {sec_api_wrapper.using_mock_api()}")
        assert sec_api_wrapper.using_mock_api() is True
        print("Mock SEC API enabled successfully")
        
        print("Testing switching to real SEC API")
        sec_api_wrapper.use_mock_sec_api(False)
        print(f"Using mock API: {sec_api_wrapper.using_mock_api()}")
        assert sec_api_wrapper.using_mock_api() is False
        print("Real SEC API enabled successfully")
        
        # Switch back to mock for testing
        sec_api_wrapper.use_mock_sec_api(True)
        print("Switched back to mock API for testing")
    except Exception as e:
        print(f"ERROR during setup: {str(e)}")
        logger.error(f"Error during setup: {str(e)}")
        traceback.print_exc()
        return
    
    # Add a callback to run tests after the GUI is fully loaded
    def run_tests():
        try:
            print("Running SEC filing extraction tests")
            logger.info("Running SEC filing extraction tests")
            
            # Test with mock data
            print("Testing with mock data")
            gui.use_mock_data_var.set(True)
            gui._toggle_mock_data()
            print(f"Using mock API after toggle: {sec_api_wrapper.using_mock_api()}")
            assert sec_api_wrapper.using_mock_api() is True
            print("Mock data toggle works correctly")
            
            # Select a test ticker
            test_ticker = "AAPL"
            print(f"Testing SEC filing extraction with mock data for {test_ticker}")
            
            # Set the ticker in the GUI
            gui.sec_ticker_var.set(test_ticker)
            print(f"Set ticker to {test_ticker}")
            
            # Set form type to 10-K
            gui.sec_form_type_var.set("10-K")
            print("Set form type to 10-K")
            
            # Extract tables
            print("Extracting 10-K tables with mock data")
            gui._extract_sec_tables_from_tab()
            
            # Wait for extraction to complete (in real usage, you'd wait for a signal)
            print("Waiting for extraction to complete...")
            time.sleep(5)  # Increased wait time
            
            # Check status
            status = gui.sec_status_var.get()
            print(f"SEC status after extraction: {status}")
            
            # Test with real data (optional - commented out to avoid hitting real SEC API)
            """
            print("Testing with real SEC API")
            gui.use_mock_data_var.set(False)
            gui._toggle_mock_data()
            assert sec_api_wrapper.using_mock_api() is False
            
            # Extract tables with real data
            print(f"Extracting 10-K tables with real SEC API for {test_ticker}")
            gui._extract_sec_tables_from_tab()
            
            # Wait for extraction to complete
            print("Waiting for extraction to complete...")
            time.sleep(10)  # Real API takes longer
            
            # Check status
            status = gui.sec_status_var.get()
            print(f"SEC status after real extraction: {status}")
            """
            
            print("Tests completed successfully")
            
            # Close the GUI after tests
            root.after(3000, root.destroy)  # Increased delay before closing
            
        except Exception as e:
            print(f"ERROR during tests: {str(e)}")
            traceback.print_exc()
            messagebox.showerror("Test Error", f"Test failed: {str(e)}")
            root.after(3000, root.destroy)
    
    # Schedule the tests to run after the GUI is fully loaded
    root.after(1000, run_tests)
    
    # Start the GUI main loop
    root.mainloop()
    
    logger.info("GUI SEC integration test completed")

if __name__ == "__main__":
    print("Starting test script")
    try:
        test_gui_sec_integration()
        print("Test completed")
    except Exception as e:
        print(f"ERROR in main: {str(e)}")
        traceback.print_exc()
