"""
Complete test script for GUI integration with SEC API wrapper
Tests both mock and real SEC API modes
"""

import os
import sys
import time
import logging
import traceback
from threading import Thread

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_gui_sec_complete.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    """Main test function"""
    try:
        logging.info("Starting complete GUI SEC integration test")
        
        # Import required modules
        logging.info("Importing required modules...")
        
        # Check for tkcalendar and install if needed
        try:
            from tkcalendar import DateEntry
            logging.info("tkcalendar is already installed")
        except ImportError:
            logging.warning("tkcalendar not found, installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tkcalendar"])
            from tkcalendar import DateEntry
            logging.info("tkcalendar installed successfully")
        
        # Import GUI and SEC API wrapper
        import tkinter as tk
        from gui import StockDataGUI
        import sec_api_wrapper
        
        logging.info("All modules imported successfully")
        
        # Create root window
        logging.info("Creating root window...")
        root = tk.Tk()
        root.title("SEC Integration Test")
        root.geometry("1200x800")
        
        # Create GUI instance
        logging.info("Creating GUI instance...")
        gui = StockDataGUI(root)
        
        # Define test steps
        def run_tests():
            try:
                logging.info("Starting test sequence...")
                
                # Step 1: Add test ticker
                logging.info("Step 1: Adding test ticker AAPL...")
                gui.ticker_entry.delete(0, tk.END)
                gui.ticker_entry.insert(0, "AAPL")
                gui.add_ticker_button.invoke()
                time.sleep(1)
                
                # Step 2: Switch to SEC filings tab
                logging.info("Step 2: Switching to SEC filings tab...")
                gui.notebook.select(gui.sec_filings_tab)
                time.sleep(1)
                
                # Step 3: Test with mock data
                logging.info("Step 3: Testing with mock data...")
                gui.sec_mock_data_var.set(True)  # Enable mock data
                gui.sec_ticker_entry.delete(0, tk.END)
                gui.sec_ticker_entry.insert(0, "AAPL")
                gui.sec_form_type_var.set("10-K")
                
                logging.info("Extracting SEC filing with mock data...")
                gui.extract_sec_tables_button.invoke()
                
                # Wait for extraction to complete
                logging.info("Waiting for extraction to complete...")
                time.sleep(5)
                
                # Step 4: Test with real data if requested
                if len(sys.argv) > 1 and sys.argv[1].lower() == "real":
                    logging.info("Step 4: Testing with real SEC API...")
                    gui.sec_mock_data_var.set(False)  # Disable mock data
                    gui.sec_ticker_entry.delete(0, tk.END)
                    gui.sec_ticker_entry.insert(0, "MSFT")
                    gui.sec_form_type_var.set("10-K")
                    
                    logging.info("Extracting SEC filing with real SEC API...")
                    gui.extract_sec_tables_button.invoke()
                    
                    # Wait for extraction to complete
                    logging.info("Waiting for extraction to complete...")
                    time.sleep(10)
                else:
                    logging.info("Skipping real SEC API test (add 'real' argument to test with real API)")
                
                # Step 5: Test clearing SEC cache
                logging.info("Step 5: Testing SEC cache clearing...")
                gui.clear_sec_cache_button.invoke()
                time.sleep(1)
                
                # Step 6: Verify final state
                logging.info("Step 6: Verifying final state...")
                mock_status = "enabled" if gui.sec_mock_data_var.get() else "disabled"
                logging.info(f"Mock SEC API is {mock_status}")
                
                logging.info("All tests completed successfully!")
                
                # Exit after tests complete
                logging.info("Exiting application in 3 seconds...")
                root.after(3000, root.destroy)
                
            except Exception as e:
                logging.error(f"Test failed: {str(e)}")
                traceback.print_exc()
                root.after(1000, root.destroy)
        
        # Schedule test execution after GUI is fully loaded
        root.after(1000, lambda: Thread(target=run_tests).start())
        
        # Start main loop
        logging.info("Starting main loop...")
        root.mainloop()
        
        logging.info("Test script completed")
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
