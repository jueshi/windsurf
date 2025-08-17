"""
Safe ticker selection wrapper for StockDataGUI
This file contains a safe wrapper method for _get_selected_tickers
that adds widget existence checks to prevent "invalid command name" errors.
"""

import logging
import tkinter as tk
from tkinter import messagebox

def safe_get_selected_tickers(self):
    """
    Safe wrapper for _get_selected_tickers that adds widget existence checks
    to prevent "invalid command name" errors when widgets are destroyed.
    """
    try:
        # Check if root window and ticker_listbox still exist
        if not hasattr(self, 'root') or not self.root.winfo_exists():
            logging.warning("Cannot get selected tickers: root window no longer exists")
            return []
            
        if not hasattr(self, 'ticker_listbox') or not self.ticker_listbox.winfo_exists():
            logging.warning("Cannot get selected tickers: ticker_listbox no longer exists")
            return []
        
        # Call the original method
        return self._original_get_selected_tickers()
        
    except tk.TclError as e:
        logging.error(f"TclError getting selected tickers: {str(e)}")
        return []

def patch_stock_data_gui(gui_instance):
    """
    Patch the StockDataGUI instance with safe ticker selection method.
    
    Args:
        gui_instance: Instance of StockDataGUI to patch
    """
    # Store reference to original method
    gui_instance._original_get_selected_tickers = gui_instance._get_selected_tickers
    
    # Replace with safe wrapper
    gui_instance._get_selected_tickers = lambda: safe_get_selected_tickers(gui_instance)
    
    logging.info("StockDataGUI patched with safe ticker selection method")
