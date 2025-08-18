"""
Minimal Fix for Stock Charts Application
This file contains minimal fixes for the 'get_data' and 'invalid command name' errors.
"""

import logging
import pandas as pd
import tkinter as tk

def apply_minimal_fixes(app, manager):
    """
    Apply minimal fixes to the application to address the most critical issues.
    
    Args:
        app: The StockDataGUI instance
        manager: The StockDataManager instance
    """
    # 1. Add get_data method to StockDataManager
    def get_data(ticker, start_date=None, end_date=None):
        """Simple get_data implementation that uses _load_stock_data with type conversion"""
        logging.info(f"Simple get_data called for {ticker}")
        try:
            if hasattr(manager, '_load_stock_data'):
                df = manager._load_stock_data(ticker)
                
                # Ensure date column is properly formatted
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                
                # Ensure numeric columns are properly typed
                for col in df.columns:
                    if col != 'Date' and df[col].dtype == 'object':
                        try:
                            df[col] = pd.to_numeric(df[col])
                        except Exception as e:
                            logging.warning(f"Could not convert column {col} to numeric: {str(e)}")
                
                return df
            else:
                logging.error("No _load_stock_data method found in StockDataManager")
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error in simple get_data: {str(e)}")
            return pd.DataFrame()
    
    # Add the method to the manager
    manager.get_data = get_data
    logging.info("Added simple get_data method to StockDataManager")
    
    # 2. Add safe wrapper for _get_selected_tickers
    if hasattr(app, '_get_selected_tickers'):
        original_get_selected_tickers = app._get_selected_tickers
        
        def safe_get_selected_tickers(*args, **kwargs):
            """Safe wrapper for _get_selected_tickers"""
            try:
                if not hasattr(app, 'root') or not app.root.winfo_exists():
                    logging.warning("Cannot get selected tickers: root window no longer exists")
                    return []
                    
                if not hasattr(app, 'ticker_listbox') or not app.ticker_listbox.winfo_exists():
                    logging.warning("Cannot get selected tickers: ticker_listbox no longer exists")
                    return []
                
                return original_get_selected_tickers(*args, **kwargs)
            except tk.TclError as e:
                logging.error(f"TclError in _get_selected_tickers: {str(e)}")
                return []
        
        # Replace the method
        app._get_selected_tickers = safe_get_selected_tickers
        logging.info("Added safe wrapper for _get_selected_tickers")
    
    # 3. Add global exception handler for Tkinter
    def add_exception_handler():
        """Add a global exception handler for Tkinter errors"""
        original_report_callback_exception = tk.Tk.report_callback_exception
        
        def custom_report_callback_exception(self, exc, val, tb):
            """Custom exception handler for Tkinter"""
            if isinstance(val, tk.TclError) and "invalid command name" in str(val):
                logging.warning(f"Caught invalid command name error: {str(val)}")
            else:
                logging.error(f"Tkinter error: {str(val)}")
                original_report_callback_exception(self, exc, val, tb)
        
        # Replace the method
        tk.Tk.report_callback_exception = custom_report_callback_exception
        logging.info("Added global exception handler for Tkinter")
    
    # 4. Add safe chart display wrapper
    if hasattr(app, '_display_chart'):
        original_display_chart = app._display_chart
        
        def safe_display_chart(ticker, *args, **kwargs):
            """Safe wrapper for _display_chart"""
            try:
                # Check if root and chart_notebook exist
                if not hasattr(app, 'root') or not app.root.winfo_exists():
                    logging.warning(f"Cannot display chart for {ticker}: root window no longer exists")
                    return
                    
                if not hasattr(app, 'chart_notebook') or not app.chart_notebook.winfo_exists():
                    logging.warning(f"Cannot display chart for {ticker}: chart notebook widget no longer exists")
                    return
                
                # Call original method with proper error handling
                try:
                    return original_display_chart(ticker, *args, **kwargs)
                except Exception as e:
                    logging.error(f"Error creating chart for {ticker}: {str(e)}")
                    # Show a message to the user if possible
                    if hasattr(app, 'status_var') and hasattr(app.status_var, 'set'):
                        app.status_var.set(f"Error creating chart for {ticker}: {str(e)}")
            except tk.TclError as e:
                logging.error(f"TclError in _display_chart: {str(e)}")
        
        # Replace the method
        app._display_chart = safe_display_chart
        logging.info("Added safe wrapper for _display_chart")
    
    # Apply the global exception handler
    add_exception_handler()
    
    logging.info("Minimal fixes applied successfully")
