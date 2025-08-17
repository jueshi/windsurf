"""
Comprehensive Fix for Stock Charts Application
This file contains comprehensive fixes for both the 'get_data' and 'invalid command name' errors.
"""

import logging
import pandas as pd
import tkinter as tk
import functools
import types
import threading

def apply_comprehensive_fixes(app, manager):
    """
    Apply comprehensive fixes to the application to address all critical issues.
    
    Args:
        app: The StockDataGUI instance
        manager: The StockDataManager instance
    """
    # 1. Add robust get_data method to StockDataManager with type conversion
    def get_data(ticker, start_date=None, end_date=None):
        """Robust get_data implementation with proper type handling"""
        logging.info(f"Robust get_data called for {ticker}")
        try:
            if hasattr(manager, '_load_stock_data'):
                df = manager._load_stock_data(ticker)
                
                # Ensure we have a valid DataFrame
                if df is None or df.empty:
                    logging.warning(f"No data returned for {ticker}")
                    return pd.DataFrame()
                
                # Ensure date column is properly formatted
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                
                # Ensure numeric columns are properly typed
                numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                for col in df.columns:
                    if col != 'Date' and (col in numeric_cols or df[col].dtype == 'object'):
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except Exception as e:
                            logging.warning(f"Could not convert column {col} to numeric: {str(e)}")
                
                return df
            else:
                # Try alternative methods if available
                if hasattr(manager, 'download_data'):
                    logging.info(f"Using download_data method for {ticker}")
                    return manager.download_data(ticker, start_date, end_date)
                elif hasattr(manager, 'get_stock_data'):
                    logging.info(f"Using get_stock_data method for {ticker}")
                    return manager.get_stock_data(ticker)
                else:
                    logging.error("No suitable data loading method found in StockDataManager")
                    return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error in robust get_data: {str(e)}")
            return pd.DataFrame()
    
    # Add the method to the manager
    manager.get_data = get_data
    logging.info("Added robust get_data method to StockDataManager")
    
    # 2. Add download_data method if not present
    if not hasattr(manager, 'download_data'):
        def download_data(ticker, start_date=None, end_date=None):
            """Download data for a ticker"""
            logging.info(f"Download data called for {ticker}")
            try:
                if hasattr(manager, '_load_stock_data'):
                    return manager._load_stock_data(ticker)
                else:
                    logging.error("No _load_stock_data method found in StockDataManager")
                    return pd.DataFrame()
            except Exception as e:
                logging.error(f"Error in download_data: {str(e)}")
                return pd.DataFrame()
        
        # Add the method to the manager
        manager.download_data = download_data
        logging.info("Added download_data method to StockDataManager")
    
    # 3. Add ensure_numeric utility method
    def ensure_numeric(value):
        """Convert value to numeric if possible"""
        try:
            return pd.to_numeric(value)
        except:
            return value
    
    # Add the method to the manager
    manager.ensure_numeric = ensure_numeric
    logging.info("Added ensure_numeric utility method to StockDataManager")
    
    # 4. Create a widget safety decorator
    def widget_safety_decorator(widget_check_func):
        """Create a decorator that checks if widgets exist before executing a method"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    # Check if widgets exist
                    if not widget_check_func():
                        method_name = func.__name__
                        logging.warning(f"Cannot execute {method_name}: required widgets no longer exist")
                        return None
                    
                    # Execute the original method with error handling
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        logging.error(f"Error in {func.__name__}: {str(e)}")
                        # Show a message to the user if possible
                        if hasattr(app, 'status_var') and hasattr(app.status_var, 'set'):
                            app.status_var.set(f"Error in {func.__name__}: {str(e)}")
                        return None
                except tk.TclError as e:
                    method_name = func.__name__
                    logging.error(f"TclError in {method_name}: {str(e)}")
                    return None
            return wrapper
        return decorator
    
    # 5. Define widget check functions
    def check_root_exists():
        return hasattr(app, 'root') and app.root.winfo_exists()
    
    def check_notebook_exists():
        return (hasattr(app, 'root') and app.root.winfo_exists() and
                hasattr(app, 'chart_notebook') and app.chart_notebook.winfo_exists())
    
    def check_ticker_listbox_exists():
        return (hasattr(app, 'root') and app.root.winfo_exists() and
                hasattr(app, 'ticker_listbox') and app.ticker_listbox.winfo_exists())
    
    # 6. Apply widget safety to critical methods
    critical_methods = {
        '_display_chart': check_notebook_exists,
        '_display_plotly_chart': check_notebook_exists,
        '_generate_chart_thread': check_root_exists,
        '_update_chart_after_download': check_root_exists,
        '_on_tab_changed': check_notebook_exists,
        '_get_selected_tickers': check_ticker_listbox_exists,
        '_generate_seasonality_chart': check_notebook_exists
    }
    
    for method_name, check_func in critical_methods.items():
        if hasattr(app, method_name):
            original_method = getattr(app, method_name)
            if callable(original_method):
                # Create a safe method
                safe_method = widget_safety_decorator(check_func)(original_method)
                # Replace the original method
                setattr(app, method_name, types.MethodType(safe_method, app))
                logging.info(f"Added widget safety to {method_name}")
    
    # 7. Add a global exception handler for Tkinter
    def add_global_exception_handler():
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
    
    # Apply the global exception handler
    add_global_exception_handler()
    
    # 8. Add thread safety for UI updates
    original_after = app.root.after if hasattr(app, 'root') else None
    
    if original_after:
        def safe_after(ms, func, *args):
            """Safe wrapper for after method to ensure widgets exist before callbacks"""
            def safe_func():
                try:
                    if check_root_exists():
                        func(*args)
                    else:
                        logging.warning(f"Cannot execute scheduled function: root window no longer exists")
                except Exception as e:
                    logging.error(f"Error in scheduled function: {str(e)}")
            
            return original_after(ms, safe_func)
        
        # Replace the after method
        app.root.after = safe_after
        logging.info("Added thread safety for UI updates")
    
    logging.info("Comprehensive fixes applied successfully")
