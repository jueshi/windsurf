"""
Direct Fix for Stock Charts Application
This file contains direct fixes for the 'get_data' and 'invalid command name' errors.
"""

import logging
import pandas as pd
import tkinter as tk
import os
import types
import pandas.api.types
from datetime import datetime

# Import Plotly modules for our direct patch
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    logging.warning("Plotly modules not available, interactive charts will be disabled")
    PLOTLY_AVAILABLE = False

def apply_direct_fixes(app, manager):
    """
    Apply direct fixes to address get_data and invalid command name errors
    
    Args:
        app (StockDataGUI): The GUI application instance
        manager (StockDataManager): The data manager instance
    """
    # 0. Patch the _display_chart method to fix date comparison issue
    def patch_date_comparison(app):
        """Patch the date comparison in _display_chart method"""
        # Get the original _is_valid_date method
        original_is_valid_date = app._is_valid_date
        
        # Create an enhanced version that converts dates properly
        def enhanced_is_valid_date(self, date_str):
            """Enhanced version of _is_valid_date that also returns the datetime object"""
            if not date_str or not isinstance(date_str, str):
                return False, None
            try:
                date_obj = pd.to_datetime(date_str)
                return True, date_obj
            except Exception:
                return False, None
        
        # Replace the original method
        app._is_valid_date = types.MethodType(enhanced_is_valid_date, app)
        
        # Get the original _display_chart method
        original_display_chart = app._display_chart
        
        # Create a patched version that handles date comparisons properly
        def patched_display_chart(self, ticker_or_path, *args, **kwargs):
            """Patched version of _display_chart that handles date comparisons properly"""
            try:
                # If this is a ticker (not a path), apply our date filtering fix
                if isinstance(ticker_or_path, str) and not os.path.isfile(ticker_or_path):
                    ticker = ticker_or_path
                    
                    # Get the data first
                    df = self.manager.get_data(ticker)
                    if df is not None and not df.empty:
                        # Apply date range filter if specified, with proper conversion
                        start_date_str = self.start_date_entry.get() if hasattr(self, 'start_date_entry') else ""
                        end_date_str = self.end_date_entry.get() if hasattr(self, 'end_date_entry') else ""
                        
                        # Convert dates properly
                        is_valid_start, start_date_obj = enhanced_is_valid_date(self, start_date_str)
                        is_valid_end, end_date_obj = enhanced_is_valid_date(self, end_date_str)
                        
                        # Ensure index is datetime
                        if not pd.api.types.is_datetime64_any_dtype(df.index):
                            df.index = pd.to_datetime(df.index)
                            
                        # Apply filters with proper datetime objects
                        if is_valid_start:
                            df = df[df.index >= start_date_obj]
                            logging.info(f"Applied start date filter: {start_date_obj}")
                            
                        if is_valid_end:
                            df = df[df.index <= end_date_obj]
                            logging.info(f"Applied end date filter: {end_date_obj}")
                        
                        # Store the filtered data for use in chart creation
                        self._filtered_data = df
                        
                # Call the original method
                return original_display_chart(ticker_or_path, *args, **kwargs)
                
            except Exception as e:
                logging.error(f"Error in patched_display_chart for {ticker_or_path}: {str(e)}")
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Error displaying chart for {ticker_or_path}: {str(e)}")
                return None
        
        # Replace the _display_chart method
        app._display_chart = types.MethodType(patched_display_chart, app)
        logging.info("Patched _display_chart method to fix date comparison issue")
    
    # Apply the date comparison patch
    patch_date_comparison(app)
    
    # 1. Add get_data method to StockDataManager
    def get_data(self, ticker, start_date=None, end_date=None):
        """Direct get_data implementation that uses _load_stock_data with enhanced type conversion"""
        logging.info(f"Direct get_data called for {ticker}")
        try:
            if hasattr(self, '_load_stock_data'):
                df = self._load_stock_data(ticker)
                
                # Ensure we have a valid DataFrame
                if df is None or df.empty:
                    logging.warning(f"No data returned for {ticker}")
                    return pd.DataFrame()
                
                # Ensure date column is properly formatted
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                
                # Define expected numeric columns
                numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                
                # Force convert all known numeric columns
                for col in df.columns:
                    if col != 'Date':
                        try:
                            # Convert all non-date columns to numeric
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            logging.info(f"Converted {col} to numeric type: {df[col].dtype}")
                        except Exception as e:
                            logging.warning(f"Could not convert column {col} to numeric: {str(e)}")
                
                # Ensure index is properly formatted if it's a DatetimeIndex
                if hasattr(df.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(df.index):
                    # Ensure the index is properly formatted
                    df.index = pd.to_datetime(df.index)
                
                logging.info(f"DataFrame dtypes after conversion: {df.dtypes}")
                
                return df
            else:
                logging.error("No _load_stock_data method found in StockDataManager")
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error in direct get_data: {str(e)}")
            return pd.DataFrame()
    
    # Add the method to the manager class
    manager.__class__.get_data = get_data
    logging.info("Added direct get_data method to StockDataManager class")
    
    # 2. Add safe wrapper for _get_selected_tickers
    if hasattr(app, '_get_selected_tickers'):
        original_get_selected_tickers = app._get_selected_tickers
        
        def safe_get_selected_tickers(self, *args, **kwargs):
            """Safe wrapper for _get_selected_tickers"""
            try:
                if not hasattr(self, 'root') or not self.root.winfo_exists():
                    logging.warning("Cannot get selected tickers: root window no longer exists")
                    return []
                    
                if not hasattr(self, 'ticker_listbox') or not self.ticker_listbox.winfo_exists():
                    logging.warning("Cannot get selected tickers: ticker_listbox no longer exists")
                    return []
                
                return original_get_selected_tickers(*args, **kwargs)
            except tk.TclError as e:
                logging.error(f"TclError in _get_selected_tickers: {str(e)}")
                return []
        
        # Replace the method
        app._get_selected_tickers = types.MethodType(safe_get_selected_tickers, app)
        logging.info("Added safe wrapper for _get_selected_tickers")
    
    # 3. Add safe wrapper for _display_chart
    if hasattr(app, '_display_chart'):
        original_display_chart = app._display_chart
        
        def safe_display_chart(self, ticker, *args, **kwargs):
            """Safe wrapper for _display_chart"""
            try:
                if not hasattr(self, 'root') or not self.root.winfo_exists():
                    logging.warning(f"Cannot display chart for {ticker}: root window no longer exists")
                    return
                    
                if not hasattr(self, 'chart_notebook') or not self.chart_notebook.winfo_exists():
                    logging.warning(f"Cannot display chart for {ticker}: chart notebook widget no longer exists")
                    return
                
                return original_display_chart(ticker, *args, **kwargs)
            except Exception as e:
                logging.error(f"Error displaying chart for {ticker}: {str(e)}")
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Error displaying chart for {ticker}: {str(e)}")
                return None
            except tk.TclError as e:
                logging.error(f"TclError in _display_chart: {str(e)}")
                return None
    
    app._original_display_chart = app._display_chart
    app._display_chart = types.MethodType(safe_display_chart, app)
    logging.info("Added safe wrapper for _display_chart")
    
    # 4. Add targeted fix for date comparison issue in chart creation
    def fix_chart_date_filtering(app):
        """Add a targeted fix for the date comparison issue in the chart creation code"""
        # Get the original _display_chart method
        original_display_chart = app._display_chart
        
        def safe_display_chart_with_date_filtering(self, ticker_or_path, *args, **kwargs):
            """Safe wrapper for date filtering in _display_chart with proper type handling"""
            try:
                # Call the original display chart method
                result = original_display_chart(ticker_or_path, *args, **kwargs)
                
                # Monkey patch the _is_valid_date method to ensure proper date handling
                original_is_valid_date = getattr(self, '_is_valid_date', None)
                
                def enhanced_is_valid_date(self, date_str):
                    """Enhanced version of _is_valid_date with better error handling"""
                    if not date_str or not isinstance(date_str, str):
                        return False
                    try:
                        pd.to_datetime(date_str)
                        return True
                    except Exception:
                        return False
                
                if original_is_valid_date:
                    self._is_valid_date = types.MethodType(enhanced_is_valid_date, self)
                
                # Monkey patch the individual chart creation code to handle date filtering properly
                def safe_date_filter(df, start_date_str, end_date_str):
                    """Safely filter DataFrame by date range with proper type handling"""
                    try:
                        filtered_df = df.copy()
                        
                        # Ensure index is datetime
                        if not pd.api.types.is_datetime64_any_dtype(filtered_df.index):
                            filtered_df.index = pd.to_datetime(filtered_df.index)
                            
                        # Apply date filters if valid
                        if start_date_str and enhanced_is_valid_date(self, start_date_str):
                            start_date = pd.to_datetime(start_date_str)
                            filtered_df = filtered_df[filtered_df.index >= start_date]
                            logging.info(f"Applied start date filter: {start_date}")
                            
                        if end_date_str and enhanced_is_valid_date(self, end_date_str):
                            end_date = pd.to_datetime(end_date_str)
                            filtered_df = filtered_df[filtered_df.index <= end_date]
                            logging.info(f"Applied end date filter: {end_date}")
                            
                        return filtered_df
                    except Exception as e:
                        logging.error(f"Error in safe_date_filter: {str(e)}")
                        return df  # Return original DataFrame if filtering fails
                
                # Add the safe_date_filter method to the app
                app.safe_date_filter = types.MethodType(safe_date_filter, app)
                
                return result
                
            except Exception as e:
                logging.error(f"Error in safe_display_chart_with_date_filtering for {ticker_or_path}: {str(e)}")
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Error displaying chart for {ticker_or_path}: {str(e)}")
                return None
            except tk.TclError as e:
                logging.error(f"TclError in safe_display_chart_with_date_filtering: {str(e)}")
                return None
        
        # Replace the _display_chart method with our safe version
        app._display_chart = types.MethodType(safe_display_chart_with_date_filtering, app)
        logging.info("Added targeted fix for date comparison issue in chart creation")
    
    # Apply the targeted fix
    fix_chart_date_filtering(app)
    
    # The targeted fix is now applied
    
    # 5. Add comprehensive fix for dtype comparison issues
    def add_comprehensive_dtype_fix(app):
        """Add a comprehensive fix for dtype comparison issues in chart creation"""
        # 1. Fix the _is_valid_date method to properly handle date conversion
        def enhanced_is_valid_date(self, date_str):
            """Enhanced version of _is_valid_date with proper date conversion"""
            if not date_str or not isinstance(date_str, str):
                return False
            try:
                pd.to_datetime(date_str)
                return True
            except Exception:
                return False
        
        # Replace the original method
        app._is_valid_date = types.MethodType(enhanced_is_valid_date, app)
        logging.info("Enhanced _is_valid_date method with proper date conversion")
        
        # 2. Add a safe DataFrame preparation method for charts
        def prepare_dataframe_for_chart(self, df, ticker):
            """Safely prepare a DataFrame for chart creation with proper dtype handling"""
            if df is None or df.empty:
                logging.warning(f"No data available for {ticker}")
                return None
            
            try:
                # Make a copy to avoid modifying the original
                chart_df = df.copy()
                
                # Ensure index is datetime
                if not pd.api.types.is_datetime64_any_dtype(chart_df.index):
                    chart_df.index = pd.to_datetime(chart_df.index)
                    logging.info(f"Converted index to datetime for {ticker}")
                
                # Force convert all numeric columns
                for col in chart_df.columns:
                    if col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                        try:
                            chart_df[col] = pd.to_numeric(chart_df[col], errors='coerce')
                            logging.info(f"Converted {col} to numeric: {chart_df[col].dtype}")
                        except Exception as e:
                            logging.warning(f"Could not convert {col} to numeric: {str(e)}")
                
                return chart_df
            except Exception as e:
                logging.error(f"Error preparing DataFrame for chart: {str(e)}")
                return None
        
        # Add the method to the app
        app.prepare_dataframe_for_chart = types.MethodType(prepare_dataframe_for_chart, app)
        logging.info("Added prepare_dataframe_for_chart method for safe dtype handling")
        
        # 3. Patch the _display_chart method to use our safe DataFrame preparation
        original_display_chart = app._display_chart
        
        def safe_display_chart(self, ticker_or_path, *args, **kwargs):
            """Safe wrapper for _display_chart with comprehensive dtype handling"""
            # If this is a file path, just pass through to original method
            if isinstance(ticker_or_path, str) and os.path.isfile(ticker_or_path):
                return original_display_chart(ticker_or_path, *args, **kwargs)
            
            try:
                # For ticker symbols, apply our dtype handling fix
                ticker = ticker_or_path
                
                # Get the data
                df = self.manager.get_data(ticker)
                if df is None or df.empty:
                    if hasattr(self, 'status_var'):
                        self.status_var.set(f"No data available for {ticker}")
                    return None
                
                # Apply date range filter with proper conversion
                start_date_str = self.start_date_entry.get() if hasattr(self, 'start_date_entry') else ""
                end_date_str = self.end_date_entry.get() if hasattr(self, 'end_date_entry') else ""
                
                # Prepare the DataFrame with proper dtype handling
                df = self.prepare_dataframe_for_chart(df, ticker)
                if df is None:
                    if hasattr(self, 'status_var'):
                        self.status_var.set(f"Error preparing data for {ticker}")
                    return None
                
                # Apply date filters with proper conversion
                if self._is_valid_date(start_date_str):
                    try:
                        start_date = pd.to_datetime(start_date_str)
                        df = df[df.index >= start_date]
                        logging.info(f"Applied start date filter: {start_date}")
                    except Exception as e:
                        logging.error(f"Error applying start date filter: {str(e)}")
                
                if self._is_valid_date(end_date_str):
                    try:
                        end_date = pd.to_datetime(end_date_str)
                        df = df[df.index <= end_date]
                        logging.info(f"Applied end date filter: {end_date}")
                    except Exception as e:
                        logging.error(f"Error applying end date filter: {str(e)}")
                
                # Store the filtered data for use in chart creation
                self._filtered_df = df
                
                # Monkey patch the _display_plotly_chart method to use our prepared DataFrame
                if hasattr(self, '_display_plotly_chart'):
                    original_display_plotly_chart = self._display_plotly_chart
                    
                    def safe_display_plotly_chart(self, ticker, tab_name):
                        """Safe wrapper for _display_plotly_chart with proper dtype handling"""
                        try:
                            # Use our prepared DataFrame if available
                            if hasattr(self, '_filtered_df') and self._filtered_df is not None:
                                # Store the original DataFrame
                                original_df = None
                                if hasattr(self.manager, 'data') and ticker in self.manager.data:
                                    original_df = self.manager.data[ticker]
                                    # Temporarily replace with our prepared DataFrame
                                    self.manager.data[ticker] = self._filtered_df
                                
                                # Call the original method
                                result = original_display_plotly_chart(ticker, tab_name)
                                
                                # Restore the original DataFrame if needed
                                if original_df is not None:
                                    self.manager.data[ticker] = original_df
                                
                                return result
                            else:
                                return original_display_plotly_chart(ticker, tab_name)
                        except Exception as e:
                            logging.error(f"Error in safe_display_plotly_chart: {str(e)}")
                            return None
                    
                    # Replace the method
                    self._display_plotly_chart = types.MethodType(safe_display_plotly_chart, self)
                    logging.info("Patched _display_plotly_chart method for safe dtype handling")
                
                # Continue with original method
                return original_display_chart(ticker_or_path, *args, **kwargs)
                
            except Exception as e:
                logging.error(f"Error in safe_display_chart: {str(e)}")
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Error displaying chart: {str(e)}")
                return None
        
        # Replace the _display_chart method
        app._display_chart = types.MethodType(safe_display_chart, app)
        logging.info("Applied comprehensive fix for dtype handling in chart creation")
        
        # 4. Add a direct fix for the Plotly chart creation code
        if hasattr(app, '_display_plotly_chart'):
            original_plotly_chart = app._display_plotly_chart
            
            def fixed_display_plotly_chart(self, ticker_or_fig, tab_name):
                """Fixed version of _display_plotly_chart with proper dtype handling"""
                try:
                    # Check if the first argument is a ticker string or a Plotly figure
                    if isinstance(ticker_or_fig, str):
                        # It's a ticker, so get the data and prepare it
                        ticker = ticker_or_fig
                        df = self.manager.get_data(ticker)
                        if df is None or df.empty:
                            if hasattr(self, 'status_var'):
                                self.status_var.set(f"No data available for {ticker}")
                            return None
                        
                        # Make a copy to avoid modifying the original
                        df = df.copy()
                        
                        # Ensure proper data types for all columns
                        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # Ensure index is datetime
                        if not pd.api.types.is_datetime64_any_dtype(df.index):
                            df.index = pd.to_datetime(df.index)
                        
                        # Instead of storing in a non-existent data attribute, 
                        # we'll create a temporary attribute to hold our filtered data
                        if not hasattr(self.manager, '_filtered_data'):
                            self.manager._filtered_data = {}
                        
                        # Store the filtered data for use in chart creation
                        self.manager._filtered_data[ticker] = df
                        
                        # Log the successful data preparation
                        logging.info(f"Prepared filtered data for {ticker} with proper dtype conversion")
                        
                        # Call the original method with the ticker
                        return original_plotly_chart(ticker, tab_name)
                    else:
                        # It's a Plotly figure, pass it directly
                        fig = ticker_or_fig
                        return original_plotly_chart(fig, tab_name)
                except Exception as e:
                    logging.error(f"Error in fixed_display_plotly_chart: {str(e)}")
                    if hasattr(self, 'status_var'):
                        if isinstance(ticker_or_fig, str):
                            self.status_var.set(f"Error displaying chart for {ticker_or_fig}: {str(e)}")
                        else:
                            self.status_var.set(f"Error displaying chart: {str(e)}")
                    return None
            
            # Only apply this fix if we haven't already patched it
            if not hasattr(app, '_filtered_df'):
                app._display_plotly_chart = types.MethodType(fixed_display_plotly_chart, app)
                logging.info("Applied direct fix for Plotly chart creation")
    
    # Apply the comprehensive dtype fix
    add_comprehensive_dtype_fix(app)
    
    # Direct patch for the date comparison in _display_chart method
    def patch_date_comparison_directly(app):
        """Directly patch the date comparison code in _display_chart method"""
        # Get the original _display_chart method
        original_display_chart = app._display_chart
        
        def patched_display_chart(self, ticker_or_path, *args, **kwargs):
            """Directly patched version of _display_chart with fixed date comparison"""
            # If this is a file path, just pass through to original method
            if isinstance(ticker_or_path, str) and os.path.isfile(ticker_or_path):
                return original_display_chart(ticker_or_path, *args, **kwargs)
            
            try:
                # For ticker symbols, apply our direct date comparison fix
                ticker = ticker_or_path
                
                # Get stock data
                df = self.manager.get_data(ticker)
                if df is None or df.empty:
                    if hasattr(self, 'status_var'):
                        self.status_var.set(f"No data available for {ticker}")
                    return None
                
                # Make a copy to avoid modifying the original
                df = df.copy()
                
                # Ensure index is datetime
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    df.index = pd.to_datetime(df.index)
                
                # Apply date range filter with proper conversion
                start_date = self.start_date_entry.get() if hasattr(self, 'start_date_entry') else ""
                end_date = self.end_date_entry.get() if hasattr(self, 'end_date_entry') else ""
                
                # Use proper date comparison with explicit conversion
                if start_date and self._is_valid_date(start_date):
                    start_date_obj = pd.to_datetime(start_date)
                    df = df[df.index >= start_date_obj]
                    logging.info(f"Applied start date filter with explicit conversion: {start_date_obj}")
                
                if end_date and self._is_valid_date(end_date):
                    end_date_obj = pd.to_datetime(end_date)
                    df = df[df.index <= end_date_obj]
                    logging.info(f"Applied end date filter with explicit conversion: {end_date_obj}")
                
                # Instead of storing in a non-existent data attribute, 
                # we'll create a temporary attribute to hold our filtered data
                if not hasattr(self.manager, '_filtered_data'):
                    self.manager._filtered_data = {}
                
                # Store the filtered data for use in chart creation
                self.manager._filtered_data[ticker] = df
                
                # Log the successful data preparation
                logging.info(f"Prepared filtered data for {ticker} with proper date conversion")
                
                # Continue with original method
                return original_display_chart(ticker_or_path, *args, **kwargs)
                
            except Exception as e:
                logging.error(f"Error in patched_display_chart: {str(e)}")
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Error displaying chart: {str(e)}")
                return None
        
        # Replace the _display_chart method
        app._display_chart = types.MethodType(patched_display_chart, app)
        logging.info("Applied direct patch for date comparison in _display_chart")
    
    # Apply the direct patch for date comparison
    patch_date_comparison_directly(app)
    
    # Direct patch for the Plotly chart creation process - more targeted approach
    def patch_plotly_chart_creation(app):
        """Directly patch the Plotly chart creation process to ensure consistent data types"""
        # Only proceed if Plotly is available
        if not PLOTLY_AVAILABLE:
            logging.warning("Skipping Plotly chart patch as Plotly is not available")
            return
            
        # Find the original method that creates the Plotly chart
        def find_plotly_chart_method(app):
            """Find the method that creates the Plotly chart"""
            # Try to find the method that creates the Plotly chart
            for method_name in dir(app):
                if method_name.startswith('_') and 'chart' in method_name.lower():
                    logging.info(f"Found potential chart method: {method_name}")
            
            # Return the display_chart method as fallback
            return app._display_chart
        
        # Get the original chart creation method
        original_method = find_plotly_chart_method(app)
        
        # Create a direct fix for the specific error
        def direct_plotly_fix(self, ticker, *args, **kwargs):
            """Direct fix for the Plotly chart creation error"""
            try:
                # Get the data
                df = None
                if hasattr(self.manager, '_filtered_data') and ticker in self.manager._filtered_data:
                    # Use the filtered data if available
                    df = self.manager._filtered_data[ticker]
                    logging.info(f"Using filtered data for {ticker}")
                else:
                    # Otherwise get the data directly
                    df = self.manager.get_data(ticker)
                    logging.info(f"Using direct data for {ticker}")
                
                if df is None or df.empty:
                    if hasattr(self, 'status_var'):
                        self.status_var.set(f"No data available for {ticker}")
                    return None
                
                # Make a copy to avoid modifying the original
                df = df.copy()
                
                # CRITICAL: Ensure index is datetime
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    df.index = pd.to_datetime(df.index)
                    logging.info(f"Converted index to datetime for {ticker}")
                
                # CRITICAL: Ensure all numeric columns are properly converted
                for col in df.columns:
                    if col != 'Date':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        logging.info(f"Converted {col} to numeric: {df[col].dtype}")
                
                # Store this properly prepared DataFrame for other methods to use
                if not hasattr(self, '_prepared_data'):
                    self._prepared_data = {}
                self._prepared_data[ticker] = df
                
                # Monkey patch the DataFrame access in the app instance
                def get_prepared_data(ticker_name):
                    if hasattr(self, '_prepared_data') and ticker_name in self._prepared_data:
                        return self._prepared_data[ticker_name]
                    return None
                
                # Temporarily attach this method to the app instance
                self._get_prepared_data = get_prepared_data
                
                # Call the original method which should now use our prepared data
                result = original_method(ticker, *args, **kwargs)
                
                return result
                
            except Exception as e:
                logging.error(f"Error in direct_plotly_fix: {str(e)}")
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Error creating chart: {str(e)}")
                return None
        
        # Apply our direct fix
        app._display_chart = types.MethodType(direct_plotly_fix, app)
        logging.info("Applied targeted direct fix for Plotly chart creation")
    
    # Apply the direct patch for Plotly chart creation
    patch_plotly_chart_creation(app)
    
    # 6. Add global exception handler for Tkinter
    def custom_report_callback_exception(self, exc, val, tb):
        """Custom exception handler for Tkinter"""
        if isinstance(val, tk.TclError) and "invalid command name" in str(val):
            logging.warning(f"Caught invalid command name error: {str(val)}")
        else:
            logging.error(f"Tkinter error: {str(val)}")
    
    # Replace the method
    tk.Tk.report_callback_exception = types.MethodType(custom_report_callback_exception, app.root.__class__)
    logging.info("Added global exception handler for Tkinter")
    
    logging.info("Direct fixes applied successfully")
