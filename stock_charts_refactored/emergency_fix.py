"""
Emergency fix for the persistent dtype comparison error in Plotly chart creation.
This module provides a direct monkey patch for the specific error location.
"""
import os
import logging
import types
import pandas as pd
import numpy as np

def apply_emergency_fix(app):
    """
    Apply an emergency fix for the dtype comparison error in Plotly chart creation.
    This directly monkey patches the specific error location.
    
    Args:
        app: The GUI application instance
    """
    logging.info("Applying emergency fix for dtype comparison error")
    
    # Store original methods
    original_display_chart = app._display_chart
    
    def emergency_display_chart(self, ticker_or_path, *args, **kwargs):
        """
        Emergency replacement for _display_chart that handles dtype issues.
        This is a minimal implementation focused on fixing the specific error.
        
        Args:
            ticker_or_path: The ticker symbol or chart path
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        try:
            logging.info(f"Emergency display chart for {ticker_or_path}")
            
            # Check if this is a ticker or a path
            if isinstance(ticker_or_path, str) and os.path.exists(ticker_or_path):
                # It's a path, use original method
                return original_display_chart(ticker_or_path, *args, **kwargs)
            
            # It's a ticker, apply our fix
            ticker = ticker_or_path
            
            try:
                # Get data safely
                df = None
                try:
                    df = self.manager.get_data(ticker)
                except Exception as e:
                    logging.error(f"Error getting data: {str(e)}")
                    if hasattr(self, 'status_var'):
                        self.status_var.set(f"Error getting data: {str(e)}")
                    return None
                
                if df is None or df.empty:
                    if hasattr(self, 'status_var'):
                        self.status_var.set(f"No data available for {ticker}")
                    return None
                
                # Make a deep copy
                df = df.copy(deep=True)
                
                # CRITICAL: Convert index to datetime
                df.index = pd.to_datetime(df.index)
                
                # CRITICAL: Convert all columns to appropriate types
                for col in df.columns:
                    if col != 'Date':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Apply date filters with string-based comparison to avoid dtype issues
                start_date = self.start_date_entry.get() if hasattr(self, 'start_date_entry') else ""
                end_date = self.end_date_entry.get() if hasattr(self, 'end_date_entry') else ""
                
                # Convert dates to strings for comparison
                if start_date:
                    start_date_obj = pd.to_datetime(start_date)
                    start_date_str = start_date_obj.strftime('%Y-%m-%d')
                    df = df[df.index.strftime('%Y-%m-%d') >= start_date_str]
                
                if end_date:
                    end_date_obj = pd.to_datetime(end_date)
                    end_date_str = end_date_obj.strftime('%Y-%m-%d')
                    df = df[df.index.strftime('%Y-%m-%d') <= end_date_str]
                
                # Create a simple static chart as fallback
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 6))
                    plt.plot(df.index, df['Close'], label=ticker)
                    plt.title(f"{ticker} Stock Price")
                    plt.xlabel("Date")
                    plt.ylabel("Price ($)")
                    plt.legend()
                    plt.grid(True)
                    
                    # Save to temp file
                    import tempfile
                    import os
                    temp_dir = tempfile.gettempdir()
                    chart_path = os.path.join(temp_dir, f"{ticker}_emergency_chart.png")
                    plt.savefig(chart_path)
                    plt.close()
                    
                    # Display static chart
                    if hasattr(self, '_display_static_chart'):
                        self._display_static_chart(chart_path)
                        if hasattr(self, 'status_var'):
                            self.status_var.set(f"Generated emergency chart for {ticker}")
                        logging.info(f"Successfully created emergency chart for {ticker}")
                        return True
                    else:
                        logging.error("_display_static_chart method not available")
                        return False
                
                except Exception as e:
                    logging.error(f"Error creating emergency chart: {str(e)}")
                    if hasattr(self, 'status_var'):
                        self.status_var.set(f"Error creating chart: {str(e)}")
                    return False
            
            except Exception as e:
                logging.error(f"Error in emergency_display_chart: {str(e)}")
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Error creating chart: {str(e)}")
                return False
        
        except Exception as e:
            logging.error(f"Critical error in emergency_display_chart: {str(e)}")
            return False
    
    # Apply the emergency fix
    app._display_chart = types.MethodType(emergency_display_chart, app)
    logging.info("Applied emergency fix for dtype comparison error")

