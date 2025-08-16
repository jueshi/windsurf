"""
Comprehensive fix for Plotly chart creation in the stock chart application.
This module provides a direct replacement for the chart creation functionality
to ensure proper data type handling and prevent dtype comparison errors.
"""

import os
import logging
import pandas as pd
import numpy as np
import types
from datetime import datetime

# Import Plotly modules
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    logging.warning("Plotly modules not available, interactive charts will be disabled")
    PLOTLY_AVAILABLE = False

def apply_plotly_fix(app):
    """
    Apply a comprehensive fix for Plotly chart creation.
    This completely replaces the chart creation functionality with a robust implementation.
    
    Args:
        app: The GUI application instance
    """
    if not PLOTLY_AVAILABLE:
        logging.warning("Skipping Plotly fix as Plotly is not available")
        return
    
    # Store the original methods
    original_display_chart = app._display_chart
    
    def safe_plotly_chart(self, ticker, *args, **kwargs):
        """
        Safe implementation of Plotly chart creation that handles dtype issues.
        
        Args:
            ticker: The ticker symbol to create a chart for
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            bool: True if chart was created successfully, False otherwise
        """
        # Add more detailed logging
        logging.info(f"===== STARTING SAFE PLOTLY CHART FOR {ticker} =====")
        try:
            logging.info(f"Creating safe Plotly chart for {ticker}")
            
            # Get the data safely
            try:
                df = self.manager.get_data(ticker)
                if df is None or df.empty:
                    if hasattr(self, 'status_var'):
                        self.status_var.set(f"No data available for {ticker}")
                    return False
            except Exception as e:
                logging.error(f"Error getting data for {ticker}: {str(e)}")
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Error getting data for {ticker}: {str(e)}")
                return False
            
            # Make a deep copy to avoid modifying the original
            df = df.copy(deep=True)
            
            # CRITICAL: Convert index to datetime explicitly
            try:
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    df.index = pd.to_datetime(df.index)
                    logging.info(f"Converted index to datetime for {ticker}")
            except Exception as e:
                logging.error(f"Error converting index to datetime: {str(e)}")
                # Continue anyway, we'll handle it later
            
            # Apply date filters safely
            try:
                # Get date range from UI if available
                start_date = self.start_date_entry.get() if hasattr(self, 'start_date_entry') else ""
                end_date = self.end_date_entry.get() if hasattr(self, 'end_date_entry') else ""
                
                # Apply filters with explicit conversion
                if start_date and hasattr(self, '_is_valid_date') and self._is_valid_date(start_date):
                    start_date_obj = pd.to_datetime(start_date)
                    # Convert to string format that matches index for safe comparison
                    start_date_str = start_date_obj.strftime('%Y-%m-%d')
                    # Filter using string comparison to avoid dtype issues
                    df = df[df.index.strftime('%Y-%m-%d') >= start_date_str]
                    logging.info(f"Applied start date filter: {start_date_str}")
                
                if end_date and hasattr(self, '_is_valid_date') and self._is_valid_date(end_date):
                    end_date_obj = pd.to_datetime(end_date)
                    # Convert to string format that matches index for safe comparison
                    end_date_str = end_date_obj.strftime('%Y-%m-%d')
                    # Filter using string comparison to avoid dtype issues
                    df = df[df.index.strftime('%Y-%m-%d') <= end_date_str]
                    logging.info(f"Applied end date filter: {end_date_str}")
            except Exception as e:
                logging.error(f"Error applying date filters: {str(e)}")
                # Continue with unfiltered data
            
            # CRITICAL: Ensure all columns are numeric
            try:
                for col in df.columns:
                    if col != 'Date':
                        # Convert to numeric with coercion to handle non-numeric values
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        logging.info(f"Converted {col} to numeric: {df[col].dtype}")
            except Exception as e:
                logging.error(f"Error converting columns to numeric: {str(e)}")
                # Continue anyway, we'll handle it in the next step
            
            # Create Plotly figure with explicit type conversion
            try:
                # Create figure with subplots
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.1, 
                                   row_heights=[0.7, 0.3])
                
                # CRITICAL: Convert all data to Python native types to avoid dtype issues
                # Log DataFrame information before conversion
                logging.info(f"DataFrame index type: {type(df.index).__name__}")
                logging.info(f"DataFrame index dtype: {df.index.dtype}")
                for col in df.columns:
                    logging.info(f"Column {col} dtype: {df[col].dtype}")
                
                # Convert index to list of strings with detailed logging
                try:
                    x_dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in df.index]
                    logging.info(f"Converted index to list of strings: {type(x_dates).__name__}, first few elements: {x_dates[:3]}")
                except Exception as e:
                    logging.error(f"Error converting index to strings: {str(e)}")
                    # Fallback conversion
                    x_dates = [str(d) for d in df.index]
                    logging.info(f"Used fallback string conversion for index")
                
                # Convert data columns to lists of native Python types with detailed logging
                try:
                    open_data = df['Open'].tolist() if 'Open' in df.columns else []
                    logging.info(f"Converted Open to list: {type(open_data).__name__}, first few elements: {open_data[:3]}")
                    
                    high_data = df['High'].tolist() if 'High' in df.columns else []
                    logging.info(f"Converted High to list: {type(high_data).__name__}, first few elements: {high_data[:3]}")
                    
                    low_data = df['Low'].tolist() if 'Low' in df.columns else []
                    logging.info(f"Converted Low to list: {type(low_data).__name__}, first few elements: {low_data[:3]}")
                    
                    close_data = df['Close'].tolist() if 'Close' in df.columns else []
                    logging.info(f"Converted Close to list: {type(close_data).__name__}, first few elements: {close_data[:3]}")
                    
                    volume_data = df['Volume'].tolist() if 'Volume' in df.columns else []
                    logging.info(f"Converted Volume to list: {type(volume_data).__name__}, first few elements: {volume_data[:3]}")
                except Exception as e:
                    logging.error(f"Error converting columns to lists: {str(e)}")
                    # Continue with empty lists as fallback
                    open_data, high_data, low_data, close_data, volume_data = [], [], [], [], []
                    logging.info("Using empty lists as fallback for data columns")
                
                # Add candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=x_dates,
                        open=open_data,
                        high=high_data,
                        low=low_data,
                        close=close_data,
                        name=ticker
                    ),
                    row=1, col=1
                )
                
                # Add volume chart
                fig.add_trace(
                    go.Bar(x=x_dates, y=volume_data, name='Volume', marker_color='rgba(0,0,255,0.5)'),
                    row=2, col=1
                )
                
                # Update layout
                fig.update_layout(
                    title=f'{ticker} Stock Price',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    yaxis2_title='Volume',
                    height=600,
                    xaxis_rangeslider_visible=False
                )
                
                # Display the chart
                if hasattr(self, '_display_plotly_chart') and callable(self._display_plotly_chart):
                    self._display_plotly_chart(fig, tab="individual")
                    if hasattr(self, 'status_var'):
                        self.status_var.set(f"Generated interactive chart for {ticker}")
                    logging.info(f"Successfully created Plotly chart for {ticker}")
                    return True
                else:
                    logging.error("_display_plotly_chart method not available")
                    return False
                
            except Exception as e:
                logging.error(f"Error creating Plotly figure: {str(e)}")
                # Fall back to original method
                return original_display_chart(ticker, *args, **kwargs)
                
        except Exception as e:
            logging.error(f"Error in safe_plotly_chart: {str(e)}")
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Error creating chart: {str(e)}")
            # Fall back to original method
            return original_display_chart(ticker, *args, **kwargs)
    
    # Replace the _display_chart method
    app._display_chart = types.MethodType(safe_plotly_chart, app)
    logging.info("Applied comprehensive Plotly chart fix")
