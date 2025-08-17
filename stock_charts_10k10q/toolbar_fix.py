"""
Toolbar fix to prevent navigation toolbar from disappearing.
This patch fixes the double packing issue in direct_chart_fix.py.
"""

import logging
import types
import tkinter as tk
from tkinter import ttk

def apply_toolbar_fix(gui_instance):
    """
    Apply a fix to prevent the navigation toolbar from disappearing.
    
    Args:
        gui_instance: The StockDataGUI instance to patch
    """
    logging.info("Applying toolbar visibility fix...")
    
    # Store the original method if it exists
    if hasattr(gui_instance, '_direct_display_chart'):
        original_direct_display_chart = gui_instance._direct_display_chart
        
        def fixed_direct_display_chart(self, ticker, frame):
            """
            Fixed version of _direct_display_chart that properly handles toolbar visibility.
            
            Args:
                ticker: Ticker symbol to display chart for
                frame: Frame to display the chart in
            """
            try:
                # Check if root window still exists before proceeding
                if not hasattr(self, 'root') or not self.root.winfo_exists():
                    logging.warning(f"Cannot display direct chart for {ticker}: root window no longer exists")
                    return False
                    
                # Check if frame still exists
                if not frame.winfo_exists():
                    logging.warning(f"Cannot display direct chart for {ticker}: frame no longer exists")
                    return False
                
                # Clear existing widgets in the frame
                for widget in frame.winfo_children():
                    widget.destroy()
                
                # Get stock data
                df = self.manager.get_data(ticker)
                if df is None or df.empty:
                    if hasattr(self, 'status_var'):
                        self.status_var.set(f"No data available for {ticker}")
                    return False
                
                # Apply date range filter if specified
                start_date = self.start_date_entry.get()
                end_date = self.end_date_entry.get()
                
                # Log original DataFrame info
                logging.info(f"Original index min: {df.index.min()}, max: {df.index.max()}")
                logging.info(f"Original index dtype: {df.index.dtype}")
                
                # Apply date filtering with proper timezone handling
                if self._is_valid_date(start_date):
                    import pandas as pd
                    from datetime import datetime, timedelta
                    import pytz
                    
                    # Convert start_date to datetime with timezone
                    start_dt = pd.to_datetime(start_date)
                    if start_dt.tzinfo is None:
                        # Make timezone-aware if it's naive
                        if df.index.tzinfo is not None:
                            start_dt = start_dt.tz_localize(df.index.tzinfo)
                        else:
                            start_dt = start_dt.tz_localize('UTC')
                    
                    logging.info(f"Using timezone-aware normalized start date: {start_dt} with tz={start_dt.tzinfo}")
                    
                    # Create mask for start date
                    start_mask = df.index >= start_dt
                    logging.info(f"Start date mask: {start_mask.sum()} rows matched out of {len(df)}")
                    
                    # Apply start date filter
                    df = df[start_mask]
                
                if self._is_valid_date(end_date):
                    import pandas as pd
                    from datetime import datetime, timedelta
                    import pytz
                    
                    # Convert end_date to datetime with timezone and add one day to include the end date
                    end_dt = pd.to_datetime(end_date) + timedelta(days=1)
                    if end_dt.tzinfo is None:
                        # Make timezone-aware if it's naive
                        if df.index.tzinfo is not None:
                            end_dt = end_dt.tz_localize(df.index.tzinfo)
                        else:
                            end_dt = end_dt.tz_localize('UTC')
                    
                    logging.info(f"Using timezone-aware normalized end date: {end_dt} with tz={end_dt.tzinfo}")
                    
                    # Create mask for end date
                    end_mask = df.index <= end_dt
                    logging.info(f"End date mask: {end_mask.sum()} rows matched out of {len(df)}")
                    
                    # Apply end date filter
                    df = df[end_mask]
                
                # Check if we have any data left after filtering
                if df.empty:
                    logging.warning("Date filtering removed all data")
                    # Revert to original data
                    df = self.manager.get_data(ticker)
                    logging.info(f"Reverting to original DataFrame with shape: {df.shape}")
                
                # Ensure numeric types for calculations
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Log DataFrame info after processing
                logging.info(f"DataFrame shape after processing: {df.shape}")
                logging.info(f"DataFrame first few rows after processing:\n{df.head()}")
                
                # Store the filtered DataFrame for later use
                self.filtered_df = df
                logging.info(f"Stored filtered DataFrame with shape {df.shape} for plotting")
                
                # Create figure with subplots
                import matplotlib.pyplot as plt
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                
                # Create figure with subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
                
                # Plot price data
                logging.info(f"Plotting {len(df)} data points for {ticker} from filtered data")
                ax1.plot(df.index, df['Close'], label=f'{ticker} Close Price')
                ax1.set_title(f'{ticker} Stock Price')
                ax1.set_ylabel('Price ($)')
                ax1.grid(True)
                ax1.legend()
                
                # Plot volume data
                ax2.bar(df.index, df['Volume'], color='blue', alpha=0.5)
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Volume')
                ax2.grid(True)
                
                # Add text label if no data in selected range
                if len(df) < 5:
                    ax1.text(0.5, 0.5, 'Insufficient data for selected date range',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax1.transAxes)
                    ax2.text(0.5, 0.5, 'Insufficient data for selected date range',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax2.transAxes)
                
                # Adjust layout
                plt.tight_layout()
                
                # Create a canvas to display the figure in the frame
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                
                # FIXED: Only pack the canvas once
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # Add toolbar
                from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
                toolbar = NavigationToolbar2Tk(canvas, frame)
                toolbar.update()
                
                # FIXED: Don't pack the canvas again, which would hide the toolbar
                # canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # This line is removed
                
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Generated chart for {ticker}")
                
                logging.info(f"Successfully displayed direct chart for {ticker}")
                return True
                
            except Exception as e:
                logging.error(f"Error displaying direct chart for {ticker}: {e}")
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Error displaying chart: {str(e)}")
                return False
        
        # Replace the original method with our fixed version
        gui_instance._direct_display_chart = types.MethodType(fixed_direct_display_chart, gui_instance)
        logging.info("Toolbar visibility fix applied successfully")
    else:
        logging.warning("Could not apply toolbar fix: _direct_display_chart method not found")

def apply_fixes(gui_instance):
    """
    Apply all fixes to the GUI instance.
    
    Args:
        gui_instance: The StockDataGUI instance to patch
    """
    # Apply the toolbar visibility fix
    apply_toolbar_fix(gui_instance)
    
    logging.info("All toolbar visibility fixes applied successfully")
