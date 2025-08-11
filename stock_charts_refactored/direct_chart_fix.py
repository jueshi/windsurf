"""
Direct fix for individual chart display issues.
This module provides a direct implementation of chart display functionality.
"""

import logging
import types
import os
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import pytz

def apply_direct_chart_fix(app):
    """
    Apply a direct fix for individual chart display issues.
    This completely replaces the chart display functionality with a reliable implementation.
    
    Args:
        app: The GUI application instance
    """
    logging.info("Applying direct fix for individual chart display")
    
    # Store original methods
    original_display_chart = app._display_chart
    
    def direct_display_chart(self, ticker_or_path, *args, **kwargs):
        """
        Direct replacement for _display_chart that ensures charts are displayed.
        
        Args:
            ticker_or_path: The ticker symbol or chart path
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        try:
            logging.info(f"Direct display chart for {ticker_or_path}")
            
            # Check if this is a ticker or a path
            if isinstance(ticker_or_path, str) and os.path.exists(ticker_or_path):
                # It's a path, use original method for static charts
                return original_display_chart(ticker_or_path, *args, **kwargs)
            
            # It's a ticker, apply our direct fix
            ticker = ticker_or_path
            self.current_chart_ticker = ticker
            
            # Check if chart_notebook widget exists
            if not hasattr(self, 'chart_notebook') or not self.chart_notebook.winfo_exists():
                logging.warning(f"Cannot display chart for {ticker}: chart notebook widget no longer exists")
                return
            
            # Switch to the appropriate tab based on active_tab
            try:
                if self.active_tab == "individual":
                    self.chart_notebook.select(0)  # Individual chart tab
                elif self.active_tab == "comparison":
                    self.chart_notebook.select(1)  # Comparison chart tab
                elif self.active_tab == "seasonality":
                    self.chart_notebook.select(2)  # Seasonality chart tab
                    self._generate_seasonality_chart(ticker)
                    return
            except tk.TclError as e:
                logging.error(f"TclError switching tabs for {ticker}: {str(e)}")
                return
            
            # For individual chart, create a direct matplotlib chart
            try:
                # Get stock data
                df = self.manager.get_data(ticker)
                if df is None or df.empty:
                    if hasattr(self, 'status_var'):
                        self.status_var.set(f"No data available for {ticker}")
                    return
                
                # Apply date range filter if specified
                start_date = self.start_date_entry.get() if hasattr(self, 'start_date_entry') else ""
                end_date = self.end_date_entry.get() if hasattr(self, 'end_date_entry') else ""
                
                # Convert dates to datetime for safe comparison
                import pandas as pd
                
                # Log DataFrame info before processing
                logging.info(f"DataFrame shape before processing: {df.shape}")
                logging.info(f"DataFrame columns: {df.columns.tolist()}")
                logging.info(f"DataFrame index type: {type(df.index).__name__}")
                logging.info(f"DataFrame first few rows:\n{df.head()}")
                
                # Check if 'Date' column exists and use it as index
                if 'Date' in df.columns:
                    logging.info("Using 'Date' column as index")
                    # Convert Date column to datetime if needed
                    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                        df['Date'] = pd.to_datetime(df['Date'])
                    # Set Date as index
                    df = df.set_index('Date')
                    logging.info(f"DataFrame after setting Date as index:\n{df.head()}")
                # Ensure index is datetime type
                elif not isinstance(df.index, pd.DatetimeIndex):
                    logging.info(f"Converting index to DatetimeIndex")
                    df.index = pd.to_datetime(df.index)
                
                # Define a local date validation function since we can't rely on self._is_valid_date
                def is_valid_date(date_str):
                    if not date_str or not date_str.strip():
                        return False
                    try:
                        pd.to_datetime(date_str)
                        return True
                    except Exception:
                        return False
                
                # Convert dates to datetime objects once for consistency
                start_date_obj = None
                end_date_obj = None
                
                # Check if custom date range is applied
                has_custom_range = False
                
                if start_date and is_valid_date(start_date):
                    has_custom_range = True
                    start_date_obj = pd.to_datetime(start_date)
                    logging.info(f"Will filter by start date: {start_date} -> {start_date_obj}")
                
                if end_date and is_valid_date(end_date):
                    has_custom_range = True
                    end_date_obj = pd.to_datetime(end_date)
                    logging.info(f"Will filter by end date: {end_date} -> {end_date_obj}")
                
                # If no custom range is applied, use all available data
                if not has_custom_range:
                    logging.info("No custom date range applied, using all available data")
                
                # Only apply date filtering if custom range is applied
                if has_custom_range:
                    try:
                        # Make a copy of the original DataFrame in case filtering fails
                        original_df = df.copy()
                        
                        # Ensure index is DatetimeIndex
                        if not isinstance(df.index, pd.DatetimeIndex):
                            logging.warning("Index is not DatetimeIndex before filtering")
                            df.index = pd.to_datetime(df.index)
                        
                        # Debug index values
                        logging.info(f"Original index min: {df.index.min()}, max: {df.index.max()}")
                        logging.info(f"Original index dtype: {df.index.dtype}")
                        
                        # Apply date filtering
                        plot_df = df.copy()
                        
                        # Apply start date filter if valid
                        if start_date_obj is not None:
                            logging.info(f"Filtering by start date: {start_date_obj}")
                            # Add timezone info to start_date_obj if the DataFrame index has timezone
                            if hasattr(plot_df.index, 'tz') and plot_df.index.tz is not None:
                                if start_date_obj.tzinfo is None:
                                    start_date_obj = pytz.utc.localize(start_date_obj)
                                    logging.info(f"Added UTC timezone to start_date: {start_date_obj}")
                            plot_df = plot_df[plot_df.index >= start_date_obj]
                            logging.info(f"After start date filter: {len(plot_df)} rows")
                        
                        # Apply end date filter if valid
                        if end_date_obj is not None:
                            logging.info(f"Filtering by end date: {end_date_obj}")
                            # Add timezone info to end_date_obj if the DataFrame index has timezone
                            if hasattr(plot_df.index, 'tz') and plot_df.index.tz is not None:
                                if end_date_obj.tzinfo is None:
                                    end_date_obj = pytz.utc.localize(end_date_obj)
                                    logging.info(f"Added UTC timezone to end_date: {end_date_obj}")
                            plot_df = plot_df[plot_df.index <= end_date_obj]
                            logging.info(f"After end date filter: {len(plot_df)} rows")
                        
                        # Log the filtered date range
                        if not plot_df.empty:
                            logging.info(f"Filtered date range: {plot_df.index.min()} to {plot_df.index.max()}")
                        else:
                            logging.warning("Date filtering resulted in empty DataFrame - falling back to all data")
                            # Fall back to using all data
                            plot_df = df.copy()
                            logging.info(f"Using all available data: {plot_df.index.min()} to {plot_df.index.max()}")
                    except Exception as e:
                        logging.error(f"Error during date filtering: {str(e)}")
                        plot_df = df.copy()  # Use original data if filtering fails
                else:
                    # No date filtering needed
                    plot_df = df.copy()
                
                # Create figure and axes before checking data
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                # Check if we have data to plot
                if plot_df.empty:
                    logging.warning(f"No data to plot for {ticker}")
                    ax1.text(0.5, 0.5, f"No data available for {ticker}", 
                             horizontalalignment='center', verticalalignment='center',
                             transform=ax1.transAxes, fontsize=14)
                    if hasattr(self, 'status_var'):
                        self.status_var.set(f"No data available for {ticker}")
                else:
                    # Ensure Close column exists
                    if 'Close' not in plot_df.columns:
                        logging.error(f"'Close' column not found in DataFrame for {ticker}")
                        if 'Adj Close' in plot_df.columns:
                            logging.info(f"Using 'Adj Close' instead of 'Close' for {ticker}")
                            plot_df['Close'] = plot_df['Adj Close']
                        else:
                            logging.error(f"No price data found for {ticker}")
                            ax1.text(0.5, 0.5, f"No price data found for {ticker}", 
                                     horizontalalignment='center', verticalalignment='center',
                                     transform=ax1.transAxes, fontsize=14)
                            if hasattr(self, 'status_var'):
                                self.status_var.set(f"No price data found for {ticker}")
                            return False
                    
                    # Plot price data
                    ax1.plot(plot_df.index, plot_df['Close'], label=f"{ticker} Close")
                    ax1.set_title(f"{ticker} Stock Price ({plot_df.index.min().strftime('%Y-%m-%d')} to {plot_df.index.max().strftime('%Y-%m-%d')})")
                    ax1.set_ylabel("Price ($)")
                    ax1.grid(True)
                    ax1.legend()
                    
                    # Get the individual chart tab frame
                    if hasattr(self, 'chart_notebook') and self.chart_notebook.winfo_exists():
                        # Get the first tab (individual chart tab)
                        try:
                            individual_frame = self.chart_notebook.winfo_children()[0]
                            
                            # Clear any existing widgets
                            for widget in individual_frame.winfo_children():
                                widget.destroy()
                            
                            # Create a fixed height container frame with increased height
                            container_frame = ttk.Frame(individual_frame, height=700)  # Increased from 400 to 500
                            container_frame.pack(fill=tk.BOTH, expand=False)
                            container_frame.pack_propagate(False)  # Prevent automatic resizing
                            
                            # Create a toolbar frame
                            toolbar_frame = ttk.Frame(container_frame)
                            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
                            
                            # Embed the matplotlib figure in the Tkinter window
                            canvas = FigureCanvasTkAgg(fig, master=container_frame)
                            canvas.draw()
                            
                            # Pack the canvas widget (but only once!)
                            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                            
                            # Add the matplotlib toolbar
                            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
                            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                            toolbar.update()
                            
                            # Make sure bottom frame is visible
                            if hasattr(self, 'bottom_frame') and self.bottom_frame.winfo_exists():
                                self.bottom_frame.update_idletasks()
                                self.bottom_frame.lift()
                                logging.info("Re-lifted bottom_frame after individual chart display")
                            
                            # Log success
                            logging.info(f"Successfully displayed individual chart for {ticker}")
                        except Exception as e:
                            logging.error(f"Error embedding matplotlib figure: {str(e)}")
                    else:
                        logging.error("Cannot display chart: chart_notebook widget not found")
            
            except Exception as e:
                logging.error(f"Error creating direct chart for {ticker}: {str(e)}")
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Error creating chart: {str(e)}")
                return False
        
        except Exception as e:
            logging.error(f"Critical error in direct_display_chart: {str(e)}")
            return False
    
    # Apply the direct chart fix
    app._display_chart = types.MethodType(direct_display_chart, app)
    logging.info("Applied direct fix for individual chart display")
