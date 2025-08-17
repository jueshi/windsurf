"""
Timeframe chart implementation for daily/weekly/monthly charts.
This module provides functionality to display stock data in all timeframes simultaneously.
"""

import logging
import types
import os
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import webbrowser

def apply_timeframe_chart_fix(app):
    """
    Apply the timeframe chart fix to add daily/weekly/monthly chart functionality.
    
    Args:
        app: The GUI application instance
    """
    logging.info("Applying timeframe chart fix...")
    
    # Add the timeframe chart tab to the notebook
    def add_timeframe_tab():
        """Add the timeframe chart tab to the notebook"""
        if not hasattr(app, 'chart_notebook') or not app.chart_notebook.winfo_exists():
            logging.error("Cannot add timeframe tab: chart_notebook widget not found")
            return False
            
        try:
            # Create timeframe chart tab
            app.timeframe_chart_frame = ttk.Frame(app.chart_notebook)
            app.chart_notebook.add(app.timeframe_chart_frame, text="Timeframe Chart")
            
            # Create a title label
            title_label = ttk.Label(app.timeframe_chart_frame, 
                                   text="All Timeframes View", 
                                   font=("Helvetica", 12, "bold"))
            title_label.pack(fill=tk.X, padx=5, pady=5)
            
            # Create containers for each timeframe chart
            # Daily chart container
            app.daily_chart_frame = ttk.LabelFrame(app.timeframe_chart_frame, text="Daily Chart")
            app.daily_chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Weekly chart container
            app.weekly_chart_frame = ttk.LabelFrame(app.timeframe_chart_frame, text="Weekly Chart")
            app.weekly_chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Monthly chart container
            app.monthly_chart_frame = ttk.LabelFrame(app.timeframe_chart_frame, text="Monthly Chart")
            app.monthly_chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            logging.info("Added timeframe chart tab with all timeframes successfully")
            return True
        except Exception as e:
            logging.error(f"Error adding timeframe chart tab: {str(e)}")
            return False
    
    # Generate all timeframe charts
    def generate_all_timeframe_charts(self, ticker):
        """Generate charts for all timeframes (daily, weekly, monthly) for the specified ticker"""
        try:
            logging.info(f"Generating all timeframe charts for {ticker}")
            
            # Store the current chart ticker
            self.current_chart_ticker = ticker
            
            # Check if we have data for this ticker
            df = None
            if hasattr(self.manager, 'get_data'):
                df = self.manager.get_data(ticker)
            else:
                df = self.manager.data.get(ticker)
                
            if df is None or df.empty:
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"No data available for {ticker}. Please download data first.")
                logging.warning(f"No data available for {ticker}")
                return False
            
            # Generate each timeframe chart
            daily_success = self._generate_timeframe_chart(ticker, "Daily", self.daily_chart_frame)
            weekly_success = self._generate_timeframe_chart(ticker, "Weekly", self.weekly_chart_frame)
            monthly_success = self._generate_timeframe_chart(ticker, "Monthly", self.monthly_chart_frame)
            
            # Update status based on success
            if daily_success and weekly_success and monthly_success:
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Displayed all timeframe charts for {ticker}")
                return True
            else:
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Some timeframe charts could not be displayed for {ticker}")
                return False
            
        except Exception as e:
            logging.error(f"Error generating all timeframe charts: {str(e)}")
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Error: {str(e)}")
            return False
    
    # Add the timeframe chart generation method
    def generate_timeframe_chart(self, ticker, timeframe, chart_frame):
        """Generate a chart for the specified ticker and timeframe"""
        try:
            logging.info(f"Generating {timeframe} chart for {ticker}")
            
            # Store the current chart ticker
            self.current_chart_ticker = ticker
            
            # Get data for the ticker
            df = None
            if hasattr(self.manager, 'get_data'):
                df = self.manager.get_data(ticker)
            else:
                df = self.manager.data.get(ticker)
                
            if df is None or df.empty:
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"No data available for {ticker}")
                return False
                
            # Ensure we have a datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df = df.set_index('Date')
                else:
                    df.index = pd.to_datetime(df.index)
            
            # Apply date filtering if needed
            start_date = self.start_date_var.get() if hasattr(self, 'start_date_var') else None
            end_date = self.end_date_var.get() if hasattr(self, 'end_date_var') else None
            
            # Create a copy of the dataframe for filtering
            plot_df = df.copy()
            
            # Define a local date validation function
            def is_valid_date(date_str):
                if not date_str or not date_str.strip():
                    return False
                try:
                    pd.to_datetime(date_str)
                    return True
                except Exception:
                    return False
            
            # Convert dates to datetime objects
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
            
            # Apply date filtering if custom range is applied
            if has_custom_range:
                try:
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
            
            # Resample data based on timeframe
            if timeframe == "Daily":
                # Daily data is already at the right frequency
                resampled_df = plot_df
            elif timeframe == "Weekly":
                # Resample to weekly frequency
                resampled_df = plot_df.resample('W').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                })
            elif timeframe == "Monthly":
                # Resample to monthly frequency
                resampled_df = plot_df.resample('M').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                })
            else:
                logging.error(f"Invalid timeframe: {timeframe}")
                return False
            
            # Create Plotly figure
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.03, 
                               row_heights=[0.7, 0.3],
                               subplot_titles=(f"{ticker} {timeframe} Chart", "Volume"))
            
            # Add price candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=resampled_df.index,
                    open=resampled_df['Open'],
                    high=resampled_df['High'],
                    low=resampled_df['Low'],
                    close=resampled_df['Close'],
                    name=ticker
                ),
                row=1, col=1
            )
            
            # Add volume bar chart
            fig.add_trace(
                go.Bar(
                    x=resampled_df.index,
                    y=resampled_df['Volume'],
                    name='Volume',
                    marker_color='rgba(0, 0, 255, 0.5)'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f"{ticker} {timeframe} Chart",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                xaxis_rangeslider_visible=False,
                height=600,
                width=900,
                showlegend=False
            )
            
            # Save to temporary HTML file
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, f"{ticker}_{timeframe}_chart.html")
            fig.write_html(html_path, auto_open=False)
            
            # Update the specified chart frame
            if chart_frame and chart_frame.winfo_exists():
                try:
                    # Clear existing widgets
                    for widget in chart_frame.winfo_children():
                        widget.destroy()
                    
                    # Create a button to open the chart in browser for better interaction
                    btn_frame = ttk.Frame(chart_frame)
                    btn_frame.pack(fill=tk.X, pady=5)
                    ttk.Button(btn_frame, text="Open in Browser", 
                              command=lambda path=html_path: webbrowser.open(f"file:///{path}")).pack()
                    
                    # Create a label to show that interactive chart is available
                    ttk.Label(chart_frame, 
                             text=f"Interactive {timeframe} chart for {ticker}\nUse mouse to zoom/pan").pack()
                    
                except tk.TclError as e:
                    logging.error(f"TclError updating {timeframe} chart frame: {str(e)}")
                    return False
            
            # Update status
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Displayed {timeframe} chart for {ticker}")
            
            # Force update of the UI to ensure buttons remain visible
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.update_idletasks()
            
            return True
            
        except Exception as e:
            logging.error(f"Error generating timeframe chart: {str(e)}")
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Error: {str(e)}")
            return False
    
    # Add the tab change handler update
    def on_tab_changed(self, event=None):
        """Handle tab change events"""
        try:
            # Get the selected tab
            selected_tab = app.chart_notebook.select()
            if not selected_tab:
                return
                
            # Get the tab index
            tab_index = app.chart_notebook.index(selected_tab)
            
            # Map tab index to tab name
            tab_names = ["individual", "comparison", "seasonality", "timeframe"]
            if tab_index < len(tab_names):
                app.active_tab = tab_names[tab_index]
                logging.info(f"Active tab is now: {app.active_tab}")
                
                # If switching to timeframe tab, update all charts if we have a current ticker
                if app.active_tab == "timeframe":
                    # Get the current ticker - either from current_chart_ticker or from the selected ticker in the listbox
                    current_ticker = None
                    
                    # First try to get from current_chart_ticker
                    if hasattr(app, 'current_chart_ticker') and app.current_chart_ticker:
                        current_ticker = app.current_chart_ticker
                        logging.info(f"Using current_chart_ticker: {current_ticker}")
                    
                    # If no current_chart_ticker, try to get from selected ticker in listbox
                    if not current_ticker and hasattr(app, 'ticker_listbox') and app.ticker_listbox.winfo_exists():
                        selected_indices = app.ticker_listbox.curselection()
                        if selected_indices:
                            ticker_text = app.ticker_listbox.get(selected_indices[0])
                            current_ticker = ticker_text.split(' - ')[0].strip()
                            logging.info(f"Using selected ticker from listbox: {current_ticker}")
                    
                    # If no ticker selected in main listbox, try watch listbox
                    if not current_ticker and hasattr(app, 'watch_listbox') and app.watch_listbox.winfo_exists():
                        selected_indices = app.watch_listbox.curselection()
                        if selected_indices:
                            ticker_text = app.watch_listbox.get(selected_indices[0])
                            current_ticker = ticker_text.split(' - ')[0].strip()
                            logging.info(f"Using selected ticker from watch listbox: {current_ticker}")
                    
                    # If we have a ticker, generate all timeframe charts
                    if current_ticker:
                        app._generate_all_timeframe_charts(current_ticker)
                    else:
                        logging.warning("No ticker selected for timeframe charts")
                        if hasattr(app, 'status_var'):
                            app.status_var.set("Please select a ticker to display timeframe charts")
            
        except Exception as e:
            logging.error(f"Error handling tab change: {str(e)}")
    
    # Apply the fixes
    try:
        # Add the timeframe tab
        add_timeframe_tab()
        
        # Add the timeframe chart generation methods
        app._generate_all_timeframe_charts = types.MethodType(generate_all_timeframe_charts, app)
        app._generate_timeframe_chart = types.MethodType(generate_timeframe_chart, app)
        
        # Update the tab change handler
        if hasattr(app, '_on_tab_changed'):
            original_on_tab_changed = app._on_tab_changed
            app._on_tab_changed = types.MethodType(on_tab_changed, app)
        
        # Update the ticker selection handlers to generate all timeframe charts
        original_on_ticker_selected = None
        if hasattr(app, '_on_ticker_selected'):
            original_on_ticker_selected = app._on_ticker_selected
            
            def enhanced_on_ticker_selected(self, event=None):
                # Call the original handler
                original_on_ticker_selected(self, event)
                
                # If timeframe tab is active, update all timeframe charts
                if hasattr(self, 'active_tab') and self.active_tab == "timeframe" and hasattr(self, 'current_chart_ticker'):
                    self._generate_all_timeframe_charts(self.current_chart_ticker)
            
            app._on_ticker_selected = types.MethodType(enhanced_on_ticker_selected, app)
        
        original_on_watch_ticker_selected = None
        if hasattr(app, '_on_watch_ticker_selected'):
            original_on_watch_ticker_selected = app._on_watch_ticker_selected
            
            def enhanced_on_watch_ticker_selected(self, event=None):
                # Call the original handler
                original_on_watch_ticker_selected(self, event)
                
                # If timeframe tab is active, update all timeframe charts
                if hasattr(self, 'active_tab') and self.active_tab == "timeframe" and hasattr(self, 'current_chart_ticker'):
                    self._generate_all_timeframe_charts(self.current_chart_ticker)
            
            app._on_watch_ticker_selected = types.MethodType(enhanced_on_watch_ticker_selected, app)
        
        logging.info("Timeframe chart fix applied successfully")
        return True
    except Exception as e:
        logging.error(f"Error applying timeframe chart fix: {str(e)}")
        return False
