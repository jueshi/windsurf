"""
Simple timeframe chart implementation for daily/weekly/monthly charts.
This module provides a direct and reliable implementation to display stock data
in all timeframes simultaneously using Plotly charts.
"""

import logging
import types
import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import webbrowser

def apply_simple_timeframe_fix(app):
    """
    Apply a simple and direct fix for timeframe charts.
    
    Args:
        app: The GUI application instance
    """
    logging.info("Applying simple timeframe chart fix...")
    
    # Create the timeframe tab
    def create_timeframe_tab():
        """Create the timeframe chart tab"""
        try:
            # Check if chart_notebook exists
            if not hasattr(app, 'chart_notebook'):
                logging.error("Cannot create timeframe tab: chart_notebook attribute not found")
                return False
                
            # Create the timeframe tab frame
            timeframe_frame = ttk.Frame(app.chart_notebook)
            app.chart_notebook.add(timeframe_frame, text="Timeframe Charts")
            
            # Store the frame reference
            app.timeframe_frame = timeframe_frame
            
            # Create frames for each timeframe
            daily_frame = ttk.LabelFrame(timeframe_frame, text="Daily Chart")
            daily_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            weekly_frame = ttk.LabelFrame(timeframe_frame, text="Weekly Chart")
            weekly_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            monthly_frame = ttk.LabelFrame(timeframe_frame, text="Monthly Chart")
            monthly_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Store frame references
            app.daily_frame = daily_frame
            app.weekly_frame = weekly_frame
            app.monthly_frame = monthly_frame
            
            # Add initial labels
            daily_label = ttk.Label(daily_frame, text="Select a ticker to display daily chart")
            daily_label.pack(pady=10)
            
            weekly_label = ttk.Label(weekly_frame, text="Select a ticker to display weekly chart")
            weekly_label.pack(pady=10)
            
            monthly_label = ttk.Label(monthly_frame, text="Select a ticker to display monthly chart")
            monthly_label.pack(pady=10)
            
            logging.info("Timeframe chart tab created successfully")
            return True
        except Exception as e:
            logging.error(f"Error creating timeframe tab: {str(e)}")
            return False
    
    # Generate timeframe charts
    def generate_timeframe_charts(self, ticker):
        """Generate charts for all timeframes"""
        try:
            logging.info(f"Generating timeframe charts for {ticker}")
            
            # Store current ticker
            self.current_chart_ticker = ticker
            
            # Get data
            df = None
            if hasattr(self.manager, 'get_data'):
                df = self.manager.get_data(ticker)
            else:
                df = self.manager.data.get(ticker)
                
            if df is None or df.empty:
                messagebox.showwarning("No Data", f"No data available for {ticker}")
                return False
            
            # Generate charts
            self._generate_daily_chart(ticker, df)
            self._generate_weekly_chart(ticker, df)
            self._generate_monthly_chart(ticker, df)
            
            return True
        except Exception as e:
            logging.error(f"Error generating timeframe charts: {str(e)}")
            messagebox.showerror("Error", f"Error generating timeframe charts: {str(e)}")
            return False
    
    # Generate daily chart
    def generate_daily_chart(self, ticker, df):
        """Generate daily chart"""
        try:
            # Clear frame
            for widget in self.daily_frame.winfo_children():
                widget.destroy()
            
            # Create figure
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=ticker
            )])
            
            # Update layout
            fig.update_layout(
                title=f"{ticker} - Daily Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                height=300
            )
            
            # Save to HTML
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, f"{ticker}_daily.html")
            fig.write_html(html_path)
            
            # Create button frame
            btn_frame = ttk.Frame(self.daily_frame)
            btn_frame.pack(fill=tk.X, pady=5)
            
            # Add title label
            ttk.Label(btn_frame, text=f"{ticker} - Daily Chart", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=5)
            
            # Add button to open in browser
            ttk.Button(btn_frame, text="Open in Browser", 
                      command=lambda: webbrowser.open(f"file:///{html_path}")).pack(side=tk.RIGHT, padx=5)
            
            # Add info label
            info_label = ttk.Label(self.daily_frame, 
                                  text="Daily chart created. Click 'Open in Browser' to view interactive chart.")
            info_label.pack(pady=10)
            
            return True
        except Exception as e:
            logging.error(f"Error generating daily chart: {str(e)}")
            ttk.Label(self.daily_frame, text=f"Error: {str(e)}", foreground="red").pack(pady=10)
            return False
    
    # Generate weekly chart
    def generate_weekly_chart(self, ticker, df):
        """Generate weekly chart"""
        try:
            # Clear frame
            for widget in self.weekly_frame.winfo_children():
                widget.destroy()
            
            # Resample to weekly
            weekly_df = df.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            
            # Create figure
            fig = go.Figure(data=[go.Candlestick(
                x=weekly_df.index,
                open=weekly_df['Open'],
                high=weekly_df['High'],
                low=weekly_df['Low'],
                close=weekly_df['Close'],
                name=ticker
            )])
            
            # Update layout
            fig.update_layout(
                title=f"{ticker} - Weekly Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                height=300
            )
            
            # Save to HTML
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, f"{ticker}_weekly.html")
            fig.write_html(html_path)
            
            # Create button frame
            btn_frame = ttk.Frame(self.weekly_frame)
            btn_frame.pack(fill=tk.X, pady=5)
            
            # Add title label
            ttk.Label(btn_frame, text=f"{ticker} - Weekly Chart", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=5)
            
            # Add button to open in browser
            ttk.Button(btn_frame, text="Open in Browser", 
                      command=lambda: webbrowser.open(f"file:///{html_path}")).pack(side=tk.RIGHT, padx=5)
            
            # Add info label
            info_label = ttk.Label(self.weekly_frame, 
                                  text="Weekly chart created. Click 'Open in Browser' to view interactive chart.")
            info_label.pack(pady=10)
            
            return True
        except Exception as e:
            logging.error(f"Error generating weekly chart: {str(e)}")
            ttk.Label(self.weekly_frame, text=f"Error: {str(e)}", foreground="red").pack(pady=10)
            return False
    
    # Generate monthly chart
    def generate_monthly_chart(self, ticker, df):
        """Generate monthly chart"""
        try:
            # Clear frame
            for widget in self.monthly_frame.winfo_children():
                widget.destroy()
            
            # Resample to monthly
            monthly_df = df.resample('M').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            
            # Create figure
            fig = go.Figure(data=[go.Candlestick(
                x=monthly_df.index,
                open=monthly_df['Open'],
                high=monthly_df['High'],
                low=monthly_df['Low'],
                close=monthly_df['Close'],
                name=ticker
            )])
            
            # Update layout
            fig.update_layout(
                title=f"{ticker} - Monthly Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                height=300
            )
            
            # Save to HTML
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, f"{ticker}_monthly.html")
            fig.write_html(html_path)
            
            # Create button frame
            btn_frame = ttk.Frame(self.monthly_frame)
            btn_frame.pack(fill=tk.X, pady=5)
            
            # Add title label
            ttk.Label(btn_frame, text=f"{ticker} - Monthly Chart", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=5)
            
            # Add button to open in browser
            ttk.Button(btn_frame, text="Open in Browser", 
                      command=lambda: webbrowser.open(f"file:///{html_path}")).pack(side=tk.RIGHT, padx=5)
            
            # Add info label
            info_label = ttk.Label(self.monthly_frame, 
                                  text="Monthly chart created. Click 'Open in Browser' to view interactive chart.")
            info_label.pack(pady=10)
            
            return True
        except Exception as e:
            logging.error(f"Error generating monthly chart: {str(e)}")
            ttk.Label(self.monthly_frame, text=f"Error: {str(e)}", foreground="red").pack(pady=10)
            return False
    
    # Handle tab change
    def handle_tab_change(self, event=None):
        """Handle tab change"""
        try:
            logging.info("Tab change handler called")
            print("DEBUG: Tab change handler called")
            
            # Call original handler if exists
            if hasattr(self, '_original_on_tab_changed'):
                print("DEBUG: Calling original tab change handler")
                self._original_on_tab_changed(event)
            
            # Get current tab
            current_tab = self.chart_notebook.select()
            tab_name = self.chart_notebook.tab(current_tab, "text")
            print(f"DEBUG: Current tab is '{tab_name}'")
            
            # Set active tab
            if tab_name == "Timeframe Charts":
                print("DEBUG: Setting active tab to 'timeframe'")
                self.active_tab = "timeframe"
                
                # Try to get current ticker from various sources
                current_ticker = None
                
                # First check if current_chart_ticker is set
                if hasattr(self, 'current_chart_ticker') and self.current_chart_ticker:
                    current_ticker = self.current_chart_ticker
                    print(f"DEBUG: Using current_chart_ticker: {current_ticker}")
                
                # If not, try to get from ticker listbox
                elif hasattr(self, 'ticker_listbox') and self.ticker_listbox.winfo_exists():
                    selected_indices = self.ticker_listbox.curselection()
                    if selected_indices:
                        ticker_text = self.ticker_listbox.get(selected_indices[0])
                        current_ticker = ticker_text.split(' - ')[0].strip()
                        print(f"DEBUG: Using ticker from listbox: {current_ticker}")
                        # Update current_chart_ticker
                        self.current_chart_ticker = current_ticker
                
                # If still not found, try watch listbox
                elif hasattr(self, 'watch_listbox') and self.watch_listbox.winfo_exists():
                    selected_indices = self.watch_listbox.curselection()
                    if selected_indices:
                        ticker_text = self.watch_listbox.get(selected_indices[0])
                        current_ticker = ticker_text.split(' - ')[0].strip()
                        print(f"DEBUG: Using ticker from watch listbox: {current_ticker}")
                        # Update current_chart_ticker
                        self.current_chart_ticker = current_ticker
                
                # Update timeframe charts if we have a ticker
                if current_ticker:
                    print(f"DEBUG: Generating timeframe charts for {current_ticker}")
                    self._generate_timeframe_charts(current_ticker)
                else:
                    print("DEBUG: No ticker available for timeframe charts")
            else:
                print(f"DEBUG: Setting active tab to '{tab_name.lower()}'")
                self.active_tab = tab_name.lower()
        except Exception as e:
            logging.error(f"Error handling tab change: {str(e)}")
            print(f"DEBUG ERROR: Error handling tab change: {str(e)}")
            import traceback
            print(traceback.format_exc())

    # Handle ticker selection
    def handle_ticker_selection(self, event=None):
        """Handle ticker selection"""
        try:
            logging.info("Ticker selection handler called")
            print("DEBUG: Ticker selection handler called")
            
            # Get selected ticker
            selected_ticker = None
            
            # Try to get from listbox
            if hasattr(self, 'ticker_listbox') and self.ticker_listbox.winfo_exists():
                selected_indices = self.ticker_listbox.curselection()
                if selected_indices:
                    ticker_text = self.ticker_listbox.get(selected_indices[0])
                    selected_ticker = ticker_text.split(' - ')[0].strip()
                    print(f"DEBUG: Selected ticker from listbox: {selected_ticker}")
            
            # Call original handler if exists
            if hasattr(self, '_original_on_ticker_selected'):
                print("DEBUG: Calling original ticker selection handler")
                self._original_on_ticker_selected(event)
            
            # Update current_chart_ticker if we found a ticker
            if selected_ticker:
                self.current_chart_ticker = selected_ticker
                print(f"DEBUG: Set current_chart_ticker to {selected_ticker}")
            
            # Update timeframe charts if active
            if hasattr(self, 'active_tab'):
                print(f"DEBUG: Active tab is {self.active_tab}")
                if self.active_tab == "timeframe":
                    print("DEBUG: Timeframe tab is active")
                    if hasattr(self, 'current_chart_ticker') and self.current_chart_ticker:
                        print(f"DEBUG: Generating timeframe charts for {self.current_chart_ticker}")
                        self._generate_timeframe_charts(self.current_chart_ticker)
                    else:
                        print("DEBUG: No current_chart_ticker available")
                else:
                    print(f"DEBUG: Active tab is not timeframe: {self.active_tab}")
            else:
                print("DEBUG: No active_tab attribute found")
        except Exception as e:
            logging.error(f"Error handling ticker selection: {str(e)}")
            print(f"DEBUG ERROR: Error handling ticker selection: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    # Apply the fix
    try:
        # Create the timeframe tab
        create_timeframe_tab()
        
        # Add methods to app
        app._generate_timeframe_charts = types.MethodType(generate_timeframe_charts, app)
        app._generate_daily_chart = types.MethodType(generate_daily_chart, app)
        app._generate_weekly_chart = types.MethodType(generate_weekly_chart, app)
        app._generate_monthly_chart = types.MethodType(generate_monthly_chart, app)
        
        # Store original handlers
        if hasattr(app, '_on_tab_changed'):
            app._original_on_tab_changed = app._on_tab_changed
            app._on_tab_changed = types.MethodType(handle_tab_change, app)
        elif hasattr(app, 'on_tab_changed'):
            app._original_on_tab_changed = app.on_tab_changed
            app.on_tab_changed = types.MethodType(handle_tab_change, app)
        else:
            # Direct binding
            app.chart_notebook.bind("<<NotebookTabChanged>>", lambda e: handle_tab_change(app, e))
        
        # Update ticker selection handlers
        if hasattr(app, '_on_ticker_selected'):
            app._original_on_ticker_selected = app._on_ticker_selected
            app._on_ticker_selected = types.MethodType(handle_ticker_selection, app)
        
        if hasattr(app, '_on_watch_ticker_selected'):
            app._original_on_watch_ticker_selected = app._on_watch_ticker_selected
            
            def enhanced_watch_handler(self, event=None):
                logging.info("Watch ticker selection handler called")
                print("DEBUG: Watch ticker selection handler called")
                
                # Get selected ticker from watch listbox
                selected_ticker = None
                if hasattr(self, 'watch_listbox') and self.watch_listbox.winfo_exists():
                    selected_indices = self.watch_listbox.curselection()
                    if selected_indices:
                        ticker_text = self.watch_listbox.get(selected_indices[0])
                        selected_ticker = ticker_text.split(' - ')[0].strip()
                        print(f"DEBUG: Selected ticker from watch listbox: {selected_ticker}")
                        # Directly set current_chart_ticker
                        self.current_chart_ticker = selected_ticker
                        print(f"DEBUG: Set current_chart_ticker to {selected_ticker}")
                
                # Call original handler
                print("DEBUG: Calling original watch ticker selection handler")
                self._original_on_watch_ticker_selected(event)
                
                # Update timeframe charts if active
                if hasattr(self, 'active_tab'):
                    print(f"DEBUG: Active tab is {self.active_tab}")
                    if self.active_tab == "timeframe":
                        print("DEBUG: Timeframe tab is active")
                        # Use the selected ticker directly if available
                        if selected_ticker:
                            print(f"DEBUG: Generating timeframe charts for selected ticker: {selected_ticker}")
                            self._generate_timeframe_charts(selected_ticker)
                        # Fall back to current_chart_ticker if no direct selection
                        elif hasattr(self, 'current_chart_ticker') and self.current_chart_ticker:
                            print(f"DEBUG: Generating timeframe charts for current_chart_ticker: {self.current_chart_ticker}")
                            self._generate_timeframe_charts(self.current_chart_ticker)
                        else:
                            print("DEBUG: No ticker available for timeframe charts")
                    else:
                        print(f"DEBUG: Active tab is not timeframe: {self.active_tab}")
                else:
                    print("DEBUG: No active_tab attribute found")
            
            app._on_watch_ticker_selected = types.MethodType(enhanced_watch_handler, app)
        
        logging.info("Simple timeframe chart fix applied successfully")
        return True
    except Exception as e:
        logging.error(f"Error applying simple timeframe chart fix: {str(e)}")
        return False
