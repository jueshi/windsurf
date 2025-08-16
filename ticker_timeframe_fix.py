"""
Ticker Timeframe Fix

This module provides a direct patch to ensure timeframe charts are updated
when a ticker is selected, regardless of which list the ticker is selected from.
"""

import logging
import types
import tkinter as tk
from tkinter import ttk
import plotly.graph_objects as go
import pandas as pd
import os

def apply_ticker_timeframe_fix(app):
    """
    Apply a direct fix to ensure timeframe charts are updated when a ticker is selected.
    
    Args:
        app: The StockDataGUI instance to patch
    """
    logging.info("Applying ticker timeframe fix...")
    
    # Store original methods
    if hasattr(app, '_on_ticker_selected'):
        original_ticker_handler = app._on_ticker_selected
        
        def patched_ticker_handler(self, event=None):
            """Patched ticker selection handler"""
            # Call original handler
            original_ticker_handler(event)
            
            # Check if timeframe tab exists and is active
            if hasattr(self, 'chart_notebook'):
                current_tab = self.chart_notebook.select()
                if current_tab:
                    tab_text = self.chart_notebook.tab(current_tab, "text")
                    if tab_text == "Timeframe Charts":
                        # Get selected ticker
                        selected_ticker = None
                        if hasattr(self, 'ticker_listbox') and self.ticker_listbox.winfo_exists():
                            selected_indices = self.ticker_listbox.curselection()
                            if selected_indices:
                                ticker_text = self.ticker_listbox.get(selected_indices[0])
                                selected_ticker = ticker_text.split(' - ')[0].strip()
                        
                        # Update timeframe charts if ticker found
                        if selected_ticker:
                            # Store as current chart ticker
                            self.current_chart_ticker = selected_ticker
                            
                            # Generate timeframe charts
                            generate_timeframe_charts(self, selected_ticker)
        
        # Replace the method
        app._on_ticker_selected = types.MethodType(patched_ticker_handler, app)
    
    # Patch watch ticker handler
    if hasattr(app, '_on_watch_ticker_selected'):
        original_watch_handler = app._on_watch_ticker_selected
        
        def patched_watch_handler(self, event=None):
            """Patched watch ticker selection handler"""
            # Call original handler
            original_watch_handler(event)
            
            # Check if timeframe tab exists and is active
            if hasattr(self, 'chart_notebook'):
                current_tab = self.chart_notebook.select()
                if current_tab:
                    tab_text = self.chart_notebook.tab(current_tab, "text")
                    if tab_text == "Timeframe Charts":
                        # Get selected ticker
                        selected_ticker = None
                        if hasattr(self, 'watch_listbox') and self.watch_listbox.winfo_exists():
                            selected_indices = self.watch_listbox.curselection()
                            if selected_indices:
                                ticker_text = self.watch_listbox.get(selected_indices[0])
                                selected_ticker = ticker_text.split(' - ')[0].strip()
                        
                        # Update timeframe charts if ticker found
                        if selected_ticker:
                            # Store as current chart ticker
                            self.current_chart_ticker = selected_ticker
                            
                            # Generate timeframe charts
                            generate_timeframe_charts(self, selected_ticker)
        
        # Replace the method
        app._on_watch_ticker_selected = types.MethodType(patched_watch_handler, app)
    
    # Patch tab change handler
    if hasattr(app, '_on_tab_changed'):
        original_tab_handler = app._on_tab_changed
        
        def patched_tab_handler(self, event=None):
            """Patched tab change handler"""
            # Call original handler
            original_tab_handler(event)
            
            # Check if timeframe tab is selected
            if hasattr(self, 'chart_notebook'):
                current_tab = self.chart_notebook.select()
                if current_tab:
                    tab_text = self.chart_notebook.tab(current_tab, "text")
                    if tab_text == "Timeframe Charts":
                        # Get current ticker from various sources
                        current_ticker = None
                        
                        # Try current_chart_ticker first
                        if hasattr(self, 'current_chart_ticker') and self.current_chart_ticker:
                            current_ticker = self.current_chart_ticker
                        
                        # Try ticker listbox
                        if not current_ticker and hasattr(self, 'ticker_listbox') and self.ticker_listbox.winfo_exists():
                            selected_indices = self.ticker_listbox.curselection()
                            if selected_indices:
                                ticker_text = self.ticker_listbox.get(selected_indices[0])
                                current_ticker = ticker_text.split(' - ')[0].strip()
                        
                        # Try watch listbox
                        if not current_ticker and hasattr(self, 'watch_listbox') and self.watch_listbox.winfo_exists():
                            selected_indices = self.watch_listbox.curselection()
                            if selected_indices:
                                ticker_text = self.watch_listbox.get(selected_indices[0])
                                current_ticker = ticker_text.split(' - ')[0].strip()
                        
                        # Update timeframe charts if ticker found
                        if current_ticker:
                            # Store as current chart ticker
                            self.current_chart_ticker = current_ticker
                            
                            # Generate timeframe charts
                            generate_timeframe_charts(self, current_ticker)
        
        # Replace the method
        app._on_tab_changed = types.MethodType(patched_tab_handler, app)

def generate_timeframe_charts(app, ticker):
    """
    Generate daily, weekly, and monthly charts for the given ticker.
    
    Args:
        app: The StockDataGUI instance
        ticker: The ticker symbol to generate charts for
    """
    try:
        logging.info(f"Generating timeframe charts for {ticker}")
        print(f"Generating timeframe charts for {ticker}")
        
        # Check if data manager exists
        if not hasattr(app, 'data_manager'):
            logging.error("Data manager not found")
            return
        
        # Get data for ticker
        df = app.data_manager.get_stock_data(ticker)
        if df is None or df.empty:
            logging.error(f"No data found for {ticker}")
            return
        
        # Create directory for charts if it doesn't exist
        charts_dir = os.path.join(os.getcwd(), "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # Generate daily chart
        daily_chart_path = os.path.join(charts_dir, f"{ticker}_daily.html")
        generate_daily_chart(df, ticker, daily_chart_path)
        
        # Generate weekly chart
        weekly_chart_path = os.path.join(charts_dir, f"{ticker}_weekly.html")
        generate_weekly_chart(df, ticker, weekly_chart_path)
        
        # Generate monthly chart
        monthly_chart_path = os.path.join(charts_dir, f"{ticker}_monthly.html")
        generate_monthly_chart(df, ticker, monthly_chart_path)
        
        # Update UI with chart paths
        update_timeframe_ui(app, ticker, daily_chart_path, weekly_chart_path, monthly_chart_path)
        
    except Exception as e:
        logging.error(f"Error generating timeframe charts: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

def generate_daily_chart(df, ticker, output_path):
    """Generate daily chart for the given ticker"""
    try:
        # Create figure
        fig = go.Figure()
        
        # Add trace
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Daily'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{ticker} - Daily Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=True
        )
        
        # Write to file
        fig.write_html(output_path)
        
        return output_path
    except Exception as e:
        logging.error(f"Error generating daily chart: {str(e)}")
        return None

def generate_weekly_chart(df, ticker, output_path):
    """Generate weekly chart for the given ticker"""
    try:
        # Resample to weekly
        weekly_df = df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        # Create figure
        fig = go.Figure()
        
        # Add trace
        fig.add_trace(go.Candlestick(
            x=weekly_df.index,
            open=weekly_df['Open'],
            high=weekly_df['High'],
            low=weekly_df['Low'],
            close=weekly_df['Close'],
            name='Weekly'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{ticker} - Weekly Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=True
        )
        
        # Write to file
        fig.write_html(output_path)
        
        return output_path
    except Exception as e:
        logging.error(f"Error generating weekly chart: {str(e)}")
        return None

def generate_monthly_chart(df, ticker, output_path):
    """Generate monthly chart for the given ticker"""
    try:
        # Resample to monthly
        monthly_df = df.resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        # Create figure
        fig = go.Figure()
        
        # Add trace
        fig.add_trace(go.Candlestick(
            x=monthly_df.index,
            open=monthly_df['Open'],
            high=monthly_df['High'],
            low=monthly_df['Low'],
            close=monthly_df['Close'],
            name='Monthly'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{ticker} - Monthly Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=True
        )
        
        # Write to file
        fig.write_html(output_path)
        
        return output_path
    except Exception as e:
        logging.error(f"Error generating monthly chart: {str(e)}")
        return None

def update_timeframe_ui(app, ticker, daily_path, weekly_path, monthly_path):
    """Update the timeframe UI with the generated charts"""
    try:
        # Find timeframe tab
        timeframe_tab = None
        for tab_id in app.chart_notebook.tabs():
            if app.chart_notebook.tab(tab_id, "text") == "Timeframe Charts":
                timeframe_tab = tab_id
                break
        
        if not timeframe_tab:
            logging.error("Timeframe tab not found")
            return
        
        # Get the timeframe frame
        timeframe_frame = None
        for child in app.chart_notebook.winfo_children():
            if str(child) == str(timeframe_tab):
                timeframe_frame = child
                break
        
        if not timeframe_frame:
            # Try to find by tab ID
            for child in app.chart_notebook.winfo_children():
                if app.chart_notebook.select() == app.chart_notebook.tabs()[app.chart_notebook.index(child)]:
                    timeframe_frame = child
                    break
        
        if not timeframe_frame:
            logging.error("Timeframe frame not found")
            return
        
        # Clear existing widgets
        for widget in timeframe_frame.winfo_children():
            widget.destroy()
        
        # Create main frame
        main_frame = ttk.Frame(timeframe_frame)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create frames for each timeframe
        daily_frame = ttk.LabelFrame(main_frame, text=f"{ticker} - Daily Chart")
        daily_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        weekly_frame = ttk.LabelFrame(main_frame, text=f"{ticker} - Weekly Chart")
        weekly_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        monthly_frame = ttk.LabelFrame(main_frame, text=f"{ticker} - Monthly Chart")
        monthly_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Add buttons to open charts in browser
        daily_button = ttk.Button(
            daily_frame, 
            text="Open Daily Chart in Browser",
            command=lambda: os.startfile(daily_path) if os.path.exists(daily_path) else None
        )
        daily_button.pack(pady=10)
        
        weekly_button = ttk.Button(
            weekly_frame, 
            text="Open Weekly Chart in Browser",
            command=lambda: os.startfile(weekly_path) if os.path.exists(weekly_path) else None
        )
        weekly_button.pack(pady=10)
        
        monthly_button = ttk.Button(
            monthly_frame, 
            text="Open Monthly Chart in Browser",
            command=lambda: os.startfile(monthly_path) if os.path.exists(monthly_path) else None
        )
        monthly_button.pack(pady=10)
        
        # Add status labels
        daily_label = ttk.Label(daily_frame, text="Daily chart generated successfully")
        daily_label.pack(pady=5)
        
        weekly_label = ttk.Label(weekly_frame, text="Weekly chart generated successfully")
        weekly_label.pack(pady=5)
        
        monthly_label = ttk.Label(monthly_frame, text="Monthly chart generated successfully")
        monthly_label.pack(pady=5)
        
        # Update status
        if hasattr(app, 'status_var'):
            app.status_var.set(f"Timeframe charts for {ticker} generated successfully")
        
    except Exception as e:
        logging.error(f"Error updating timeframe UI: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
