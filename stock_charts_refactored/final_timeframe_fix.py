"""
Comprehensive Timeframe Chart Fix

This module provides a single, robust implementation for the timeframe charts feature.
It consolidates the logic from previous fixes, corrects critical bugs in data fetching
and UI updates, and ensures the feature works as intended.
"""

import logging
import types
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import webbrowser
import os

def apply_final_timeframe_fix(app):
    """
    Applies the comprehensive fix for the timeframe charts.

    This function will:
    1. Create a new "Timeframe Charts" tab.
    2. Create and store stable frames for daily, weekly, and monthly charts.
    3. Add a new method to the app to generate and display the charts.
    4. Correctly patch event handlers to trigger chart updates.

    Args:
        app: The StockDataGUI application instance.
    """
    logging.info("Applying the final, comprehensive timeframe chart fix...")

    # --- 1. Create UI Elements ---
    try:
        if not hasattr(app, 'chart_notebook'):
            logging.error("Cannot create timeframe tab: chart_notebook not found.")
            return

        # Create the main frame for the timeframe tab
        timeframe_tab_frame = ttk.Frame(app.chart_notebook)
        app.chart_notebook.add(timeframe_tab_frame, text="Timeframe Charts")

        # Create and store dedicated frames for each chart
        app.daily_chart_frame = ttk.LabelFrame(timeframe_tab_frame, text="Daily Chart")
        app.daily_chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        app.weekly_chart_frame = ttk.LabelFrame(timeframe_tab_frame, text="Weekly Chart")
        app.weekly_chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        app.monthly_chart_frame = ttk.LabelFrame(timeframe_tab_frame, text="Monthly Chart")
        app.monthly_chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Set initial labels
        ttk.Label(app.daily_chart_frame, text="Select a ticker to display the daily chart.").pack(pady=10)
        ttk.Label(app.weekly_chart_frame, text="Select a ticker to display the weekly chart.").pack(pady=10)
        ttk.Label(app.monthly_chart_frame, text="Select a ticker to display the monthly chart.").pack(pady=10)

        logging.info("Timeframe chart tab and frames created successfully.")

    except Exception as e:
        logging.error(f"Error creating timeframe UI: {e}")
        return

    # --- 2. Define Chart Generation and Display Logic ---

    def _display_chart_in_frame(frame, fig, ticker, timeframe):
        """Helper function to display a Plotly figure in a given Tkinter frame."""
        try:
            # Clear any existing widgets in the frame
            for widget in frame.winfo_children():
                widget.destroy()

            # Save chart to a temporary HTML file
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, f"{ticker}_{timeframe}_chart.html")
            fig.write_html(html_path)

            # Create a container for the button and title
            header_frame = ttk.Frame(frame)
            header_frame.pack(fill=tk.X, pady=5, padx=5)

            # Add a title label
            title = f"{ticker} - {timeframe.capitalize()} Chart"
            ttk.Label(header_frame, text=title, font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)

            # Add a button to open the interactive chart in a browser
            browser_button = ttk.Button(header_frame, text="Open in Browser",
                                        command=lambda p=html_path: webbrowser.open(f"file:///{os.path.abspath(p)}"))
            browser_button.pack(side=tk.RIGHT)

            # Add a placeholder label indicating success
            ttk.Label(frame, text=f"Interactive {timeframe} chart generated.").pack(pady=10)
            frame.update_idletasks()
        except Exception as e:
            logging.error(f"Error displaying {timeframe} chart in frame: {e}")
            ttk.Label(frame, text=f"Error displaying chart: {e}", foreground="red").pack(pady=10)

    def _generate_and_display_timeframe_charts(self, ticker):
        """
        Fetches data and generates daily, weekly, and monthly charts for the given ticker.
        This method is intended to be bound to the StockDataGUI instance.
        """
        logging.info(f"Generating all timeframe charts for {ticker}...")
        self.current_chart_ticker = ticker

        # Correctly fetch data using the existing manager and method
        df = self.manager.load_data(ticker)

        if df is None or df.empty:
            messagebox.showwarning("No Data", f"No data available for {ticker}. Cannot generate timeframe charts.")
            return

        # --- Generate Daily Chart ---
        try:
            fig_daily = go.Figure(data=[go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']
            )])
            fig_daily.update_layout(title_text=f"{ticker} Daily", xaxis_rangeslider_visible=False, margin=dict(t=30, b=10, l=20, r=20))
            _display_chart_in_frame(self.daily_chart_frame, fig_daily, ticker, "daily")
        except Exception as e:
            logging.error(f"Failed to generate daily chart for {ticker}: {e}")

        # --- Generate Weekly Chart ---
        try:
            weekly_df = df.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
            fig_weekly = go.Figure(data=[go.Candlestick(
                x=weekly_df.index, open=weekly_df['Open'], high=weekly_df['High'], low=weekly_df['Low'], close=weekly_df['Close']
            )])
            fig_weekly.update_layout(title_text=f"{ticker} Weekly", xaxis_rangeslider_visible=False, margin=dict(t=30, b=10, l=20, r=20))
            _display_chart_in_frame(self.weekly_chart_frame, fig_weekly, ticker, "weekly")
        except Exception as e:
            logging.error(f"Failed to generate weekly chart for {ticker}: {e}")

        # --- Generate Monthly Chart ---
        try:
            monthly_df = df.resample('M').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
            fig_monthly = go.Figure(data=[go.Candlestick(
                x=monthly_df.index, open=monthly_df['Open'], high=monthly_df['High'], low=monthly_df['Low'], close=monthly_df['Close']
            )])
            fig_monthly.update_layout(title_text=f"{ticker} Monthly", xaxis_rangeslider_visible=False, margin=dict(t=30, b=10, l=20, r=20))
            _display_chart_in_frame(self.monthly_chart_frame, fig_monthly, ticker, "monthly")
        except Exception as e:
            logging.error(f"Failed to generate monthly chart for {ticker}: {e}")

        self.status_var.set(f"Timeframe charts for {ticker} displayed.")

    # Bind the new method to the application instance
    app._generate_and_display_timeframe_charts = types.MethodType(_generate_and_display_timeframe_charts, app)

    # --- 3. Patch Event Handlers ---

    # Store original handlers if they exist
    original_ticker_selected = app._on_ticker_selected
    original_watch_ticker_selected = app._on_watch_ticker_selected
    original_tab_changed = app._on_tab_changed

    def patched_on_ticker_selected(self, event=None):
        """Patched handler for ticker selection from the main list."""
        # Call the original handler first to maintain other functionalities
        original_ticker_selected(event)

        # Now, add the new functionality
        if self.active_tab == "timeframe":
            selected_indices = self.ticker_listbox.curselection()
            if not selected_indices:
                return
            ticker_text = self.ticker_listbox.get(selected_indices[0])
            ticker = ticker_text.split(' - ')[0].strip()
            if ticker:
                self._generate_and_display_timeframe_charts(ticker)

    def patched_on_watch_ticker_selected(self, event=None):
        """Patched handler for ticker selection from the watch list."""
        original_watch_ticker_selected(event)
        if self.active_tab == "timeframe":
            selected_indices = self.watch_listbox.curselection()
            if not selected_indices:
                return
            ticker = self.watch_listbox.get(selected_indices[0]).strip()
            if ticker:
                self._generate_and_display_timeframe_charts(ticker)

    def patched_on_tab_changed(self, event=None):
        """
        Patched handler for tab changes. This version correctly handles the
        control flow to prevent the UI from switching back to the individual tab.
        """
        try:
            selected_tab_widget = self.chart_notebook.select()
            selected_tab_text = self.chart_notebook.tab(selected_tab_widget, "text")
        except tk.TclError:
            # This can happen during shutdown, so we fail gracefully.
            return

        # Check if our custom tab is the one selected.
        if selected_tab_text == "Timeframe Charts":
            # If it is, we take full control.
            self.active_tab = "timeframe"
            logging.info("Switched to Timeframe Charts tab. Intercepting tab change logic.")

            # Run the logic to update the charts in this tab.
            ticker_to_load = None
            main_selection = self.ticker_listbox.curselection()
            watch_selection = self.watch_listbox.curselection()

            if main_selection:
                ticker_text = self.ticker_listbox.get(main_selection[0])
                ticker_to_load = ticker_text.split(' - ')[0].strip()
            elif watch_selection:
                ticker_to_load = self.watch_listbox.get(watch_selection[0]).strip()
            elif hasattr(self, 'current_chart_ticker') and self.current_chart_ticker:
                # Fallback to the last known ticker if none is selected
                ticker_to_load = self.current_chart_ticker

            if ticker_to_load:
                self._generate_and_display_timeframe_charts(ticker_to_load)
        else:
            # If any other tab is selected, we let the original handler do its job.
            # This prevents the unwanted side effects.
            logging.info(f"Switched to '{selected_tab_text}' tab. Running original handler.")
            original_tab_changed(event)

    # Apply the new patched handlers
    app._on_ticker_selected = types.MethodType(patched_on_ticker_selected, app)
    app._on_watch_ticker_selected = types.MethodType(patched_on_watch_ticker_selected, app)
    app._on_tab_changed = types.MethodType(patched_on_tab_changed, app)

    # Re-bind to ensure the new handlers are used, as the original might have been bound directly
    app.ticker_listbox.bind("<<ListboxSelect>>", app._on_ticker_selected)
    app.watch_listbox.bind("<<ListboxSelect>>", app._on_watch_ticker_selected)
    app.chart_notebook.bind("<<NotebookTabChanged>>", app._on_tab_changed)

    # Set the active_tab attribute to "timeframe" if the timeframe tab is currently selected
    # This ensures that if the app starts with the timeframe tab selected, the logic will work correctly
    current_tab_text = app.chart_notebook.tab(app.chart_notebook.select(), "text")
    if current_tab_text == "Timeframe Charts":
        app.active_tab = "timeframe"


    logging.info("Comprehensive timeframe chart fix applied successfully.")
