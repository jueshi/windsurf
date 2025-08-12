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
import plotly.io as pio
from plotly.subplots import make_subplots
import tempfile
import webbrowser
import os
import io
from PIL import Image, ImageTk

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

        # Create a top frame to hold the daily and weekly charts side-by-side
        top_frame = ttk.Frame(timeframe_tab_frame)
        top_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))

        # Create and store dedicated frames for each chart, updating titles
        app.daily_chart_frame = ttk.LabelFrame(top_frame, text="Daily Chart (Last 6 Months)")
        app.daily_chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        app.weekly_chart_frame = ttk.LabelFrame(top_frame, text="Weekly Chart (Last 3 Years)")
        app.weekly_chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Create the monthly chart frame below the top frame
        app.monthly_chart_frame = ttk.LabelFrame(timeframe_tab_frame, text="Monthly Chart (All Time)")
        app.monthly_chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 5))

        # Set initial labels
        ttk.Label(app.daily_chart_frame, text="Select a ticker.").pack(pady=10)
        ttk.Label(app.weekly_chart_frame, text="Select a ticker.").pack(pady=10)
        ttk.Label(app.monthly_chart_frame, text="Select a ticker.").pack(pady=10)

        logging.info("Timeframe chart tab and frames created successfully.")

    except Exception as e:
        logging.error(f"Error creating timeframe UI: {e}")
        return

    # --- 2. Define Chart Generation and Display Logic ---

    def _display_chart_in_frame(frame, fig, ticker, timeframe):
        """
        Helper function to display a Plotly figure in a given Tkinter frame.
        It now displays a static image preview directly in the GUI.
        """
        try:
            # Clear any existing widgets in the frame
            for widget in frame.winfo_children():
                widget.destroy()

            # Save chart to a temporary HTML file for the "Open in Browser" button
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, f"{ticker}_{timeframe}_chart.html")
            fig.write_html(html_path)

            # --- Header ---
            header_frame = ttk.Frame(frame)
            header_frame.pack(fill=tk.X, pady=5, padx=5)
            title = f"{ticker} - {timeframe.capitalize()} Chart"
            ttk.Label(header_frame, text=title, font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
            browser_button = ttk.Button(header_frame, text="Open in Browser",
                                        command=lambda p=html_path: webbrowser.open(f"file:///{os.path.abspath(p)}"))
            browser_button.pack(side=tk.RIGHT)

            # --- Static Image Preview ---
            try:
                # Generate a static image from the figure
                img_bytes = pio.to_image(fig, format='png', width=800, height=250)

                # Convert bytes to a PIL Image
                img_data = io.BytesIO(img_bytes)
                pil_img = Image.open(img_data)

                # Convert PIL Image to a PhotoImage for Tkinter
                photo_img = ImageTk.PhotoImage(pil_img)

                # Create a label to hold the image
                img_label = ttk.Label(frame, image=photo_img)
                img_label.image = photo_img  # Keep a reference!
                img_label.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

            except Exception as img_e:
                # This can happen if the 'kaleido' package is not installed
                logging.warning(f"Could not generate static image for {ticker} {timeframe}: {img_e}")
                fallback_label = ttk.Label(frame, text="Chart preview not available.\n(Requires the 'kaleido' package).\nUse 'Open in Browser' to view.")
                fallback_label.pack(pady=10, padx=5)

            frame.update_idletasks()
        except Exception as e:
            logging.error(f"Error displaying {timeframe} chart in frame: {e}")
            ttk.Label(frame, text=f"Error displaying chart: {e}", foreground="red").pack(pady=10)

    def _generate_and_display_timeframe_charts(self, ticker):
        """
        Fetches data and generates daily, weekly, and monthly charts for the given ticker,
        applying specific date ranges for each timeframe.
        """
        logging.info(f"Generating all timeframe charts for {ticker} with date filtering...")
        self.current_chart_ticker = ticker

        # Correctly fetch data using the existing manager and method
        df = self.manager.load_data(ticker)

        if df is None or df.empty:
            messagebox.showwarning("No Data", f"No data available for {ticker}. Cannot generate timeframe charts.")
            return

        # --- Generate Daily Chart (Last 6 Months) ---
        try:
            # Filter data for the last 6 months
            daily_df_filtered = df[df.index >= (df.index.max() - pd.DateOffset(months=6))]
            fig_daily = go.Figure(data=[go.Candlestick(
                x=daily_df_filtered.index, open=daily_df_filtered['Open'], high=daily_df_filtered['High'],
                low=daily_df_filtered['Low'], close=daily_df_filtered['Close']
            )])
            fig_daily.update_layout(title_text=f"{ticker} Daily", xaxis_rangeslider_visible=False, margin=dict(t=30, b=10, l=20, r=20))
            _display_chart_in_frame(self.daily_chart_frame, fig_daily, ticker, "daily")
        except Exception as e:
            logging.error(f"Failed to generate daily chart for {ticker}: {e}")

        # --- Generate Weekly Chart (Last 3 Years) ---
        try:
            weekly_df = df.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
            # Filter data for the last 3 years
            weekly_df_filtered = weekly_df[weekly_df.index >= (weekly_df.index.max() - pd.DateOffset(years=3))]
            fig_weekly = go.Figure(data=[go.Candlestick(
                x=weekly_df_filtered.index, open=weekly_df_filtered['Open'], high=weekly_df_filtered['High'],
                low=weekly_df_filtered['Low'], close=weekly_df_filtered['Close']
            )])
            fig_weekly.update_layout(title_text=f"{ticker} Weekly", xaxis_rangeslider_visible=False, margin=dict(t=30, b=10, l=20, r=20))
            _display_chart_in_frame(self.weekly_chart_frame, fig_weekly, ticker, "weekly")
        except Exception as e:
            logging.error(f"Failed to generate weekly chart for {ticker}: {e}")

        # --- Generate Monthly Chart (All Time) ---
        try:
            monthly_df = df.resample('M').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
            # No filtering for the monthly chart
            fig_monthly = go.Figure(data=[go.Candlestick(
                x=monthly_df.index, open=monthly_df['Open'], high=monthly_df['High'],
                low=monthly_df['Low'], close=monthly_df['Close']
            )])
            fig_monthly.update_layout(title_text=f"{ticker} Monthly (Log Scale)", xaxis_rangeslider_visible=False,
                                      margin=dict(t=30, b=10, l=20, r=20), yaxis_type='log')
            _display_chart_in_frame(self.monthly_chart_frame, fig_monthly, ticker, "monthly")
        except Exception as e:
            logging.error(f"Failed to generate monthly chart for {ticker}: {e}")

        self.status_var.set(f"Timeframe charts for {ticker} displayed.")

    # Bind the new method to the application instance
    app._generate_and_display_timeframe_charts = types.MethodType(_generate_and_display_timeframe_charts, app)

    # --- 3. Definitive Event Handlers ---

    # Store the original tab change handler, as its core logic is still needed for other tabs.
    original_tab_changed = app._on_tab_changed

    def definitive_on_ticker_selected(self, event, source_listbox):
        """
        This is the single, definitive event handler for ticker selection.
        It replaces the broken patch chain and correctly routes actions for all tabs.
        """
        selected_indices = source_listbox.curselection()
        if not selected_indices:
            return

        ticker_text = source_listbox.get(selected_indices[0])
        ticker = ticker_text.split(' - ')[0].strip()

        if not ticker:
            return

        logging.info(f"Definitive handler triggered for ticker '{ticker}' with active tab '{self.active_tab}'")

        # Branch logic based on the active tab
        if self.active_tab == "timeframe":
            self._generate_and_display_timeframe_charts(ticker)
        elif self.active_tab == "seasonality":
            self._generate_seasonality_chart(ticker)
        elif self.active_tab == "comparison":
            self._compare_percentage_performance()
        else:  # Default to "individual"
            self._display_chart(ticker)

    def definitive_on_tab_changed(self, event=None):
        """
        This is the single, definitive event handler for tab changes.
        It correctly sets the active tab state and calls the appropriate update logic.
        """
        try:
            selected_tab_widget = self.chart_notebook.select()
            selected_tab_text = self.chart_notebook.tab(selected_tab_widget, "text")
        except tk.TclError:
            return # Fails gracefully on shutdown

        # First, set the active tab state. This is crucial.
        if selected_tab_text == "Timeframe Charts":
            self.active_tab = "timeframe"
        # For other tabs, we rely on the original handler's logic.
        # However, to be safe, we can set them here too.
        elif selected_tab_text == "Individual Chart":
            self.active_tab = "individual"
        elif selected_tab_text == "Comparison Chart":
            self.active_tab = "comparison"
        elif selected_tab_text == "Seasonality Chart":
            self.active_tab = "seasonality"

        logging.info(f"Definitive tab handler set active tab to: '{self.active_tab}'")

        # Now, decide what to do.
        if self.active_tab == "timeframe":
            # For our custom tab, we trigger our own logic.
            ticker_to_load = None
            if self.ticker_listbox.curselection():
                ticker_text = self.ticker_listbox.get(self.ticker_listbox.curselection()[0])
                ticker_to_load = ticker_text.split(' - ')[0].strip()
            elif self.watch_listbox.curselection():
                 ticker_to_load = self.watch_listbox.get(self.watch_listbox.curselection()[0]).strip()
            elif hasattr(self, 'current_chart_ticker') and self.current_chart_ticker:
                ticker_to_load = self.current_chart_ticker

            if ticker_to_load:
                self._generate_and_display_timeframe_charts(ticker_to_load)
        else:
            # For all other tabs, we let the original logic run.
            # This is safer than trying to replicate its functionality.
            original_tab_changed(event)

    # Bind the definitive methods to the app instance, completely replacing the old ones.
    app.definitive_on_ticker_selected = types.MethodType(definitive_on_ticker_selected, app)
    app._on_ticker_selected = lambda event, s=app: s.definitive_on_ticker_selected(event, s.ticker_listbox)
    app._on_watch_ticker_selected = lambda event, s=app: s.definitive_on_ticker_selected(event, s.watch_listbox)
    app._on_tab_changed = types.MethodType(definitive_on_tab_changed, app)

    # Re-bind the events to ensure our new handlers are called.
    app.ticker_listbox.bind("<<ListboxSelect>>", app._on_ticker_selected)
    app.watch_listbox.bind("<<ListboxSelect>>", app._on_watch_ticker_selected)
    app.chart_notebook.bind("<<NotebookTabChanged>>", app._on_tab_changed)

    logging.info("Definitive event handlers applied successfully.")
