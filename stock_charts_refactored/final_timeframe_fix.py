"""
Comprehensive Timeframe Chart Fix (Refactored for Testability & Matplotlib)

This module provides a single, robust implementation for the timeframe charts feature.
It replaces the Plotly/Kaleido dependency with Matplotlib for image generation
to remove the need for an external Chrome installation.
"""

import logging
import types
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import tempfile
import webbrowser
import os
from PIL import Image, ImageTk

# ==============================================================================
# Standalone, Testable Core Logic using Matplotlib
# ==============================================================================

def display_chart_in_frame(app, frame, fig, ticker, timeframe):
    """
    Helper function to display a Matplotlib figure in a given Tkinter frame.
    """
    try:
        for widget in frame.winfo_children():
            widget.destroy()

        temp_dir = tempfile.gettempdir()
        # The HTML path is no longer relevant for Matplotlib, but we can keep the button
        # to open the generated PNG file itself.
        img_path = os.path.join(temp_dir, f"{ticker}_{timeframe}_image.png")
        fig.savefig(img_path, dpi=100, bbox_inches='tight')
        plt.close(fig) # Close the figure to free up memory

        header_frame = ttk.Frame(frame)
        header_frame.pack(fill=tk.X, pady=5, padx=5)
        title = f"{ticker} - {timeframe.capitalize()} Chart"
        ttk.Label(header_frame, text=title, font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        browser_button = ttk.Button(header_frame, text="Open Image",
                                    command=lambda p=img_path: webbrowser.open(f"file:///{os.path.abspath(p)}"))
        browser_button.pack(side=tk.RIGHT)

        pil_img = Image.open(img_path)
        # Resize image to fit the frame, e.g., max width 800, max height 250
        pil_img.thumbnail((800, 250), Image.LANCZOS)
        photo_img = ImageTk.PhotoImage(pil_img)

        app.timeframe_chart_images.append(photo_img)

        img_label = ttk.Label(frame, image=photo_img)
        img_label.image = photo_img
        img_label.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

        frame.update_idletasks()
    except Exception as e:
        logging.error(f"Error displaying {timeframe} chart in frame: {e}")
        ttk.Label(frame, text=f"Error displaying chart: {e}", foreground="red").pack(pady=10)

def generate_and_display_timeframe_charts(app, ticker):
    """
    Fetches data and generates daily, weekly, and monthly charts using Matplotlib.
    """
    app.timeframe_chart_images.clear()
    logging.info(f"Generating all timeframe charts for {ticker} with Matplotlib...")
    app.current_chart_ticker = ticker

    df = app.manager.load_data(ticker)
    if df is None or df.empty:
        messagebox.showwarning("No Data", f"No data available for {ticker}.")
        return

    # --- Generate Daily Chart (Last 6 Months) ---
    try:
        daily_df_filtered = df[df.index >= (df.index.max() - pd.DateOffset(months=6))]
        fig_daily, ax_daily = plt.subplots(figsize=(8, 2.5))
        ax_daily.plot(daily_df_filtered.index, daily_df_filtered['Close'])
        ax_daily.set_title(f"{ticker} Daily", fontsize=10)
        ax_daily.grid(True, alpha=0.3)
        ax_daily.xaxis.set_major_formatter(DateFormatter("%b %Y"))
        fig_daily.tight_layout()
        display_chart_in_frame(app, app.daily_chart_frame, fig_daily, ticker, "daily")
    except Exception as e:
        logging.error(f"Failed to generate daily chart for {ticker} with Matplotlib: {e}")

    # --- Generate Weekly Chart (Last 3 Years) ---
    try:
        weekly_df = df.resample('W').agg({'Close': 'last'}).dropna()
        weekly_df_filtered = weekly_df[weekly_df.index >= (weekly_df.index.max() - pd.DateOffset(years=3))]
        fig_weekly, ax_weekly = plt.subplots(figsize=(8, 2.5))
        ax_weekly.plot(weekly_df_filtered.index, weekly_df_filtered['Close'])
        ax_weekly.set_title(f"{ticker} Weekly", fontsize=10)
        ax_weekly.grid(True, alpha=0.3)
        ax_weekly.xaxis.set_major_formatter(DateFormatter("%Y"))
        fig_weekly.tight_layout()
        display_chart_in_frame(app, app.weekly_chart_frame, fig_weekly, ticker, "weekly")
    except Exception as e:
        logging.error(f"Failed to generate weekly chart for {ticker} with Matplotlib: {e}")

    # --- Generate Monthly Chart (All Time, Log Scale) ---
    try:
        monthly_df = df.resample('ME').agg({'Close': 'last'}).dropna()
        fig_monthly, ax_monthly = plt.subplots(figsize=(8, 2.5))
        ax_monthly.semilogy(monthly_df.index, monthly_df['Close']) # Use log scale
        ax_monthly.set_title(f"{ticker} Monthly (Log Scale)", fontsize=10)
        ax_monthly.grid(True, which='both', alpha=0.3)
        ax_monthly.xaxis.set_major_formatter(DateFormatter("%Y"))
        fig_monthly.tight_layout()
        display_chart_in_frame(app, app.monthly_chart_frame, fig_monthly, ticker, "monthly")
    except Exception as e:
        logging.error(f"Failed to generate monthly chart for {ticker} with Matplotlib: {e}")

    app.status_var.set(f"Timeframe charts for {ticker} displayed.")

# ==============================================================================
# Patcher and Event Handlers (Unchanged from previous version)
# ==============================================================================

def apply_final_timeframe_fix(app):
    logging.info("Applying the final, comprehensive timeframe chart fix (using Matplotlib)...")
    app.timeframe_chart_images = []

    try:
        if not hasattr(app, 'chart_notebook'):
            logging.error("Cannot create timeframe tab: chart_notebook not found.")
            return
        timeframe_tab_frame = ttk.Frame(app.chart_notebook)
        app.chart_notebook.add(timeframe_tab_frame, text="Timeframe Charts")
        top_frame = ttk.Frame(timeframe_tab_frame)
        top_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))
        app.daily_chart_frame = ttk.LabelFrame(top_frame, text="Daily Chart (Last 6 Months)")
        app.daily_chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        app.weekly_chart_frame = ttk.LabelFrame(top_frame, text="Weekly Chart (Last 3 Years)")
        app.weekly_chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        app.monthly_chart_frame = ttk.LabelFrame(timeframe_tab_frame, text="Monthly Chart (All Time)")
        app.monthly_chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 5))
        ttk.Label(app.daily_chart_frame, text="Select a ticker.").pack(pady=10)
        ttk.Label(app.weekly_chart_frame, text="Select a ticker.").pack(pady=10)
        ttk.Label(app.monthly_chart_frame, text="Select a ticker.").pack(pady=10)
        logging.info("Timeframe chart tab and frames created successfully.")
    except Exception as e:
        logging.error(f"Error creating timeframe UI: {e}")
        return

    app._generate_and_display_timeframe_charts = types.MethodType(generate_and_display_timeframe_charts, app)

    original_tab_changed = app._on_tab_changed

    def definitive_on_ticker_selected(self, event, source_listbox):
        selected_indices = source_listbox.curselection()
        if not selected_indices: return
        ticker = source_listbox.get(selected_indices[0]).split(' - ')[0].strip()
        if not ticker: return
        logging.info(f"Definitive handler triggered for ticker '{ticker}' with active tab '{self.active_tab}'")
        if self.active_tab == "timeframe":
            self._generate_and_display_timeframe_charts(ticker)
        elif self.active_tab == "seasonality":
            self._generate_seasonality_chart(ticker)
        elif self.active_tab == "comparison":
            self._compare_percentage_performance()
        else:
            self._display_chart(ticker)

    def definitive_on_tab_changed(self, event=None):
        try:
            selected_tab_widget = self.chart_notebook.select()
            selected_tab_text = self.chart_notebook.tab(selected_tab_widget, "text")
        except tk.TclError:
            return
        tab_map = {"Timeframe Charts": "timeframe", "Individual Chart": "individual", "Comparison Chart": "comparison", "Seasonality Chart": "seasonality"}
        self.active_tab = tab_map.get(selected_tab_text, "individual")
        logging.info(f"Definitive tab handler set active tab to: '{self.active_tab}'")
        if self.active_tab == "timeframe":
            ticker_to_load = None
            if self.ticker_listbox.curselection():
                ticker_to_load = self.ticker_listbox.get(self.ticker_listbox.curselection()[0]).split(' - ')[0].strip()
            elif self.watch_listbox.curselection():
                ticker_to_load = self.watch_listbox.get(self.watch_listbox.curselection()[0]).strip()
            elif hasattr(self, 'current_chart_ticker') and self.current_chart_ticker:
                ticker_to_load = self.current_chart_ticker
            if ticker_to_load:
                self._generate_and_display_timeframe_charts(ticker_to_load)
        else:
            original_tab_changed(event)

    app.definitive_on_ticker_selected = types.MethodType(definitive_on_ticker_selected, app)
    app._on_ticker_selected = lambda event, s=app: s.definitive_on_ticker_selected(event, s.ticker_listbox)
    app._on_watch_ticker_selected = lambda event, s=app: s.definitive_on_ticker_selected(event, s.watch_listbox)
    app._on_tab_changed = types.MethodType(definitive_on_tab_changed, app)
    app.ticker_listbox.bind("<<ListboxSelect>>", app._on_ticker_selected)
    app.watch_listbox.bind("<<ListboxSelect>>", app._on_watch_ticker_selected)
    app.chart_notebook.bind("<<NotebookTabChanged>>", app._on_tab_changed)
    logging.info("Definitive event handlers applied successfully.")
