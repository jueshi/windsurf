import os
import sys
import re
import json
import time
import logging
import importlib
import inspect
import threading
import webbrowser
from queue import Queue
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from tkcalendar import DateEntry
import time
import pytz
import yfinance as yf
from typing import Optional, List, Any
from random import uniform
import math

from data_manager import StockDataManager

class StockDataGUI:
    """GUI for Stock Data Manager"""

    def __init__(self, root, manager):
        """Initialize the GUI."""
        self.root = root
        self.manager = manager
        self.current_tickers = []
        self.watch_list = []
        self.current_image = None  # Store reference to prevent garbage collection
        self.active_tab = "individual"  # Track which tab is active: "individual" or "comparison"

        # Load ticker lists from ticker_lists.py
        self.ticker_lists = {}
        self._load_ticker_lists_from_module()

        # Load watch list from ticker_lists.py if it exists
        try:
            import ticker_lists
            if hasattr(ticker_lists, 'watch_list'):
                self.watch_list = ticker_lists.watch_list.copy()
                logging.info(f"Loaded {len(self.watch_list)} tickers from watch list")
        except Exception as e:
            logging.error(f"Error loading watch list: {e}")

        self._create_widgets()

    def _load_ticker_lists_from_module(self):
        """Load all ticker lists from ticker_lists module"""
        # Import or reload the ticker_lists module to get fresh data
        try:
            # First, try to completely remove the module from sys.modules
            if 'ticker_lists' in sys.modules:
                del sys.modules['ticker_lists']

            # Now import it fresh
            import ticker_lists

            # Get the module
            current_module = ticker_lists

            # Clear existing ticker lists
            self.ticker_lists = {}

            # Get the source code of the module to check for commented lines
            module_file = inspect.getfile(current_module)
            with open(module_file, 'r', encoding='utf-8') as f:
                source_lines = f.readlines()

            # Extract active variable names (not commented out)
            active_vars = set()
            for line in source_lines:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    var_name = line.split('=')[0].strip()
                    active_vars.add(var_name)

            # Load all list objects from the module that are not commented out
            for name in dir(current_module):
                # Skip private/special variables, functions, and commented-out variables
                if name.startswith('__') or callable(getattr(current_module, name)) or name not in active_vars:
                    continue

                try:
                    obj = getattr(current_module, name)
                    if isinstance(obj, list):
                        self.ticker_lists[name] = obj
                except Exception as e:
                    logging.debug(f"Error accessing {name}: {e}")

            logging.info(f"Loaded {len(self.ticker_lists)} ticker lists from ticker_lists.py")
        except Exception as e:
            logging.error(f"Error loading ticker lists: {e}")
            messagebox.showerror("Error", f"Failed to load ticker lists: {e}")

    # def _refresh_ticker_lists(self):
    #     """Reload ticker lists from ticker_lists.py"""
    #     try:
    #         # Remember current selection
    #         current_selection = self.ticker_list_var.get()

    #         # Clear filter if any
    #         if hasattr(self, 'list_filter_var'):
    #             self.list_filter_var.set('')

    #         # Reload ticker lists from module
    #         self._load_ticker_lists_from_module()

    #         # Update the dropdown values
    #         self.ticker_list_combo['values'] = list(self.ticker_lists.keys())

    #         # Restore previous selection if it still exists
    #         if current_selection and current_selection in self.ticker_lists:
    #             self.ticker_list_var.set(current_selection)

    #         # Also refresh watch list if it exists in ticker_lists.py
    #         try:
    #             import ticker_lists
    #             importlib.reload(ticker_lists)
    #             if hasattr(ticker_lists, 'watch_list'):
    #                 self.watch_list = ticker_lists.watch_list.copy()
    #                 # Update watch list display
    #                 self.watch_listbox.delete(0, tk.END)
    #                 for ticker in self.watch_list:
    #                     self.watch_listbox.insert(tk.END, ticker)
    #                 logging.info(f"Refreshed watch list with {len(self.watch_list)} tickers")
    #         except Exception as e:
    #             logging.error(f"Error refreshing watch list: {e}")

    #         # Update status
    #         self.status_var.set(f"Refreshed {len(self.ticker_lists)} ticker lists from ticker_lists.py")

    #     except Exception as e:
    #         logging.error(f"Error refreshing ticker lists: {str(e)}")
    #         messagebox.showerror("Error", f"Failed to refresh ticker lists: {str(e)}")

    def _filter_ticker_lists(self, event=None):
        """Filter ticker lists dropdown based on filter text"""
        filter_text = self.list_filter_var.get().lower()

        if not filter_text:
            # If filter is empty, show all ticker lists
            self.ticker_list_combo['values'] = list(self.ticker_lists.keys())
        else:
            # Filter ticker lists that contain the filter text
            filtered_lists = [name for name in self.ticker_lists.keys()
                            if filter_text in name.lower()]
            self.ticker_list_combo['values'] = filtered_lists

            # If there's a match and the current selection doesn't match the filter,
            # update the selection to the first match
            if filtered_lists and self.ticker_list_var.get() not in filtered_lists:
                self.ticker_list_var.set(filtered_lists[0])

    def _refresh_ticker_lists(self):
        """Reload ticker lists from ticker_lists.py"""
        try:
            # Remember current selection
            current_selection = self.ticker_list_var.get()

            # Clear filter if any
            if hasattr(self, 'list_filter_var'):
                self.list_filter_var.set('')

            # Store old ticker lists for comparison
            old_ticker_lists = set(self.ticker_lists.keys())

            # Reload ticker lists from module
            self._load_ticker_lists_from_module()

            # Get new ticker lists
            new_ticker_lists = set(self.ticker_lists.keys())

            # Find removed and added lists
            removed_lists = old_ticker_lists - new_ticker_lists
            added_lists = new_ticker_lists - old_ticker_lists

            # Update the dropdown values
            self.ticker_list_combo['values'] = list(self.ticker_lists.keys())

            # Restore previous selection if it still exists
            if current_selection and current_selection in self.ticker_lists:
                self.ticker_list_var.set(current_selection)
            elif removed_lists and current_selection in removed_lists:
                # If current selection was removed, select first available list
                if self.ticker_lists:
                    self.ticker_list_var.set(list(self.ticker_lists.keys())[0])
                    self._load_ticker_list()  # Load the newly selected list

            # Also refresh watch list if it exists in ticker_lists.py
            try:
                import ticker_lists
                importlib.reload(ticker_lists)
                if hasattr(ticker_lists, 'watch_list'):
                    self.watch_list = ticker_lists.watch_list.copy()
                    # Update watch list display
                    self.watch_listbox.delete(0, tk.END)
                    for ticker in self.watch_list:
                        self.watch_listbox.insert(tk.END, ticker)
                    logging.info(f"Refreshed watch list with {len(self.watch_list)} tickers")
            except Exception as e:
                logging.error(f"Error refreshing watch list: {e}")

            # Update status with information about changes
            status_msg = f"Refreshed {len(self.ticker_lists)} ticker lists from ticker_lists.py"
            if removed_lists:
                status_msg += f" (Removed: {', '.join(removed_lists)})"
            if added_lists:
                status_msg += f" (Added: {', '.join(added_lists)})"
            self.status_var.set(status_msg)

        except Exception as e:
            logging.error(f"Error refreshing ticker lists: {str(e)}")
            messagebox.showerror("Error", f"Failed to refresh ticker lists: {str(e)}")

    def _create_widgets(self):
        """Create all GUI widgets"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create top frame for ticker list selection
        top_frame = ttk.Frame(main_frame, padding="10")
        top_frame.pack(fill=tk.X, pady=5)

        # Ticker list selection with filter
        ttk.Label(top_frame, text="Ticker List:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

        # Create a frame for the dropdown and its filter
        dropdown_frame = ttk.Frame(top_frame)
        dropdown_frame.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Add filter entry for ticker list dropdown
        self.list_filter_var = tk.StringVar()
        list_filter_entry = ttk.Entry(dropdown_frame, textvariable=self.list_filter_var, width=20)
        list_filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        list_filter_entry.bind("<KeyRelease>", self._filter_ticker_lists)

        # Create the combobox for ticker lists
        self.ticker_list_var = tk.StringVar()
        self.ticker_list_combo = ttk.Combobox(dropdown_frame, textvariable=self.ticker_list_var,
                                        values=list(self.ticker_lists.keys()), width=60)
        self.ticker_list_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        self.ticker_list_combo.bind("<<ComboboxSelected>>", self._on_list_selected)

        # Create a frame for the buttons
        button_frame = ttk.Frame(top_frame)
        button_frame.grid(row=0, column=2, padx=5, pady=5)

        # Load List button loads the selected list
        ttk.Button(button_frame, text="Load List", command=self._load_ticker_list).pack(side=tk.LEFT, padx=(0, 5))

        # Refresh Lists button reloads ticker lists from ticker_lists.py
        ttk.Button(button_frame, text="Refresh Lists", command=self._refresh_ticker_lists).pack(side=tk.LEFT)

        # Add manual ticker entry
        ttk.Label(top_frame, text="Add Ticker:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.manual_ticker_var = tk.StringVar()
        manual_ticker_entry = ttk.Entry(top_frame, textvariable=self.manual_ticker_var, width=80)
        manual_ticker_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(top_frame, text="Add", command=self._add_manual_ticker).grid(row=1, column=2, padx=5, pady=5)

        # Add list name entry and save button
        ttk.Label(top_frame, text="New List Name:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.list_name_var = tk.StringVar()
        list_name_entry = ttk.Entry(top_frame, textvariable=self.list_name_var, width=80)
        list_name_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(top_frame, text="Save List", command=self._save_ticker_list).grid(row=2, column=2, padx=5, pady=5)

        # Create middle frame with three sections: available tickers, watch list, and chart display
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Left section for available tickers (limited width)
        left_frame = ttk.LabelFrame(middle_frame, text="Available Tickers", padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        # Add filter entry for ticker list
        filter_frame = ttk.Frame(left_frame)
        filter_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar()
        filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var, width=8)
        filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Bind filter entry to update the list as user types
        self.filter_var.trace_add("write", self._apply_ticker_filter)

        # Create ticker listbox with scrollbar
        ticker_frame = ttk.Frame(left_frame)
        ticker_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(ticker_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Limit width to 5 letters (approximately 40 pixels)
        self.ticker_listbox = tk.Listbox(ticker_frame, selectmode=tk.EXTENDED, height=20, width=10)
        self.ticker_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.ticker_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.ticker_listbox.yview)

        # Middle section for watch list (limited width)
        middle_list_frame = ttk.LabelFrame(middle_frame, text="Watch List", padding="5")
        middle_list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        # Create watch list listbox with scrollbar
        watch_frame = ttk.Frame(middle_list_frame)
        watch_frame.pack(fill=tk.BOTH, expand=True)

        watch_scrollbar = ttk.Scrollbar(watch_frame)
        watch_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Limit width to 5 letters (approximately 40 pixels)
        self.watch_listbox = tk.Listbox(watch_frame, selectmode=tk.EXTENDED, height=20, width=10)
        self.watch_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right section for chart display (takes remaining space)
        self.chart_frame = ttk.LabelFrame(middle_frame, text="Chart Display", padding="5")
        self.chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 0))

        # Add date range controls at the top of chart frame
        date_range_frame = ttk.Frame(self.chart_frame, padding="5")
        date_range_frame.pack(fill=tk.X, expand=False, pady=(0, 5))

        # Start date entry with calendar widget
        ttk.Label(date_range_frame, text="Start Date:").pack(side=tk.LEFT, padx=(0, 5))
        self.start_date_var = tk.StringVar()
        self.start_date_entry = DateEntry(date_range_frame, textvariable=self.start_date_var, width=12,
                                        date_pattern='yyyy-mm-dd', background='darkblue', foreground='white',
                                        borderwidth=2, locale='en_US')
        self.start_date_entry.pack(side=tk.LEFT, padx=(0, 10))

        # End date entry with calendar widget
        ttk.Label(date_range_frame, text="End Date:").pack(side=tk.LEFT, padx=(0, 5))
        self.end_date_var = tk.StringVar()
        self.end_date_entry = DateEntry(date_range_frame, textvariable=self.end_date_var, width=12,
                                      date_pattern='yyyy-mm-dd', background='darkblue', foreground='white',
                                      borderwidth=2, locale='en_US')
        self.end_date_entry.pack(side=tk.LEFT, padx=(0, 10))

        # Apply date range button
        ttk.Button(date_range_frame, text="Apply Date Range", command=self._apply_date_range).pack(side=tk.LEFT)

        # Create a notebook with tabs for individual, comparison, and seasonality charts
        self.chart_notebook = ttk.Notebook(self.chart_frame)
        self.chart_notebook.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # Create individual chart tab
        self.individual_chart_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.individual_chart_frame, text="Individual Chart")

        # Create comparison chart tab
        self.comparison_chart_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.comparison_chart_frame, text="Comparison Chart")
        
        # Create seasonality chart tab
        self.seasonality_chart_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.seasonality_chart_frame, text="Seasonality Chart")
        
        # Create a frame for year selection in seasonality tab
        self.seasonality_controls_frame = ttk.Frame(self.seasonality_chart_frame)
        self.seasonality_controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create year selection dropdown
        ttk.Label(self.seasonality_controls_frame, text="Select Year:").pack(side=tk.LEFT, padx=(0, 5))
        self.year_var = tk.StringVar(value="All Years")
        self.year_dropdown = ttk.Combobox(self.seasonality_controls_frame, textvariable=self.year_var, state="readonly", width=15)
        self.year_dropdown.pack(side=tk.LEFT)
        self.year_dropdown.bind("<<ComboboxSelected>>", self._on_year_selected)

        # Bind tab change event
        self.chart_notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Create labels to display charts in each tab
        self.chart_label = ttk.Label(self.individual_chart_frame)
        self.chart_label.pack(fill=tk.BOTH, expand=True)

        self.comparison_chart_label = ttk.Label(self.comparison_chart_frame)
        self.comparison_chart_label.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for the seasonality chart
        self.seasonality_chart_container = ttk.Frame(self.seasonality_chart_frame)
        self.seasonality_chart_container.pack(fill=tk.BOTH, expand=True)
        
        # Create label to display seasonality chart
        self.seasonality_chart_label = ttk.Label(self.seasonality_chart_container)
        self.seasonality_chart_label.pack(fill=tk.BOTH, expand=True)

        self.watch_listbox.config(yscrollcommand=watch_scrollbar.set)
        watch_scrollbar.config(command=self.watch_listbox.yview)

        # Populate watch list listbox with loaded watch list
        for ticker in self.watch_list:
            self.watch_listbox.insert(tk.END, ticker)

        # Create right-click context menu for ticker listbox
        self.ticker_context_menu = tk.Menu(self.ticker_listbox, tearoff=0)
        self.ticker_context_menu.add_command(label="Copy to Watch List", command=self._copy_to_watch_list)

        # Create right-click context menu for watch list
        self.watch_context_menu = tk.Menu(self.watch_listbox, tearoff=0)
        self.watch_context_menu.add_command(label="Delete from Watch List", command=self._delete_from_watch_list)

        # Bind right-click events
        self.ticker_listbox.bind("<Button-3>", self._show_ticker_context_menu)
        self.watch_listbox.bind("<Button-3>", self._show_watch_context_menu)

        # Bind selection events to display charts
        self.ticker_listbox.bind("<<ListboxSelect>>", self._on_ticker_selected)
        self.watch_listbox.bind("<<ListboxSelect>>", self._on_watch_ticker_selected)

        # Create bottom frame for actions
        bottom_frame = ttk.Frame(main_frame, padding="10")
        bottom_frame.pack(fill=tk.X, pady=5)

        # Force download toggle
        self.force_download_var = tk.BooleanVar(value=False)
        force_download_check = ttk.Checkbutton(bottom_frame, text="Force Download", variable=self.force_download_var)
        force_download_check.pack(side=tk.RIGHT, padx=5)
        ttk.Label(bottom_frame, text="Options:").pack(side=tk.RIGHT, padx=5)

        # Action buttons
        ttk.Button(bottom_frame, text="Download/Update Data", command=self._download_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Visualize Daily/Weekly/Monthly", command=self._visualize_all_timeframes).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="View HTML Report", command=self._view_html_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Compare % Performance", command=self._compare_percentage_performance).pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _on_list_selected(self, event):
        """Handle ticker list selection and auto-load the selected list"""
        selected_list = self.ticker_list_var.get()
        if selected_list in self.ticker_lists:
            self.status_var.set(f"Selected list: {selected_list} with {len(self.ticker_lists[selected_list])} tickers")
            # Auto-load the selected ticker list
            self._load_ticker_list()

    # def _refresh_ticker_lists(self):
    #     """Reload ticker lists from ticker_lists.py"""
    #     try:
    #         # Remember current selection
    #         current_selection = self.ticker_list_var.get()

    #         # Clear filter if any
    #         if hasattr(self, 'list_filter_var'):
    #             self.list_filter_var.set('')

    #         # Reload ticker lists from module
    #         self._load_ticker_lists_from_module()

    #         # Update the dropdown values
    #         self.ticker_list_combo['values'] = list(self.ticker_lists.keys())

    #         # Restore previous selection if it still exists
    #         if current_selection and current_selection in self.ticker_lists:
    #             self.ticker_list_var.set(current_selection)

    #         # Also refresh watch list if it exists in ticker_lists.py
    #         try:
    #             import ticker_lists
    #             importlib.reload(ticker_lists)
    #             if hasattr(ticker_lists, 'watch_list'):
    #                 self.watch_list = ticker_lists.watch_list.copy()
    #                 # Update watch list display
    #                 self.watch_listbox.delete(0, tk.END)
    #                 for ticker in self.watch_list:
    #                     self.watch_listbox.insert(tk.END, ticker)
    #                 logging.info(f"Refreshed watch list with {len(self.watch_list)} tickers")
    #         except Exception as e:
    #             logging.error(f"Error refreshing watch list: {e}")

    #         # Update status
    #         self.status_var.set(f"Refreshed {len(self.ticker_lists)} ticker lists from ticker_lists.py")

    #     except Exception as e:
    #         logging.error(f"Error refreshing ticker lists: {str(e)}")
    #         messagebox.showerror("Error", f"Failed to refresh ticker lists: {str(e)}")

    # def _filter_ticker_lists(self, event=None):
    #     """Filter the ticker lists dropdown based on input"""
    #     filter_text = self.list_filter_var.get().lower()

    #     if not filter_text:
    #         # If filter is empty, show all lists
    #         self.ticker_list_combo['values'] = list(self.ticker_lists.keys())
    #         return

    #     # Filter lists based on input
    #     filtered_lists = [name for name in self.ticker_lists.keys()
    #                      if filter_text in name.lower()]

    #     # Update dropdown values
    #     self.ticker_list_combo['values'] = filtered_lists

    #     # If exactly one match, select it and load it automatically
    #     if len(filtered_lists) == 1:
    #         self.ticker_list_var.set(filtered_lists[0])
    #         self._load_ticker_list()
    #     else:
    #         # Reset to show all lists
    #         self.ticker_list_combo['values'] = list(self.ticker_lists.keys())

    #     # Update status
    #     if filter_text:
    #         self.status_var.set(f"List filter: '{filter_text}' - {len(self.ticker_list_combo['values'])} matches")

    def _apply_ticker_filter(self, *args):
        """Filter the ticker list based on filter text"""
        filter_text = self.filter_var.get().strip().upper()

        # If no current tickers or no filter, don't do anything
        if not hasattr(self, 'current_tickers') or not self.current_tickers:
            return

        # Get the currently selected list
        selected_list = self.ticker_list_var.get()
        if not selected_list or selected_list not in self.ticker_lists:
            return

        # Get the full list of tickers
        tickers = self.current_tickers

        # Clear the listbox
        self.ticker_listbox.delete(0, tk.END)

        # Apply filter and update listbox
        filtered_count = 0
        for ticker in tickers:
            # Apply filter
            if filter_text and filter_text not in ticker.upper():
                continue

            # Add ticker to listbox
            if 'tickers_comment_dict' in globals() and ticker in tickers_comment_dict:
                self.ticker_listbox.insert(tk.END, f"{ticker} - {tickers_comment_dict[ticker]}")
            else:
                self.ticker_listbox.insert(tk.END, ticker)
            filtered_count += 1

        # Update status
        if filter_text:
            self.status_var.set(f"Filter '{filter_text}': showing {filtered_count}/{len(tickers)} tickers from {selected_list}")
        else:
            self.status_var.set(f"Showing all {len(tickers)} tickers from {selected_list}")

    def _load_ticker_list(self):
        """Load selected ticker list into listbox"""
        selected_list = self.ticker_list_var.get()
        if not selected_list:
            messagebox.showwarning("No List Selected", "Please select a ticker list first.")
            return

        if selected_list in self.ticker_lists:
            tickers = self.ticker_lists[selected_list]
            self.current_tickers = tickers

            # Reset filter when loading a new list
            if hasattr(self, 'filter_var'):
                self.filter_var.set('')

            # Update listbox
            self.ticker_listbox.delete(0, tk.END)
            for ticker in tickers:
                # Check if we have a comment for this ticker
                if 'tickers_comment_dict' in globals() and ticker in tickers_comment_dict:
                    self.ticker_listbox.insert(tk.END, f"{ticker} - {tickers_comment_dict[ticker]}")
                else:
                    self.ticker_listbox.insert(tk.END, ticker)

            self.status_var.set(f"Loaded {len(tickers)} tickers from {selected_list}")

    def _add_manual_ticker(self):
        """Add manually entered ticker(s)"""
        ticker_input = self.manual_ticker_var.get().strip()
        if not ticker_input:
            return

        # Remove brackets and split by commas
        ticker_input = ticker_input.replace('[', '').replace(']', '')
        tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

        if not tickers:
            return

        added_count = 0
        for ticker in tickers:
            # Remove any quotes around the ticker
            ticker = ticker.strip('\'"')

            # Skip empty tickers
            if not ticker:
                continue

            # Add to current tickers if not already present
            if ticker not in self.current_tickers:
                self.current_tickers.append(ticker)
                self.ticker_listbox.insert(tk.END, ticker)
                added_count += 1

        if added_count == 1:
            self.status_var.set(f"Added ticker: {tickers[0]}")
        else:
            self.status_var.set(f"Added {added_count} tickers")

        # Clear entry field
        self.manual_ticker_var.set("")

    def _save_ticker_list(self):
        """Save current tickers as a new list in ticker_lists.py"""
        list_name = self.list_name_var.get().strip()
        if not list_name:
            messagebox.showwarning("No List Name", "Please enter a name for the ticker list.")
            return

        if not self.current_tickers:
            messagebox.showwarning("No Tickers", "Please add tickers to the list before saving.")
            return

        # Format list name to be a valid Python variable name
        list_name = list_name.replace(" ", "_").replace("-", "_")
        if not list_name[0].isalpha() and list_name[0] != '_':
            list_name = "ticker_" + list_name

        # Create Python code for the new list
        tickers_str = ", ".join([f"\"{ticker}\"" for ticker in self.current_tickers])
        new_list_code = f"\n{list_name}_stocks = [{tickers_str}]\n"

        try:
            # Read the current content of ticker_lists.py
            ticker_lists_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_lists.py")
            with open(ticker_lists_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Find the position of the first function definition
            function_pattern = re.compile(r'\n# Function to')
            match = function_pattern.search(content)

            if match:
                # Insert the new list before the function definition
                insert_position = match.start()
                new_content = content[:insert_position] + new_list_code + content[insert_position:]

                # Write the modified content back to the file
                with open(ticker_lists_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
            else:
                # If no function definition found, append to the end of the file
                with open(ticker_lists_path, "a", encoding="utf-8") as f:
                    f.write(new_list_code)

            # Update the ticker lists dictionary
            self.ticker_lists[list_name + "_stocks"] = self.current_tickers
            self.ticker_list_var.set(list_name + "_stocks")

            # Update the dropdown menu
            self.ticker_list_combo['values'] = list(self.ticker_lists.keys())

            self.status_var.set(f"Saved {len(self.current_tickers)} tickers as '{list_name}_stocks'")
            messagebox.showinfo("List Saved", f"Ticker list saved as '{list_name}_stocks' in ticker_lists.py")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving ticker list: {str(e)}")
            logging.error(f"Error saving ticker list: {e}")

    def _show_ticker_context_menu(self, event):
        """Show context menu on right-click in ticker listbox"""
        # Only show context menu if there are selected items
        if self.ticker_listbox.curselection():
            try:
                self.ticker_context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.ticker_context_menu.grab_release()

    def _show_watch_context_menu(self, event):
        """Show context menu on right-click in watch list"""
        # Only show context menu if there are selected items
        if self.watch_listbox.curselection():
            try:
                self.watch_context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.watch_context_menu.grab_release()

    def _delete_from_watch_list(self):
        """Delete selected tickers from watch list"""
        selected_indices = self.watch_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select at least one ticker to delete.")
            return

        # Get selected tickers
        selected_tickers = [self.watch_listbox.get(i) for i in selected_indices]

        # Confirm deletion
        if len(selected_tickers) == 1:
            confirm = messagebox.askyesno("Confirm Delete", f"Delete {selected_tickers[0]} from watch list?")
        else:
            confirm = messagebox.askyesno("Confirm Delete", f"Delete {len(selected_tickers)} tickers from watch list?")

        if not confirm:
            return

        # Delete from watch list (in reverse order to maintain correct indices)
        for i in sorted(selected_indices, reverse=True):
            ticker = self.watch_listbox.get(i)
            self.watch_listbox.delete(i)
            if ticker in self.watch_list:
                self.watch_list.remove(ticker)

        # Save the updated watch list
        self._save_watch_list()

        if len(selected_tickers) == 1:
            self.status_var.set(f"Deleted {selected_tickers[0]} from watch list")
        else:
            self.status_var.set(f"Deleted {len(selected_tickers)} tickers from watch list")

    def _copy_to_watch_list(self):
        """Copy selected tickers to watch list and save to ticker_lists.py"""
        selected_tickers = self._get_selected_tickers()
        if not selected_tickers:
            return

        # Add selected tickers to watch list if not already present
        added_count = 0
        for ticker in selected_tickers:
            if ticker not in self.watch_list:
                self.watch_list.append(ticker)
                self.watch_listbox.insert(tk.END, ticker)
                added_count += 1

        if added_count > 0:
            # Save the updated watch list to ticker_lists.py
            self._save_watch_list()

            if added_count == 1:
                self.status_var.set(f"Added {selected_tickers[0]} to watch list and saved")
            else:
                self.status_var.set(f"Added {added_count} tickers to watch list and saved")
        else:
            self.status_var.set("All selected tickers already in watch list")

    def _save_watch_list(self):
        """Save the watch list to ticker_lists.py"""
        if not self.watch_list:
            return

        # Use 'watch_list' as the name for the list in ticker_lists.py
        list_name = "watch_list"

        # Create Python code for the watch list
        tickers_str = ", ".join([f"\"{ticker}\"" for ticker in self.watch_list])
        new_list_code = f"\n{list_name} = [{tickers_str}]\n"

        try:
            # Read the current content of ticker_lists.py
            ticker_lists_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_lists.py")
            with open(ticker_lists_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check if watch_list already exists in the file
            watch_list_pattern = re.compile(r'\nwatch_list\s*=\s*\[.*?\]', re.DOTALL)
            match = watch_list_pattern.search(content)

            if match:
                # Replace the existing watch_list
                new_content = content[:match.start()] + new_list_code + content[match.end():]
            else:
                # Find the position of the first function definition
                function_pattern = re.compile(r'\n# Function to')
                func_match = function_pattern.search(content)

                if func_match:
                    # Insert the watch list before the function definition
                    insert_position = func_match.start()
                    new_content = content[:insert_position] + new_list_code + content[insert_position:]
                else:
                    # If no function definition found, append to the end of the file
                    new_content = content + new_list_code

            # Write the modified content back to the file
            with open(ticker_lists_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            # Update the ticker lists dictionary if it's not already there
            if list_name not in self.ticker_lists:
                self.ticker_lists[list_name] = self.watch_list
                # Update the dropdown menu
                self.ticker_list_dropdown['values'] = list(self.ticker_lists.keys())
            else:
                # Just update the existing entry
                self.ticker_lists[list_name] = self.watch_list

            logging.info(f"Saved watch list with {len(self.watch_list)} tickers to ticker_lists.py")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving watch list: {str(e)}")
            logging.error(f"Error saving watch list: {e}")

    def _get_selected_tickers(self):
        """Get selected tickers from listbox"""
        selected_indices = self.ticker_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select at least one ticker.")
            return []

        selected_tickers = []
        for i in selected_indices:
            # Extract ticker symbol (it might include a comment after a dash)
            ticker_text = self.ticker_listbox.get(i)
            ticker = ticker_text.split(' - ')[0].strip()
            selected_tickers.append(ticker)

        return selected_tickers

    def _download_data(self):
        """Download or update data for selected tickers"""
        selected_tickers = self._get_selected_tickers()
        if not selected_tickers:
            return

        # Get force download setting
        force_download = self.force_download_var.get()
        mode_text = "force downloading" if force_download else "updating"

        self.status_var.set(f"{mode_text.capitalize()} data for {len(selected_tickers)} tickers...")
        self.root.update_idletasks()

        success_count = 0
        for ticker in selected_tickers:
            try:
                data = self.manager.update_data(ticker, force_download=force_download)
                if data is not None and not data.empty:
                    success_count += 1
                    self.status_var.set(f"{mode_text.capitalize()} data for {ticker} ({success_count}/{len(selected_tickers)})")
                else:
                    self.status_var.set(f"No data available for {ticker}")
                self.root.update_idletasks()
            except Exception as e:
                messagebox.showerror("Error", f"Error {mode_text} data for {ticker}: {str(e)}")

        self.status_var.set(f"Completed: {mode_text.capitalize()} data for {success_count}/{len(selected_tickers)} tickers")

    def _visualize_daily_weekly(self):
        """Visualize daily vs weekly charts for selected tickers"""
        selected_tickers = self._get_selected_tickers()
        if not selected_tickers:
            return

        for ticker in selected_tickers:
            try:
                self.status_var.set(f"Visualizing daily vs weekly for {ticker}...")
                self.root.update_idletasks()

                self.manager.visualize_daily_vs_weekly(ticker)

                # Open the saved chart in the default web browser
                chart_path = os.path.join(self.manager.plot_save_path, f"{ticker}_daily_vs_weekly_price.png")
                if os.path.exists(chart_path):
                    webbrowser.open(f"file:///{os.path.abspath(chart_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error visualizing {ticker}: {str(e)}")

        self.status_var.set(f"Completed visualization for {len(selected_tickers)} tickers")

    def _visualize_all_timeframes(self):
        """Visualize daily, weekly, and monthly charts for selected tickers"""
        selected_tickers = self._get_selected_tickers()
        if not selected_tickers:
            return

        for ticker in selected_tickers:
            try:
                self.status_var.set(f"Visualizing all timeframes for {ticker}...")
                self.root.update_idletasks()

                # Use the existing visualize_daily_vs_weekly method which already shows daily, weekly, and monthly data
                self.manager.visualize_daily_vs_weekly(ticker)

                # Open the saved chart in the default web browser
                chart_path = os.path.join(self.manager.plot_save_path, f"{ticker}_daily_vs_weekly_price.png")
                if os.path.exists(chart_path):
                    webbrowser.open(f"file:///{os.path.abspath(chart_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error visualizing {ticker}: {str(e)}")

        self.status_var.set(f"Completed visualization for {len(selected_tickers)} tickers")

    def _generate_seasonality_chart(self, ticker):
        """Generate and display seasonality chart for the selected ticker
        
        Args:
            ticker (str): Ticker symbol to generate seasonality chart for
        """
        try:
            # Set matplotlib to use non-interactive backend for thread safety
            import matplotlib
            original_backend = matplotlib.get_backend()
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Load data for the ticker
            data = self.manager.load_data(ticker)
            if data is None or len(data) < 252:  # Need at least a year of data
                messagebox.showwarning("Insufficient Data", f"Not enough data available for {ticker} to generate a seasonality chart.")
                self.status_var.set(f"Not enough data for {ticker} seasonality chart")
                return
            
            # Apply date filters if set
            if self.manager.start_date:
                start_date = pd.Timestamp(self.manager.start_date)
                if start_date > data.index.min():
                    logging.info(f"Applying start date filter: {self.manager.start_date}, data range: {data.index.min()} to {data.index.max()}")
                    data = data[data.index >= start_date]
                    logging.info(f"After start date filter: data range: {data.index.min()} to {data.index.max()}, rows: {len(data)}")
            
            if self.manager.end_date:
                end_date = pd.Timestamp(self.manager.end_date)
                if end_date < data.index.max():
                    logging.info(f"Applying end date filter: {self.manager.end_date}, data range: {data.index.min()} to {data.index.max()}")
                    data = data[data.index <= end_date]
                    logging.info(f"After end date filter: data range: {data.index.min()} to {data.index.max()}, rows: {len(data)}")
            
            # Extract year from each date and create a new column
            data['Year'] = data.index.year
            
            # Get unique years in the data
            years = sorted(data['Year'].unique())
            
            # Store data for each year
            year_data = {}
            all_days = set()
            
            # Calculate percentage change for each year
            for year in years:
                year_df = data[data['Year'] == year].copy()
                if len(year_df) < 30:  # Skip years with insufficient data
                    continue
                    
                # Reset index to get day of year
                year_df = year_df.reset_index()
                year_df['DayOfYear'] = year_df['Date'].dt.dayofyear
                
                # Calculate percentage change from first day of the year
                first_close = year_df['Close'].iloc[0]
                year_df['PctChange'] = ((year_df['Close'] - first_close) / first_close) * 100
                
                # Store data for this year
                year_data[year] = year_df[['DayOfYear', 'PctChange']].set_index('DayOfYear')['PctChange']
                all_days.update(year_df['DayOfYear'])
            
            if not year_data:
                messagebox.showwarning("Insufficient Data", f"No complete years of data available for {ticker}.")
                self.status_var.set(f"No complete years of data for {ticker}")
                return
            
            # Update the year dropdown with available years
            year_options = ["All Years"] + [str(year) for year in sorted(year_data.keys())]
            self.year_dropdown['values'] = year_options
            
            # If the current selection is not in the list, reset to "All Years"
            if self.year_var.get() not in year_options:
                self.year_var.set("All Years")
            
            # Get the selected year from the dropdown
            selected_year = self.year_var.get()
            
            # Create a new figure
            plt.figure(figsize=(10, 6))
            
            # Generate distinct colors for each year
            colors = plt.cm.tab10.colors
            if len(year_data) > len(colors):
                colors = plt.cm.tab20.colors
            
            # Plot based on the selected year
            if selected_year == "All Years":
                # Plot all years
                for i, (year, pct_change) in enumerate(year_data.items()):
                    color_idx = i % len(colors)
                    plt.plot(pct_change.index, pct_change.values, label=f'{year}', color=colors[color_idx], alpha=0.7)
                
                # Calculate and plot the average percentage change across all years
                # Create a DataFrame with all days and all years
                avg_df = pd.DataFrame(index=sorted(all_days))
                
                # Add each year's data
                for year, pct_change in year_data.items():
                    avg_df[year] = pct_change
                
                # Calculate the average across years, ignoring NaN values
                avg_df['Average'] = avg_df.mean(axis=1)
                
                # Plot the average line with thicker line width and transparency
                plt.plot(avg_df.index, avg_df['Average'], label='Average', color='black', linewidth=2.5, alpha=0.7)
                
                chart_title = f'{ticker} Seasonality Chart - Yearly Percentage Performance'
            else:
                # Plot only the selected year
                year = int(selected_year)
                if year in year_data:
                    plt.plot(year_data[year].index, year_data[year].values, label=f'{year}', color=colors[0], linewidth=2.5)
                    chart_title = f'{ticker} Seasonality Chart - {year} Percentage Performance'
                else:
                    self.status_var.set(f"Year {year} data not available for {ticker}")
                    return
            
            # Add chart details
            plt.title(chart_title)
            plt.xlabel('Day of Year')
            plt.ylabel('Percentage Change (%)')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)  # Add horizontal line at 0%
            
            # Save the chart
            plots_dir = self.manager.plot_save_path
            os.makedirs(plots_dir, exist_ok=True)
            chart_path = os.path.join(plots_dir, f"{ticker}_seasonality_chart.png")
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Restore original backend
            matplotlib.use(original_backend)
            
            # Display the chart in the seasonality tab
            if os.path.exists(chart_path):
                # Load and resize the image
                img = Image.open(chart_path)
                
                # Get the chart frame size
                chart_width = self.seasonality_chart_container.winfo_width()
                chart_height = self.seasonality_chart_container.winfo_height()
                
                # If the frame hasn't been rendered yet, use default size
                if chart_width <= 1:
                    chart_width = 800
                if chart_height <= 1:
                    chart_height = 600
                
                # Resize image to fit the frame while maintaining aspect ratio
                img_width, img_height = img.size
                aspect_ratio = img_width / img_height
                
                if chart_width / chart_height > aspect_ratio:
                    # Frame is wider than image
                    new_height = chart_height
                    new_width = int(new_height * aspect_ratio)
                else:
                    # Frame is taller than image
                    new_width = chart_width
                    new_height = int(new_width / aspect_ratio)
                
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Convert to PhotoImage and display in seasonality tab
                photo = ImageTk.PhotoImage(img)
                self.seasonality_chart_label.config(image=photo)
                self.seasonality_chart_label.image = photo  # Keep a reference to prevent garbage collection
                
                # Store the current ticker for reference across tab changes and year selections
                self.current_chart_ticker = ticker
                
                self.status_var.set(f"Generated seasonality chart for {ticker} with {len(year_data)} years of data")
            else:
                self.status_var.set(f"Error: Seasonality chart file for {ticker} not created")
                
        except Exception as e:
            error_msg = f"Error generating seasonality chart for {ticker}: {str(e)}"
            messagebox.showerror("Error", error_msg)
            logging.error(error_msg)
            self.status_var.set(f"Error generating seasonality chart for {ticker}")
            plt.close()  # Ensure figure is closed
    
    def _on_ticker_selected(self, event):
        """Handle ticker selection from available tickers list"""
        selected_indices = self.ticker_listbox.curselection()
        if not selected_indices:
            return

        # Get the selected ticker
        ticker_text = self.ticker_listbox.get(selected_indices[0])
        ticker = ticker_text.split(' - ')[0].strip()

        # Display chart for the selected ticker
        self._display_chart(ticker)

    def _on_watch_ticker_selected(self, event):
        """Handle ticker selection from watch list"""
        selected_indices = self.watch_listbox.curselection()
        if not selected_indices:
            return

        # Get the selected ticker
        ticker = self.watch_listbox.get(selected_indices[0])

        # Display chart for the selected ticker
        self._display_chart(ticker)

    def _apply_date_range(self):
        """Apply the selected date range and refresh the current chart"""
        try:
            # Get date inputs from calendar widgets
            start_date = self.start_date_var.get().strip()
            end_date = self.end_date_var.get().strip()

            # If both dates are valid, store them in the manager
            self.manager.start_date = start_date if start_date else None
            self.manager.end_date = end_date if end_date else None

            # Check which tab is active and update the appropriate chart
            if self.active_tab == "comparison":
                # If comparison tab is active, update the comparison chart
                logging.info("Comparison tab is active, updating comparison chart with new date range")
                self._compare_percentage_performance()
            else:
                # If individual tab is active or no tab is set, update the individual chart
                # Refresh the current chart if there's a ticker selected
                selected_indices = self.ticker_listbox.curselection()
                if selected_indices:
                    ticker_text = self.ticker_listbox.get(selected_indices[0])
                    ticker = ticker_text.split(' - ')[0].strip()
                    self._display_chart(ticker)
                else:
                    # Check watch list selection
                    selected_indices = self.watch_listbox.curselection()
                    if selected_indices:
                        ticker = self.watch_listbox.get(selected_indices[0])
                        self._display_chart(ticker)
                    else:
                        self.status_var.set("Date range set. Select a ticker to display chart.")
        except Exception as e:
            messagebox.showerror("Error", f"Error applying date range: {str(e)}")
            logging.error(f"Error applying date range: {e}")

    def _on_ticker_selected(self, event):
        """Handle ticker selection event from main ticker listbox

        Args:
            event: The selection event
        """
        try:
            # Get selected ticker indices
            selected_indices = self.ticker_listbox.curselection()
            if not selected_indices:
                return

            # Get selected tickers
            selected_tickers = []
            for i in selected_indices:
                ticker_text = self.ticker_listbox.get(i)
                # Extract ticker symbol (it might have a comment after it)
                ticker = ticker_text.split(' ')[0].strip()
                selected_tickers.append(ticker)

            logging.info(f"Selected tickers from main list: {selected_tickers}")

            # Update chart based on active tab
            if hasattr(self, 'active_tab'):
                if self.active_tab == "comparison":
                    # If comparison tab is active and multiple tickers selected, update comparison chart
                    self._compare_percentage_performance()
                elif self.active_tab == "seasonality" and selected_tickers:
                    # If seasonality tab is active and a ticker is selected, update seasonality chart
                    self._generate_seasonality_chart(selected_tickers[0])
                elif self.active_tab == "individual" and selected_tickers:
                    # If individual tab is active and a ticker is selected, update individual chart
                    self._display_chart(selected_tickers[0])
            else:
                # Default to individual chart if active_tab is not set
                if selected_tickers:
                    self._display_chart(selected_tickers[0])
        except Exception as e:
            logging.error(f"Error handling ticker selection: {e}")

    def _on_watch_ticker_selected(self, event):
        """Handle ticker selection event from watch list

        Args:
            event: The selection event
        """
        try:
            # Get selected ticker indices
            selected_indices = self.watch_listbox.curselection()
            if not selected_indices:
                return

            # Get selected tickers
            selected_tickers = []
            for i in selected_indices:
                ticker = self.watch_listbox.get(i).strip()
                selected_tickers.append(ticker)

            logging.info(f"Selected tickers from watch list: {selected_tickers}")

            # Update chart based on active tab
            if hasattr(self, 'active_tab'):
                if self.active_tab == "comparison":
                    # If comparison tab is active and multiple tickers selected, update comparison chart
                    self._compare_percentage_performance()
                elif self.active_tab == "seasonality" and selected_tickers:
                    # If seasonality tab is active and a ticker is selected, update seasonality chart
                    self._generate_seasonality_chart(selected_tickers[0])
                elif self.active_tab == "individual" and selected_tickers:
                    # If individual tab is active and a ticker is selected, update individual chart
                    self._display_chart(selected_tickers[0])
            else:
                # Otherwise update individual chart for the first selected ticker
                if selected_tickers:
                    self._display_chart(selected_tickers[0])
        except Exception as e:
            logging.error(f"Error handling watch ticker selection: {e}")

    def _on_year_selected(self, event):
        """Handle year selection event from the seasonality chart year dropdown
        
        Args:
            event: The combobox selection event
        """
        try:
            # Try to get the current ticker from different sources
            current_ticker = None
            
            # First check if we have a current chart ticker (highest priority)
            if hasattr(self, 'current_chart_ticker') and self.current_chart_ticker:
                current_ticker = self.current_chart_ticker
                logging.info(f"Using current_chart_ticker: {current_ticker}")
            
            # If no current chart ticker, check main ticker list selection
            if not current_ticker:
                main_selected_indices = self.ticker_listbox.curselection()
                if main_selected_indices:
                    ticker_text = self.ticker_listbox.get(main_selected_indices[0])
                    current_ticker = ticker_text.split(' - ')[0].strip()
                    logging.info(f"Using main ticker list selection: {current_ticker}")
            
            # If still no ticker, check watch list selection
            if not current_ticker:
                watch_selected_indices = self.watch_listbox.curselection()
                if watch_selected_indices:
                    current_ticker = self.watch_listbox.get(watch_selected_indices[0]).strip()
                    logging.info(f"Using watch list selection: {current_ticker}")
            
            if not current_ticker:
                messagebox.showwarning("No Selection", "Please select a ticker first.")
                return
                
            # Regenerate the seasonality chart with the selected year
            self._generate_seasonality_chart(current_ticker)
            
            # Store this ticker as the current chart ticker
            self.current_chart_ticker = current_ticker
            
        except Exception as e:
            logging.error(f"Error handling year selection: {e}")
            messagebox.showerror("Error", f"Error updating chart: {str(e)}")
            self.status_var.set(f"Error updating chart: {str(e)}")

    def _on_tab_changed(self, event):
        """Handle tab change event

        Args:
            event: The tab change event
        """
        try:
            # Get the currently selected tab
            selected_tab = self.chart_notebook.select()

            # Update active tab tracking based on which tab is selected
            if selected_tab == str(self.individual_chart_frame):
                self.active_tab = "individual"
                logging.info("Switched to individual chart tab")
            elif selected_tab == str(self.comparison_chart_frame):
                self.active_tab = "comparison"
                logging.info("Switched to comparison chart tab")
            elif selected_tab == str(self.seasonality_chart_frame):
                self.active_tab = "seasonality"
                logging.info("Switched to seasonality chart tab")

            logging.info(f"Active tab is now: {self.active_tab}")

            # Update chart based on current selection and active tab
            selected_tickers = self._get_selected_tickers()
            if self.active_tab == "comparison" and len(selected_tickers) > 1:
                # If comparison tab is active and multiple tickers selected, update comparison chart
                self._compare_percentage_performance()
            elif self.active_tab == "seasonality":
                # If seasonality tab is active, use current_chart_ticker if available, otherwise use selection
                ticker_to_use = None
                if hasattr(self, 'current_chart_ticker') and self.current_chart_ticker:
                    ticker_to_use = self.current_chart_ticker
                elif selected_tickers:
                    ticker_to_use = selected_tickers[0]
                    
                if ticker_to_use:
                    # Update seasonality chart with the appropriate ticker
                    self._generate_seasonality_chart(ticker_to_use)
            elif self.active_tab == "individual" and selected_tickers:
                # If individual tab is active and a ticker is selected, update individual chart
                self._display_chart(selected_tickers[0])
        except Exception as e:
            logging.error(f"Error handling tab change: {e}")

    def _is_valid_date(self, date_str):
        """Check if a string is a valid date in YYYY-MM-DD format"""
        try:
            if date_str:
                datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False

    def _update_chart_after_download(self, ticker):
        """Update chart display after background download completes

        Args:
            ticker (str): Ticker symbol to update chart for
        """
        try:
            self.status_var.set(f"Generating chart for {ticker} in background...")
            self.root.update_idletasks()

            # Create a background thread for chart generation
            def generate_chart_thread():
                try:
                    # Set matplotlib to use non-interactive backend for thread safety
                    import matplotlib
                    original_backend = matplotlib.get_backend()
                    matplotlib.use('Agg')  # Use non-interactive backend in thread

                    # Generate the chart
                    self.manager.visualize_daily_vs_weekly(ticker)

                    # Restore original backend
                    matplotlib.use(original_backend)

                    # After chart generation completes, update the display
                    self.root.after(100, lambda: self._display_chart_after_generation(ticker))
                except Exception as e:
                    # Handle any exceptions in the thread
                    logging.error(f"Error generating chart for {ticker} in background thread: {e}")
                    self.root.after(0, lambda: self.status_var.set(f"Error generating chart for {ticker}: {str(e)}"))

            # Start the chart generation thread
            chart_thread = threading.Thread(target=generate_chart_thread)
            chart_thread.daemon = True
            chart_thread.start()

        except Exception as e:
            error_msg = f"Error updating chart after download for {ticker}: {str(e)}"
            self.status_var.set(error_msg)
            logging.error(error_msg)

    def _display_chart_after_generation(self, ticker):
        """Display chart after background generation completes

        Args:
            ticker (str): Ticker symbol to display chart for
        """
        try:
            self.status_var.set(f"Chart generation for {ticker} complete. Displaying...")
            self.root.update_idletasks()

            # Display the chart without regenerating it
            plots_dir = self.manager.plot_save_path
            chart_path = os.path.join(plots_dir, f"{ticker}_daily_weekly_monthly.png")

            if os.path.exists(chart_path):
                # Load and resize the image
                img = Image.open(chart_path)

                # Get the chart frame size
                chart_width = self.individual_chart_frame.winfo_width()
                chart_height = self.individual_chart_frame.winfo_height()

                # If the frame hasn't been rendered yet, use default size
                if chart_width <= 1:
                    chart_width = 800
                if chart_height <= 1:
                    chart_height = 600

                # Resize image to fit the frame while maintaining aspect ratio
                img_width, img_height = img.size
                aspect_ratio = img_width / img_height

                if chart_width / chart_height > aspect_ratio:
                    # Frame is wider than image
                    new_height = chart_height
                    new_width = int(new_height * aspect_ratio)
                else:
                    # Frame is taller than image
                    new_width = chart_width
                    new_height = int(new_width / aspect_ratio)

                img = img.resize((new_width, new_height), Image.LANCZOS)

                # Convert to PhotoImage and display
                photo = ImageTk.PhotoImage(img)
                self.chart_label.config(image=photo)
                self.chart_label.image = photo  # Keep a reference to prevent garbage collection

                # Select the individual chart tab if we're not already on another tab
                if self.active_tab != "seasonality" and self.active_tab != "comparison":
                    self.chart_notebook.select(self.individual_chart_frame)
                    self.active_tab = "individual"
                
                # Store the current ticker for reference across tab changes and year selections
                self.current_chart_ticker = ticker

                self.status_var.set(f"Displaying chart for {ticker}")
            else:
                self.status_var.set(f"Error: Chart for {ticker} not found")

        except Exception as e:
            error_msg = f"Error displaying chart after generation for {ticker}: {str(e)}"
            self.status_var.set(error_msg)
            logging.error(error_msg)

    def _display_chart(self, ticker_or_path):
        """Display chart for the selected ticker or direct path

        Args:
            ticker_or_path (str): Either a ticker symbol or a full path to an image file
        """
        # Initialize variables that might be referenced in exception handler
        is_direct_path = False
        chart_path = None
        ticker = None

        try:
            # Determine if this is a direct path to an image file or a ticker symbol
            is_direct_path = ticker_or_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))

            if is_direct_path:
                # Direct path to image file
                chart_path = ticker_or_path
                if not os.path.exists(chart_path):
                    self.status_var.set(f"Chart file not found: {os.path.basename(chart_path)}")
                    return
            else:
                # This is a ticker symbol
                ticker = ticker_or_path

                # Check if data exists for this ticker
                data_path = self.manager._get_data_path(ticker)
                if not os.path.exists(data_path):
                    # Download data if it doesn't exist
                    self.status_var.set(f"Downloading data for {ticker} in background...")
                    self.root.update_idletasks()

                    # Create a background thread for downloading data
                    def download_data_thread():
                        try:
                            self.manager.update_data(ticker, force_download=True)
                            # After download completes, update the chart
                            self.root.after(100, lambda: self._update_chart_after_download(ticker))
                        except Exception as e:
                            # Handle any exceptions in the thread
                            logging.error(f"Error downloading data for {ticker} in background thread: {e}")
                            self.root.after(0, lambda: self.status_var.set(f"Error downloading data for {ticker}: {str(e)}"))

                    # Start the download thread
                    download_thread = threading.Thread(target=download_data_thread)
                    download_thread.daemon = True
                    download_thread.start()

                    # Return early - the chart will be updated when download completes
                    return

                # Generate or update chart if needed
                plots_dir = self.manager.plot_save_path
                os.makedirs(plots_dir, exist_ok=True)

                chart_path = os.path.join(plots_dir, f"{ticker}_daily_weekly_monthly.png")
                chart_outdated = False

                # If chart doesn't exist, it needs to be generated
                if not os.path.exists(chart_path):
                    chart_outdated = True
                # If chart exists, check if data file is newer than chart file
                elif os.path.exists(data_path):
                    chart_mod_time = os.path.getmtime(chart_path)
                    data_mod_time = os.path.getmtime(data_path)

                    # If data file is newer, chart is outdated
                    if data_mod_time > chart_mod_time:
                        chart_outdated = True

                # Generate chart if needed
                if chart_outdated:
                    self.status_var.set(f"Generating chart for {ticker} in background...")
                    self.root.update_idletasks()

                    # Create a background thread for chart generation
                    def generate_chart_thread():
                        try:
                            self.manager.visualize_daily_vs_weekly(ticker)
                            # After chart generation completes, update the display
                            self.root.after(100, lambda: self._display_chart_after_generation(ticker))
                        except Exception as e:
                            # Handle any exceptions in the thread
                            logging.error(f"Error generating chart for {ticker} in background thread: {e}")
                            self.root.after(0, lambda: self.status_var.set(f"Error generating chart for {ticker}: {str(e)}"))

                    # Start the chart generation thread
                    chart_thread = threading.Thread(target=generate_chart_thread)
                    chart_thread.daemon = True
                    chart_thread.start()

                    # Return early - the chart will be displayed when generation completes
                    return

            # Display the chart in the chart_label
            if os.path.exists(chart_path):
                # Load and resize the image
                img = Image.open(chart_path)

                # Get the chart frame size
                chart_width = self.chart_frame.winfo_width()
                chart_height = self.chart_frame.winfo_height()

                # If the frame hasn't been rendered yet, use default size
                if chart_width <= 1:
                    chart_width = 800
                if chart_height <= 1:
                    chart_height = 600

                # Resize image to fit the frame while maintaining aspect ratio
                img_width, img_height = img.size
                aspect_ratio = img_width / img_height

                if chart_width / chart_height > aspect_ratio:
                    # Frame is wider than image
                    new_height = chart_height
                    new_width = int(new_height * aspect_ratio)
                else:
                    # Frame is taller than image
                    new_width = chart_width
                    new_height = int(new_width / aspect_ratio)

                img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Convert to PhotoImage and display
                photo = ImageTk.PhotoImage(img)
                self.chart_label.config(image=photo)
                self.chart_label.image = photo  # Keep a reference to prevent garbage collection

                # Only select the individual chart tab if we're not already on another tab
                if self.active_tab != "seasonality" and self.active_tab != "comparison":
                    self.chart_notebook.select(self.individual_chart_frame)
                    self.active_tab = "individual"
                
                # Store the current ticker for reference across tab changes and year selections
                if not is_direct_path and ticker:
                    self.current_chart_ticker = ticker

                # Use the appropriate name for status message
                if is_direct_path:
                    chart_name = os.path.basename(chart_path)
                    self.status_var.set(f"Displaying chart: {chart_name}")
                else:
                    self.status_var.set(f"Displaying chart for {ticker}")
            else:
                # Use the appropriate name for error message
                if is_direct_path:
                    chart_name = os.path.basename(chart_path)
                    self.status_var.set(f"Error: Chart file not found: {chart_name}")
                else:
                    self.status_var.set(f"Error: Chart for {ticker} not found")

        except Exception as e:
            # Use the appropriate name for error message
            try:
                if is_direct_path and chart_path:
                    chart_name = os.path.basename(chart_path or ticker_or_path)
                    messagebox.showerror("Error", f"Error displaying chart: {chart_name}: {str(e)}")
                    self.status_var.set(f"Error displaying chart: {chart_name}")
                else:
                    # Either it's a ticker or we couldn't determine the type
                    chart_name = ticker or ticker_or_path
                    messagebox.showerror("Error", f"Error displaying chart for {chart_name}: {str(e)}")
                    self.status_var.set(f"Error displaying chart for {chart_name}")
            except Exception:
                # Fallback error handling if anything goes wrong in the error handler
                messagebox.showerror("Error", f"Error displaying chart: {str(e)}")
                self.status_var.set("Error displaying chart")

    def _download_data_in_background(self, tickers):
        """Download data for multiple tickers in a background thread

        Args:
            tickers (list): List of ticker symbols to download data for
        """
        if not tickers:
            return

        # Create a queue to store results
        self.download_queue = Queue()

        # Create and start the download thread
        download_thread = threading.Thread(
            target=self._download_worker,
            args=(tickers, self.download_queue),
            daemon=True
        )
        download_thread.start()

        # Schedule periodic checks for download completion
        self._check_download_progress(download_thread, len(tickers))

    def _download_worker(self, tickers, queue):
        """Worker function to download data for multiple tickers

        Args:
            tickers (list): List of ticker symbols to download data for
            queue (Queue): Queue to store results
        """
        try:
            total = len(tickers)
            completed = 0

            for ticker in tickers:
                try:
                    # Update status in the queue
                    queue.put(("status", f"Downloading data for {ticker}... ({completed}/{total})"))

                    # Download data with force_download=True
                    self.manager.update_data(ticker, force_download=True)

                    # Update completed count
                    completed += 1
                    queue.put(("progress", completed))

                except Exception as e:
                    # Report error for this ticker
                    queue.put(("error", f"Error downloading {ticker}: {str(e)}"))

            # Signal completion
            queue.put(("complete", f"Downloaded data for {completed}/{total} tickers"))

        except Exception as e:
            # Report critical error
            queue.put(("critical", f"Critical error in download thread: {str(e)}"))

    def _check_download_progress(self, thread, total_tickers, check_interval=100):
        """Periodically check download progress and update the UI

        Args:
            thread (Thread): The download thread to monitor
            total_tickers (int): Total number of tickers being downloaded
            check_interval (int): How often to check progress in milliseconds
        """
        try:
            # Process all available messages from the queue
            while not self.download_queue.empty():
                msg_type, msg = self.download_queue.get_nowait()

                if msg_type == "status":
                    # Update status message
                    self.status_var.set(msg)
                elif msg_type == "progress":
                    # Update progress (could be used for a progress bar)
                    progress = int((msg / total_tickers) * 100)
                    self.status_var.set(f"Downloading... {progress}% complete ({msg}/{total_tickers})")
                elif msg_type == "error":
                    # Log error but continue
                    logging.error(msg)
                elif msg_type == "complete":
                    # Download complete
                    self.status_var.set(msg)
                elif msg_type == "critical":
                    # Critical error
                    logging.error(msg)
                    messagebox.showerror("Download Error", msg)
                    return

            # If thread is still alive, schedule another check
            if thread.is_alive():
                self.root.after(check_interval,
                                lambda: self._check_download_progress(thread, total_tickers, check_interval))
            else:
                # Thread completed, final update
                self.status_var.set(f"Download completed for {total_tickers} tickers")

        except Exception as e:
            logging.error(f"Error checking download progress: {str(e)}")

    def cleanup(self):
        """Clean up resources before application exit"""
        try:
            # First, clear any image references which often cause issues
            if hasattr(self, 'chart_label') and hasattr(self.chart_label, 'image'):
                self.chart_label.image = None

            # Clear listbox contents
            if hasattr(self, 'ticker_listbox'):
                self.ticker_listbox.delete(0, tk.END)
            if hasattr(self, 'watch_listbox'):
                self.watch_listbox.delete(0, tk.END)

            # Destroy all widgets explicitly to prevent reference cycles
            for widget in self.root.winfo_children():
                if widget.winfo_exists():
                    widget.destroy()

            # Set Tkinter variables to None instead of deleting them
            # This helps prevent 'main thread is not in main loop' errors
            if hasattr(self, 'status_var'):
                self.status_var.set('')
                self.status_var = None
            if hasattr(self, 'ticker_list_var'):
                self.ticker_list_var.set('')
                self.ticker_list_var = None
            if hasattr(self, 'force_download_var'):
                self.force_download_var.set(False)
                self.force_download_var = None
            if hasattr(self, 'manual_ticker_var'):
                self.manual_ticker_var.set('')
                self.manual_ticker_var = None
            if hasattr(self, 'list_name_var'):
                self.list_name_var.set('')
                self.list_name_var = None

            # Clear other references
            if hasattr(self, 'ticker_context_menu'):
                self.ticker_context_menu = None
            if hasattr(self, 'watch_context_menu'):
                self.watch_context_menu = None

        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            # Don't re-raise the exception as we're already in cleanup

    def _compare_percentage_performance(self):
        """Generate overlayed percentage comparison chart for selected tickers
        using the common available data range"""
        # Get selected tickers
        try:
            selected_tickers = self._get_selected_tickers()
            if not selected_tickers:
                return

            if len(selected_tickers) < 1:
                messagebox.showwarning("Insufficient Selection", "Please select at least one ticker to compare.")
                return

            # Set active tab to comparison
            self.active_tab = "comparison"
            self.chart_notebook.select(self.comparison_chart_frame)

            # If only one ticker is selected, we'll still create a percentage chart for it
            # in the comparison tab (not switching to individual chart)

            logging.info(f"Selected tickers for comparison: {selected_tickers}")
        except Exception as e:
            messagebox.showerror("Error", f"Error getting selected tickers: {str(e)}")
            logging.error(f"Error getting selected tickers: {str(e)}")
            return

        # Update status
        self.status_var.set(f"Generating percentage comparison chart for {len(selected_tickers)} tickers...")
        self.root.update_idletasks()

        # Check for missing data
        try:
            missing_tickers = []
            for ticker in selected_tickers:
                data_path = self.manager._get_data_path(ticker)
                if not os.path.exists(data_path):
                    missing_tickers.append(ticker)

            if missing_tickers:
                self.status_var.set(f"Downloading missing data for {len(missing_tickers)} tickers in background...")
                self.root.update_idletasks()
                self._download_data_in_background(missing_tickers)
                messagebox.showinfo("Download in Progress",
                                   "Some ticker data is being downloaded in the background. "
                                   "Please try generating the comparison chart again once the download completes.")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Error checking for missing data: {str(e)}")
            logging.error(f"Error checking for missing data: {str(e)}")
            return

        # Load data for all tickers
        try:
            ticker_data = {}
            for ticker_symbol in selected_tickers:
                try:
                    data = self.manager.load_data(ticker_symbol)
                    if data is not None and not data.empty:
                        # Ensure index is datetime
                        if not isinstance(data.index, pd.DatetimeIndex):
                            data.index = pd.to_datetime(data.index)
                        ticker_data[ticker_symbol] = data
                    else:
                        logging.warning(f"No data available for {ticker_symbol}. Skipping.")
                except Exception as e:
                    logging.error(f"Error loading data for {ticker_symbol}: {str(e)}. Skipping.")

            if len(ticker_data) < 1:
                messagebox.showwarning("Insufficient Data", "Need at least one ticker with valid data to generate comparison.")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Error loading ticker data: {str(e)}")
            logging.error(f"Error loading ticker data: {str(e)}")
            return

        # Find common date range
        try:
            # Get min and max dates for each ticker
            start_dates = []
            end_dates = []

            # Debug log each ticker's date range
            for ticker_symbol, df in ticker_data.items():
                ticker_start = df.index.min()
                ticker_end = df.index.max()
                logging.info(f"Ticker {ticker_symbol} date range: {ticker_start} to {ticker_end}")
                start_dates.append(ticker_start)
                end_dates.append(ticker_end)

            if not start_dates or not end_dates:
                messagebox.showwarning("Data Error", "Could not determine valid date ranges for the selected tickers.")
                return

            # Ensure all dates are timezone-naive for consistent comparison
            start_dates = [date.tz_localize(None) if hasattr(date, 'tz_localize') else date for date in start_dates]
            end_dates = [date.tz_localize(None) if hasattr(date, 'tz_localize') else date for date in end_dates]

            # Find common range (or just use the single ticker's range if only one ticker)
            if len(start_dates) == 1:
                common_start = start_dates[0]
                common_end = end_dates[0]
            else:
                common_start = max(start_dates)
                common_end = min(end_dates)

            # Apply user-specified date range if set
            if self.manager.start_date:
                user_start = pd.Timestamp(self.manager.start_date).tz_localize(None)
                logging.info(f"Applying user-specified start date: {user_start} to comparison chart")
                common_start = max(common_start, user_start)

            if self.manager.end_date:
                user_end = pd.Timestamp(self.manager.end_date).tz_localize(None)
                logging.info(f"Applying user-specified end date: {user_end} to comparison chart")
                common_end = min(common_end, user_end)

            # Ensure we have proper datetime objects
            common_start = pd.to_datetime(common_start)
            common_end = pd.to_datetime(common_end)

            if common_start >= common_end:
                messagebox.showwarning("No Common Range", "Selected tickers don't have a common date range for comparison.")
                return

            logging.info(f"Common date range: {common_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}")

            # Verify the common range is reasonable (not epoch dates)
            if common_start.year < 1980 or common_end.year < 1980:
                logging.warning(f"Suspicious date range detected: {common_start} to {common_end}")
                messagebox.showwarning("Date Range Issue",
                                     "The common date range appears to be invalid. Please try different tickers.")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Error determining common date range: {str(e)}")
            logging.error(f"Error determining common date range: {str(e)}")
            return

        # Create the plot
        try:
            # Set matplotlib to use non-interactive backend for thread safety
            import matplotlib
            original_backend = matplotlib.get_backend()
            matplotlib.use('Agg')  # Use non-interactive backend for thread safety

            plt.figure(figsize=(12, 8))

            # Plot each ticker's percentage change
            plotted_tickers = []

            for ticker_symbol, data in ticker_data.items():
                try:
                    # Filter to common date range - ensure timezone consistency
                    # Convert index to timezone-naive for comparison
                    data_index_naive = data.index.tz_localize(None) if hasattr(data.index, 'tz_localize') else data.index

                    # Create a mask for filtering with consistent timezone handling
                    mask = (data_index_naive >= common_start) & (data_index_naive <= common_end)
                    filtered_data = data.loc[mask].copy()

                    # Debug log the filtered data range
                    if not filtered_data.empty:
                        logging.info(f"Filtered data for {ticker_symbol}: {filtered_data.index.min()} to {filtered_data.index.max()}, {len(filtered_data)} rows")

                    if not filtered_data.empty:
                        # Calculate percentage change from first day
                        first_close = filtered_data['Close'].iloc[0]
                        filtered_data['pct_change'] = ((filtered_data['Close'] - first_close) / first_close) * 100

                        # Plot the percentage change
                        plt.plot(filtered_data.index, filtered_data['pct_change'], label=ticker_symbol)
                        plotted_tickers.append(ticker_symbol)
                        logging.info(f"Successfully plotted {ticker_symbol}")
                    else:
                        logging.warning(f"No data in common range for {ticker_symbol}")
                except Exception as e:
                    logging.error(f"Error plotting {ticker_symbol}: {str(e)}")

            if not plotted_tickers:
                messagebox.showwarning("Plot Error", "Could not plot any tickers. Please try different tickers.")
                plt.close()  # Close the figure to avoid memory leak
                return

            # Add chart details
            start_date_str = pd.Timestamp(common_start).strftime('%Y-%m-%d')
            end_date_str = pd.Timestamp(common_end).strftime('%Y-%m-%d')
            plt.title(f'Percentage Performance Comparison ({start_date_str} to {end_date_str})')
            plt.xlabel('Date')
            plt.ylabel('Percentage Change (%)')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            plt.gcf().autofmt_xdate()
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)  # Add horizontal line at 0%
        except Exception as e:
            messagebox.showerror("Error", f"Error creating plot: {str(e)}")
            logging.error(f"Error creating plot: {str(e)}")
            plt.close()  # Close the figure to avoid memory leak
            return

        # Save and display the chart
        try:
            # Create directory if needed
            plots_dir = self.manager.plot_save_path
            os.makedirs(plots_dir, exist_ok=True)

            # Create filename from plotted tickers
            tickers_str = '_'.join(plotted_tickers)
            if len(tickers_str) > 50:  # Avoid excessively long filenames
                tickers_str = f"{len(plotted_tickers)}_tickers_comparison"

            # Save the chart
            chart_path = os.path.join(plots_dir, f"pct_comparison_{tickers_str}.png")
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()  # Close the figure to free memory

            # Restore original backend
            matplotlib.use(original_backend)

            # Display the chart in the comparison tab
            if os.path.exists(chart_path):
                # Load and resize the image
                img = Image.open(chart_path)

                # Get the chart frame size
                chart_width = self.comparison_chart_frame.winfo_width()
                chart_height = self.comparison_chart_frame.winfo_height()

                # If the frame hasn't been rendered yet, use default size
                if chart_width <= 1:
                    chart_width = 800
                if chart_height <= 1:
                    chart_height = 600

                # Resize image to fit the frame while maintaining aspect ratio
                img_width, img_height = img.size
                aspect_ratio = img_width / img_height

                if chart_width / chart_height > aspect_ratio:
                    # Frame is wider than image
                    new_height = chart_height
                    new_width = int(new_height * aspect_ratio)
                else:
                    # Frame is taller than image
                    new_width = chart_width
                    new_height = int(new_width / aspect_ratio)

                img = img.resize((new_width, new_height), Image.LANCZOS)

                # Convert to PhotoImage and display in comparison tab
                photo = ImageTk.PhotoImage(img)
                self.comparison_chart_label.config(image=photo)
                self.comparison_chart_label.image = photo  # Keep a reference to prevent garbage collection

                # Select the comparison chart tab
                self.chart_notebook.select(self.comparison_chart_frame)
                # Update active tab tracking
                self.active_tab = "comparison"

                # Initialize current chart ticker tracking
                self.current_chart_ticker = None

                self.status_var.set(f"Generated percentage comparison chart for {len(plotted_tickers)} tickers")
            else:
                self.status_var.set("Error: Chart file not created")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving or displaying chart: {str(e)}")
            logging.error(f"Error saving or displaying chart: {str(e)}")
            plt.close()  # Ensure figure is closed
            self.status_var.set("Error generating comparison chart")

    def _view_html_report(self):
        """Generate and view HTML report for the current ticker list"""
        selected_tickers = self._get_selected_tickers()
        if not selected_tickers:
            messagebox.showwarning("No Tickers Selected", "Please select at least one ticker from the list.")
            return

        try:
            # Create plots directory if it doesn't exist
            plots_dir = self.manager.plot_save_path
            os.makedirs(plots_dir, exist_ok=True)

            # Check for missing data and download automatically
            missing_tickers = []
            for ticker in selected_tickers:
                data_path = self.manager._get_data_path(ticker)
                if not os.path.exists(data_path):
                    missing_tickers.append(ticker)

            # If there are missing tickers, download their data in background
            if missing_tickers:
                self.status_var.set(f"Downloading missing data for {len(missing_tickers)} tickers in background...")
                self.root.update_idletasks()
                self._download_data_in_background(missing_tickers)

            # Check for missing or outdated visualizations and generate them
            for ticker in selected_tickers:
                timeframe_plot_path = os.path.join(plots_dir, f"{ticker}_daily_weekly_monthly.png")
                data_path = self.manager._get_data_path(ticker)

                # Check if chart needs to be generated or updated
                chart_outdated = False

                # If chart doesn't exist, it needs to be generated
                if not os.path.exists(timeframe_plot_path):
                    chart_outdated = True
                # If chart exists, check if data file is newer than chart file
                elif os.path.exists(data_path):
                    chart_mod_time = os.path.getmtime(timeframe_plot_path)
                    data_mod_time = os.path.getmtime(data_path)

                    # If data file is newer, chart is outdated
                    if data_mod_time > chart_mod_time:
                        chart_outdated = True
                        self.status_var.set(f"Chart for {ticker} is outdated. Regenerating...")
                        self.root.update_idletasks()

                # Generate chart if needed
                if chart_outdated:
                    self.status_var.set(f"Generating visualizations for {ticker}...")
                    self.root.update_idletasks()
                    self.manager.visualize_daily_vs_weekly(ticker)

            # Get the current ticker list name
            current_list_name = self.ticker_list_var.get() or "custom_list"

            # Get current date in YYYY-MM-DD format
            current_date = datetime.now().strftime("%Y-%m-%d")

            # Create a filename with list name and date
            report_filename = f"stock_analysis_{current_list_name}_{current_date}.html"

            # Generate HTML report with custom filename and selected tickers
            self.status_var.set(f"Generating HTML report for {current_list_name}...")
            self.root.update_idletasks()
            report_path = self.manager.generate_html_report(plots_dir, report_filename, selected_tickers)

            # Open the HTML report in Microsoft Edge
            if os.path.exists(report_path):
                try:
                    # Register and use Microsoft Edge
                    webbrowser.register('edge', None, webbrowser.BackgroundBrowser(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'))
                    webbrowser.get('edge').open(f"file:///{os.path.abspath(report_path)}")
                    self.status_var.set(f"HTML report for {current_list_name} opened in Edge browser")
                except Exception as browser_error:
                    # Fall back to default browser if Edge registration fails
                    logging.warning(f"Could not open Edge browser: {browser_error}. Using default browser.")
                    webbrowser.open(f"file:///{os.path.abspath(report_path)}")
                    self.status_var.set(f"HTML report for {current_list_name} opened in default browser")
            else:
                self.status_var.set("Error: HTML report not found")

        except Exception as e:
            messagebox.showerror("Error", f"Error generating HTML report: {str(e)}")
            self.status_var.set("Error generating HTML report")

    def _get_selected_tickers(self):
        """Get selected tickers from listbox"""
        selected_indices = self.ticker_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select at least one ticker.")
            return []

        selected_tickers = []
        for i in selected_indices:
            # Extract ticker symbol (it might include a comment after a dash)
            ticker_text = self.ticker_listbox.get(i)
            ticker = ticker_text.split(' - ')[0].strip()
            selected_tickers.append(ticker)

        return selected_tickers

    def _download_data(self):
        """Download or update data for selected tickers"""
        selected_tickers = self._get_selected_tickers()
        if not selected_tickers:
            return

        # Get force download setting
        force_download = self.force_download_var.get()
        mode_text = "force downloading" if force_download else "updating"

        self.status_var.set(f"{mode_text.capitalize()} data for {len(selected_tickers)} tickers...")
        self.root.update_idletasks()

        success_count = 0
        for ticker in selected_tickers:
            try:
                data = self.manager.update_data(ticker, force_download=force_download)
                if data is not None and not data.empty:
                    success_count += 1
                    self.status_var.set(f"{mode_text.capitalize()} data for {ticker} ({success_count}/{len(selected_tickers)})")
                else:
                    self.status_var.set(f"No data available for {ticker}")
                self.root.update_idletasks()
            except Exception as e:
                messagebox.showerror("Error", f"Error {mode_text} data for {ticker}: {str(e)}")

        self.status_var.set(f"Completed: {mode_text.capitalize()} data for {success_count}/{len(selected_tickers)} tickers")
