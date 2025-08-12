import os
import sys
import io
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

# Plotly imports for interactive charts
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.offline as pyo
from plotly.offline import plot
import webbrowser
import tempfile

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
        self.seasonality_pil_img = None  # To store the high-res seasonality chart
        self._debounce_job = None       # For debouncing resize events
        self.year_selection_vars = {}  # For multi-select year checkbuttons
        self.seasonality_year_menubutton = None # The new menubutton for year selection
        self.year_menu = None # A direct reference to the year selection menu

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

    def _display_plotly_chart(self, fig, tab="individual"):
        """Display a Plotly chart in the specified tab
        
        Args:
            fig: Plotly figure to display
            tab: Tab to display the chart in ("individual", "comparison", or "seasonality")
        """
        try:
            # Check if root window still exists before proceeding
            if not hasattr(self, 'root') or not self.root.winfo_exists():
                logging.warning(f"Cannot display Plotly chart: root window no longer exists")
                return
                
            # Create a temporary HTML file to display the chart
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, f"stock_chart_{tab}.html")
            
            # Configure the figure for better display
            fig.update_layout(
                autosize=True,
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(rangeslider=dict(visible=False)),  # Hide default range slider
                template="plotly_white"
            )
            
            # Add range selector buttons for time periods
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            
            # Save the figure to HTML with full HTML header for browser display
            plot(fig, filename=html_path, auto_open=False)
            
            # Create a frame to embed the HTML - check if widgets exist first
            if tab == "individual":
                if not hasattr(self, 'individual_chart_frame') or not self.individual_chart_frame.winfo_exists():
                    logging.warning("Cannot update individual chart frame: widget no longer exists")
                    return
                    
                try:
                    for widget in self.individual_chart_frame.winfo_children():
                        widget.destroy()
                        
                    # Create a button to open the chart in browser for better interaction
                    btn_frame = ttk.Frame(self.individual_chart_frame)
                    btn_frame.pack(fill=tk.X, pady=5)
                    ttk.Button(btn_frame, text="Open in Browser", 
                              command=lambda: webbrowser.open(f"file:///{html_path}")).pack()
                    
                    # Create a label to show that interactive chart is available
                    ttk.Label(self.individual_chart_frame, 
                             text=f"Interactive chart for {self.current_chart_ticker if hasattr(self, 'current_chart_ticker') else ''}\nUse mouse to zoom/pan").pack()
                except tk.TclError as e:
                    logging.error(f"TclError updating individual chart frame: {str(e)}")
                    return
                
            elif tab == "comparison":
                if not hasattr(self, 'comparison_chart_frame') or not self.comparison_chart_frame.winfo_exists():
                    logging.warning("Cannot update comparison chart frame: widget no longer exists")
                    return
                    
                try:
                    for widget in self.comparison_chart_frame.winfo_children():
                        widget.destroy()
                        
                    # Create a button to open the chart in browser for better interaction
                    btn_frame = ttk.Frame(self.comparison_chart_frame)
                    btn_frame.pack(fill=tk.X, pady=5)
                    ttk.Button(btn_frame, text="Open in Browser", 
                              command=lambda: webbrowser.open(f"file:///{html_path}")).pack()
                    
                    # Create a label to show that interactive chart is available
                    ttk.Label(self.comparison_chart_frame, 
                             text="Interactive comparison chart\nUse mouse to zoom/pan").pack()
                except tk.TclError as e:
                    logging.error(f"TclError updating comparison chart frame: {str(e)}")
                    return
                
            # The 'seasonality' case is now handled entirely within _generate_seasonality_chart
                         
            # Update status if status_var exists
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Displayed interactive {tab} chart")
                
            # Force update of the UI to ensure buttons remain visible
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.update_idletasks()
            
        except Exception as e:
            logging.error(f"Error displaying Plotly chart: {e}")
            messagebox.showerror("Error", f"Failed to display interactive chart: {e}")
    
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

        # Create a frame for the second row with Add Ticker and New List Name
        second_row_frame = ttk.Frame(top_frame)
        second_row_frame.grid(row=1, column=0, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Add manual ticker entry (left side of second row)
        ticker_frame = ttk.Frame(second_row_frame)
        ticker_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(ticker_frame, text="Add Ticker:").pack(side=tk.LEFT, padx=(0, 5))
        self.manual_ticker_var = tk.StringVar()
        manual_ticker_entry = ttk.Entry(ticker_frame, textvariable=self.manual_ticker_var, width=30)
        manual_ticker_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(ticker_frame, text="Add", command=self._add_manual_ticker).pack(side=tk.LEFT)
        
        # Add separator
        ttk.Separator(second_row_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Add list name entry and save button (right side of second row)
        list_frame = ttk.Frame(second_row_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(list_frame, text="New List Name:").pack(side=tk.LEFT, padx=(0, 5))
        self.list_name_var = tk.StringVar()
        list_name_entry = ttk.Entry(list_frame, textvariable=self.list_name_var, width=30)
        list_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(list_frame, text="Save List", command=self._save_ticker_list).pack(side=tk.LEFT)

        # Create a PanedWindow for resizable sections
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=5)

        # --- Left Pane: Available Tickers ---
        left_pane_frame = ttk.Frame(paned_window)
        paned_window.add(left_pane_frame, weight=1) # Give less weight initially

        left_frame = ttk.LabelFrame(left_pane_frame, text="Available Tickers", padding="5")
        left_frame.pack(fill=tk.BOTH, expand=True)

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

        self.ticker_listbox = tk.Listbox(ticker_frame, selectmode=tk.EXTENDED, height=20, width=10)
        self.ticker_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.ticker_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.ticker_listbox.yview)

        # --- Middle Pane: Watch List ---
        middle_pane_frame = ttk.Frame(paned_window)
        paned_window.add(middle_pane_frame, weight=1) # Give less weight initially

        middle_list_frame = ttk.LabelFrame(middle_pane_frame, text="Watch List", padding="5")
        middle_list_frame.pack(fill=tk.BOTH, expand=True)

        # Create watch list listbox with scrollbar
        watch_frame = ttk.Frame(middle_list_frame)
        watch_frame.pack(fill=tk.BOTH, expand=True)

        watch_scrollbar = ttk.Scrollbar(watch_frame)
        watch_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.watch_listbox = tk.Listbox(watch_frame, selectmode=tk.EXTENDED, height=20, width=10)
        self.watch_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Right Pane: Chart Display ---
        right_pane_frame = ttk.Frame(paned_window)
        paned_window.add(right_pane_frame, weight=6) # Give more weight to the chart

        self.chart_frame = ttk.LabelFrame(right_pane_frame, text="Chart Display", padding="5")
        self.chart_frame.pack(fill=tk.BOTH, expand=True)

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
        
        # Create year selection dropdown using a Menubutton for multi-select
        ttk.Label(self.seasonality_controls_frame, text="Select Years:").pack(side=tk.LEFT, padx=(0, 5))
        self.seasonality_year_menubutton = ttk.Menubutton(self.seasonality_controls_frame, text="Select Years")
        self.seasonality_year_menubutton.pack(side=tk.LEFT)

        # The menu itself will be created and populated dynamically in _generate_seasonality_chart
        self.year_menu = tk.Menu(self.seasonality_year_menubutton, tearoff=0)
        self.seasonality_year_menubutton.config(menu=self.year_menu)

        # Bind tab change event
        self.chart_notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Create labels to display charts in each tab
        self.chart_label = ttk.Label(self.individual_chart_frame)
        self.chart_label.pack(fill=tk.BOTH, expand=True)

        self.comparison_chart_label = ttk.Label(self.comparison_chart_frame)
        self.comparison_chart_label.pack(fill=tk.BOTH, expand=True)
        
        # Create a container for the seasonality chart display
        self.seasonality_chart_container = ttk.Frame(self.seasonality_chart_frame)
        self.seasonality_chart_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.seasonality_chart_container.bind("<Configure>", self._on_seasonality_resize)

        # --- Create persistent widgets for the seasonality chart view ---
        self.seasonality_header_frame = ttk.Frame(self.seasonality_chart_container)
        self.seasonality_header_frame.pack(fill=tk.X)

        self.seasonality_title_label = ttk.Label(self.seasonality_header_frame, text="Select a ticker", font=("Helvetica", 10, "bold"))
        self.seasonality_title_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.seasonality_browser_button = ttk.Button(self.seasonality_header_frame, text="Open in Browser", state="disabled")
        self.seasonality_browser_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.seasonality_img_label = ttk.Label(self.seasonality_chart_container, text="Chart will be displayed here.")
        self.seasonality_img_label.pack(fill=tk.BOTH, expand=True, pady=5)

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

    def _generate_seasonality_chart(self, ticker, is_new_ticker=False):
        """
        Generate and display interactive seasonality chart with multi-year selection.
        
        Args:
            ticker (str): Ticker symbol to generate seasonality chart for.
            is_new_ticker (bool): Flag to indicate if this is the first load for a new ticker.
        """
        try:
            if not hasattr(self, 'root') or not self.root.winfo_exists():
                return

            self.status_var.set(f"Generating seasonality chart for {ticker}...")
            self.root.update_idletasks()
            
            data = self.manager.load_data(ticker)
            if data is None or len(data) < 252:
                messagebox.showwarning("Insufficient Data", f"Not enough data for {ticker} seasonality.")
                return
            
            data['Year'] = data.index.year
            all_available_years = sorted([int(y) for y in data['Year'].unique()])

            # --- Update Year Selection Menu ---
            if is_new_ticker:
                self.year_selection_vars.clear()
                for year in all_available_years:
                    self.year_selection_vars[year] = tk.BooleanVar()
                # Default to last 5 years
                self._select_last_five_years(all_available_years)

            self._update_year_selection_menu(all_available_years)
            
            # --- Get Selected Years ---
            selected_years = [year for year, var in self.year_selection_vars.items() if var.get()]
            if not selected_years:
                messagebox.showwarning("No Years Selected", "Please select at least one year to display.")
                self.seasonality_img_label.config(image="", text="No years selected.")
                self.seasonality_pil_img = None
                return

            # --- Process Data for Selected Years ---
            year_data = {}
            all_trading_days = set()
            for year in selected_years:
                year_df = data[data['Year'] == year].copy()
                if len(year_df) < 30: continue
                
                year_df = year_df.sort_values('Date').reset_index()
                year_df['TradingDayNum'] = range(1, len(year_df) + 1)
                first_close = float(year_df['Close'].iloc[0])
                year_df['PctChange'] = ((year_df['Close'] - first_close) / first_close) * 100
                
                year_data[year] = year_df[['TradingDayNum', 'PctChange', 'Date']].set_index('TradingDayNum')
                all_trading_days.update(year_df['TradingDayNum'])

            if not year_data:
                messagebox.showwarning("Insufficient Data", f"No valid years with enough data for {ticker}.")
                return
                
            # --- Create Plotly Figure ---
            fig = go.Figure()
            plotly_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            for i, (year, year_df) in enumerate(year_data.items()):
                hover_text = [f"Date: {date.strftime('%Y-%m-%d')}<br>Change: {pct:.2f}%" for date, pct in zip(year_df['Date'], year_df['PctChange'])]
                fig.add_trace(go.Scatter(x=year_df.index, y=year_df['PctChange'], mode='lines', name=f'{year}',
                                         line=dict(color=plotly_colors[i % len(plotly_colors)], width=1.5), opacity=0.7,
                                         hovertext=hover_text, hoverinfo='text'))

            # --- Calculate and Plot Average ---
            if len(selected_years) > 1:
                avg_df = pd.DataFrame(index=sorted(all_trading_days))
                for year, year_df in year_data.items():
                    avg_df[year] = year_df['PctChange']
                avg_df['Average'] = avg_df.mean(axis=1)
                if len(avg_df) > 5:
                    avg_df['Average'] = avg_df['Average'].rolling(window=3, min_periods=1, center=True).mean()
                fig.add_trace(go.Scatter(x=avg_df.index, y=avg_df['Average'], mode='lines', name='Average', line=dict(color='black', width=3), opacity=0.8))

            # --- Finalize and Display Figure ---
            fig.add_shape(type="line", x0=min(all_trading_days), y0=0, x1=max(all_trading_days), y1=0, line=dict(color="black", width=1, dash="dash"))
            fig.update_layout(title=f'{ticker} Seasonality - Selected Years', xaxis_title='Trading Day Number', yaxis_title='Percentage Change (%)',
                              height=600, hovermode='closest', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                              margin=dict(l=50, r=50, t=80, b=50))
            
            self.current_chart_ticker = ticker
            self._display_seasonality_figure(fig)

        except Exception as e:
            logging.error(f"Error in _generate_seasonality_chart: {e}")
            messagebox.showerror("Error", f"Could not generate seasonality chart: {e}")

    def _update_year_selection_menu(self, available_years):
        """Dynamically populates the year selection menu."""
        menu = self.year_menu
        if not menu:
            logging.error("Year menu has not been initialized.")
            return
        menu.delete(0, tk.END)

        # Chain commands to select years and then trigger the update
        menu.add_command(label="Select Last 5 Years", command=lambda: (self._select_last_five_years(available_years), self._on_year_selection_change()))
        menu.add_command(label="Select All", command=lambda: (self._select_all_years(available_years), self._on_year_selection_change()))
        menu.add_command(label="Deselect All", command=lambda: (self._deselect_all_years(available_years), self._on_year_selection_change()))
        menu.add_separator()

        for year in available_years:
            menu.add_checkbutton(label=str(year), variable=self.year_selection_vars[year],
                                 command=self._on_year_selection_change)

    def _select_all_years(self, years):
        for year in years:
            if year in self.year_selection_vars: self.year_selection_vars[year].set(True)

    def _select_last_five_years(self, years):
        for year in years:
            if year in self.year_selection_vars: self.year_selection_vars[year].set(False)
        last_five_years = sorted(years, reverse=True)[:5]
        for year in last_five_years:
            if year in self.year_selection_vars: self.year_selection_vars[year].set(True)

    def _deselect_all_years(self, years):
        for year in years:
            if year in self.year_selection_vars: self.year_selection_vars[year].set(False)

    def _on_year_selection_change(self):
        """Handles the change in year selection and regenerates the chart."""
        if self.current_chart_ticker:
            self._generate_seasonality_chart(self.current_chart_ticker)
            
    def _display_seasonality_figure(self, fig):
        """Handles the rendering of the Plotly figure to an image and displaying it."""
        try:
            container = self.seasonality_chart_container
            if not container.winfo_exists():
                logging.warning("Cannot update seasonality chart container: widget no longer exists")
                return

            # Save to HTML for the 'Open in Browser' button
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, "stock_chart_seasonality.html")
            plot(fig, filename=html_path, auto_open=False)

            self.seasonality_browser_button.config(command=lambda: webbrowser.open(f"file:///{html_path}"), state="normal")

            # Generate and store high-res image
            img_bytes = pio.to_image(fig, format='png', width=1200, height=600)
            self.seasonality_pil_img = Image.open(io.BytesIO(img_bytes))
            
            # Trigger resize to fit current window
            class FakeEvent:
                def __init__(self, w, h): self.width = w; self.height = h
            self.root.update_idletasks()
            self._on_seasonality_resize(FakeEvent(container.winfo_width(), container.winfo_height()))
            
            self.status_var.set(f"Generated seasonality chart for {self.current_chart_ticker}")
        except Exception as img_e:
            logging.warning(f"Could not generate static image for seasonality chart: {img_e}")
            self.seasonality_pil_img = None
            self.seasonality_img_label.config(image="", text="Chart preview not available.\n(Requires 'kaleido' package).")
            messagebox.showwarning("Preview Generation Failed", "Please ensure 'kaleido' is installed.")

    def _on_seasonality_resize(self, event):
        """Debounce and handle the resize event for the seasonality chart."""
        if self._debounce_job:
            self.root.after_cancel(self._debounce_job)
        self._debounce_job = self.root.after(200, lambda: self._resize_seasonality_image(event))

    def _resize_seasonality_image(self, event):
        """Resize the seasonality PIL image to fit the container."""
        if not self.seasonality_pil_img:
            return

        # Get container dimensions, with some padding
        container_width = event.width - 10
        container_height = event.height - 10

        if container_width <= 1 or container_height <= 1:
            return

        try:
            # Create a copy to resize
            img_copy = self.seasonality_pil_img.copy()

            # Use thumbnail to resize while maintaining aspect ratio
            img_copy.thumbnail((container_width, container_height), Image.LANCZOS)

            # Update the label with the new image
            photo_img = ImageTk.PhotoImage(img_copy)
            self.seasonality_img_label.config(image=photo_img)
            self.seasonality_img_label.image = photo_img
        except Exception as e:
            logging.error(f"Error resizing seasonality image: {e}")

    def _on_ticker_selected(self, event):
        """Handle ticker selection from available tickers list"""
        # ... (rest of the code remains the same)
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
                    self._generate_seasonality_chart(selected_tickers[0], is_new_ticker=True)
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
                    self._generate_seasonality_chart(selected_tickers[0], is_new_ticker=True)
                elif self.active_tab == "individual" and selected_tickers:
                    # If individual tab is active and a ticker is selected, update individual chart
                    self._display_chart(selected_tickers[0])
            else:
                # Otherwise update individual chart for the first selected ticker
                if selected_tickers:
                    self._display_chart(selected_tickers[0])
        except Exception as e:
            logging.error(f"Error handling watch ticker selection: {e}")

    def _on_tab_changed(self, event):
        """Handle tab change event

        Args:
            event: The tab change event
        """
        try:
            # Check if widgets still exist before proceeding
            if not hasattr(self, 'chart_notebook') or not self.chart_notebook.winfo_exists():
                logging.warning("Cannot handle tab change: chart notebook widget no longer exists")
                return
                
            try:
                # Get the currently selected tab
                selected_tab = self.chart_notebook.select()
                
                # Check if frame widgets exist before comparing
                if hasattr(self, 'individual_chart_frame') and self.individual_chart_frame.winfo_exists() and \
                   selected_tab == str(self.individual_chart_frame):
                    self.active_tab = "individual"
                    logging.info("Switched to individual chart tab")
                elif hasattr(self, 'comparison_chart_frame') and self.comparison_chart_frame.winfo_exists() and \
                     selected_tab == str(self.comparison_chart_frame):
                    self.active_tab = "comparison"
                    logging.info("Switched to comparison chart tab")
                elif hasattr(self, 'seasonality_chart_frame') and self.seasonality_chart_frame.winfo_exists() and \
                     selected_tab == str(self.seasonality_chart_frame):
                    self.active_tab = "seasonality"
                    logging.info("Switched to seasonality chart tab")
            except tk.TclError as e:
                logging.error(f"TclError in tab change handler: {str(e)}")
                return

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
            
    def _compare_percentage_performance(self):
        """Generate and display an interactive comparison chart showing percentage performance of multiple stocks"""
        try:
            # Get selected tickers
            selected_tickers = self._get_selected_tickers()
            
            if not selected_tickers or len(selected_tickers) < 2:
                self.status_var.set("Please select at least two tickers for comparison")
                return
                
            self.status_var.set(f"Generating comparison chart for {', '.join(selected_tickers)}...")
            self.root.update_idletasks()
            
            # Create a Plotly figure for the comparison chart
            fig = go.Figure()
            
            # Apply date range filter if specified
            start_date = self.start_date_entry.get()
            end_date = self.end_date_entry.get()
            
            # Process each ticker
            for ticker in selected_tickers:
                try:
                    # Get data for this ticker
                    df = self.manager.get_data(ticker)
                    
                    if df is None or df.empty:
                        logging.warning(f"No data available for {ticker}")
                        continue
                        
                    # Apply date filters if specified
                    if self._is_valid_date(start_date):
                        # Convert to datetime and normalize for consistent comparison
                        start_date_obj = pd.to_datetime(start_date)
                        start_date_norm = pd.Timestamp(start_date_obj.date())
                        logging.info(f"Comparison chart: Filtering by start date: {start_date_norm}")
                        
                        # Ensure index is DatetimeIndex
                        if not isinstance(df.index, pd.DatetimeIndex):
                            df.index = pd.to_datetime(df.index)
                            
                        # Use normalized dates for comparison
                        mask = df.index.normalize() >= start_date_norm
                        df = df[mask]
                        logging.info(f"Comparison chart: DataFrame shape after start date filter: {df.shape}")
                        
                    if self._is_valid_date(end_date):
                        # Convert to datetime and normalize for consistent comparison
                        end_date_obj = pd.to_datetime(end_date)
                        # Add one day to include the end date in the results
                        end_date_norm = pd.Timestamp(end_date_obj.date()) + pd.Timedelta(days=1)
                        logging.info(f"Comparison chart: Filtering by end date: {end_date_norm}")
                        
                        # Ensure index is DatetimeIndex
                        if not isinstance(df.index, pd.DatetimeIndex):
                            df.index = pd.to_datetime(df.index)
                            
                        # Use normalized dates for comparison
                        mask = df.index.normalize() < end_date_norm
                        df = df[mask]
                        logging.info(f"Comparison chart: DataFrame shape after end date filter: {df.shape}")
                        
                    if df.empty:
                        logging.warning(f"No data available for {ticker} in the specified date range")
                        continue
                        
                    # Ensure numeric types
                    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                    
                    # Calculate percentage change from first day
                    first_close = df['Close'].iloc[0]
                    df['pct_change'] = ((df['Close'] - first_close) / first_close) * 100
                    
                    # Add trace to the figure
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['pct_change'],
                        mode='lines',
                        name=ticker
                    ))
                    
                except Exception as e:
                    logging.error(f"Error processing {ticker} for comparison chart: {str(e)}")
                    
            # Update layout with interactive features
            fig.update_layout(
                title='Percentage Performance Comparison',
                xaxis_title='Date',
                yaxis_title='Percentage Change (%)',
                height=600,
                legend_title='Tickers',
                hovermode='x unified'
            )
            
            # Add range slider and selector
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            
            # Display the interactive chart
            self._display_plotly_chart(fig, tab="comparison")
            self.status_var.set(f"Generated interactive comparison chart for {len(selected_tickers)} tickers")
            
        except Exception as e:
            error_msg = f"Error generating comparison chart: {str(e)}"
            self.status_var.set(error_msg)
            logging.error(error_msg)

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
            # Verify that the root window still exists and is not being destroyed
            if not self.root.winfo_exists():
                logging.warning(f"Cannot update chart for {ticker}: root window no longer exists")
                return
                
            self.status_var.set(f"Generating chart for {ticker} in background...")
            self.root.update_idletasks()

            # Create a thread to generate the chart
            chart_thread = threading.Thread(
                target=self._generate_chart_thread,
                args=(ticker,),
                daemon=True
            )
            chart_thread.start()
            
        except tk.TclError as e:
            logging.error(f"TclError updating chart for {ticker}: {str(e)}")
        except Exception as e:
            error_msg = f"Error updating chart after download for {ticker}: {str(e)}"
            self.status_var.set(error_msg)
            logging.error(error_msg)
            
    def _generate_chart_thread(self, ticker):
        """Generate chart in a background thread and display it when done
        
        Args:
            ticker (str): Ticker symbol to generate chart for
        """
        try:
            # Set matplotlib to use non-interactive backend for thread safety
            import matplotlib
            original_backend = matplotlib.get_backend()
            matplotlib.use('Agg')  # Use non-interactive backend in thread

            # Generate the chart
            self.manager.visualize_daily_vs_weekly(ticker)

            # Restore original backend
            matplotlib.use(original_backend)
            
            # Schedule display of the chart in the main thread, but first check if root still exists
            def safe_display_chart():
                try:
                    # Check if root window still exists before updating UI
                    if hasattr(self, 'root') and self.root.winfo_exists():
                        self._display_chart(ticker)
                    else:
                        logging.warning(f"Cannot display chart for {ticker}: root window no longer exists")
                except tk.TclError as e:
                    logging.error(f"TclError displaying chart for {ticker}: {str(e)}")
                except Exception as e:
                    logging.error(f"Error displaying chart for {ticker}: {str(e)}")
            
            self.root.after(100, safe_display_chart)
            
        except Exception as e:
            # Handle any exceptions in the thread
            logging.error(f"Error generating chart for {ticker} in background thread: {e}")
            
            # Safely update status if root still exists
            def safe_update_status():
                try:
                    if hasattr(self, 'root') and self.root.winfo_exists() and hasattr(self, 'status_var'):
                        self.status_var.set(f"Error generating chart for {ticker}: {str(e)}")
                except Exception as inner_e:
                    logging.error(f"Error updating status: {str(inner_e)}")
                    
            self.root.after(0, safe_update_status)
            
    def _display_static_chart(self, image_path):
        """Display a static chart image in the appropriate chart container
        
        Args:
            image_path (str): Path to the image file to display
        """
        try:
            # Clear the current chart container
            for widget in self.chart_frame.winfo_children():
                widget.destroy()
                
            # Load the image
            img = Image.open(image_path)
            
            # Resize image to fit the chart frame while maintaining aspect ratio
            frame_width = self.chart_frame.winfo_width() or 800
            frame_height = self.chart_frame.winfo_height() or 600
            
            # Calculate resize dimensions while maintaining aspect ratio
            img_width, img_height = img.size
            ratio = min(frame_width/img_width, frame_height/img_height)
            new_width = int(img_width * ratio * 0.9)  # 90% of available space
            new_height = int(img_height * ratio * 0.9)  # 90% of available space
            
            # Resize the image
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage for Tkinter
            photo = ImageTk.PhotoImage(img)
            
            # Create a label to display the image
            img_label = tk.Label(self.chart_frame, image=photo)
            img_label.image = photo  # Keep a reference to prevent garbage collection
            img_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Update the status
            self.status_var.set(f"Displayed chart from {os.path.basename(image_path)}")
            
        except Exception as e:
            error_msg = f"Error displaying static chart: {str(e)}"
            self.status_var.set(error_msg)
            logging.error(error_msg)
    def _display_chart(self, ticker_or_path):
        """Display interactive chart for the selected ticker or direct path

        Args:
            ticker_or_path (str): Either a ticker symbol or a full path to an image file
        """
        try:
            # First check if the root window and widgets still exist
            if not hasattr(self, 'root') or not self.root.winfo_exists():
                logging.warning(f"Cannot display chart for {ticker_or_path}: root window no longer exists")
                return
                
            # Determine if input is a ticker or a path
            if os.path.exists(ticker_or_path):
                # It's a path to an image file, display it directly
                self._display_static_chart(ticker_or_path)
                # Extract ticker from filename if possible
                ticker = os.path.basename(ticker_or_path).split('_')[0]
                self.current_chart_ticker = ticker
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Displayed chart for {ticker}")
                return
            
            # It's a ticker symbol, generate interactive chart
            ticker = ticker_or_path
            self.current_chart_ticker = ticker
            
            # Check if chart_notebook widget exists before trying to access it
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
            
            # For individual chart, create interactive Plotly chart
            try:
                # Get stock data
                df = self.manager.get_data(ticker)
                if df is None or df.empty:
                    self.status_var.set(f"No data available for {ticker}")
                    return
                
                # Apply date range filter if specified
                start_date = self.start_date_entry.get()
                end_date = self.end_date_entry.get()
                if self._is_valid_date(start_date):
                    df = df[df.index >= start_date]
                if self._is_valid_date(end_date):
                    df = df[df.index <= end_date]
                
                # Ensure numeric types for calculations
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Create interactive Plotly figure with subplots
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.1, 
                                   row_heights=[0.7, 0.3])
                
                # Add price candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name=ticker
                    ),
                    row=1, col=1
                )
                
                # Add volume bar chart
                fig.add_trace(
                    go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='rgba(0,0,255,0.5)'),
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
                
                # Display the interactive chart
                self._display_plotly_chart(fig, tab="individual")
                self.status_var.set(f"Generated interactive chart for {ticker}")
                
            except Exception as e:
                # If Plotly chart fails, fall back to static chart
                logging.error(f"Error creating Plotly chart for {ticker}: {str(e)}")
                self.status_var.set(f"Using static chart for {ticker} due to error: {str(e)}")
                
                # Generate static chart as fallback
                plots_dir = self.manager.plot_save_path
                chart_path = os.path.join(plots_dir, f"{ticker}_daily_weekly_monthly.png")
                
                if os.path.exists(chart_path):
                    self._display_static_chart(chart_path)
                else:
                    self.status_var.set(f"Error: No chart available for {ticker}")
        
        except Exception as e:
            chart_name = ticker_or_path
            error_msg = f"Error displaying chart for {chart_name}: {str(e)}"
            self.status_var.set(error_msg)
            logging.error(error_msg)
            messagebox.showerror("Error", error_msg)

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
