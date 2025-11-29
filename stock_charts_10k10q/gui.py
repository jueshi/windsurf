import os
import sys
import io
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
import requests
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import logging
import json
import threading
import queue
import webbrowser
from PIL import Image, ImageTk
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.offline import plot
from tkcalendar import DateEntry
from data_manager import StockDataManager
import google.generativeai as genai
import gemini_analyzer
import buffett_canslim
import news_fetcher
import sec_filing_extractor
import sec_api_wrapper
import webbrowser
import tempfile
from live_chart_generator import generate_chart_html
from multi_tf_charts import generate_multi_timeframe_chart_html, generate_multi_timeframe_linechart_html
from generate_stockcharts_gallery import (
    generate_multi_timeframe_stockcharts_html,
    generate_multi_timeframe_stockcharts_line_html,
)
import subprocess
from thread_safe_tkinter import (
    setup_thread_safe_tkinter,
    safe_update_text_widget,
    safe_update_status,
    safe_show_message,
    thread_safe
)
from tooltip_manager import TooltipManager

class StockDataGUI:
    """GUI for Stock Data Manager"""

    def __init__(self, root, manager):
        """Initialize the GUI."""
        self.root = root
        self.manager = manager
        self.current_tickers = []
        self.watch_list = []
        
        # Set up thread-safe Tkinter updates
        setup_thread_safe_tkinter(root)
        self.current_image = None  # Store reference to prevent garbage collection
        self.active_tab = "individual"  # Track which tab is active: "individual" or "comparison"
        self.seasonality_pil_img = None  # To store the high-res seasonality chart
        self.seasonality_chart_ticker = None # Tracks the ticker for the current seasonality chart
        self._debounce_job = None       # For debouncing resize events
        self.year_selection_vars = {}  # For multi-select year checkbuttons
        self.seasonality_year_menubutton = None # The new menubutton for year selection
        self.year_menu = None # A direct reference to the year selection menu
        # https://stockcharts.com/freecharts/seasonality.php?symbol=OKTA&compare=SPY
        self.fundamental_data_cache = [] # Cache for fundamental data
        self.fundamental_filter_var = tk.StringVar() # Filter for fundamental data
        self.business_analysis_filter_var = tk.StringVar() # Filter for business analysis data
        self.business_analysis_original_text = "" # Store original text for filtering
        self.market_news_original_text = ""  # Cache latest market news summary
        self.stock_news_temp_tickers = []  # Temporary tickers detected from Finviz v=3

        self.show_tooltips = tk.BooleanVar(value=True)
        self.tooltip_manager = TooltipManager(self.root)
        self.show_tooltips.trace_add("write", self._update_tooltip_state)
        self._update_tooltip_state()

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
        
        # Auto-load the first ticker list after GUI is created
        self.root.after(200, self._load_first_ticker_list)

    def _update_tooltip_state(self, *args):
        try:
            enabled = bool(self.show_tooltips.get())
            if hasattr(self, "tooltip_manager"):
                self.tooltip_manager.set_enabled(enabled)
        except Exception as exc:
            logging.debug(f"Tooltip toggle update failed: {exc}")

    def _attach_tooltip(self, widget, *, text=None, text_provider=None, tooltip_id=None):
        if not widget or not hasattr(self, "tooltip_manager"):
            return
        try:
            self.tooltip_manager.attach(
                widget,
                text=text,
                text_provider=text_provider,
                tooltip_id=tooltip_id,
            )
        except Exception as exc:
            logging.debug(f"Failed to attach tooltip to {widget}: {exc}")

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

    def _set_initial_sash_positions(self):
        """Set the initial positions of the sashes to make list panes very narrow"""
        try:
            # Get the total width of the paned window
            total_width = self.paned_window.winfo_width()
            
            if total_width > 0:
                # Set first sash position (between left and middle panes)
                self.paned_window.sashpos(0, 120)  # 120 pixels from left (doubled from 60)
                
                # Set second sash position (between middle and right panes)
                self.paned_window.sashpos(1, 240)  # 240 pixels from left (doubled from 120)
                
                logging.info(f"Set initial sash positions: 120, 240 (total width: {total_width})")
            else:
                # If window width is not yet available, try again after a delay
                self.root.after(100, self._set_initial_sash_positions)
        except Exception as e:
            logging.error(f"Error setting sash positions: {e}")
    
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

    def _open_ticker_list_in_notepadpp(self):
        """Open ticker_lists.py in Notepad++ if available; otherwise offer default editor."""
        try:
            ticker_lists_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_lists.py")
            if not os.path.exists(ticker_lists_path):
                messagebox.showerror("File Not Found", f"Could not find ticker list file at:\n{ticker_lists_path}")
                return

            candidates = [
                r"C:\\Program Files\\Notepad++\\notepad++.exe",
                r"C:\\Program Files (x86)\\Notepad++\\notepad++.exe",
            ]

            launched = False
            for exe in candidates:
                if os.path.isfile(exe):
                    subprocess.Popen([exe, ticker_lists_path])
                    launched = True
                    break

            if not launched:
                try:
                    subprocess.Popen(["notepad++", ticker_lists_path])
                    launched = True
                except FileNotFoundError:
                    launched = False

            if not launched:
                if messagebox.askyesno("Notepad++ Not Found", "Notepad++ was not found. Open with the default editor instead?"):
                    try:
                        os.startfile(ticker_lists_path)
                    except Exception as e:
                        messagebox.showerror("Open Failed", f"Failed to open file: {e}")
            else:
                self.status_var.set("Opened ticker_lists.py in Notepad++")
        except Exception as e:
            logging.error(f"Error opening ticker list in Notepad++: {e}")
            messagebox.showerror("Error", f"Failed to open ticker list: {e}")

    def _copy_current_list_to_clipboard(self):
        """Copy all current tickers to the clipboard, one per line."""
        try:
            if not self.current_tickers:
                messagebox.showinfo("No Tickers", "There are no tickers in the current list to copy.")
                return
            text = "\n".join(self.current_tickers)
            try:
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
                self.root.update_idletasks()
            except Exception:
                # Fallback using Tk's clipboard methods
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
            self.status_var.set(f"Copied {len(self.current_tickers)} tickers to clipboard")
        except Exception as e:
            logging.error(f"Error copying tickers to clipboard: {e}")
            messagebox.showerror("Error", f"Failed to copy to clipboard: {e}")

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
            
        except ImportError as ie:
            error_msg = f"Missing required package for interactive charts: {ie}"
            logging.error(error_msg)
            if hasattr(self, 'status_var'):
                self.status_var.set(error_msg)
            messagebox.showerror("Package Error", f"Missing required package for interactive charts: {ie}\n\nPlease install plotly with: pip install plotly")
            
        except tk.TclError as te:
            error_msg = f"Tkinter error displaying chart: {te}"
            logging.error(error_msg)
            # Don't show messagebox for Tkinter errors as they're often related to widget destruction
            
        except Exception as e:
            error_msg = f"Error displaying Plotly chart: {e}"
            logging.error(error_msg)
            if hasattr(self, 'status_var'):
                self.status_var.set(error_msg)
            messagebox.showerror("Chart Error", f"Failed to display interactive chart: {e}\n\nTry refreshing the data or restarting the application.")
    
    def _create_widgets(self):
        """Create all GUI widgets"""
        # Create a vertical paned window so the action bar can be resized
        self.layout_paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        self.layout_paned.pack(fill=tk.BOTH, expand=True)

        # Create main frame for the core content (top pane)
        main_frame = ttk.Frame(self.layout_paned, padding="10")
        self.layout_paned.add(main_frame, weight=1)

        # Create bottom frame for actions (bottom pane) but populate later
        bottom_frame = ttk.Frame(self.layout_paned, padding=(10, 2))
        self.layout_paned.add(bottom_frame, weight=0)
        try:
            self.layout_paned.paneconfig(bottom_frame, minsize=55)
        except Exception:
            pass
        self.bottom_frame = bottom_frame
        self._bottom_sash_initialized = False

        def _init_bottom_sash(event=None):
            if self._bottom_sash_initialized:
                return
            paned = getattr(self, 'layout_paned', None)
            if not paned:
                return
            bottom = getattr(self, 'bottom_frame', None)
            if not bottom:
                return
            paned.update_idletasks()
            height = paned.winfo_height()
            bottom_height = bottom.winfo_reqheight()
            if height <= 0:
                return
            try:
                desired = max(height - bottom_height - 4, 0)
                paned.sashpos(0, desired)
                self._bottom_sash_initialized = True
            except Exception:
                pass

        self.layout_paned.bind("<Configure>", _init_bottom_sash)

        # Create top frame for ticker list selection
        top_frame = ttk.Frame(main_frame, padding="10")
        top_frame.pack(fill=tk.X, pady=5)

        # Ticker list selection with filter
        ticker_list_label = ttk.Label(top_frame, text="Ticker List:")
        ticker_list_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self._attach_tooltip(
            ticker_list_label,
            text="Select which saved ticker list (from ticker_lists.py) should populate the workspace.",
            tooltip_id="ticker_list.label",
        )

        # Create a frame for the dropdown and its filter
        dropdown_frame = ttk.Frame(top_frame)
        dropdown_frame.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Add filter entry for ticker list dropdown
        self.list_filter_var = tk.StringVar()
        list_filter_entry = ttk.Entry(dropdown_frame, textvariable=self.list_filter_var, width=20)
        list_filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._attach_tooltip(
            list_filter_entry,
            text="Type to filter saved list names. Supports partial matches and updates immediately.",
            tooltip_id="ticker_list.filter",
        )
        list_filter_entry.bind("<KeyRelease>", self._filter_ticker_lists)

        # Create the combobox for ticker lists
        self.ticker_list_var = tk.StringVar()
        self.ticker_list_combo = ttk.Combobox(
            dropdown_frame,
            textvariable=self.ticker_list_var,
            values=list(self.ticker_lists.keys()),
            width=60,
        )
        self.ticker_list_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        self.ticker_list_combo.bind("<<ComboboxSelected>>", self._on_list_selected)
        self._attach_tooltip(
            self.ticker_list_combo,
            text="Choose from discovered ticker lists. Selecting a list enables Load/Prev/Next actions.",
            tooltip_id="ticker_list.combo",
        )

        # Create a frame for the buttons
        button_frame = ttk.Frame(top_frame)
        button_frame.grid(row=0, column=2, padx=5, pady=5)

        tooltip_toggle = ttk.Checkbutton(
            top_frame,
            text="Show Tooltips",
            variable=self.show_tooltips,
        )
        tooltip_toggle.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        self._attach_tooltip(
            tooltip_toggle,
            text="Toggle inline guidance bubbles. Shift+F1 on any control also shows its tip.",
            tooltip_id="settings.tooltips_toggle",
        )

        # Load List button loads the selected list
        load_button = ttk.Button(button_frame, text="Load", command=self._load_ticker_list)
        load_button.pack(side=tk.LEFT, padx=(0, 5))
        self._attach_tooltip(
            load_button,
            text="Load the highlighted ticker list into the Available/Watch panes.",
            tooltip_id="ticker_list.load",
        )

        refresh_button = ttk.Button(button_frame, text="Refresh", command=self._refresh_ticker_lists)
        refresh_button.pack(side=tk.LEFT, padx=(0, 5))
        self._attach_tooltip(
            refresh_button,
            text="Reload ticker_lists.py from disk to pick up edits or new watch lists.",
            tooltip_id="ticker_list.refresh",
        )

        prev_button = ttk.Button(button_frame, text="Prev", command=self._go_prev_list)
        prev_button.pack(side=tk.LEFT, padx=(0, 5))
        self._attach_tooltip(
            prev_button,
            text="Jump to the previous list based on the combo ordering.",
            tooltip_id="ticker_list.prev",
        )

        next_button = ttk.Button(button_frame, text="Next", command=self._go_next_list)
        next_button.pack(side=tk.LEFT, padx=(0, 5))
        self._attach_tooltip(
            next_button,
            text="Jump forward to the next saved list without opening the dropdown.",
            tooltip_id="ticker_list.next",
        )

        remove_button = ttk.Button(button_frame, text="Remove", command=self._remove_current_list)
        remove_button.pack(side=tk.LEFT)
        self._attach_tooltip(
            remove_button,
            text="Delete the selected list from ticker_lists.py (asks for confirmation).",
            tooltip_id="ticker_list.remove",
        )

        daily_button = ttk.Button(
            button_frame,
            text="Daily charts",
            command=lambda: self._open_live_charts_for_current_list(time_frame="d"),
        )
        daily_button.pack(side=tk.LEFT, padx=(5, 0))
        self._attach_tooltip(
            daily_button,
            text="Launch browser charts for each ticker using daily candles.",
            tooltip_id="ticker_list.daily",
        )

        weekly_button = ttk.Button(
            button_frame,
            text="Weekly",
            command=lambda: self._open_live_charts_for_current_list(time_frame="w"),
        )
        weekly_button.pack(side=tk.LEFT, padx=(5, 0))
        self._attach_tooltip(
            weekly_button,
            text="Open StockCharts multi-symbol layout with weekly timeframe.",
            tooltip_id="ticker_list.weekly",
        )

        monthly_button = ttk.Button(
            button_frame,
            text="Monthly",
            command=lambda: self._open_live_charts_for_current_list(time_frame="m"),
        )
        monthly_button.pack(side=tk.LEFT, padx=(5, 0))
        self._attach_tooltip(
            monthly_button,
            text="Open monthly candle view for the active list.",
            tooltip_id="ticker_list.monthly",
        )

        multi_tf_button = ttk.Button(
            button_frame,
            text="Multi-TF",
            command=self._open_multi_timeframe_gallery_for_current_list,
        )
        multi_tf_button.pack(side=tk.LEFT, padx=(5, 0))
        self._attach_tooltip(
            multi_tf_button,
            text="Generate StockCharts gallery with daily/weekly/monthly snapshots.",
            tooltip_id="ticker_list.multi_tf",
        )

        linecharts_button = ttk.Button(
            button_frame,
            text="LineCharts",
            command=self._open_linecharts_gallery_for_current_list,
        )
        linecharts_button.pack(side=tk.LEFT, padx=(5, 0))
        self._attach_tooltip(
            linecharts_button,
            text="Render comparison-focused line chart gallery for the list.",
            tooltip_id="ticker_list.linecharts",
        )

        stockcharts_button = ttk.Button(
            button_frame,
            text="StockCharts",
            command=self._open_stockcharts_gallery_for_current_list,
        )
        stockcharts_button.pack(side=tk.LEFT, padx=(5, 0))
        self._attach_tooltip(
            stockcharts_button,
            text="Open the traditional StockCharts gallery for this list.",
            tooltip_id="ticker_list.stockcharts",
        )

        # StockCharts line style ID entry (for SC-Line gallery)
        self.stockcharts_line_style_var = tk.StringVar(value="t3327397499c")

        # Clickable label that opens the StockCharts style page in the browser
        sc_style_label = ttk.Label(button_frame, text="SC style id:", foreground="blue", cursor="hand2")
        sc_style_label.pack(side=tk.LEFT, padx=(10, 2))
        sc_style_label.bind(
            "<Button-1>",
            lambda e: webbrowser.open("https://stockcharts.com/sc3/ui/?s=ALAB"),
        )
        self._attach_tooltip(
            sc_style_label,
            text="Click to open StockCharts style builder in your browser.",
            tooltip_id="ticker_list.sc_style_label",
        )

        sc_style_entry = ttk.Entry(button_frame, textvariable=self.stockcharts_line_style_var, width=14)
        sc_style_entry.pack(side=tk.LEFT, padx=(0, 5))
        self._attach_tooltip(
            sc_style_entry,
            text="Paste a StockCharts style ID (t###########c) used for SC-Line gallery requests.",
            tooltip_id="ticker_list.sc_style_entry",
        )

        sc_line_button = ttk.Button(
            button_frame,
            text="SC-Line",
            command=self._open_stockcharts_line_gallery_for_current_list,
        )
        sc_line_button.pack(side=tk.LEFT, padx=(5, 0))
        self._attach_tooltip(
            sc_line_button,
            text="Generate SC-Line gallery with the supplied style ID.",
            tooltip_id="ticker_list.sc_line",
        )

        open_ticker_button = ttk.Button(button_frame, text="Open Ticker", command=self._open_ticker_list_in_notepadpp)
        open_ticker_button.pack(side=tk.LEFT, padx=(5, 0))
        self._attach_tooltip(
            open_ticker_button,
            text="Open ticker_lists.py in Notepad++ (or default editor) for quick edits.",
            tooltip_id="ticker_list.open_file",
        )

        copy_list_button = ttk.Button(button_frame, text="Copy List", command=self._copy_current_list_to_clipboard)
        copy_list_button.pack(side=tk.LEFT, padx=(5, 0))
        self._attach_tooltip(
            copy_list_button,
            text="Copy all tickers in the current list to clipboard, one per line.",
            tooltip_id="ticker_list.copy",
        )

        # Create a frame for the second row with Add Ticker and New List Name
        second_row_frame = ttk.Frame(top_frame)
        second_row_frame.grid(row=1, column=0, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Add manual ticker entry (left side of second row)
        ticker_frame = ttk.Frame(second_row_frame)
        ticker_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        add_ticker_label = ttk.Label(ticker_frame, text="Add Ticker:")
        add_ticker_label.pack(side=tk.LEFT, padx=(0, 5))
        self._attach_tooltip(
            add_ticker_label,
            text="Manually append a symbol to the currently loaded list.",
            tooltip_id="ticker_list.add_label",
        )
        self.manual_ticker_var = tk.StringVar()
        manual_ticker_entry = ttk.Entry(ticker_frame, textvariable=self.manual_ticker_var, width=30)
        manual_ticker_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self._attach_tooltip(
            manual_ticker_entry,
            text="Enter a ticker symbol (comma-separated for multiples) to append to the list.",
            tooltip_id="ticker_list.manual_entry",
        )
        add_ticker_button = ttk.Button(ticker_frame, text="Add", command=self._add_manual_ticker)
        add_ticker_button.pack(side=tk.LEFT)
        self._attach_tooltip(
            add_ticker_button,
            text="Append the typed ticker(s) to the current list immediately.",
            tooltip_id="ticker_list.manual_add",
        )
        
        # Add separator
        ttk.Separator(second_row_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Add list name entry and save button (right side of second row)
        list_frame = ttk.Frame(second_row_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        new_list_label = ttk.Label(list_frame, text="New List Name:")
        new_list_label.pack(side=tk.LEFT, padx=(0, 5))
        self._attach_tooltip(
            new_list_label,
            text="Provide a Python-safe variable name for the list saved to ticker_lists.py.",
            tooltip_id="ticker_list.new_label",
        )
        self.list_name_var = tk.StringVar()
        list_name_entry = ttk.Entry(list_frame, textvariable=self.list_name_var, width=30)
        list_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self._attach_tooltip(
            list_name_entry,
            text="Name used for the new ticker list variable. Avoid spaces or special characters.",
            tooltip_id="ticker_list.new_entry",
        )
        create_list_button = ttk.Button(list_frame, text="Create List", command=self._save_ticker_list)
        create_list_button.pack(side=tk.LEFT)
        self._attach_tooltip(
            create_list_button,
            text="Persist the current set of tickers under the provided list name.",
            tooltip_id="ticker_list.create",
        )

        # Create a PanedWindow for resizable sections
        self.paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, pady=5)

        # --- Left Pane: Available Tickers ---
        left_pane_frame = ttk.Frame(self.paned_window, width=50)  # Set very narrow fixed width
        left_pane_frame.pack_propagate(False)  # Prevent child widgets from changing frame size
        self.paned_window.add(left_pane_frame, weight=1) # Very minimal weight for ticker list

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

        self.ticker_listbox = tk.Listbox(ticker_frame, selectmode=tk.EXTENDED, height=20, width=2)
        self.ticker_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.ticker_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.ticker_listbox.yview)

        # Add buttons for reordering and sorting tickers
        ticker_buttons_frame = ttk.Frame(left_frame)
        ticker_buttons_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(ticker_buttons_frame, text="A-Z", command=self._sort_tickers, width=10).pack(side=tk.LEFT)
        ttk.Button(ticker_buttons_frame, text="↑", command=self._move_ticker_up, width=10).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(ticker_buttons_frame, text="↓", command=self._move_ticker_down, width=10).pack(side=tk.LEFT, padx=(0, 2))

        # --- Middle Pane: Watch List ---
        middle_pane_frame = ttk.Frame(self.paned_window, width=50)  # Set very narrow fixed width
        middle_pane_frame.pack_propagate(False)  # Prevent child widgets from changing frame size
        self.paned_window.add(middle_pane_frame, weight=1) # Very minimal weight for watch list

        middle_list_frame = ttk.LabelFrame(middle_pane_frame, text="Watch List", padding="5")
        middle_list_frame.pack(fill=tk.BOTH, expand=True)

        # Create watch list listbox with scrollbar
        watch_frame = ttk.Frame(middle_list_frame)
        watch_frame.pack(fill=tk.BOTH, expand=True)

        watch_scrollbar = ttk.Scrollbar(watch_frame)
        watch_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.watch_listbox = tk.Listbox(watch_frame, selectmode=tk.EXTENDED, height=20, width=2)
        self.watch_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Right Pane: Chart Display ---
        right_pane_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(right_pane_frame, weight=20) # Greatly increased weight for chart display
        
        # Store references to pane frames for later use
        self.left_pane_frame = left_pane_frame
        self.middle_pane_frame = middle_pane_frame
        self.right_pane_frame = right_pane_frame

        self.chart_frame = ttk.LabelFrame(right_pane_frame, text="Chart Display", padding="5")
        self.chart_frame.pack(fill=tk.BOTH, expand=True)

        # Add date range controls at the top of chart frame
        date_range_frame = ttk.Frame(self.chart_frame, padding="5")
        date_range_frame.pack(fill=tk.X, expand=False, pady=(0, 5))

        # Start date entry with calendar widget
        start_label = ttk.Label(date_range_frame, text="Start Date:")
        start_label.pack(side=tk.LEFT, padx=(0, 5))
        self._attach_tooltip(
            start_label,
            text="Beginning of the date window applied to chart data.",
            tooltip_id="charts.start_label",
        )
        self.start_date_var = tk.StringVar()
        self.start_date_entry = DateEntry(date_range_frame, textvariable=self.start_date_var, width=12,
                                        date_pattern='yyyy-mm-dd', background='darkblue', foreground='white',
                                      borderwidth=2, locale='en_US')
        self.start_date_entry.pack(side=tk.LEFT, padx=(0, 10))
        self._attach_tooltip(
            self.start_date_entry,
            text="Pick the earliest date for charts. Defaults to earliest available data.",
            tooltip_id="charts.start_entry",
        )

        # End date entry with calendar widget
        end_label = ttk.Label(date_range_frame, text="End Date:")
        end_label.pack(side=tk.LEFT, padx=(0, 5))
        self._attach_tooltip(
            end_label,
            text="Set the latest date to include in the chart window.",
            tooltip_id="charts.end_label",
        )
        self.end_date_var = tk.StringVar()
        self.end_date_entry = DateEntry(date_range_frame, textvariable=self.end_date_var, width=12,
                                      date_pattern='yyyy-mm-dd', background='darkblue', foreground='white',
                                      borderwidth=2, locale='en_US')
        self.end_date_entry.pack(side=tk.LEFT, padx=(0, 10))
        self._attach_tooltip(
            self.end_date_entry,
            text="Choose the final date included in generated charts.",
            tooltip_id="charts.end_entry",
        )

        apply_range_button = ttk.Button(date_range_frame, text="Apply Date Range", command=self._apply_date_range)
        apply_range_button.pack(side=tk.LEFT)
        self._attach_tooltip(
            apply_range_button,
            text="Rebuild charts using the specified start/end dates.",
            tooltip_id="charts.apply_range",
        )

        reset_range_button = ttk.Button(date_range_frame, text="Reset Date Range", command=self._reset_date_range)
        reset_range_button.pack(side=tk.LEFT, padx=(10, 0))
        self._attach_tooltip(
            reset_range_button,
            text="Clear manual bounds and show the full data history.",
            tooltip_id="charts.reset_range",
        )

        quick_label = ttk.Label(date_range_frame, text="Quick:")
        quick_label.pack(side=tk.LEFT, padx=(10, 5))
        self._attach_tooltip(
            quick_label,
            text="Choose a predefined range shortcut.",
            tooltip_id="charts.quick_label",
        )

        quick_6m = ttk.Button(date_range_frame, text="6M", width=4, command=lambda: self._set_quick_range(days=182))
        quick_6m.pack(side=tk.LEFT)
        self._attach_tooltip(quick_6m, text="Show the last six months of data.", tooltip_id="charts.quick_6m")

        quick_1y = ttk.Button(date_range_frame, text="1Y", width=4, command=lambda: self._set_quick_range(days=365))
        quick_1y.pack(side=tk.LEFT, padx=(5, 0))
        self._attach_tooltip(quick_1y, text="Display trailing twelve months of history.", tooltip_id="charts.quick_1y")

        quick_3y = ttk.Button(date_range_frame, text="3Y", width=4, command=lambda: self._set_quick_range(days=365*3))
        quick_3y.pack(side=tk.LEFT, padx=(5, 0))
        self._attach_tooltip(quick_3y, text="Limit view to the last three years.", tooltip_id="charts.quick_3y")

        quick_5y = ttk.Button(date_range_frame, text="5Y", width=4, command=lambda: self._set_quick_range(days=365*5))
        quick_5y.pack(side=tk.LEFT, padx=(5, 0))
        self._attach_tooltip(quick_5y, text="Trim dataset to the past five years.", tooltip_id="charts.quick_5y")

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

        # Create fundamental analysis tab
        self.fundamental_analysis_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.fundamental_analysis_frame, text="Fundamental Analysis")

        # Create business analysis tab
        self.business_analysis_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.business_analysis_frame, text="Business Analysis")

        # Create Market News tab
        self.market_news_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.market_news_frame, text="Market News Blog")
        
        # Create Buffett & CANSLIM tab
        self.buffett_canslim_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.buffett_canslim_frame, text="Buffett & CANSLIM")

        # Layout for Buffett & CANSLIM tab
        bc_outer = ttk.Frame(self.buffett_canslim_frame, padding="10")
        bc_outer.pack(fill=tk.BOTH, expand=True)

        bc_controls = ttk.Frame(bc_outer)
        bc_controls.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(bc_controls, text="Analyze Selected", command=self._analyze_buffett_canslim_current).pack(side=tk.LEFT)
        self.bc_status_var = tk.StringVar(value="Select a ticker and click Analyze, or select while this tab is active.")
        ttk.Label(bc_controls, textvariable=self.bc_status_var).pack(side=tk.LEFT, padx=10)

        self.bc_content = ttk.PanedWindow(bc_outer, orient=tk.HORIZONTAL)
        self.bc_content.pack(fill=tk.BOTH, expand=True)
        # Track sash initialization
        self._bc_sash_initialized = False

        # Left: chart image
        self.bc_left_frame = ttk.LabelFrame(self.bc_content, text="Radar & Trend")
        self.bc_content.add(self.bc_left_frame, weight=2)
        self.bc_chart_label = ttk.Label(self.bc_left_frame, width=48)
        self.bc_chart_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self._bc_chart_photo = None  # keep reference
        # Zoom state for Buffett & CANSLIM chart
        self._bc_base_image = None   # PIL.Image for original chart
        self._bc_zoom_scale = 0.5    # current zoom scale
        self._bc_user_zoomed = False # whether user has manually zoomed
        self._bc_last_ticker = None
        self._bc_last_explanation = ""
        # Refit image to container when layout changes, unless user has zoomed
        def _bc_on_container_resize(event=None):
            try:
                if self._bc_base_image is not None and not self._bc_user_zoomed:
                    self._update_bc_chart_image()
            except Exception:
                pass
        try:
            self.bc_left_frame.bind('<Configure>', _bc_on_container_resize)
        except Exception:
            pass

        # Right: explanation text
        bc_right = ttk.LabelFrame(self.bc_content, text="Explanation")
        self.bc_content.add(bc_right, weight=3)
        self.bc_text = tk.Text(bc_right, wrap=tk.WORD)
        self.bc_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # Make sure the right pane is visible: set min size and an initial sash position
        try:
            self.bc_content.paneconfig(bc_right, minsize=220)
            self.bc_content.paneconfig(self.bc_left_frame, minsize=220)
        except Exception:
            pass
        # Set a larger initial sash position; and ensure it's applied after layout via Configure
        try:
            self.root.after(100, lambda: self.bc_content.sashpos(0, 400))
            def _adjust_bc_sash_once(event=None):
                if getattr(self, '_bc_sash_initialized', False):
                    return
                try:
                    w = self.bc_content.winfo_width()
                    if w and w > 0:
                        self.bc_content.sashpos(0, int(w * 0.55))
                        self._bc_sash_initialized = True
                except Exception:
                    pass
            self.bc_content.bind('<Configure>', _adjust_bc_sash_once)
        except Exception:
            pass

        # Create SEC filings tab
        self.sec_filings_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.sec_filings_frame, text="SEC Filings")
        
        # Configure SEC filings tab
        sec_frame = ttk.Frame(self.sec_filings_frame, padding="10")
        sec_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top control frame for SEC filings tab
        sec_control_frame = ttk.Frame(sec_frame)
        sec_control_frame.pack(fill=tk.X, pady=5)
        
        # Ticker label and selection
        ttk.Label(sec_control_frame, text="Ticker:").pack(side=tk.LEFT, padx=(0, 5))
        self.sec_ticker_var = tk.StringVar()
        self.sec_ticker_entry = ttk.Entry(sec_control_frame, textvariable=self.sec_ticker_var, width=10)
        self.sec_ticker_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        # Form type selection
        ttk.Label(sec_control_frame, text="Form Type:").pack(side=tk.LEFT, padx=(0, 5))
        self.sec_form_type_var = tk.StringVar(value="10-K")
        form_type_combo = ttk.Combobox(sec_control_frame, textvariable=self.sec_form_type_var, 
                                     values=["10-K", "10-Q"], width=5, state="readonly")
        form_type_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Mock data checkbox
        self.use_mock_data_var = tk.BooleanVar(value=False)
        mock_data_check = ttk.Checkbutton(sec_control_frame, text="Use Mock Data", 
                                       variable=self.use_mock_data_var,
                                       command=self._toggle_mock_data)
        mock_data_check.pack(side=tk.LEFT, padx=(0, 10))
        
        # Extract button
        ttk.Button(sec_control_frame, text="Extract Tables", 
                  command=self._extract_sec_tables_from_tab).pack(side=tk.LEFT, padx=(0, 10))
        
        # Open folder button
        ttk.Button(sec_control_frame, text="Open Output Folder", 
                  command=self._open_sec_output_folder).pack(side=tk.LEFT, padx=(0, 10))
        
        # Status label
        self.sec_status_var = tk.StringVar(value="Select a ticker and form type, then click 'Extract Tables'")
        ttk.Label(sec_control_frame, textvariable=self.sec_status_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Second row for API status and cache info
        sec_status_frame = ttk.Frame(sec_frame)
        sec_status_frame.pack(fill=tk.X, pady=(0, 5))
        
        # API status label
        self.sec_api_status_var = tk.StringVar(value="Using real SEC API with caching (recommended for production)")
        ttk.Label(sec_status_frame, textvariable=self.sec_api_status_var, 
                 font=("Helvetica", 9, "italic")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Clear cache button
        ttk.Button(sec_status_frame, text="Clear SEC Cache", 
                 command=self._clear_sec_cache).pack(side=tk.RIGHT)
        
        # Create a paned window to split the view
        sec_paned = ttk.PanedWindow(sec_frame, orient=tk.HORIZONTAL)
        sec_paned.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Left side - Table list
        sec_left_frame = ttk.LabelFrame(sec_paned, text="Available Tables")
        sec_paned.add(sec_left_frame, weight=1)
        
        # Create a listbox for tables with scrollbar
        sec_list_frame = ttk.Frame(sec_left_frame)
        sec_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        sec_list_scrollbar = ttk.Scrollbar(sec_list_frame)
        sec_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.sec_table_listbox = tk.Listbox(sec_list_frame, selectmode=tk.SINGLE, height=20)
        self.sec_table_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.sec_table_listbox.config(yscrollcommand=sec_list_scrollbar.set)
        sec_list_scrollbar.config(command=self.sec_table_listbox.yview)
        
        # Bind selection event
        self.sec_table_listbox.bind("<<ListboxSelect>>", self._on_sec_table_selected)
        
        # Right side - Table view
        sec_right_frame = ttk.LabelFrame(sec_paned, text="Table Content")
        sec_paned.add(sec_right_frame, weight=3)
        
        # Create a treeview for table content with scrollbars
        sec_tree_frame = ttk.Frame(sec_right_frame)
        sec_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Horizontal scrollbar
        sec_tree_xscrollbar = ttk.Scrollbar(sec_tree_frame, orient="horizontal")
        sec_tree_xscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Vertical scrollbar
        sec_tree_yscrollbar = ttk.Scrollbar(sec_tree_frame)
        sec_tree_yscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Treeview for table data
        self.sec_table_tree = ttk.Treeview(sec_tree_frame, style="Custom.Treeview")
        self.sec_table_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbars
        self.sec_table_tree.config(xscrollcommand=sec_tree_xscrollbar.set, yscrollcommand=sec_tree_yscrollbar.set)
        sec_tree_xscrollbar.config(command=self.sec_table_tree.xview)
        sec_tree_yscrollbar.config(command=self.sec_table_tree.yview)
        
        # Bottom frame for export options
        sec_bottom_frame = ttk.Frame(sec_frame)
        sec_bottom_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(sec_bottom_frame, text="Export to Excel", 
                  command=self._export_sec_table_to_excel).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(sec_bottom_frame, text="Copy to Clipboard", 
                  command=self._copy_sec_table_to_clipboard).pack(side=tk.LEFT, padx=(0, 10))

        # --- Fundamental Analysis Tab Widgets ---
        # Configure a custom style for the Treeview for a larger font
        style = ttk.Style()
        style.configure("Custom.Treeview", font=('Helvetica', 12))  # Set font size to 12
        style.configure("Custom.Treeview.Heading", font=('Helvetica', 14, 'bold')) # Set heading font size

        # Create a frame for the filter widget
        fa_filter_frame = ttk.Frame(self.fundamental_analysis_frame)
        fa_filter_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(fa_filter_frame, text="Filter Metric:").pack(side=tk.LEFT, padx=(0, 5))
        fa_filter_entry = ttk.Entry(fa_filter_frame, textvariable=self.fundamental_filter_var)
        fa_filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        fa_filter_entry.bind("<KeyRelease>", self._populate_fundamental_treeview)
        
        # Add a small help label
        ttk.Label(fa_filter_frame, text="(Multiple terms use OR logic, ! for exclusion, * to show all)", 
                 font=("Helvetica", 8)).pack(side=tk.LEFT, padx=5)
        
        # Add Save and Load filter buttons
        save_filter_button = ttk.Button(fa_filter_frame, text="Save Filter", width=10, command=self._save_filter)
        save_filter_button.pack(side=tk.LEFT, padx=5)
        
        load_filter_button = ttk.Button(fa_filter_frame, text="Load Filter", width=10, command=self._load_filter)
        load_filter_button.pack(side=tk.LEFT, padx=5)

        # Create a frame to hold the treeview and scrollbar
        fa_tree_frame = ttk.Frame(self.fundamental_analysis_frame)
        fa_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create a treeview to display fundamental data in a table
        self.fundamental_data_tree = ttk.Treeview(fa_tree_frame, columns=('Metric', 'Value'), show='headings', style="Custom.Treeview")
        self.fundamental_data_tree.heading('Metric', text='Metric')
        self.fundamental_data_tree.heading('Value', text='Value')
        self.fundamental_data_tree.column('Metric', width=200)
        self.fundamental_data_tree.column('Value', width=400)

        # Add a vertical scrollbar
        vsb = ttk.Scrollbar(fa_tree_frame, orient="vertical", command=self.fundamental_data_tree.yview)
        self.fundamental_data_tree.configure(yscrollcommand=vsb.set)

        # Pack the scrollbar and the treeview
        vsb.pack(side='right', fill='y')
        self.fundamental_data_tree.pack(side='left', fill='both', expand=True)

        # Define important fundamental metrics to highlight
        self.important_metrics = [
            'longName', 'sector', 'industry', 'marketCap', 'trailingPE',
            'forwardPE', 'dividendYield', 'beta', 'fiftyTwoWeekHigh',
            'fiftyTwoWeekLow', 'longBusinessSummary'
        ]

        # Configure a tag for bold text in the fundamental data view
        self.fundamental_data_tree.tag_configure("bold", font=("Helvetica", 10, "bold"))

        # --- Business Analysis Tab Widgets ---
        ba_frame = ttk.Frame(self.business_analysis_frame, padding="10")
        ba_frame.pack(fill=tk.BOTH, expand=True)

        ba_button_frame = ttk.Frame(ba_frame)
        ba_button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(ba_button_frame, text="Run BA", command=self._run_business_analysis).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(ba_button_frame, text="Conduct News Search", command=self._run_news_search).pack(side=tk.LEFT, padx=(0, 5))        
        
        ttk.Button(ba_button_frame, text="10-Q Study", command=self._run_10q_study).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(ba_button_frame, text="10K Study", command=self._run_10k_study).pack(side=tk.LEFT, padx=(0, 5))
        # SEC filing extraction buttons
        ttk.Button(ba_button_frame, text="Extract 10-K Tables", command=lambda: self._extract_sec_filing('10-K')).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(ba_button_frame, text="Extract 10-Q Tables", command=lambda: self._extract_sec_filing('10-Q')).pack(side=tk.LEFT, padx=(0, 5))


        self.general_search_var = tk.StringVar()
        general_search_entry = ttk.Entry(ba_button_frame, textvariable=self.general_search_var, width=40)
        general_search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        ttk.Button(ba_button_frame, text="AI Search", command=self._run_general_search).pack(side=tk.LEFT)
        
        # Add filter frame for business analysis
        ba_filter_frame = ttk.Frame(ba_frame)
        ba_filter_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(ba_filter_frame, text="Filter Metric:").pack(side=tk.LEFT, padx=(0, 5))
        filter_entry = ttk.Entry(ba_filter_frame, textvariable=self.business_analysis_filter_var, width=40)
        filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        filter_entry.bind("<KeyRelease>", self._apply_business_analysis_filter)
        
        # Add a small help label
        ttk.Label(ba_filter_frame, text="(Multiple terms use AND logic, ! for exclusion)", 
                 font=("Helvetica", 8)).pack(side=tk.LEFT, padx=5)
        
        # BA tuning controls: freshness (days), history depth, and show/hide change-over-time
        ba_opts_frame = ttk.Frame(ba_frame)
        ba_opts_frame.pack(fill=tk.X, pady=(0,5))
        
        # Variables
        self.ba_freshness_days_var = tk.IntVar(value=30)
        self.ba_history_max_items_var = tk.IntVar(value=5)
        self.ba_show_change_var = tk.BooleanVar(value=True)
        
        ttk.Label(ba_opts_frame, text="Freshness (days):").pack(side=tk.LEFT, padx=(0,5))
        try:
            freshness_spin = ttk.Spinbox(ba_opts_frame, from_=1, to=365, width=5, textvariable=self.ba_freshness_days_var)
        except Exception:
            freshness_spin = tk.Spinbox(ba_opts_frame, from_=1, to=365, width=5, textvariable=self.ba_freshness_days_var)
        freshness_spin.pack(side=tk.LEFT)
        
        ttk.Label(ba_opts_frame, text="History (max):").pack(side=tk.LEFT, padx=(10,5))
        try:
            history_spin = ttk.Spinbox(ba_opts_frame, from_=1, to=20, width=5, textvariable=self.ba_history_max_items_var)
        except Exception:
            history_spin = tk.Spinbox(ba_opts_frame, from_=1, to=20, width=5, textvariable=self.ba_history_max_items_var)
        history_spin.pack(side=tk.LEFT)
        
        ttk.Checkbutton(ba_opts_frame, text="Show Change Over Time", variable=self.ba_show_change_var).pack(side=tk.LEFT, padx=(10,0))
        
        ba_text_frame = ttk.Frame(ba_frame)
        ba_text_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        if sys.platform == "win32":
            font_name = "Consolas"
        elif sys.platform == "darwin":
            font_name = "Menlo"
        else: # linux
            font_name = "DejaVu Sans Mono"

        self.business_analysis_text = tk.Text(ba_text_frame, wrap=tk.WORD, height=20, width=80, font=(font_name, 12))
        self.business_analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ba_scrollbar = ttk.Scrollbar(ba_text_frame, command=self.business_analysis_text.yview)
        ba_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.business_analysis_text.config(yscrollcommand=ba_scrollbar.set)

        # --- Market News Tab Widgets ---
        mn_outer = ttk.Frame(self.market_news_frame, padding="10")
        mn_outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(mn_outer, text="Latest Market News Blog", font=("Helvetica", 12, "bold"))\
            .pack(anchor=tk.W, pady=(0, 5))

        mn_text_frame = ttk.Frame(mn_outer)
        mn_text_frame.pack(fill=tk.BOTH, expand=True)

        if sys.platform == "win32":
            mn_font = "Consolas"
        elif sys.platform == "darwin":
            mn_font = "Menlo"
        else:
            mn_font = "DejaVu Sans Mono"

        self.market_news_text = tk.Text(mn_text_frame, wrap=tk.WORD, height=25, width=90, font=(mn_font, 11))
        self.market_news_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        mn_scroll = ttk.Scrollbar(mn_text_frame, command=self.market_news_text.yview)
        mn_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.market_news_text.config(yscrollcommand=mn_scroll.set)
        
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

        # Use a fixed container to prevent label/image from resizing the parent frame
        self.comparison_chart_container = ttk.Frame(self.comparison_chart_frame)
        self.comparison_chart_container.pack(fill=tk.BOTH, expand=True)
        # Disable geometry propagation so the container size is driven by layout, not image natural size
        try:
            self.comparison_chart_container.pack_propagate(False)
        except Exception:
            pass
        self.comparison_chart_label = ttk.Label(self.comparison_chart_container)
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
        self.ticker_context_menu.add_command(label="Remove Ticker", command=self._remove_ticker)

        # Create right-click context menu for watch list
        self.watch_context_menu = tk.Menu(self.watch_listbox, tearoff=0)
        self.watch_context_menu.add_command(label="Delete from Watch List", command=self._delete_from_watch_list)
        
        # Set initial sash positions after a short delay to ensure the window is fully created
        self.root.after(100, self._set_initial_sash_positions)
        self.root.after(150, self._ensure_bottom_frame_layout)

        # Bind right-click events
        self.ticker_listbox.bind("<Button-3>", self._show_ticker_context_menu)
        self.watch_listbox.bind("<Button-3>", self._show_watch_context_menu)

        # Bind selection events to display charts
        self.ticker_listbox.bind("<ButtonRelease-1>", self._on_ticker_selected)
        self.watch_listbox.bind("<ButtonRelease-1>", self._on_watch_ticker_selected)

        # Bottom frame for actions (already added to vertical paned layout)
        bottom_frame = self.bottom_frame

        # Keep actions and status separated so the status text remains visible within the bottom pane
        # Row 1: Data actions and options
        actions_row1 = ttk.Frame(bottom_frame)
        actions_row1.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))

        # Force download toggle (right side of row 1)
        self.force_download_var = tk.BooleanVar(value=False)
        force_download_check = ttk.Checkbutton(actions_row1, text="Force Download", variable=self.force_download_var)
        force_download_check.pack(side=tk.RIGHT, padx=5)
        ttk.Label(actions_row1, text="Options:").pack(side=tk.RIGHT, padx=5)

        # Data action buttons (left side of row 1)
        ttk.Button(actions_row1, text="Download/Update Data", command=self._download_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_row1, text="Visualize Daily/Weekly/Monthly", command=self._visualize_all_timeframes).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_row1, text="View HTML Report", command=self._view_html_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_row1, text="Compare % Performance", command=self._compare_percentage_performance).pack(side=tk.LEFT, padx=5)

        # Row 2: News summarization buttons
        actions_row2 = ttk.Frame(bottom_frame)
        actions_row2.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))

        ttk.Label(actions_row2, text="Summarize:").pack(side=tk.LEFT, padx=(5, 2))
        ttk.Button(actions_row2, text="Market News", command=self._summarize_market_news).pack(side=tk.LEFT, padx=3)
        ttk.Button(actions_row2, text="Stock News", command=self._summarize_stock_news).pack(side=tk.LEFT, padx=3)
        ttk.Button(actions_row2, text="ETF News", command=self._summarize_etf_news).pack(side=tk.LEFT, padx=3)
        ttk.Button(actions_row2, text="Crypto News", command=self._summarize_crypto_news).pack(side=tk.LEFT, padx=3)
        ttk.Button(actions_row2, text="📋 Clipboard", command=self._summarize_clipboard_content).pack(side=tk.LEFT, padx=3)

        # Status bar hosted inside the bottom pane to prevent it from being obscured
        self.status_var = tk.StringVar(value="Ready")

        status_style = ttk.Style()
        status_style.configure(
            "StatusBar.TLabel",
            padding=(10, 3),
            relief=tk.SUNKEN,
            background="#f2f2f2",
        )

        status_frame = ttk.Frame(bottom_frame)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))
        ttk.Separator(bottom_frame, orient=tk.HORIZONTAL).pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0))

        status_bar = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            style="StatusBar.TLabel",
            anchor=tk.W,
        )
        status_bar.pack(fill=tk.X)

        # Sash initialization is handled via the <Configure> binding above so
        # the action bar stays compact by default while remaining resizable.

    def _ensure_bottom_frame_layout(self):
        """Keep the bottom action bar attached to the vertical paned window."""
        paned = getattr(self, "layout_paned", None)
        bottom = getattr(self, "bottom_frame", None)
        if not paned or not bottom:
            return
        if not paned.winfo_exists() or not bottom.winfo_exists():
            return

        parent_widget = None
        try:
            parent_name = bottom.winfo_parent()
            if parent_name:
                parent_widget = self.root.nametowidget(parent_name)
        except Exception:
            parent_widget = None

        if parent_widget is not paned:
            # Detach from whatever geometry manager was applied and re-add to the paned window
            try:
                bottom.pack_forget()
            except Exception:
                pass
            try:
                paned.add(bottom, weight=0)
            except Exception:
                try:
                    paned.forget(bottom)
                except Exception:
                    pass
                paned.add(bottom, weight=0)

        try:
            paned.paneconfig(bottom, minsize=55)
        except Exception:
            pass

        try:
            paned.update_idletasks()
            total_height = paned.winfo_height()
            bottom_height = bottom.winfo_reqheight() or 0
            if total_height > 0:
                desired = max(total_height - bottom_height - 4, 0)
                paned.sashpos(0, desired)
        except Exception as exc:
            logging.debug("Bottom sash adjustment skipped: %s", exc)

    def _on_list_selected(self, event):
        """Handle ticker list selection and auto-load the selected list"""
        selected_list = self.ticker_list_var.get()
        if selected_list in self.ticker_lists:
            self.status_var.set(f"Selected list: {selected_list} with {len(self.ticker_lists[selected_list])} tickers")
            # Auto-load the selected ticker list
            self._load_ticker_list()

    def _go_prev_list(self):
        try:
            values = list(self.ticker_list_combo['values']) if self.ticker_list_combo else []
            if not values:
                messagebox.showwarning("No Lists", "There are no ticker lists to navigate.")
                return
            current = self.ticker_list_var.get()
            idx = values.index(current) if current in values else 0
            new_idx = (idx - 1) % len(values)
            new_name = values[new_idx]
            self.ticker_list_var.set(new_name)
            self._load_ticker_list()
            self.status_var.set(f"Switched to list: {new_name} ({len(self.ticker_lists.get(new_name, []))} tickers)")
        except Exception as e:
            logging.error(f"Error going to previous list: {e}")

    def _go_next_list(self):
        try:
            values = list(self.ticker_list_combo['values']) if self.ticker_list_combo else []
            if not values:
                messagebox.showwarning("No Lists", "There are no ticker lists to navigate.")
                return
            current = self.ticker_list_var.get()
            idx = values.index(current) if current in values else -1
            new_idx = (idx + 1) % len(values)
            new_name = values[new_idx]
            self.ticker_list_var.set(new_name)
            self._load_ticker_list()
            self.status_var.set(f"Switched to list: {new_name} ({len(self.ticker_lists.get(new_name, []))} tickers)")
        except Exception as e:
            logging.error(f"Error going to next list: {e}")

    def _open_live_charts_for_current_list(self, time_frame="d"):
        try:
            selected_list = self.ticker_list_var.get()
            if not selected_list or selected_list not in self.ticker_lists:
                messagebox.showwarning("No List Selected", "Please select a ticker list first.")
                return
            tickers = [t for t in self.ticker_lists.get(selected_list, []) if isinstance(t, str) and t.strip()]
            if not tickers:
                messagebox.showwarning("Empty List", "The selected ticker list is empty.")
                return
            columns = 4
            output_path = os.path.join(tempfile.gettempdir(), f"{selected_list}_stock_charts.html")
            generate_chart_html(tickers, columns, output_path, time_frame=time_frame)
            try:
                webbrowser.register('edge', None, webbrowser.BackgroundBrowser(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'))
                webbrowser.get('edge').open(f"file:///{os.path.abspath(output_path)}")
                self.status_var.set(f"Live charts for {selected_list} opened in Edge browser")
            except Exception as browser_error:
                logging.warning(f"Could not open Edge browser: {browser_error}. Using default browser.")
                webbrowser.open(f"file:///{os.path.abspath(output_path)}")
                self.status_var.set(f"Live charts for {selected_list} opened in default browser")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating live charts: {str(e)}")
            self.status_var.set("Error generating live charts")

    def _open_multi_timeframe_gallery_for_current_list(self):
        try:
            selected_list = self.ticker_list_var.get()
            if not selected_list or selected_list not in self.ticker_lists:
                messagebox.showwarning("No List Selected", "Please select a ticker list first.")
                return
            tickers = [t for t in self.ticker_lists.get(selected_list, []) if isinstance(t, str) and t.strip()]
            if not tickers:
                messagebox.showwarning("Empty List", "The selected ticker list is empty.")
                return
            output_path = os.path.join(tempfile.gettempdir(), f"{selected_list}_multi_timeframe_charts.html")
            generate_multi_timeframe_chart_html(tickers, output_filename=output_path)
            try:
                webbrowser.register('edge', None, webbrowser.BackgroundBrowser(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'))
                webbrowser.get('edge').open(f"file:///{os.path.abspath(output_path)}")
                self.status_var.set(f"Multi-timeframe gallery for {selected_list} opened in Edge browser")
            except Exception as browser_error:
                logging.warning(f"Could not open Edge browser: {browser_error}. Using default browser.")
                webbrowser.open(f"file:///{os.path.abspath(output_path)}")
                self.status_var.set(f"Multi-timeframe gallery for {selected_list} opened in default browser")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating multi-timeframe gallery: {str(e)}")
            self.status_var.set("Error generating multi-timeframe gallery")

    def _open_stockcharts_gallery_for_current_list(self):
        """Generate and open StockCharts.com gallery for the current ticker list"""
        try:
            selected_list = self.ticker_list_var.get()
            if not selected_list or selected_list not in self.ticker_lists:
                messagebox.showwarning("No List Selected", "Please select a ticker list first.")
                return
            
            tickers = [t for t in self.ticker_lists.get(selected_list, []) if isinstance(t, str) and t.strip()]
            if not tickers:
                messagebox.showwarning("Empty List", "The selected ticker list is empty.")
                return
            
            # Generate HTML file in temp directory
            output_path = os.path.join(tempfile.gettempdir(), f"{selected_list}_stockcharts_gallery.html")
            generate_multi_timeframe_stockcharts_html(tickers, output_filename=output_path)
            
            # Open in browser
            try:
                webbrowser.register('edge', None, webbrowser.BackgroundBrowser(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'))
                webbrowser.get('edge').open(f"file:///{os.path.abspath(output_path)}")
                self.status_var.set(f"StockCharts gallery for {selected_list} ({len(tickers)} tickers) opened in Edge browser")
            except Exception as browser_error:
                logging.warning(f"Could not open Edge browser: {browser_error}. Using default browser.")
                webbrowser.open(f"file:///{os.path.abspath(output_path)}")
                self.status_var.set(f"StockCharts gallery for {selected_list} ({len(tickers)} tickers) opened in default browser")
                
            logging.info(f"Generated StockCharts gallery for {selected_list} with {len(tickers)} tickers")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating StockCharts gallery: {str(e)}")
            self.status_var.set("Error generating StockCharts gallery")
            logging.error(f"Error generating StockCharts gallery: {e}")

    def _open_stockcharts_line_gallery_for_current_list(self):
        """Generate and open StockCharts.com line-chart gallery for the current ticker list"""
        try:
            selected_list = self.ticker_list_var.get()
            if not selected_list or selected_list not in self.ticker_lists:
                messagebox.showwarning("No List Selected", "Please select a ticker list first.")
                return

            tickers = [t for t in self.ticker_lists.get(selected_list, []) if isinstance(t, str) and t.strip()]
            if not tickers:
                messagebox.showwarning("Empty List", "The selected ticker list is empty.")
                return

            # Generate HTML file in temp directory
            output_path = os.path.join(tempfile.gettempdir(), f"{selected_list}_stockcharts_line_gallery.html")

            # Use the current StockCharts style ID from the GUI entry (fallback to default if empty)
            style_id = getattr(self, "stockcharts_line_style_var", None)
            style_id_value = style_id.get().strip() if style_id is not None else "t3327397499c"
            if not style_id_value:
                style_id_value = "t3327397499c"

            generate_multi_timeframe_stockcharts_line_html(
                tickers,
                output_filename=output_path,
                style_id=style_id_value,
            )

            # Open in browser
            try:
                webbrowser.register('edge', None, webbrowser.BackgroundBrowser(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'))
                webbrowser.get('edge').open(f"file:///{os.path.abspath(output_path)}")
                self.status_var.set(f"StockCharts line gallery for {selected_list} ({len(tickers)} tickers) opened in Edge browser")
            except Exception as browser_error:
                logging.warning(f"Could not open Edge browser: {browser_error}. Using default browser.")
                webbrowser.open(f"file:///{os.path.abspath(output_path)}")
                self.status_var.set(f"StockCharts line gallery for {selected_list} ({len(tickers)} tickers) opened in default browser")

            logging.info(f"Generated StockCharts line gallery for {selected_list} with {len(tickers)} tickers")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating StockCharts line gallery: {str(e)}")
            self.status_var.set("Error generating StockCharts line gallery")
            logging.error(f"Error generating StockCharts line gallery: {e}")

    def _open_linecharts_gallery_for_current_list(self):
        """Generate and open Finviz line chart gallery for the current ticker list"""
        try:
            selected_list = self.ticker_list_var.get()
            if not selected_list or selected_list not in self.ticker_lists:
                messagebox.showwarning("No List Selected", "Please select a ticker list first.")
                return

            tickers = [t for t in self.ticker_lists.get(selected_list, []) if isinstance(t, str) and t.strip()]
            if not tickers:
                messagebox.showwarning("Empty List", "The selected ticker list is empty.")
                return

            # Generate HTML file in temp directory
            output_path = os.path.join(tempfile.gettempdir(), f"{selected_list}_finviz_linecharts_gallery.html")
            generate_multi_timeframe_linechart_html(tickers, output_filename=output_path)

            # Open in browser
            try:
                webbrowser.register('edge', None, webbrowser.BackgroundBrowser(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'))
                webbrowser.get('edge').open(f"file:///{os.path.abspath(output_path)}")
                self.status_var.set(f"Finviz line chart gallery for {selected_list} ({len(tickers)} tickers) opened in Edge browser")
            except Exception as browser_error:
                logging.warning(f"Could not open Edge browser: {browser_error}. Using default browser.")
                webbrowser.open(f"file:///{os.path.abspath(output_path)}")
                self.status_var.set(f"Finviz line chart gallery for {selected_list} ({len(tickers)} tickers) opened in default browser")

            logging.info(f"Generated Finviz line chart gallery for {selected_list} with {len(tickers)} tickers")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating Finviz line chart gallery: {str(e)}")
            self.status_var.set("Error generating Finviz line chart gallery")
            logging.error(f"Error generating Finviz line chart gallery: {e}")

    def _apply_ticker_filter(self, *args):
        """Filter the ticker list based on filter text
        
        Spaces in filter text are treated as OR logic, allowing multiple search terms
        Example: 'A TSL' will match tickers containing either 'A' or 'TSL'
        """
        filter_text = self.filter_var.get().strip().upper()

        # If no current tickers or no filter, don't do anything
        if not hasattr(self, 'current_tickers') or not self.current_tickers:
            return

        # Get the currently selected list
        selected_list = self.ticker_list_var.get()
        if not selected_list or selected_list not in self.ticker_lists:
            return
        
        # Get the tickers for the selected list
        tickers = self.ticker_lists[selected_list]
        
        # Clear the listbox
        self.ticker_listbox.delete(0, tk.END)
        
        # Split filter text by spaces to implement OR logic
        filter_terms = filter_text.split() if filter_text else []
        
        # Filter tickers based on input and update listbox
        filtered_count = 0
        for ticker in tickers:
            ticker_upper = ticker.upper()
            # If no filter terms or any of the filter terms match the ticker
            if not filter_terms or any(term in ticker_upper for term in filter_terms):
                self.ticker_listbox.insert(tk.END, ticker)
                filtered_count += 1
        
        # Update status
        if filter_text:
            terms_display = ' OR '.join(f"'{term}'" for term in filter_terms)
            self.status_var.set(f"Filter {terms_display}: showing {filtered_count}/{len(tickers)} tickers from {selected_list}")
        else:
            self.status_var.set(f"Showing all {len(tickers)} tickers from {selected_list}")

    def _load_first_ticker_list(self):
        """Automatically load the first ticker list on startup"""
        if self.ticker_lists:
            # Get the first list name
            first_list = list(self.ticker_lists.keys())[0]
            # Set it as the selected value in the combobox
            self.ticker_list_var.set(first_list)
            # Load the list
            self._load_ticker_list()
            logging.info(f"Auto-loaded first ticker list: {first_list}")

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

        # Get the current selected list name
        current_list_name = self.ticker_list_var.get()
        
        # Save the updated list back to the file if it's a named list (not 'all_tickers') and tickers were added
        if current_list_name and current_list_name != 'all_tickers' and added_count > 0:
            try:
                # Read the current content of ticker_lists.py
                ticker_lists_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_lists.py")
                with open(ticker_lists_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Create the updated list code
                tickers_str = ", ".join([f"\"{ticker}\"" for ticker in self.current_tickers])
                updated_list_code = f"{current_list_name} = [{tickers_str}]\n"
                
                # Find the existing list definition
                list_pattern = re.compile(f"\n{current_list_name}\s*=\s*\[.*?\]", re.DOTALL)
                match = list_pattern.search(content)
                
                if match:
                    # Replace the existing list
                    new_content = content[:match.start()] + f"\n{updated_list_code}" + content[match.end():]
                    
                    # Write the modified content back to the file
                    with open(ticker_lists_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    
                    # Update the ticker lists dictionary
                    self.ticker_lists[current_list_name] = self.current_tickers
                    
                    logging.info(f"Updated {current_list_name} with {len(self.current_tickers)} tickers in ticker_lists.py")
                else:
                    logging.warning(f"Could not find {current_list_name} in ticker_lists.py to update")
            except Exception as e:
                messagebox.showerror("Error", f"Error updating ticker list: {str(e)}")
                logging.error(f"Error updating ticker list: {e}")

        if added_count == 1:
            self.status_var.set(f"Added ticker: {tickers[0]}")
        else:
            self.status_var.set(f"Added {added_count} tickers")

        # Clear entry field
        self.manual_ticker_var.set("")

    def _save_ticker_list(self):
        """Create a new ticker list in ticker_lists.py and load it immediately"""
        list_name = self.list_name_var.get().strip()
        if not list_name:
            messagebox.showwarning("No List Name", "Please enter a name for the ticker list.")
            return

        # Format list name to be a valid Python variable name
        list_name = list_name.replace(" ", "_").replace("-", "_")
        if not list_name[0].isalpha() and list_name[0] != '_':
            list_name = "ticker_" + list_name
            
        # Add _stocks suffix if not already present
        if not list_name.endswith("_stocks"):
            list_name = f"{list_name}_stocks"

        # Create Python code for the new empty list
        new_list_code = f"\n{list_name} = []\n"

        try:
            # Check if list already exists
            if list_name in self.ticker_lists:
                if not messagebox.askyesno("List Exists", f"The list '{list_name}' already exists. Do you want to overwrite it?"): 
                    return
            
            # Read the current content of ticker_lists.py
            ticker_lists_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_lists.py")
            with open(ticker_lists_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check if the list already exists in the file
            list_pattern = re.compile(f"\n{list_name}\s*=\s*\[.*?\]", re.DOTALL)
            match_existing = list_pattern.search(content)
            
            if match_existing:
                # Replace the existing list
                new_content = content[:match_existing.start()] + f"\n{list_name} = []" + content[match_existing.end():]
                
                # Write the modified content back to the file
                with open(ticker_lists_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
            else:
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

            # Update the ticker lists dictionary with an empty list
            self.ticker_lists[list_name] = []
            
            # Clear the current tickers and update the listbox
            self.current_tickers = []
            self.ticker_listbox.delete(0, tk.END)
            
            # Set the newly created list as the current selection
            self.ticker_list_var.set(list_name)

            # Update the dropdown menu
            self.ticker_list_combo['values'] = list(self.ticker_lists.keys())

            self.status_var.set(f"Created empty list '{list_name}' and loaded it")
            messagebox.showinfo("List Created", f"Empty ticker list '{list_name}' created and loaded")
            
            # Clear the list name entry
            self.list_name_var.set("")
        except Exception as e:
            messagebox.showerror("Error", f"Error creating ticker list: {str(e)}")
            logging.error(f"Error creating ticker list: {e}")

    def _setup_ticker_context_menu(self):
        """Set up the context menu for the ticker listbox"""
        # Bind right-click event
        self.ticker_listbox.bind("<Button-3>", self._show_ticker_context_menu)
    
    def _setup_watch_context_menu(self):
        """Set up the context menu for the watch listbox"""
        # Bind right-click event
        self.watch_listbox.bind("<Button-3>", self._show_watch_context_menu)
    
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
        
        # Convert to list and sort in reverse order to avoid index shifting during deletion
        indices = sorted(list(selected_indices), reverse=True)
        
        # Remove from listbox and watch_list
        for i in indices:
            ticker = self.watch_listbox.get(i).strip()
            self.watch_listbox.delete(i)
            if ticker in self.watch_list:
                self.watch_list.remove(ticker)
                
        # Save updated watch list
        self._save_watch_list()

    def _get_selected_tickers(self, show_warning=True):
        """Get selected tickers preferring the main list; fall back to watch list if needed."""
        selected_indices = self.ticker_listbox.curselection()
        if not selected_indices:
            # Fall back to watch list selection if present
            try:
                watch_indices = self.watch_listbox.curselection()
            except Exception:
                watch_indices = ()

            if watch_indices:
                selected_tickers = []
                for i in watch_indices:
                    ticker_text = self.watch_listbox.get(i)
                    ticker = ticker_text.split(' - ')[0].strip()
                    selected_tickers.append(ticker)
                return selected_tickers

            if show_warning:
                messagebox.showwarning("No Selection", "Please select at least one ticker.")
            return []

        selected_tickers = []
        for i in selected_indices:
            # Extract ticker symbol (it might include a comment after a dash)
            ticker_text = self.ticker_listbox.get(i)
            ticker = ticker_text.split(' - ')[0].strip()
            selected_tickers.append(ticker)

        return selected_tickers

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
            
    def _remove_ticker(self):
        """Remove selected tickers from the available tickers list and save changes"""
        selected_tickers = self._get_selected_tickers(show_warning=False)
        if not selected_tickers:
            return
            
        # Ask for confirmation
        if len(selected_tickers) == 1:
            message = f"Are you sure you want to remove {selected_tickers[0]} from the available tickers?"
        else:
            message = f"Are you sure you want to remove {len(selected_tickers)} tickers from the available tickers?"
            
        if not messagebox.askyesno("Confirm Removal", message):
            return
            
        # Convert to list and sort in reverse order to avoid index shifting during deletion
        selected_indices = sorted(list(self.ticker_listbox.curselection()), reverse=True)
        
        # Remove from listbox and current_tickers list
        for i in selected_indices:
            ticker_text = self.ticker_listbox.get(i)
            ticker = ticker_text.split(' - ')[0].strip()
            self.ticker_listbox.delete(i)
            if ticker in self.current_tickers:
                self.current_tickers.remove(ticker)
        
        # Get the current selected list name
        current_list_name = self.ticker_list_var.get()
        
        # Save the updated list back to the file if it's a named list (not 'all_tickers')
        if current_list_name and current_list_name != 'all_tickers':
            try:
                # Read the current content of ticker_lists.py
                ticker_lists_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_lists.py")
                with open(ticker_lists_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Create the updated list code
                tickers_str = ", ".join([f"\"{ticker}\"" for ticker in self.current_tickers])
                updated_list_code = f"{current_list_name} = [{tickers_str}]\n"
                
                # Find the existing list definition
                list_pattern = re.compile(f"\n{current_list_name}\s*=\s*\[.*?\]", re.DOTALL)
                match = list_pattern.search(content)
                
                if match:
                    # Replace the existing list
                    new_content = content[:match.start()] + f"\n{updated_list_code}" + content[match.end():]
                    
                    # Write the modified content back to the file
                    with open(ticker_lists_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    
                    # Update the ticker lists dictionary
                    self.ticker_lists[current_list_name] = self.current_tickers
                    
                    logging.info(f"Updated {current_list_name} with {len(self.current_tickers)} tickers in ticker_lists.py")
                else:
                    logging.warning(f"Could not find {current_list_name} in ticker_lists.py to update")
            except Exception as e:
                messagebox.showerror("Error", f"Error updating ticker list: {str(e)}")
                logging.error(f"Error updating ticker list: {e}")
                
        # Update status
        if len(selected_tickers) == 1:
            self.status_var.set(f"Removed {selected_tickers[0]} from available tickers")
        else:
            self.status_var.set(f"Removed {len(selected_tickers)} tickers from available tickers")

    def _move_ticker_up(self):
        """Move selected ticker up in the list"""
        selected_indices = self.ticker_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select a ticker to move.")
            return
        
        # Only move the first selected ticker
        index = selected_indices[0]
        
        # Can't move up if already at the top
        if index == 0:
            return
        
        # Get the ticker at the current position
        ticker = self.ticker_listbox.get(index)
        
        # Remove from current position
        self.ticker_listbox.delete(index)
        self.current_tickers.pop(index)
        
        # Insert at new position (one up)
        new_index = index - 1
        self.ticker_listbox.insert(new_index, ticker)
        self.current_tickers.insert(new_index, ticker)
        
        # Select the ticker at its new position
        self.ticker_listbox.selection_set(new_index)
        self.ticker_listbox.see(new_index)
        
        # Save the updated list
        self._save_current_ticker_list_order()
        
    def _move_ticker_down(self):
        """Move selected ticker down in the list"""
        selected_indices = self.ticker_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select a ticker to move.")
            return
        
        # Only move the first selected ticker
        index = selected_indices[0]
        
        # Can't move down if already at the bottom
        if index >= self.ticker_listbox.size() - 1:
            return
        
        # Get the ticker at the current position
        ticker = self.ticker_listbox.get(index)
        
        # Remove from current position
        self.ticker_listbox.delete(index)
        self.current_tickers.pop(index)
        
        # Insert at new position (one down)
        new_index = index + 1
        self.ticker_listbox.insert(new_index, ticker)
        self.current_tickers.insert(new_index, ticker)
        
        # Select the ticker at its new position
        self.ticker_listbox.selection_set(new_index)
        self.ticker_listbox.see(new_index)
        
        # Save the updated list
        self._save_current_ticker_list_order()
        
    def _sort_tickers(self):
        """Sort tickers alphabetically"""
        if not self.current_tickers:
            return
        
        # Sort the current tickers list
        self.current_tickers.sort()
        
        # Clear and repopulate the listbox
        self.ticker_listbox.delete(0, tk.END)
        for ticker in self.current_tickers:
            self.ticker_listbox.insert(tk.END, ticker)
        
        # Save the updated list
        self._save_current_ticker_list_order()
        
        self.status_var.set(f"Sorted {len(self.current_tickers)} tickers alphabetically")
        
    def _save_current_ticker_list_order(self):
        """Save the current ticker list order to ticker_lists.py"""
        current_list_name = self.ticker_list_var.get()
        
        # Only save if it's a named list (not 'all_tickers')
        if not current_list_name or current_list_name == 'all_tickers':
            return
        
        try:
            # Read the current content of ticker_lists.py
            ticker_lists_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_lists.py")
            with open(ticker_lists_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Create the updated list code
            tickers_str = ", ".join([f"\"{ticker}\"" for ticker in self.current_tickers])
            updated_list_code = f"{current_list_name} = [{tickers_str}]\n"
            
            # Find the existing list definition
            list_pattern = re.compile(f"\n{current_list_name}\s*=\s*\[.*?\]", re.DOTALL)
            match = list_pattern.search(content)
            
            if match:
                # Replace the existing list
                new_content = content[:match.start()] + f"\n{updated_list_code}" + content[match.end():]
                
                # Write the modified content back to the file
                with open(ticker_lists_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                
                # Update the ticker lists dictionary
                self.ticker_lists[current_list_name] = self.current_tickers.copy()
                
                logging.info(f"Updated {current_list_name} order in ticker_lists.py")
            else:
                logging.warning(f"Could not find {current_list_name} in ticker_lists.py to update")
        except Exception as e:
            messagebox.showerror("Error", f"Error updating ticker list: {str(e)}")
            logging.error(f"Error updating ticker list: {e}")

    def _remove_current_list(self):
        """Remove the currently selected ticker list from ticker_lists.py"""
        # Get the current selected list name
        current_list_name = self.ticker_list_var.get()
        
        # Check if a list is selected
        if not current_list_name:
            messagebox.showwarning("No List Selected", "Please select a ticker list to remove.")
            return
            
        # Don't allow removing watch_list through this method (it has its own management)
        if current_list_name == "watch_list":
            messagebox.showwarning("Cannot Remove", "The watch list cannot be removed through this button.\n\nTo clear the watch list, remove all tickers from it.")
            return
            
        # Ask for confirmation
        if not messagebox.askyesno("Confirm Removal", f"Are you sure you want to remove the list '{current_list_name}'?\n\nThis action cannot be undone."):
            return
            
        try:
            # Read the current content of ticker_lists.py
            ticker_lists_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_lists.py")
            with open(ticker_lists_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Find the existing list definition
            list_pattern = re.compile(f"\n{current_list_name}\s*=\s*\[.*?\]", re.DOTALL)
            match = list_pattern.search(content)
            
            if match:
                # Remove the existing list
                new_content = content[:match.start()] + content[match.end():]
                
                # Write the modified content back to the file
                with open(ticker_lists_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                
                # Remove from the ticker lists dictionary
                if current_list_name in self.ticker_lists:
                    del self.ticker_lists[current_list_name]
                
                # Update the dropdown values
                self.ticker_list_combo['values'] = list(self.ticker_lists.keys())
                
                # Clear the current selection if it was the removed list
                if self.ticker_list_var.get() == current_list_name:
                    self.ticker_list_var.set("")
                    # Clear the ticker listbox
                    self.ticker_listbox.delete(0, tk.END)
                    self.current_tickers = []
                
                # Update status
                self.status_var.set(f"Removed list '{current_list_name}' from ticker_lists.py")
                logging.info(f"Removed list '{current_list_name}' from ticker_lists.py")
            else:
                messagebox.showwarning("List Not Found", f"Could not find list '{current_list_name}' in ticker_lists.py")
                
        except Exception as e:
            error_msg = f"Error removing ticker list: {str(e)}"
            messagebox.showerror("Error", error_msg)
            logging.error(error_msg)
    
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

    def _get_selected_tickers(self, show_warning=True):
        """Get selected tickers preferring the main list; fall back to watch list if needed."""
        selected_indices = self.ticker_listbox.curselection()
        if not selected_indices:
            # Fall back to watch list selection if present
            try:
                watch_indices = self.watch_listbox.curselection()
            except Exception:
                watch_indices = ()

            if watch_indices:
                selected_tickers = []
                for i in watch_indices:
                    ticker_text = self.watch_listbox.get(i)
                    ticker = ticker_text.split(' - ')[0].strip()
                    selected_tickers.append(ticker)
                return selected_tickers

            if show_warning:
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
        """Download or update data for selected tickers using background thread"""
        selected_tickers = self._get_selected_tickers()
        if not selected_tickers:
            return
            
        # Get force download setting
        force_download = self.force_download_var.get()
        
        # Use the background download functionality instead of blocking the UI thread
        self._download_data_with_force_option(selected_tickers, force_download)
        
    def _download_data_with_force_option(self, tickers, force_download=False):
        """Download data for multiple tickers in a background thread with force option
        
        Args:
            tickers (list): List of ticker symbols to download data for
            force_download (bool): Whether to force download new data
        """
        if not tickers:
            return
            
        mode_text = "force downloading" if force_download else "updating"
        self.status_var.set(f"{mode_text.capitalize()} data for {len(tickers)} tickers in background...")
        self.root.update_idletasks()
        
        # Create a queue to store results
        self.download_queue = Queue()
        
        # Create and start the download thread
        download_thread = threading.Thread(
            target=self._download_worker_with_force_option,
            args=(tickers, self.download_queue, force_download),
            daemon=True
        )
        download_thread.name = f"DownloadThread-{'-'.join(tickers[:2])}"
        download_thread.start()
        
        # Schedule periodic checks for download completion
        self._check_download_progress(download_thread, len(tickers))
        
        # Show a message to the user that download is happening in background
        messagebox.showinfo("Download Started", 
                          f"{mode_text.capitalize()} data for {len(tickers)} tickers in the background.\n\n"
                          f"The status bar will update as the process completes.")
        
    def _download_worker_with_force_option(self, tickers, queue, force_download=False):
        """Worker function to download data for multiple tickers with force option
        
        Args:
            tickers (list): List of ticker symbols to download data for
            queue (Queue): Queue to store results
            force_download (bool): Whether to force download new data
        """
        try:
            total = len(tickers)
            completed = 0
            success_count = 0
            mode_text = "force downloading" if force_download else "updating"
            
            for ticker in tickers:
                try:
                    # Update status in the queue
                    queue.put(("status", f"{mode_text.capitalize()} data for {ticker}... ({completed}/{total})"))
                    
                    # Download data with specified force_download option
                    data = self.manager.update_data(ticker, force_download=force_download)
                    
                    # Check if data was successfully retrieved
                    if data is not None and not data.empty:
                        success_count += 1
                        
                    # Update completed count
                    completed += 1
                    queue.put(("progress", completed))
                    queue.put(("success_count", success_count))
                    
                except Exception as e:
                    # Report error for this ticker
                    queue.put(("error", f"Error {mode_text} data for {ticker}: {str(e)}"))
                    completed += 1
            
            # Signal completion
            queue.put(("complete", f"Completed: {mode_text.capitalize()} data for {success_count}/{total} tickers"))
            
        except Exception as e:
            # Report critical error
            queue.put(("critical", f"Critical error in download thread: {str(e)}"))

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

                # Generate the chart
                self.manager.visualize_daily_vs_weekly(ticker)

                # Get the path to the generated chart
                chart_path = os.path.join(self.manager.plot_save_path, f"{ticker}_daily_weekly_monthly.png")

                # Display the chart in the GUI
                if os.path.exists(chart_path):
                    # Switch to the individual chart tab before displaying
                    self.chart_notebook.select(self.individual_chart_frame)
                    self._display_chart(chart_path)
                else:
                    messagebox.showerror("Error", f"Chart file not found for {ticker}")

            except Exception as e:
                messagebox.showerror("Error", f"Error visualizing {ticker}: {str(e)}")

        self.status_var.set(f"Completed visualization for {len(selected_tickers)} tickers")

    def _generate_seasonality_chart(self, ticker):
        """
        Generate and display interactive seasonality chart with multi-year selection.
        
        Args:
            ticker (str): Ticker symbol to generate seasonality chart for.
        """
        try:
            if not hasattr(self, 'root') or not self.root.winfo_exists():
                return

            is_new_ticker = (ticker != self.seasonality_chart_ticker)
            self.seasonality_chart_ticker = ticker

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
                # Convert Close column to float to avoid type mismatch
                year_df['Close'] = pd.to_numeric(year_df['Close'], errors='coerce')
                # Drop any rows where conversion failed
                year_df = year_df.dropna(subset=['Close'])
                if len(year_df) < 30: continue
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
        container = self.seasonality_chart_container
        if not container.winfo_exists():
            logging.warning("Cannot update seasonality chart container: widget no longer exists")
            return

        # Save to HTML for the 'Open in Browser' button
        temp_dir = tempfile.gettempdir()
        html_path = os.path.join(temp_dir, "stock_chart_seasonality.html")
        
        # Try to generate HTML file for browser viewing
        try:
            # Use plotly.offline.plot to generate HTML
            plot(fig, filename=html_path, auto_open=False)
            self.seasonality_browser_button.config(command=lambda: webbrowser.open(f"file:///{html_path}"), state="normal")
        except Exception as html_e:
            logging.error(f"Error generating HTML for seasonality chart: {html_e}")
            self.seasonality_browser_button.config(state="disabled")

        # Try to generate static image preview
        try:
            # Check if kaleido is available by trying to import it
            import kaleido
            img_bytes = pio.to_image(fig, format='png', width=1200, height=600)
            self.seasonality_pil_img = Image.open(io.BytesIO(img_bytes))
            
            # Trigger resize to fit current window
            class FakeEvent:
                def __init__(self, w, h): self.width = w; self.height = h
            self.root.update_idletasks()
            self._on_seasonality_resize(FakeEvent(container.winfo_width(), container.winfo_height()))
            
            self.status_var.set(f"Generated seasonality chart for {self.current_chart_ticker}")
        except (ImportError, ModuleNotFoundError) as ke:
            logging.warning(f"Kaleido package not installed, cannot generate static image: {ke}")
            self.seasonality_pil_img = None
            self.seasonality_img_label.config(image="", text="Chart preview not available.\n(Requires 'kaleido' package).")
            # Don't show warning if it's just a missing package
            self.status_var.set(f"Generated seasonality chart for {self.current_chart_ticker} (browser only)")
        except Exception as img_e:
            logging.warning(f"Could not generate static image for seasonality chart: {img_e}")
            self.seasonality_pil_img = None
            self.seasonality_img_label.config(image="", text="Chart preview not available.\n(Error generating image)")
            messagebox.showwarning("Preview Generation Failed", f"Error generating chart preview: {img_e}")

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
        """Handle ticker selection event from main ticker listbox, including deselection."""
        try:
            selected_indices = self.ticker_listbox.curselection()
            if not selected_indices:
                return

            selected_tickers = [self.ticker_listbox.get(i).split(' ')[0].strip() for i in selected_indices]
            logging.info(f"Selected tickers from main list: {selected_tickers}")

            current_tab_index = self.chart_notebook.index("current")

            # If Buffett & CANSLIM tab is active, use cache if available; do not auto re-run
            if hasattr(self, 'buffett_canslim_frame') and str(self.chart_notebook.select()) == str(self.buffett_canslim_frame):
                if selected_tickers:
                    ticker = selected_tickers[0]
                    if self._bc_base_image is not None and self._bc_last_ticker == ticker:
                        try:
                            self._show_cached_buffett_canslim()
                        except Exception:
                            pass
                    else:
                        self.bc_text.delete("1.0", tk.END)
                        self.bc_text.insert(tk.END, f"Ready to analyze {ticker}. Click Analyze Selected.")
                        self.bc_status_var.set("Idle. Not re-running automatically.")
                return

            # Tab indices are: 0: Individual, 1: Comparison, 2: Seasonality, 3: Fundamental, 4: Business Analysis
            if current_tab_index == 4:
                if selected_tickers:
                    ticker = selected_tickers[0]
                    self._load_cached_analysis(ticker)
            elif current_tab_index == 3:
                self._display_fundamental_data(selected_tickers)
            elif selected_tickers:
                if current_tab_index == 1:
                    self._compare_percentage_performance(tickers=selected_tickers)
                elif current_tab_index == 2:
                    self._generate_seasonality_chart(selected_tickers[0])
                elif current_tab_index == 0:
                    self._display_chart(selected_tickers[0])
        except Exception as e:
            logging.error(f"Error handling ticker selection: {e}")

    def _on_watch_ticker_selected(self, event):
        """Handle ticker selection event from watch list, including deselection."""
        try:
            selected_indices = self.watch_listbox.curselection()
            if not selected_indices:
                return

            selected_tickers = [self.watch_listbox.get(i).strip() for i in selected_indices]
            logging.info(f"Selected tickers from watch list: {selected_tickers}")

            current_tab_index = self.chart_notebook.index("current")

            # If Buffett & CANSLIM tab is active, run analysis and return
            if hasattr(self, 'buffett_canslim_frame') and str(self.chart_notebook.select()) == str(self.buffett_canslim_frame):
                if selected_tickers:
                    self._analyze_buffett_canslim_current()
                return

            # Tab indices are: 0: Individual, 1: Comparison, 2: Seasonality, 3: Fundamental, 4: Business Analysis
            if current_tab_index == 4:
                if selected_tickers:
                    ticker = selected_tickers[0]
                    self._load_cached_analysis(ticker)
            elif current_tab_index == 3:
                self._display_fundamental_data(selected_tickers)
            elif selected_tickers:
                if current_tab_index == 1:
                    self._compare_percentage_performance(tickers=selected_tickers)
                elif current_tab_index == 2:
                    self._generate_seasonality_chart(selected_tickers[0])
                elif current_tab_index == 0:
                    self._display_chart(selected_tickers[0])
        except Exception as e:
            logging.error(f"Error handling watch ticker selection: {e}")

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

    def _set_quick_range(self, days: int):
        """Set the date range to [today - days, today] and apply it.

        Args:
            days: Number of days back from today for the start date.
        """
        try:
            today = datetime.today().date()
            start = today - timedelta(days=days)
            if hasattr(self, 'start_date_var'):
                self.start_date_var.set(start.strftime('%Y-%m-%d'))
            if hasattr(self, 'end_date_var'):
                self.end_date_var.set(today.strftime('%Y-%m-%d'))
            # Apply immediately
            self._apply_date_range()
        except Exception as e:
            logging.error(f"Error setting quick date range: {e}")

    def _reset_date_range(self):
        """Reset start/end dates to use the maximum available data range and refresh chart."""
        try:
            # Clear UI date fields
            if hasattr(self, 'start_date_var'):
                self.start_date_var.set("")
            if hasattr(self, 'end_date_var'):
                self.end_date_var.set("")

            # Clear manager filters so full range is used
            if hasattr(self, 'manager'):
                self.manager.start_date = None
                self.manager.end_date = None

            # Refresh based on active tab and selection
            if self.active_tab == "comparison":
                self.status_var.set("Reset date range. Using full history for comparison chart.")
                self._compare_percentage_performance()
                return

            # Try to refresh currently selected single ticker chart
            ticker = self._get_selected_single_ticker()
            if ticker:
                self.status_var.set(f"Reset date range. Using full history for {ticker}.")
                if self.active_tab == "seasonality":
                    self._generate_seasonality_chart(ticker)
                else:
                    self._display_chart(ticker)
            else:
                self.status_var.set("Reset date range. Select a ticker to display chart with full history.")
        except Exception as e:
            logging.error(f"Error resetting date range: {e}")
            messagebox.showerror("Error", f"Error resetting date range: {str(e)}")

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
                elif hasattr(self, 'fundamental_analysis_frame') and self.fundamental_analysis_frame.winfo_exists() and \
                        selected_tab == str(self.fundamental_analysis_frame):
                    self.active_tab = "fundamental"
                    logging.info("Switched to fundamental analysis tab")
                elif hasattr(self, 'business_analysis_frame') and self.business_analysis_frame.winfo_exists() and \
                        selected_tab == str(self.business_analysis_frame):
                    self.active_tab = "business_analysis"
                    logging.info("Switched to business analysis tab")
                elif hasattr(self, 'buffett_canslim_frame') and self.buffett_canslim_frame.winfo_exists() and \
                        selected_tab == str(self.buffett_canslim_frame):
                    self.active_tab = "buffett_canslim"
                    logging.info("Switched to Buffett & CANSLIM tab")
                elif hasattr(self, 'sec_filings_frame') and self.sec_filings_frame.winfo_exists() and \
                        selected_tab == str(self.sec_filings_frame):
                    self.active_tab = "sec_filings"
                    logging.info("Switched to SEC filings tab")
                elif hasattr(self, 'market_news_frame') and self.market_news_frame.winfo_exists() and \
                        selected_tab == str(self.market_news_frame):
                    self.active_tab = "market_news"
                    logging.info("Switched to market news tab")
            except tk.TclError as e:
                logging.error(f"TclError in tab change handler: {str(e)}")
                return

            logging.info(f"Active tab is now: {self.active_tab}")

            # Update view based on the new active tab and current selection
            # Get selected tickers from either listbox without showing a popup
            selected_tickers = []
            selected_indices = self.ticker_listbox.curselection()
            if selected_indices:
                for i in selected_indices:
                    ticker_text = self.ticker_listbox.get(i)
                    selected_tickers.append(ticker_text.split(' ')[0].strip())
            else:
                watch_indices = self.watch_listbox.curselection()
                for i in watch_indices:
                    selected_tickers.append(self.watch_listbox.get(i).strip())

            logging.info(f"Tab changed to {self.active_tab}. Current selection: {selected_tickers}")

            if self.active_tab == "fundamental":
                self._display_fundamental_data(selected_tickers)
            elif self.active_tab == "business_analysis":
                if selected_tickers:
                    ticker = selected_tickers[0]
                    self._load_cached_analysis(ticker)
                    # Use the new method if it exists, otherwise just continue
                    if hasattr(self, '_check_for_cached_10k'):
                        self._check_for_cached_10k(ticker)
                else:
                    self.business_analysis_text.delete("1.0", tk.END)
                    self.business_analysis_text.insert(tk.END, "Select a ticker to view analysis.")
                    # Check if open_10k_button exists before trying to access it
                    if hasattr(self, 'open_10k_button') and self.open_10k_button.winfo_exists():
                        self.open_10k_button.config(state="disabled")
            elif self.active_tab == "buffett_canslim":
                # Ensure right explanation pane is visible when entering the tab
                try:
                    if hasattr(self, 'bc_content') and self.bc_content.winfo_exists():
                        w = self.bc_content.winfo_width()
                        if w and w > 0:
                            self.bc_content.sashpos(0, int(w * 0.55))
                            self._bc_sash_initialized = True
                except Exception:
                    pass
                if selected_tickers and self._bc_base_image is not None and \
                   self._bc_last_ticker == selected_tickers[0]:
                    try:
                        self._show_cached_buffett_canslim()
                    except Exception:
                        pass
                elif not selected_tickers:
                    self.bc_text.delete("1.0", tk.END)
                    self.bc_text.insert(tk.END, "Select a ticker to analyze.")
                    self.bc_status_var.set("Waiting for ticker selection")
                else:
                    self.bc_status_var.set(f"Cached chart available for {self._bc_last_ticker}" if self._bc_base_image is not None else "Select Analyze to run study")
            elif selected_tickers: # For other tabs, only update if there is a selection
                if self.active_tab == "comparison":
                    self._compare_percentage_performance(tickers=selected_tickers)
                elif self.active_tab == "seasonality":
                    self._generate_seasonality_chart(selected_tickers[0])
                elif self.active_tab == "individual":
                    self._display_chart(selected_tickers[0])
        except Exception as e:
            logging.error(f"Error handling tab change: {e}")

    def _get_selected_single_ticker(self) -> Optional[str]:
        """Return the first selected ticker from main or watch list, or None."""
        sel = self.ticker_listbox.curselection()
        if sel:
            t = self.ticker_listbox.get(sel[0])
            return t.split(' ')[0].strip()
        wsel = self.watch_listbox.curselection()
        if wsel:
            return self.watch_listbox.get(wsel[0]).strip()
        return None

    def _analyze_buffett_canslim_current(self):
        """Trigger analysis for the currently selected ticker in the Buffett & CANSLIM tab."""
        try:
            ticker = self._get_selected_single_ticker()
            if not ticker:
                self.bc_status_var.set("No ticker selected")
                return
            self.bc_status_var.set(f"Analyzing {ticker} ...")

            def worker():
                try:
                    result = buffett_canslim.analyze_stock_scores(ticker)
                    # Build figure
                    fig = buffett_canslim.build_analysis_figure(
                        ticker,
                        result['buffett_scores'],
                        result['buffett_total'],
                        result['canslim_scores'],
                        result['canslim_total'],
                        history_df=None,
                    )
                    try:
                        buffett_canslim.save_study_markdown(ticker, result, fig)
                    except Exception:
                        pass
                    # Render to PNG in-memory (PIL only; do not create Tk objects in worker thread)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    buf.seek(0)
                    img = Image.open(buf)

                    explanation = result['raw_text']

                    def on_ui(msg=explanation):
                        if not hasattr(self, 'bc_chart_label') or not self.bc_chart_label.winfo_exists():
                            return
                        # Ensure text widget exists
                        if not hasattr(self, 'bc_text') or not self.bc_text.winfo_exists():
                            return
                        # Cache last analysis state
                        self._bc_last_ticker = ticker
                        self._bc_last_explanation = msg or ""
                        # Store base image and initial zoom
                        self._bc_base_image = img
                        # Start with fit-to-container scale and allow user zoom later
                        self._bc_user_zoomed = False
                        self._bc_zoom_scale = 1.0
                        self._update_bc_chart_image()
                        # Make the chart image zoomable
                        self.bc_chart_label.configure(state='normal')
                        # Linux scroll events
                        self.bc_chart_label.bind('<Button-4>', self._zoom_chart_in)
                        self.bc_chart_label.bind('<Button-5>', self._zoom_chart_out)
                        # Windows/macOS MouseWheel
                        self.bc_chart_label.bind('<MouseWheel>', self._on_mouse_wheel_zoom)
                        # Populate explanation text reliably
                        try:
                            self.bc_text.configure(state='normal')
                        except Exception:
                            pass
                        self.bc_text.delete('1.0', tk.END)
                        # Format Chinese text to be more readable
                        formatted = (msg or "")  # preserve original line breaks
                        try:
                            logging.info(f"BC explanation length: {len(formatted)}")
                        except Exception:
                            pass
                        if not formatted.strip():
                            formatted = "No explanation text was returned by the analyzer."
                        self.bc_text.insert(tk.END, formatted)
                        try:
                            self.bc_text.see('1.0')
                        except Exception:
                            pass
                        try:
                            self.bc_text.update_idletasks()
                        except Exception:
                            pass
                        self.bc_status_var.set(f"Completed analysis for {ticker}")

                    self.root.after(0, on_ui)
                except Exception as e:
                    err_msg = f"Error in Buffett & CANSLIM analysis: {e}"
                    logging.error(err_msg)
                    def on_err(msg=err_msg):
                        self.bc_status_var.set(f"Error: {msg}")
                    self.root.after(0, on_err)

            # Show immediate placeholder so users see progress and to verify text area renders
            try:
                if hasattr(self, 'bc_text') and self.bc_text.winfo_exists():
                    self.bc_text.configure(state='normal')
                    self.bc_text.delete('1.0', tk.END)
                    self.bc_text.insert(tk.END, f"Analyzing {ticker}…")
                    self.bc_text.see('1.0')
            except Exception:
                pass

            threading.Thread(target=worker, daemon=True).start()
        except Exception as e:
            logging.error(f"Error starting Buffett & CANSLIM analysis: {e}")
            
    def _update_bc_chart_image(self):
        """Re-render the Buffett & CANSLIM chart image based on current zoom scale."""
        try:
            if not self._bc_base_image or not hasattr(self, 'bc_chart_label') or not self.bc_chart_label.winfo_exists():
                return
            # Determine target scale: fit image to container if user hasn't zoomed
            scale = self._bc_zoom_scale
            if not getattr(self, '_bc_user_zoomed', False):
                try:
                    # Use the label size as the actual drawing area
                    avail_w = max(1, self.bc_chart_label.winfo_width())
                    avail_h = max(1, self.bc_chart_label.winfo_height())
                    # Keep some padding
                    pad = 16
                    avail_w = max(1, avail_w - pad)
                    avail_h = max(1, avail_h - pad)
                    base_w, base_h = self._bc_base_image.size
                    if base_w > 0 and base_h > 0 and avail_w > 0 and avail_h > 0:
                        fit_scale = min(avail_w / base_w, avail_h / base_h)
                        # Avoid oversizing
                        scale = max(0.1, min(fit_scale, 3.0))
                        self._bc_zoom_scale = scale
                except Exception:
                    pass
            # Clamp final scale
            scale = max(0.1, min(scale, 5.0))
            base_w, base_h = self._bc_base_image.size
            new_size = (max(1, int(base_w * scale)), max(1, int(base_h * scale)))
            resized = self._bc_base_image.resize(new_size, Image.LANCZOS)
            self._bc_chart_photo = ImageTk.PhotoImage(resized)
            self.bc_chart_label.configure(image=self._bc_chart_photo)
        except Exception as e:
            logging.error(f"Error updating BC chart image: {e}")

    def _show_cached_buffett_canslim(self):
        """Show cached Buffett & CANSLIM chart and explanation without re-running analysis."""
        try:
            if not hasattr(self, 'bc_chart_label') or not self.bc_chart_label.winfo_exists():
                return
            if self._bc_base_image is None:
                return
            # Maintain current zoom; just redraw
            self._update_bc_chart_image()
            # Update explanation text
            if hasattr(self, 'bc_text') and self.bc_text.winfo_exists():
                try:
                    self.bc_text.configure(state='normal')
                except Exception:
                    pass
                self.bc_text.delete('1.0', tk.END)
                self.bc_text.insert(tk.END, self._bc_last_explanation or "")
                try:
                    self.bc_text.see('1.0')
                except Exception:
                    pass
            if hasattr(self, 'bc_status_var'):
                self.bc_status_var.set(f"Showing cached analysis for {self._bc_last_ticker}")
        except Exception as e:
            logging.error(f"Error showing cached Buffett & CANSLIM view: {e}")

    def _zoom_chart_in(self, event=None):
        """Zoom in the Buffett & CANSLIM chart."""
        self._bc_user_zoomed = True
        self._bc_zoom_scale *= 1.1
        self._update_bc_chart_image()

    def _zoom_chart_out(self, event=None):
        """Zoom out the Buffett & CANSLIM chart."""
        self._bc_user_zoomed = True
        self._bc_zoom_scale /= 1.1
        self._update_bc_chart_image()

    def _on_mouse_wheel_zoom(self, event):
        """Mouse wheel zoom handler for Windows/macOS."""
        try:
            if event.delta > 0:
                self._zoom_chart_in()
            else:
                self._zoom_chart_out()
        except Exception as e:
            logging.error(f"Mouse wheel zoom error: {e}")
    def _compare_percentage_performance0(self, tickers=None):
        """Generate and display an interactive comparison chart showing percentage performance of multiple stocks"""
        try:
            # If tickers aren't passed directly, get them from the listbox selection
            selected_tickers = tickers if tickers is not None else self._get_selected_tickers()
            
            if not selected_tickers:
                self.status_var.set("Please select at least one ticker for comparison")
                return
                
            self.status_var.set(f"Generating comparison chart for {', '.join(selected_tickers)}...")
            self.root.update_idletasks()
            
            # Create a Plotly figure for the comparison chart
            fig = go.Figure()
            
            # Apply date range filter if specified
            start_date = self.start_date_entry.get()
            end_date = self.end_date_entry.get()
            
            # Track successfully plotted tickers
            plotted_tickers = []
            
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
                    
                    # Drop NaN values that might have been introduced by coercion
                    df = df.dropna(subset=['Close'])
                    
                    if df.empty or len(df) < 2:
                        logging.warning(f"Insufficient valid data for {ticker}")
                        continue
                    
                    # Calculate percentage change from first day
                    first_close = df['Close'].iloc[0]
                    if pd.isna(first_close) or first_close == 0:
                        logging.warning(f"Invalid first close value for {ticker}: {first_close}")
                        continue
                        
                    df['pct_change'] = ((df['Close'] - first_close) / first_close) * 100
                    
                    # Add trace to the figure
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['pct_change'],
                        mode='lines',
                        name=ticker
                    ))
                    
                    # Track successfully plotted ticker
                    plotted_tickers.append(ticker)
                    
                except Exception as e:
                    logging.error(f"Error processing {ticker} for comparison chart: {str(e)}")
            
            # Check if we have any data to plot
            if not plotted_tickers:
                self.status_var.set("No valid data available for the selected tickers in the specified date range")
                messagebox.showwarning("No Data", "No valid data available for the selected tickers in the specified date range")
                return
                    
            # Update layout with interactive features
            fig.update_layout(
                title=f'Percentage Performance Comparison ({len(plotted_tickers)} tickers)',
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
            
            # Add horizontal line at 0%
            fig.add_shape(
                type="line",
                x0=fig.data[0].x[0],
                y0=0,
                x1=fig.data[0].x[-1],
                y1=0,
                line=dict(color="black", width=1, dash="dash")
            )
            
            # Display the interactive chart
            self._display_plotly_chart(fig, tab="comparison")
            self.status_var.set(f"Generated interactive comparison chart for {len(plotted_tickers)} tickers")
            
        except Exception as e:
            error_msg = f"Error generating comparison chart: {str(e)}"
            self.status_var.set(error_msg)
            logging.error(error_msg)
            messagebox.showerror("Chart Error", f"Error generating comparison chart: {str(e)}")
            return

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
        """Display a static chart image in the individual chart tab.
        
        Args:
            image_path (str): Path to the image file to display
        """
        try:
            # Load the image
            img = Image.open(image_path)
            
            # Get the frame size
            frame_width = self.individual_chart_frame.winfo_width() or 800
            frame_height = self.individual_chart_frame.winfo_height() or 600
            
            # Resize image to fit the frame while maintaining aspect ratio
            img.thumbnail((frame_width, frame_height), Image.LANCZOS)
            
            # Convert to PhotoImage for Tkinter
            photo = ImageTk.PhotoImage(img)
            
            # Update the label with the new image
            self.chart_label.config(image=photo)
            self.chart_label.image = photo  # Keep a reference
            
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
            error_msg = f"Error displaying chart for {ticker_or_path}: {str(e)}"
            logging.error(error_msg)
            if hasattr(self, 'status_var'):
                self.status_var.set(error_msg)

    def _display_fundamental_data(self, tickers):
        """Fetch and display fundamental data for the selected tickers. Clears the view if tickers list is empty.
        Preserves the current filter when switching tickers.
        """
        try:
            # Store the current filter before updating the data
            current_filter = self.fundamental_filter_var.get() if hasattr(self, 'fundamental_filter_var') else ""
            
            # Clear the fundamental data cache
            self.fundamental_data_cache = []
            
            # If no tickers are selected, reset the view and return
            if not tickers:
                self._populate_fundamental_treeview() # This will show an empty table
                self.status_var.set("Select a ticker to view fundamental data.")
                return
            
            # Dynamically configure columns for side-by-side comparison
            columns = ['Metric'] + tickers
            self.fundamental_data_tree['columns'] = columns
            for col in columns:
                self.fundamental_data_tree.heading(col, text=col)
                self.fundamental_data_tree.column(col, width=150 if len(tickers) > 1 else 400)
            self.fundamental_data_tree.column('Metric', width=200)

            # Fetch fundamental data for all tickers
            self.status_var.set(f"Fetching fundamental data for {', '.join(tickers)}...")
            self.root.update_idletasks()

            all_data = {ticker: self.manager.get_fundamental_data(ticker) for ticker in tickers}

            # Get a union of all keys from all tickers' data
            all_keys = set()
            for ticker in tickers:
                if all_data[ticker]:
                    all_keys.update(all_data[ticker].keys())

            # Populate the cache
            for key in sorted(list(all_keys)):
                values = [key]
                for ticker in tickers:
                    if all_data[ticker]:
                        values.append(all_data[ticker].get(key, 'N/A'))
                    else:
                        values.append('N/A')

                tags = ("bold",) if key in self.important_metrics else ()
                self.fundamental_data_cache.append((values, tags))

            # Restore the filter if it existed
            if current_filter:
                self.fundamental_filter_var.set(current_filter)
                
            # Populate the treeview from the cache (this will apply any filter)
            self._populate_fundamental_treeview()
            self.status_var.set(f"Displayed fundamental data for {', '.join(tickers)}")
            
        except Exception as e:
            error_msg = f"Error displaying fundamental data: {str(e)}"
            self.status_var.set(error_msg)
            logging.error(error_msg)
            messagebox.showerror("Error", error_msg)

    def _save_filter(self):
        """Save the current filter to a file (ticker-agnostic)"""
        try:
            # Get the current filter text
            filter_text = self.fundamental_filter_var.get().strip()
            
            # If the filter is empty, inform the user
            if not filter_text:
                messagebox.showinfo("Save Filter", "No filter to save.")
                return
            
            # Create a filters directory if it doesn't exist
            filters_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "filters")
            os.makedirs(filters_dir, exist_ok=True)
            
            # Ask the user for a filter name
            filter_name = simpledialog.askstring("Save Filter", "Enter a name for this filter:")
            if not filter_name:
                return  # User cancelled
                
            # Create a filename with just the filter name (ticker-agnostic)
            filename = os.path.join(filters_dir, f"{filter_name}.filter")
            
            # Check if file already exists and confirm overwrite
            if os.path.exists(filename):
                if not messagebox.askyesno("Confirm Overwrite", f"Filter '{filter_name}' already exists. Overwrite?"): 
                    return  # User cancelled overwrite
            
            # Save the filter to the file
            with open(filename, "w") as f:
                f.write(filter_text)
                
            messagebox.showinfo("Save Filter", f"Filter saved as '{filter_name}'")
            logging.info(f"Filter saved to {filename}")
            
        except Exception as e:
            error_msg = f"Error saving filter: {str(e)}"
            logging.error(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def _load_filter(self):
        """Load a saved filter from a file (ticker-agnostic)"""
        try:
            # Get the filters directory
            filters_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "filters")
            
            # Check if the directory exists
            if not os.path.exists(filters_dir):
                messagebox.showinfo("Load Filter", "No saved filters found.")
                return
                
            # Get a list of all filter files
            filter_files = [f for f in os.listdir(filters_dir) if f.endswith(".filter")]
                
            # If no filters found, inform the user
            if not filter_files:
                messagebox.showinfo("Load Filter", "No saved filters found.")
                return
                
            # Extract filter names from filenames
            filter_names = [f.replace(".filter", "") for f in filter_files]
            
            # Sort filter names alphabetically
            filter_names.sort()
            
            # Ask the user to select a filter
            selected_filter = simpledialog.askstring(
                "Load Filter", 
                "Select a filter to load:", 
                initialvalue=filter_names[0] if filter_names else "")
                
            if not selected_filter or selected_filter not in filter_names:
                return  # User cancelled or invalid selection
                
            # Get the selected filter file
            selected_file = os.path.join(filters_dir, f"{selected_filter}.filter")
            
            # Load the filter from the file
            with open(selected_file, "r") as f:
                filter_text = f.read().strip()
                
            # Set the filter text
            self.fundamental_filter_var.set(filter_text)
            
            # Apply the filter
            self._populate_fundamental_treeview()
            
            messagebox.showinfo("Load Filter", f"Filter '{selected_filter}' loaded successfully.")
            logging.info(f"Filter loaded from {selected_file}")
            
        except Exception as e:
            error_msg = f"Error loading filter: {str(e)}"
            logging.error(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def _populate_fundamental_treeview(self, event=None):
        """Populate the fundamental data treeview from the cache, applying the current filter.
        
        Supports:
        - Multiple filter terms separated by spaces (OR logic)
        - Exclusion with ! prefix
        - Case-insensitive matching
        - Filtering applied to the Metric column
        - * wildcard to show all other rows below the matching ones
        """
        try:
            # Clear previous data from the treeview
            for item in self.fundamental_data_tree.get_children():
                self.fundamental_data_tree.delete(item)

            filter_text = self.fundamental_filter_var.get().strip().lower()

            # If the cache is empty, ensure the view is empty and columns are reset
            if not self.fundamental_data_cache:
                columns = ['Metric', 'Value']
                self.fundamental_data_tree['columns'] = columns
                self.fundamental_data_tree.heading('Metric', text='Metric')
                self.fundamental_data_tree.heading('Value', text='Value')
                self.fundamental_data_tree.column('Metric', width=200)
                self.fundamental_data_tree.column('Value', width=400)
                return
                
            # Check if wildcard is present
            show_all_others = '*' in filter_text
            
            # Remove the wildcard from filter terms if present
            filter_text = filter_text.replace('*', ' ').strip()
                
            # Split the filter text into terms
            if filter_text:
                filter_terms = filter_text.split()
                
                # Separate inclusion and exclusion terms
                include_terms = [term for term in filter_terms if not term.startswith('!')]
                exclude_terms = [term[1:] for term in filter_terms if term.startswith('!')]
            else:
                include_terms = []
                exclude_terms = []

            # First pass: add matching items
            matching_items = []
            non_matching_items = []
            
            for values, tags in self.fundamental_data_cache:
                # The metric name is the first item in the values list
                metric_name = str(values[0]).lower()
                
                # Check exclusion terms first (any match excludes the item)
                excluded = any(term in metric_name for term in exclude_terms if term)
                
                if excluded:
                    if show_all_others:
                        non_matching_items.append((values, tags))
                    continue
                    
                # Check inclusion terms (any match for OR logic)
                included = any(term in metric_name for term in include_terms if term)
                
                # If no filter or any inclusion term matches, add the item to matching
                if not filter_text or included or not include_terms:
                    matching_items.append((values, tags))
                elif show_all_others:
                    non_matching_items.append((values, tags))
            
            # Add matching items first
            for values, tags in matching_items:
                self.fundamental_data_tree.insert('', tk.END, values=values, tags=tags)
            
            # If wildcard is present, add a separator and then non-matching items
            if show_all_others and non_matching_items:
                # Add a separator row
                separator_values = ["--- Other Metrics ---", ""]
                self.fundamental_data_tree.insert('', tk.END, values=separator_values, tags=("separator",))
                
                # Configure a tag for the separator
                self.fundamental_data_tree.tag_configure("separator", background="#e0e0e0", font=("Helvetica", 10, "bold"))
                
                # Add non-matching items
                for values, tags in non_matching_items:
                    self.fundamental_data_tree.insert('', tk.END, values=values, tags=tags)
        except Exception as e:
            logging.error(f"Error populating fundamental treeview: {e}")
            messagebox.showerror("Error", f"Could not update fundamental data view: {e}")

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
        """Worker function to download data for multiple tickers with force_download=True

        Args:
            tickers (list): List of ticker symbols to download data for
            queue (Queue): Queue to store results
        """
        # Call the more flexible worker with force_download=True
        self._download_worker_with_force_option(tickers, queue, force_download=True)

    def _check_download_progress(self, thread, total_tickers, check_interval=100):
        """Periodically check download progress and update the UI

        Args:
            thread (Thread): The download thread to monitor
            total_tickers (int): Total number of tickers being downloaded
            check_interval (int): How often to check progress in milliseconds
        """
        try:
            # Process all available messages from the queue
            success_count = 0
            while not self.download_queue.empty():
                msg_type, msg = self.download_queue.get_nowait()

                if msg_type == "status":
                    # Update status message
                    self.status_var.set(msg)
                elif msg_type == "progress":
                    # Update progress (could be used for a progress bar)
                    progress = int((msg / total_tickers) * 100)
                    self.status_var.set(f"Downloading... {progress}% complete ({msg}/{total_tickers})")
                elif msg_type == "success_count":
                    # Track successful downloads
                    success_count = msg
                elif msg_type == "error":
                    # Log error but continue
                    logging.error(msg)
                    # Show error in a non-blocking way
                    self.root.after(1, lambda m=msg: self._show_non_blocking_error(m))
                elif msg_type == "complete":
                    # Download complete
                    self.status_var.set(msg)
                elif msg_type == "critical":
                    # Critical error
                    logging.error(msg)
                    self.root.after(1, lambda m=msg: messagebox.showerror("Download Error", m))
                    return

            # If thread is still alive, schedule another check
            if thread.is_alive():
                self.root.after(check_interval,
                                lambda: self._check_download_progress(thread, total_tickers, check_interval))
            else:
                # Thread completed, final update
                if not self.download_queue.empty():
                    # Process any remaining messages
                    self.root.after(10, lambda: self._check_download_progress(thread, total_tickers, check_interval))
                else:
                    # All done, update final status
                    self.status_var.set(f"Download completed: {success_count}/{total_tickers} tickers successful")

        except Exception as e:
            logging.error(f"Error checking download progress: {str(e)}")
            
    def _show_non_blocking_error(self, message):
        """Show an error message in a non-blocking way
        
        Args:
            message (str): Error message to display
        """
        # Log the error
        logging.error(message)
        
        # Update status bar instead of showing a modal dialog
        current_status = self.status_var.get()
        self.status_var.set(f"Error: {message}")
        
        # Schedule restoration of previous status after a few seconds
        self.root.after(5000, lambda prev=current_status: self.status_var.set(prev))

    def _run_general_search(self):
        """Run a general AI search using the query in the search field without requiring ticker selection"""
        # Get the query from the search field
        query = self.general_search_var.get().strip()
        
        # Validate query
        if not query:
            safe_show_message('warning', "Empty Query", "Please enter a search query.")
            return
            
        # Update status
        safe_update_status(self.status_var, f"Running AI search for query: '{query}'...")
        self.root.update_idletasks()
        
        # Clear previous results
        safe_update_text_widget(self.business_analysis_text, 'delete', "1.0", tk.END)
        safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, f"Running AI search for query: '{query}'...\nPlease wait...")
        
        def search_thread():
            try:
                # Run the search using the gemini_analyzer module with the new general_ai_search function
                logging.info(f"Running general AI search with query: {query}")
                from gemini_analyzer import general_ai_search
                result = general_ai_search(query)
                
                if result.startswith("Error:"):
                    safe_show_message('error', "API Error", result)
                    safe_update_status(self.status_var, f"Failed to run AI search: {result}")
                    return
                    
                # Display the result in the text widget using thread-safe methods
                safe_update_text_widget(self.business_analysis_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, 
                                       f"# AI Search Results for: '{query}'\n{result}")
                safe_update_text_widget(self.business_analysis_text, 'see', "1.0")
                
                # Update status using thread-safe method
                safe_update_status(self.status_var, f"Completed AI search for query: '{query}'")
                
            except Exception as e:
                error_msg = f"Error running AI search: {str(e)}"
                safe_update_status(self.status_var, error_msg)
                logging.error(error_msg)
                safe_update_text_widget(self.business_analysis_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, f"Error: {error_msg}")
                safe_show_message('error', "Error", error_msg)
        
        # Run the search in a separate thread to avoid freezing the GUI
        search_thread_obj = threading.Thread(target=search_thread, daemon=True)
        search_thread_obj.name = f"AISearch-General-{query[:10]}"
        search_thread_obj.start()
    
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

    def _compare_percentage_performance(self, tickers=None):
        """Generate overlayed percentage comparison chart for selected tickers
        using the common available data range"""
        # Get selected tickers
        try:
            if tickers is None:
                selected_tickers = self._get_selected_tickers()
                if not selected_tickers:
                    return
            else:
                selected_tickers = tickers

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

                # Get the comparison container size (fixed, non-propagating)
                target_widget = getattr(self, 'comparison_chart_container', self.comparison_chart_frame)
                chart_width = target_widget.winfo_width()
                chart_height = target_widget.winfo_height()

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

    def _get_ba_cache_file(self, ticker: str) -> str:
        """Return a unique timestamped BA markdown cache file path for a ticker."""
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(cache_dir, f"{ticker}_business_analysis_{ts}.md")

    def _find_latest_ba_cache_file(self, ticker: str) -> str | None:
        """Return the most recent BA markdown cache file path for a ticker, or None if none found."""
        try:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
            if not os.path.isdir(cache_dir):
                return None
            prefix = f"{ticker}_business_analysis"
            candidates = [
                os.path.join(cache_dir, name)
                for name in os.listdir(cache_dir)
                if name.startswith(prefix) and name.endswith(".md")
            ]
            if not candidates:
                return None
            return max(candidates, key=lambda p: os.path.getmtime(p))
        except Exception:
            return None

    def _list_ba_cache_files(self, ticker: str) -> list[str]:
        """Return all BA markdown files for ticker sorted by mtime descending (latest first)."""
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        if not os.path.isdir(cache_dir):
            return []
        prefix = f"{ticker}_business_analysis"
        paths = [
            os.path.join(cache_dir, name)
            for name in os.listdir(cache_dir)
            if name.startswith(prefix) and name.endswith('.md')
        ]
        return sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True)

    def _build_ba_change_over_time_section(self, ticker: str, max_items: int = 5) -> str:
        """Build a markdown section summarizing change over time using historical BA files."""
        try:
            files = self._list_ba_cache_files(ticker)
            if len(files) < 2:
                return ""
            files = files[:max_items]

            from difflib import ndiff, SequenceMatcher
            lines = ["## Change Over Time (last {} versions)".format(len(files))]
            # Add list of versions with timestamps
            for idx, path in enumerate(files):
                ts = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S')
                label = "latest" if idx == 0 else ""
                lines.append(f"- {ts} {label}")

            # Compare adjacent versions for quick stats (latest vs previous, etc.)
            lines.append("")
            for i in range(len(files) - 1):
                newer = files[i]
                older = files[i + 1]
                with open(newer, 'r', encoding='utf-8') as f1, open(older, 'r', encoding='utf-8') as f2:
                    new_txt = f1.read().splitlines()
                    old_txt = f2.read().splitlines()
                diff = list(ndiff(old_txt, new_txt))
                added = sum(1 for d in diff if d.startswith('+ '))
                removed = sum(1 for d in diff if d.startswith('- '))
                ratio = SequenceMatcher(None, '\n'.join(old_txt), '\n'.join(new_txt)).ratio()
                new_ts = datetime.fromtimestamp(os.path.getmtime(newer)).strftime('%Y-%m-%d %H:%M:%S')
                old_ts = datetime.fromtimestamp(os.path.getmtime(older)).strftime('%Y-%m-%d %H:%M:%S')
                lines.append(f"- Δ {old_ts} → {new_ts}: +{added} / -{removed}, similarity {ratio:.2f}")

            return "\n".join(lines)
        except Exception as e:
            logging.error(f"Error building BA change-over-time section: {e}")
            return ""

    def _is_cache_fresh(self, path: str, days: int = 30) -> bool:
        """Check if file at path was modified within the last `days`."""
        try:
            mtime = os.path.getmtime(path)
            return (datetime.now() - datetime.fromtimestamp(mtime)) <= timedelta(days=days)
        except Exception:
            return False

    def _beautify_business_analysis(self, raw_text: str, ticker: str, company_info: Optional[Dict[str, Any]] = None) -> str:
        """Transform Gemini output into a structured, infographic-style layout."""
        if not raw_text or not raw_text.strip():
            return raw_text

        normalized_text = raw_text.replace("\r\n", "\n").strip()
        lower_text = normalized_text.lower()

        if lower_text.startswith("error") or lower_text.startswith("an error"):
            return raw_text

        # Skip if already formatted
        if "BUSINESS SNAPSHOT" in raw_text and "BUSINESS ANALYSIS" in raw_text:
            return raw_text

        company_info = company_info or self.manager.get_fundamental_data(ticker)

        sections = self._extract_ba_sections(normalized_text)
        if not sections:
            return raw_text

        width = 82
        banner_lines = self._build_ba_banner(ticker, company_info, width)
        snapshot_lines = self._build_business_snapshot_panel(company_info, width)

        icon_map = {
            "BUSINESS MODEL": "🛡",
            "REVENUE STREAMS": "💰",
            "COMPETITIVE LANDSCAPE": "⚔️",
            "FINANCIAL HEALTH": "📊",
            "GROWTH PROSPECTS": "🚀",
            "POTENTIAL RISKS": "⚠️",
        }

        formatted_lines = banner_lines
        if snapshot_lines:
            formatted_lines.extend(["", *snapshot_lines, ""])

        for idx, section in enumerate(sections, 1):
            title = self._normalize_section_title(section["title"])
            icon = icon_map.get(title, "")
            heading = f"{idx}. {title}" if title else f"Section {idx}"
            if icon:
                heading = f"{icon}  {heading}"

            formatted_lines.append(heading)
            formatted_lines.append("─" * min(len(heading) + 2, width))

            for item_type, text in section["body"]:
                if not text:
                    continue
                if item_type == "bullet":
                    formatted_lines.append(f"   • {text}")
                else:
                    formatted_lines.append(f"   {text}")

            formatted_lines.append("")

        return "\n".join(line.rstrip() for line in formatted_lines).strip()

    def _build_ba_banner(self, ticker: str, company_info: Optional[Dict[str, Any]], width: int) -> List[str]:
        """Create a banner headline for the business analysis card."""
        company_name = (company_info or {}).get('longName') or ticker.upper()
        title = f"{company_name.upper()} ({ticker.upper()})"
        subtitle = "BUSINESS ANALYSIS"

        top = "╔" + "═" * (width - 2) + "╗"
        bottom = "╚" + "═" * (width - 2) + "╝"
        middle_title = f"║ {title.center(width - 4)} ║"
        middle_subtitle = f"║ {subtitle.center(width - 4)} ║"

        return [top, middle_title, middle_subtitle, bottom]

    def _build_business_snapshot_panel(self, company_info: Optional[Dict[str, Any]], width: int) -> List[str]:
        """Build a quick facts panel similar to an infographic callout."""
        if not company_info:
            return []

        fields = [
            ("Sector", company_info.get('sector')),
            ("Industry", company_info.get('industry')),
            ("Market Cap", company_info.get('marketCap')),
            ("Trailing P/E", company_info.get('trailingPE')),
            ("Forward P/E", company_info.get('forwardPE')),
            ("Dividend Yield", company_info.get('dividendYield')),
            ("Beta", company_info.get('beta')),
            ("52W High", company_info.get('fiftyTwoWeekHigh')),
            ("52W Low", company_info.get('fiftyTwoWeekLow')),
        ]

        panel_width = width - 2  # number of dashes between corners
        inner_width = panel_width - 2  # available characters between vertical bars

        lines = [
            "┌" + "─" * panel_width + "┐",
            f"│ { 'BUSINESS SNAPSHOT'.center(inner_width) } │",
            "├" + "─" * panel_width + "┤",
        ]

        for label, value in fields:
            display_value = self._format_snapshot_value(label, value)
            content = f"{label:<15}: {display_value}"
            truncated = content[:inner_width].ljust(inner_width)
            lines.append(f"│ {truncated} │")

        lines.append("└" + "─" * panel_width + "┘")
        return lines

    def _format_snapshot_value(self, label: str, value: Any) -> str:
        """Format numeric snapshot values into human-friendly strings."""
        if value in (None, "", "N/A"):
            return "N/A"

        if isinstance(value, (int, float)):
            if "yield" in label.lower():
                pct = value * 100 if abs(value) < 1 else value
                return f"{pct:.2f}%"
            if label.lower().startswith("beta"):
                return f"{value:.2f}"
            if "52w" in label.lower():
                return f"{value:,.2f}"
            return self._format_human_number(value)

        return str(value)

    def _format_human_number(self, value: Any) -> str:
        """Convert large numbers into compact representations (e.g., 405B)."""
        try:
            num = float(value)
        except (TypeError, ValueError):
            return str(value)

        abs_num = abs(num)
        if abs_num >= 1e12:
            return f"{num / 1e12:.2f}T"
        if abs_num >= 1e9:
            return f"{num / 1e9:.2f}B"
        if abs_num >= 1e6:
            return f"{num / 1e6:.2f}M"
        if abs_num >= 1e3:
            return f"{num:,.0f}"
        return f"{num:.2f}"

    def _extract_ba_sections(self, text: str) -> List[Dict[str, Any]]:
        """Parse markdown-ish Gemini output into ordered sections."""
        sections: List[Dict[str, Any]] = []
        current_section: Optional[Dict[str, Any]] = None

        for raw_line in text.split('\n'):
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith('#'):
                heading = line.lstrip('#').strip()
                if not heading:
                    continue
                current_section = {"title": heading, "body": []}
                sections.append(current_section)
                continue

            if current_section is None:
                current_section = {"title": "Overview", "body": []}
                sections.append(current_section)

            if line[0] in {'*', '-', '•'}:
                bullet_text = line.lstrip('*-•\t ').strip()
                current_section["body"].append(("bullet", self._strip_markdown_emphasis(bullet_text)))
            else:
                current_section["body"].append(("text", self._strip_markdown_emphasis(line)))

        return sections

    def _normalize_section_title(self, title: str) -> str:
        """Normalize headings to uppercase card titles."""
        if not title:
            return ""
        cleaned = re.sub(r"^\d+[\.)]\s*", "", title).strip().rstrip(':')
        return cleaned.upper()

    def _strip_markdown_emphasis(self, text: str) -> str:
        """Remove simple markdown emphasis markers for cleaner bullets."""
        if not text:
            return text
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"__(.*?)__", r"\1", text)
        text = text.replace('`', '')
        return text.replace('**', '').strip()

    def _run_business_analysis(self):
        """Runs the business analysis for the selected ticker."""
        selected_tickers = self._get_selected_tickers(show_warning=True)
        if not selected_tickers:
            # Check watch list if no ticker is selected in the main list
            selected_indices = self.watch_listbox.curselection()
            if not selected_indices:
                safe_show_message('warning', "No Selection", "Please select a ticker from the 'Available Tickers' or 'Watch List'.")
                return
            selected_tickers = [self.watch_listbox.get(i).strip() for i in selected_indices]

        ticker = selected_tickers[0]
        # Load fresh cached BA markdown (<= 30 days) if available
        try:
            cache_file = self._find_latest_ba_cache_file(ticker)
            if cache_file and os.path.exists(cache_file) and self._is_cache_fresh(cache_file, days=self.ba_freshness_days_var.get()):
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_md = f.read()
                formatted_md = self._beautify_business_analysis(cached_md, ticker)
                self.business_analysis_original_text = formatted_md
                self.business_analysis_filter_var.set("")
                safe_update_text_widget(self.business_analysis_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, formatted_md)
                safe_update_text_widget(self.business_analysis_text, 'see', "1.0")
                # Append change-over-time section if enabled and history exists
                if self.ba_show_change_var.get():
                    change_md = self._build_ba_change_over_time_section(ticker, max_items=self.ba_history_max_items_var.get())
                    if change_md:
                        safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, "\n\n---\n" + change_md)
                safe_update_status(self.status_var, f"Loaded cached Business Analysis for {ticker} (fresh)")
                return
        except Exception as e:
            logging.error(f"Error reading BA cache: {e}")
        # Use thread-safe text widget updates
        safe_update_text_widget(self.business_analysis_text, 'delete', "1.0", tk.END)
        safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, f"Running business analysis for {ticker}...")
        self.root.update_idletasks()

        def analysis_thread():
            try:
                logging.info(f"Fetching fundamental data for {ticker}")
                company_info = self.manager.get_fundamental_data(ticker)
                if not company_info:
                    safe_update_text_widget(self.business_analysis_text, 'delete', "1.0", tk.END)
                    safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, f"Could not retrieve fundamental data for {ticker}.")
                    return

                logging.info(f"Running analysis for {ticker}")
                analysis_result = gemini_analyzer.analyze_ticker(ticker, company_info)
                beautified_result = self._beautify_business_analysis(analysis_result, ticker, company_info)
                
                # Store the original text for filtering
                self.business_analysis_original_text = beautified_result
                
                # Reset filter when loading new content
                if threading.current_thread() is threading.main_thread():
                    self.business_analysis_filter_var.set("")
                else:
                    # Use thread-safe method to update StringVar
                    self.root.after(0, lambda: self.business_analysis_filter_var.set(""))
                
                # Update UI with results using thread-safe methods
                safe_update_text_widget(self.business_analysis_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, beautified_result)
                safe_update_text_widget(self.business_analysis_text, 'see', "1.0")
                # Append change-over-time section based on saved history if enabled
                try:
                    if self.ba_show_change_var.get():
                        change_md = self._build_ba_change_over_time_section(ticker, max_items=self.ba_history_max_items_var.get())
                        if change_md:
                            safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, "\n\n---\n" + change_md)
                except Exception:
                    pass

                # Save the analysis to a markdown cache file
                try:
                    analysis_file = self._get_ba_cache_file(ticker)
                    with open(analysis_file, "w", encoding="utf-8") as f:
                        # Optionally add a simple header
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        f.write(f"# Business Analysis: {ticker}\n\n")
                        f.write(f"_Generated: {timestamp}_\n\n")
                        f.write(beautified_result)
                    safe_update_status(self.status_var, f"Saved Business Analysis markdown for {ticker}")
                except Exception as e:
                    logging.error(f"Could not save analysis: {e}")
                    safe_show_message('error', "Error", f"Could not save analysis: {e}")
            except Exception as e:
                logging.error(f"Error in business analysis thread: {e}")
                safe_update_text_widget(self.business_analysis_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, f"Error during analysis: {e}")

        # Run the analysis in a separate thread to avoid freezing the GUI
        analysis_thread_obj = threading.Thread(target=analysis_thread, daemon=True)
        analysis_thread_obj.name = f"BusinessAnalysis-{ticker}"
        analysis_thread_obj.start()

    def _apply_business_analysis_filter(self, event=None):
        """Apply filter to business analysis text content
        
        Supports:
        - Multiple filter terms separated by spaces (AND logic)
        - Exclusion with ! prefix
        - Case-insensitive matching
        - Filtering applied to the Metric column
        """
        # Get filter text
        filter_text = self.business_analysis_filter_var.get().strip()
        
        # If no original text, do nothing
        if not self.business_analysis_original_text:
            return
            
        # If no filter, restore original text
        if not filter_text:
            self.business_analysis_text.delete("1.0", tk.END)
            self.business_analysis_text.insert(tk.END, self.business_analysis_original_text)
            return
            
        # Split the filter text into terms
        filter_terms = filter_text.lower().split()
        
        # Separate inclusion and exclusion terms
        include_terms = [term for term in filter_terms if not term.startswith('!')]
        exclude_terms = [term[1:] for term in filter_terms if term.startswith('!')]
        
        # Process the original text line by line
        lines = self.business_analysis_original_text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line_lower = line.lower()
            
            # Check if this is a metric line (contains a colon)
            if ':' in line:
                # Check exclusion terms first (any match excludes the line)
                excluded = any(term in line_lower for term in exclude_terms if term)
                
                if excluded:
                    continue
                    
                # Check inclusion terms (all must match for AND logic)
                included = all(term in line_lower for term in include_terms if term)
                
                if included or not include_terms:
                    filtered_lines.append(line)
            else:
                # Non-metric lines are always included
                filtered_lines.append(line)
        
        # Update the text widget with filtered content
        self.business_analysis_text.delete("1.0", tk.END)
        self.business_analysis_text.insert(tk.END, '\n'.join(filtered_lines))
    
    def _load_cached_analysis(self, ticker):
        """Load cached business analysis for a ticker"""
        try:
            # Check if we have a fresh cached analysis (load latest)
            cache_file = self._find_latest_ba_cache_file(ticker)
            if cache_file and os.path.exists(cache_file) and self._is_cache_fresh(cache_file, days=self.ba_freshness_days_var.get()):
                with open(cache_file, "r", encoding="utf-8") as f:
                    analysis = f.read()
                formatted_analysis = self._beautify_business_analysis(analysis, ticker)
                
                # Store the original text for filtering
                self.business_analysis_original_text = formatted_analysis
                
                # Reset filter when loading new content
                self.business_analysis_filter_var.set("")
                
                # Use thread-safe text widget updates
                safe_update_text_widget(self.business_analysis_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, formatted_analysis)
                # Append change-over-time section if enabled and history exists
                if self.ba_show_change_var.get():
                    change_md = self._build_ba_change_over_time_section(ticker, max_items=self.ba_history_max_items_var.get())
                    if change_md:
                        safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, "\n\n---\n" + change_md)
                safe_update_status(self.status_var, f"Loaded cached business analysis for {ticker} (fresh)")
                return True
            else:
                safe_update_text_widget(self.business_analysis_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(
                    self.business_analysis_text,
                    'insert',
                    tk.END,
                    f"No fresh cached analysis found for {ticker}. Click 'Run BA' to generate or refresh."
                )
                return False
        except Exception as e:
            logging.error(f"Error loading cached analysis: {e}")
            safe_update_text_widget(self.business_analysis_text, 'delete', "1.0", tk.END)
            safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, f"Error loading cached analysis: {e}")
            return False
            
    def _check_for_cached_10k(self, ticker):
        """Check if a cached 10-K report exists for the given ticker and enable/disable the Open 10-K button accordingly"""
        try:
            # Check if we have a cached 10-K report
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{ticker}_10k.txt")
            
            if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
                # Enable the Open 10-K button using thread-safe method
                if hasattr(self, 'open_10k_button'):
                    if threading.current_thread() is threading.main_thread():
                        self.open_10k_button.config(state=tk.NORMAL)
                    else:
                        tk_update_queue.put((self.open_10k_button.config, (), {'state': tk.NORMAL}))
                return True
            else:
                # Disable the Open 10-K button using thread-safe method
                if hasattr(self, 'open_10k_button'):
                    if threading.current_thread() is threading.main_thread():
                        self.open_10k_button.config(state=tk.DISABLED)
                    else:
                        tk_update_queue.put((self.open_10k_button.config, (), {'state': tk.DISABLED}))
                return False
        except Exception as e:
            logging.error(f"Error checking for cached 10-K: {e}")
            # Disable the Open 10-K button on error using thread-safe method
            if hasattr(self, 'open_10k_button'):
                if threading.current_thread() is threading.main_thread():
                    self.open_10k_button.config(state=tk.DISABLED)
                else:
                    tk_update_queue.put((self.open_10k_button.config, (), {'state': tk.DISABLED}))
            return False
            
    def _run_10k_study(self):
        """
        Runs the 10-K study for the selected ticker by searching the web.
        """
        selected_tickers = self._get_selected_tickers(show_warning=True)
        if not selected_tickers:
            return

        ticker = selected_tickers[0]
        # Use thread-safe text widget updates
        safe_update_text_widget(self.business_analysis_text, 'delete', "1.0", tk.END)
        safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, 
                               f"Running 10-K study for {ticker} by searching online...")
        self.root.update_idletasks()

        def study_thread():
            try:
                logging.info(f"Running 10-K study for {ticker}")
                analysis_result = gemini_analyzer.analyze_10k_report(ticker)
                beautified_result = self._beautify_business_analysis(analysis_result, ticker)
                
                # Store the original text for filtering
                self.business_analysis_original_text = beautified_result
                
                # Reset filter when loading new content
                if threading.current_thread() is threading.main_thread():
                    self.business_analysis_filter_var.set("")
                else:
                    # Use thread-safe method to update StringVar
                    self.root.after(0, lambda: self.business_analysis_filter_var.set(""))
                
                # Update UI with results using thread-safe methods
                safe_update_text_widget(self.business_analysis_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, beautified_result)
                safe_update_text_widget(self.business_analysis_text, 'see', "1.0")
                
                # Update status using thread-safe method
                safe_update_status(self.status_var, f"Completed 10-K study for {ticker}")
            except Exception as e:
                logging.error(f"Error in 10-K study thread: {e}")
                safe_update_text_widget(self.business_analysis_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, f"Error during 10-K study: {e}")
                safe_update_status(self.status_var, f"Error in 10-K study: {str(e)}")

        # Run the study in a separate thread to avoid freezing the GUI
        study_thread_obj = threading.Thread(target=study_thread, daemon=True)
        study_thread_obj.name = f"10KStudy-{ticker}"
        study_thread_obj.start()


    def _run_news_search(self):
        """Runs the news search for the selected ticker."""
        selected_tickers = self._get_selected_tickers(show_warning=True)
        if not selected_tickers:
            safe_show_message('warning', "No Selection", "Please select a ticker.")
            return

        ticker = selected_tickers[0]
        # Use thread-safe text widget updates
        safe_update_text_widget(self.business_analysis_text, 'delete', "1.0", tk.END)
        safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, f"Running news search for {ticker}...")
        self.root.update_idletasks()

        def search_thread():
            try:
                # Use thread-safe text widget updates
                safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, f"\nFetching news for {ticker}...")
                
                logging.info(f"Fetching news for {ticker}")
                news_articles = news_fetcher.fetch_news(ticker)

                if not news_articles or "error" in news_articles[0]:
                    error_msg = news_articles[0]['error'] if news_articles else "Could not fetch news."
                    safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, f"\n{error_msg}")
                    return

                safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, 
                                       f"\nFound {len(news_articles)} news articles. Analyzing...")
                
                logging.info(f"Analyzing {len(news_articles)} news articles for {ticker}")
                analysis_result = gemini_analyzer.analyze_news(news_articles)
                
                # Update UI with results using thread-safe methods
                safe_update_text_widget(self.business_analysis_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, analysis_result)
                safe_update_text_widget(self.business_analysis_text, 'see', "1.0")
                
                # Save the analysis to a file
                try:
                    analysis_dir = os.path.join("stock_data", "business_analysis")
                    os.makedirs(analysis_dir, exist_ok=True)
                    analysis_file = os.path.join(analysis_dir, f"{ticker}_news_analysis.txt")
                    with open(analysis_file, "w", encoding="utf-8") as f:
                        f.write(analysis_result)
                    safe_update_status(self.status_var, f"Saved news analysis for {ticker}")
                except Exception as e:
                    logging.error(f"Could not save news analysis: {e}")
            except Exception as e:
                logging.error(f"Error in news search thread: {e}")
                safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, f"\nError: {e}")

        # Run the search in a separate thread to avoid freezing the GUI
        search_thread_obj = threading.Thread(target=search_thread, daemon=True)
        search_thread_obj.name = f"NewsSearch-{ticker}"
        search_thread_obj.start()

    def _summarize_market_news(self):
        """Summarize overall market news (base feed) using the Finviz blog workflow."""
        try:
            self.chart_notebook.select(self.market_news_frame)
        except Exception:
            pass

        safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
        safe_update_text_widget(
            self.market_news_text,
            'insert',
            tk.END,
            "正在收集 Finviz 市场新闻，请稍候...\nGathering Finviz market news feed, please wait..."
        )
        self.root.update_idletasks()

        def news_thread():
            try:
                articles = self._fetch_market_news_articles(limit=12)
            except Exception as e:
                logging.error(f"Error fetching market news: {e}")
                safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(
                    self.market_news_text,
                    'insert',
                    tk.END,
                    f"无法获取市场新闻：{e}\nCould not fetch market news: {e}"
                )
                self._update_news_temp_list([])
                safe_show_message('error', "Market News", f"Could not fetch Finviz market news: {e}")
                return

            if not articles:
                safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(
                    self.market_news_text,
                    'insert',
                    tk.END,
                    "未找到最新市场新闻。\nNo market news items were found."
                )
                self._update_news_temp_list([])
                safe_update_status(self.status_var, "No market news detected")
                return

            tickers = self._collect_unique_tickers(articles)
            self._update_news_temp_list(tickers)

            safe_update_status(self.status_var, "Summarizing Finviz market news via Gemini...")

            try:
                summary = gemini_analyzer.summarize_market_news(articles, tickers)
            except Exception as e:
                logging.error(f"Error summarizing market news: {e}")
                safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(
                    self.market_news_text,
                    'insert',
                    tk.END,
                    f"市场新闻总结失败：{e}\nFailed to summarize market news: {e}"
                )
                safe_show_message('error', "Market News", f"Failed to summarize Finviz market news: {e}")
                return

            formatted = summary or "Gemini did not return any content."
            self.market_news_original_text = formatted

            safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
            safe_update_text_widget(self.market_news_text, 'insert', tk.END, formatted)
            safe_update_text_widget(self.market_news_text, 'see', "1.0")
            safe_update_status(self.status_var, "Market news blog updated")

        threading.Thread(target=news_thread, daemon=True, name="FinvizMarketNewsSummary").start()

    def _summarize_stock_news(self):
        """Summarize Finviz v=3 stock news feed without requiring ticker selection."""
        try:
            self.chart_notebook.select(self.market_news_frame)
        except Exception:
            pass

        safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
        safe_update_text_widget(
            self.market_news_text,
            'insert',
            tk.END,
            "正在收集 Finviz 股票新闻，请稍候...\nGathering Finviz stock news feed, please wait..."
        )
        self.root.update_idletasks()

        def stock_news_thread():
            try:
                articles = self._fetch_stock_news_articles(limit=12)
            except Exception as e:
                logging.error(f"Error fetching stock news feed: {e}")
                safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(
                    self.market_news_text,
                    'insert',
                    tk.END,
                    f"无法获取股票新闻：{e}\nCould not fetch stock news: {e}"
                )
                self._update_news_temp_list([])
                safe_show_message('error', "Stock News", f"Could not fetch Finviz stock news: {e}")
                return

            if not articles:
                safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(
                    self.market_news_text,
                    'insert',
                    tk.END,
                    "未找到最新股票新闻。\nNo stock news items were found."
                )
                self._update_news_temp_list([])
                safe_update_status(self.status_var, "No stock news detected")
                return

            tickers = self._collect_unique_tickers(articles)
            self._update_news_temp_list(tickers)

            safe_update_status(self.status_var, "Summarizing Finviz stock news via Gemini...")

            try:
                summary = gemini_analyzer.summarize_stock_news(articles, tickers)
            except Exception as e:
                logging.error(f"Error summarizing stock news feed: {e}")
                safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(
                    self.market_news_text,
                    'insert',
                    tk.END,
                    f"总结股票新闻失败：{e}\nFailed to summarize stock news: {e}"
                )
                safe_show_message('error', "Stock News", f"Failed to summarize Finviz stock news: {e}")
                return

            formatted = summary or "Gemini did not return any content."
            self.market_news_original_text = formatted

            safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
            safe_update_text_widget(self.market_news_text, 'insert', tk.END, formatted)
            safe_update_text_widget(self.market_news_text, 'see', "1.0")
            safe_update_status(self.status_var, "Stock news blog updated")

        threading.Thread(target=stock_news_thread, daemon=True, name="FinvizStockNewsSummary").start()

    def _summarize_etf_news(self):
        """Summarize Finviz v=4 ETF news feed without requiring ticker selection."""
        try:
            self.chart_notebook.select(self.market_news_frame)
        except Exception:
            pass

        safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
        safe_update_text_widget(
            self.market_news_text,
            'insert',
            tk.END,
            "正在收集 Finviz ETF 新闻，请稍候...\nGathering Finviz ETF news feed, please wait..."
        )
        self.root.update_idletasks()

        def etf_news_thread():
            try:
                articles = self._fetch_etf_news_articles(limit=12)
            except Exception as e:
                logging.error(f"Error fetching ETF news feed: {e}")
                safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(
                    self.market_news_text,
                    'insert',
                    tk.END,
                    f"无法获取ETF新闻：{e}\nCould not fetch ETF news: {e}"
                )
                self._update_news_temp_list([])
                safe_show_message('error', "ETF News", f"Could not fetch Finviz ETF news: {e}")
                return

            if not articles:
                safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(
                    self.market_news_text,
                    'insert',
                    tk.END,
                    "未找到最新ETF新闻。\nNo ETF news items were found."
                )
                self._update_news_temp_list([])
                safe_update_status(self.status_var, "No ETF news detected")
                return

            tickers = self._collect_unique_tickers(articles)
            self._update_news_temp_list(tickers)

            safe_update_status(self.status_var, "Summarizing Finviz ETF news via Gemini...")

            try:
                summary = gemini_analyzer.summarize_etf_news(articles, tickers)
            except Exception as e:
                logging.error(f"Error summarizing ETF news feed: {e}")
                safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(
                    self.market_news_text,
                    'insert',
                    tk.END,
                    f"总结ETF新闻失败：{e}\nFailed to summarize ETF news: {e}"
                )
                safe_show_message('error', "ETF News", f"Failed to summarize Finviz ETF news: {e}")
                return

            formatted = summary or "Gemini did not return any content."
            self.market_news_original_text = formatted

            safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
            safe_update_text_widget(self.market_news_text, 'insert', tk.END, formatted)
            safe_update_text_widget(self.market_news_text, 'see', "1.0")
            safe_update_status(self.status_var, "ETF news blog updated")

        threading.Thread(target=etf_news_thread, daemon=True, name="FinvizEtfNewsSummary").start()

    def _summarize_crypto_news(self):
        """Summarize Finviz v=5 crypto news feed without requiring ticker selection."""
        try:
            self.chart_notebook.select(self.market_news_frame)
        except Exception:
            pass

        safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
        safe_update_text_widget(
            self.market_news_text,
            'insert',
            tk.END,
            "正在收集 Finviz 加密货币新闻，请稍候...\nGathering Finviz crypto news feed, please wait..."
        )
        self.root.update_idletasks()

        def crypto_news_thread():
            try:
                articles = self._fetch_crypto_news_articles(limit=12)
            except Exception as e:
                logging.error(f"Error fetching crypto news feed: {e}")
                safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(
                    self.market_news_text,
                    'insert',
                    tk.END,
                    f"无法获取加密货币新闻：{e}\nCould not fetch crypto news: {e}"
                )
                self._update_news_temp_list([])
                safe_show_message('error', "Crypto News", f"Could not fetch Finviz crypto news: {e}")
                return

            if not articles:
                safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(
                    self.market_news_text,
                    'insert',
                    tk.END,
                    "未找到最新加密货币新闻。\nNo crypto news items were found."
                )
                self._update_news_temp_list([])
                safe_update_status(self.status_var, "No crypto news detected")
                return

            tickers = self._collect_unique_tickers(articles)
            self._update_news_temp_list(tickers)

            safe_update_status(self.status_var, "Summarizing Finviz crypto news via Gemini...")

            try:
                summary = gemini_analyzer.summarize_crypto_news(articles, tickers)
            except Exception as e:
                logging.error(f"Error summarizing crypto news feed: {e}")
                safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(
                    self.market_news_text,
                    'insert',
                    tk.END,
                    f"总结加密货币新闻失败：{e}\nFailed to summarize crypto news: {e}"
                )
                safe_show_message('error', "Crypto News", f"Failed to summarize Finviz crypto news: {e}")
                return

            formatted = summary or "Gemini did not return any content."
            self.market_news_original_text = formatted

            safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
            safe_update_text_widget(self.market_news_text, 'insert', tk.END, formatted)
            safe_update_text_widget(self.market_news_text, 'see', "1.0")
            safe_update_status(self.status_var, "Crypto news blog updated")

        threading.Thread(target=crypto_news_thread, daemon=True, name="FinvizCryptoNewsSummary").start()

    def _summarize_clipboard_content(self):
        """Summarize content from clipboard which may contain URLs or direct text."""
        try:
            self.chart_notebook.select(self.market_news_frame)
        except Exception:
            pass

        # Get clipboard content
        try:
            clipboard_content = self.root.clipboard_get()
        except tk.TclError:
            safe_show_message('warning', "Clipboard", "Clipboard is empty or contains non-text content.")
            return

        if not clipboard_content or not clipboard_content.strip():
            safe_show_message('warning', "Clipboard", "Clipboard is empty.")
            return

        safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
        safe_update_text_widget(
            self.market_news_text,
            'insert',
            tk.END,
            "正在处理剪贴板内容，请稍候...\nProcessing clipboard content, please wait..."
        )
        self.root.update_idletasks()

        def clipboard_thread():
            # Detect URLs in clipboard content
            url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
            detected_urls = re.findall(url_pattern, clipboard_content)
            
            fetched_content = ""
            valid_urls = []
            
            if detected_urls:
                safe_update_status(self.status_var, f"Found {len(detected_urls)} URL(s), fetching content...")
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                }
                
                for url in detected_urls[:5]:  # Limit to 5 URLs
                    try:
                        # Ensure URL has scheme
                        if url.startswith('www.'):
                            url = 'https://' + url
                        
                        response = requests.get(url, headers=headers, timeout=15)
                        response.raise_for_status()
                        
                        soup = BeautifulSoup(response.text, "html.parser")
                        
                        # Remove script and style elements
                        for script in soup(["script", "style", "nav", "footer", "header"]):
                            script.decompose()
                        
                        # Get text content
                        text = soup.get_text(separator='\n', strip=True)
                        
                        # Clean up excessive whitespace
                        lines = [line.strip() for line in text.splitlines() if line.strip()]
                        text = '\n'.join(lines)
                        
                        if text:
                            fetched_content += f"\n\n--- Content from {url} ---\n{text[:5000]}"
                            valid_urls.append(url)
                    except Exception as e:
                        logging.warning(f"Could not fetch URL {url}: {e}")
                        continue
            
            # Use fetched content if available, otherwise use raw clipboard content
            content_to_summarize = fetched_content if fetched_content else clipboard_content
            
            # Extract stock tickers from both fetched content AND original clipboard content
            all_tickers = []
            # First extract from fetched URL content
            if fetched_content:
                url_tickers = self._extract_tickers_from_text(fetched_content)
                all_tickers.extend(url_tickers)
            # Also extract from original clipboard content (may contain tickers directly)
            clipboard_tickers = self._extract_tickers_from_text(clipboard_content)
            for t in clipboard_tickers:
                if t not in all_tickers:
                    all_tickers.append(t)
            
            self._update_news_temp_list(all_tickers)
            logging.info(f"Extracted {len(all_tickers)} tickers from clipboard/URLs")
            
            safe_update_status(self.status_var, "Summarizing clipboard content via Gemini...")

            try:
                summary = gemini_analyzer.summarize_clipboard_content(content_to_summarize, valid_urls if valid_urls else None)
            except Exception as e:
                logging.error(f"Error summarizing clipboard content: {e}")
                safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(
                    self.market_news_text,
                    'insert',
                    tk.END,
                    f"总结剪贴板内容失败：{e}\nFailed to summarize clipboard content: {e}"
                )
                safe_show_message('error', "Clipboard Summary", f"Failed to summarize clipboard content: {e}")
                return

            formatted = summary or "Gemini did not return any content."
            self.market_news_original_text = formatted

            safe_update_text_widget(self.market_news_text, 'delete', "1.0", tk.END)
            safe_update_text_widget(self.market_news_text, 'insert', tk.END, formatted)
            safe_update_text_widget(self.market_news_text, 'see', "1.0")
            safe_update_status(self.status_var, "Clipboard content summarized")

        threading.Thread(target=clipboard_thread, daemon=True, name="ClipboardSummary").start()

    def _fetch_market_news_articles(self, limit: int = 12) -> List[Dict[str, str]]:
        """Fetch latest market news items from the Finviz base feed."""
        url = "https://elite.finviz.com/news.ashx"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://elite.finviz.com/",
        }
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Error fetching Finviz news: {e}") from e

        soup = BeautifulSoup(response.text, "html.parser")
        return self._parse_finviz_news_articles(soup, limit)

    def _fetch_stock_news_articles(self, limit: int = 12) -> List[Dict[str, str]]:
        """Fetch the latest stock-specific headlines from Finviz (v=3) without needing a ticker."""
        url = "https://elite.finviz.com/news.ashx?v=3"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://elite.finviz.com/",
        }
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Error fetching Finviz stock news: {e}") from e

        soup = BeautifulSoup(response.text, "html.parser")
        return self._parse_finviz_news_articles(soup, limit)

    def _fetch_etf_news_articles(self, limit: int = 12) -> List[Dict[str, str]]:
        """Fetch the latest ETF-specific headlines from Finviz (v=4)."""
        url = "https://elite.finviz.com/news.ashx?v=4"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://elite.finviz.com/",
        }
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Error fetching Finviz ETF news: {e}") from e

        soup = BeautifulSoup(response.text, "html.parser")
        return self._parse_finviz_news_articles(soup, limit)

    def _fetch_crypto_news_articles(self, limit: int = 12) -> List[Dict[str, str]]:
        """Fetch the latest crypto-specific headlines from Finviz (v=5)."""
        url = "https://elite.finviz.com/news.ashx?v=5"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://elite.finviz.com/",
        }
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Error fetching Finviz crypto news: {e}") from e

        soup = BeautifulSoup(response.text, "html.parser")
        return self._parse_finviz_news_articles(soup, limit)

    def _parse_finviz_news_articles(self, soup, limit: int) -> List[Dict[str, str]]:
        articles: List[Dict[str, str]] = []
        ticker_pattern = re.compile(r"t=([A-Za-z0-9\.-]+)")

        def extract_tickers(cell) -> List[str]:
            tickers = set()
            for link in cell.find_all("a"):
                href = link.get("href", "")
                match = ticker_pattern.search(href)
                if match:
                    tickers.add(match.group(1).upper())
                    continue
                text = (link.get_text(strip=True) or "")
                if self._looks_like_ticker(text):
                    tickers.add(text.upper())

            cell_text = cell.get_text(" ", strip=True)
            for match in re.findall(r"\(([A-Z]{1,5})\)", cell_text):
                tickers.add(match.upper())

            return sorted(tickers)

        rows = soup.select("table.fullview-news-outer tr")
        if not rows:
            rows = soup.select("#news table.styled-table-new tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            link_cell = None
            link = None
            for cell in cells[1:]:
                candidate = cell.find("a")
                if candidate:
                    link = candidate
                    link_cell = cell
                    break
            if not link or not link_cell:
                continue
            title = link.get_text(strip=True)
            href = link.get("href", "")
            if not title or not href:
                continue
            snippet = link_cell.get_text(" ", strip=True)
            time_text = cells[0].get_text(strip=True)
            article_tickers = extract_tickers(link_cell)
            articles.append({
                "title": title,
                "url": href,
                "timestamp": time_text,
                "snippet": snippet,
                "tickers": article_tickers,
            })
            if len(articles) >= limit:
                break

        if articles:
            return articles

        # Fallback: general anchors if expected table is missing
        for link in soup.select("a"):
            href = link.get("href", "")
            if not href.startswith("http"):
                continue
            title = link.get_text(strip=True)
            if not title:
                continue
            parent_text = link.parent.get_text(" ", strip=True) if link.parent else title
            detected = []
            match = ticker_pattern.search(href)
            if match:
                detected.append(match.group(1).upper())
            articles.append({
                "title": title,
                "url": href,
                "timestamp": "",
                "snippet": parent_text,
                "tickers": detected,
            })
            if len(articles) >= limit:
                break

        return articles

    def _looks_like_ticker(self, text: str) -> bool:
        if not text:
            return False
        return bool(re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z]{1,2})?", text.strip()))

    def _collect_unique_tickers(self, articles: List[Dict[str, Any]]) -> List[str]:
        seen = []
        for article in articles:
            for ticker in article.get("tickers", []) or []:
                ticker = ticker.upper()
                if ticker and ticker not in seen:
                    seen.append(ticker)
        return seen

    def _extract_tickers_from_text(self, text: str) -> List[str]:
        """Extract potential stock tickers from raw text content."""
        # Common words that look like tickers but aren't (deduplicated)
        common_words = {
            # Single letters and pronouns
            'A', 'I', 'AM', 'PM', 'AN', 'AS', 'AT', 'BE', 'BY', 'DO', 'GO', 'HE', 'IF', 'IN', 'IS', 'IT',
            'ME', 'MY', 'NO', 'OF', 'OK', 'ON', 'OR', 'SO', 'TO', 'UP', 'US', 'WE',
            # Common verbs and articles
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD', 'HER', 'WAS', 'ONE',
            'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD',
            'SEE', 'WAY', 'WHO', 'BOY', 'DID', 'OWN', 'SAY', 'SHE', 'TOO', 'USE',
            # Business titles
            'CEO', 'CFO', 'COO', 'CTO', 'CIO', 'CMO', 'HR', 'IT', 'PR', 'VP', 'SVP', 'EVP',
            # Financial terms
            'IPO', 'ETF', 'SEC', 'FDA', 'FED', 'GDP', 'AI', 'API',
            'Q1', 'Q2', 'Q3', 'Q4', 'YOY', 'QOQ', 'MOM', 'YTD', 'MTD', 'WTD',
            'EPS', 'PE', 'ROI', 'ROE', 'ROA', 'EBITDA', 'GAAP', 'NON', 'VS', 'EST', 'AVG', 'MAX', 'MIN', 'PCT', 'BPS',
            # Currencies and countries
            'USA', 'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'YEN', 'GBX', 'CHF',
            # Company suffixes
            'LLC', 'INC', 'LTD', 'PLC', 'AG', 'SA', 'NV', 'BV', 'AB', 'ASA', 'OYJ', 'SE', 'SPA', 'SARL',
            # HTML/Web terms often found in scraped content
            'HTTP', 'HTTPS', 'HTML', 'CSS', 'JSON', 'XML', 'URL', 'WWW', 'COM', 'ORG', 'NET', 'EDU', 'GOV',
        }
        
        # Pattern: 1-5 uppercase letters, optionally followed by .A or .B (for share classes)
        ticker_pattern = r'\b([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\b'
        
        candidates = re.findall(ticker_pattern, text)
        
        seen = []
        for ticker in candidates:
            ticker = ticker.upper()
            if ticker not in common_words and ticker not in seen and len(ticker) >= 2:
                seen.append(ticker)
        
        # Limit to reasonable number of tickers
        return seen[:50]

    def _save_temp_stock_list(self, tickers: List[str]):
        """Persist detected Finviz tickers into ticker_lists.temp_stocks."""
        try:
            import ticker_lists
            ticker_lists.temp_stocks = tickers.copy()
            logging.info("Saved %d Finviz ticker(s) to temp_stocks list", len(tickers))
        except Exception as e:
            logging.error(f"Failed to update temp_stocks list: {e}")

        self._persist_temp_stocks_to_file(tickers)

        def _switch_to_temp_list():
            try:
                import ticker_lists
                importlib.reload(ticker_lists)
                refreshed = getattr(ticker_lists, "temp_stocks", tickers) or tickers
                self.ticker_lists["temp_stocks"] = refreshed
                if hasattr(self, "ticker_list_combo"):
                    current_values = list(self.ticker_lists.keys())
                    self.ticker_list_combo['values'] = current_values
                self.ticker_list_var.set("temp_stocks")
                self._load_ticker_list()
                self.status_var.set(f"Switched to temp_stocks ({len(refreshed)} tickers)")
            except Exception as e:
                logging.error(f"Failed to reload temp_stocks list: {e}")

        try:
            self.root.after(0, _switch_to_temp_list)
        except Exception:
            _switch_to_temp_list()

    def _persist_temp_stocks_to_file(self, tickers: List[str]):
        ticker_lists_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_lists.py")
        try:
            with open(ticker_lists_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except OSError as e:
            logging.error(f"Could not read ticker_lists.py to persist temp_stocks: {e}")
            return

        new_lines = self._format_temp_stock_lines(tickers)

        start_idx = end_idx = None
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("temp_stocks") and "=" in stripped:
                start_idx = idx
                depth = line.count("[") - line.count("]")
                end_idx = idx
                while depth > 0 and end_idx + 1 < len(lines):
                    end_idx += 1
                    depth += lines[end_idx].count("[") - lines[end_idx].count("]")
                break

        if start_idx is not None and end_idx is not None:
            lines[start_idx:end_idx + 1] = new_lines
        else:
            if lines and lines[-1].strip():
                lines.append("")
            lines.extend(new_lines)

        new_content = "\n".join(lines) + "\n"

        try:
            with open(ticker_lists_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            logging.info("Persisted %d Finviz ticker(s) to ticker_lists.py", len(tickers))
        except OSError as e:
            logging.error(f"Failed to write temp_stocks to ticker_lists.py: {e}")

    def _format_temp_stock_lines(self, tickers: List[str]) -> List[str]:
        tickers = tickers or []
        if not tickers:
            return ["temp_stocks = []"]

        lines = ["temp_stocks = ["]
        for idx, ticker in enumerate(tickers):
            suffix = "," if idx < len(tickers) - 1 else ""
            lines.append(f'    "{ticker}"{suffix}')
        lines.append("]")
        return lines

    def _update_news_temp_list(self, tickers: List[str]):
        tickers = tickers or []
        self.stock_news_temp_tickers = tickers
        self._save_temp_stock_list(tickers)

        if tickers:
            status = f"Saved {len(tickers)} Finviz ticker(s) to 'temp_stocks'"
        else:
            status = "Cleared 'temp_stocks' (no Finviz tickers detected)"

        def _apply():
            safe_update_status(self.status_var, status)

        try:
            self.root.after(0, _apply)
        except Exception:
            _apply()

    def _run_10q_study(self):
        """
        Runs the 10-Q study for the selected ticker by searching the web.
        """
        selected_tickers = self._get_selected_tickers(show_warning=True)
        if not selected_tickers:
            # Check watch list if no ticker is selected in the main list
            selected_indices = self.watch_listbox.curselection()
            if not selected_indices:
                safe_show_message('warning', "No Selection", "Please select a ticker from the 'Available Tickers' or 'Watch List'.")
                return
            selected_tickers = [self.watch_listbox.get(i).strip() for i in selected_indices]

        ticker = selected_tickers[0]
        # Use thread-safe text widget updates
        safe_update_text_widget(self.business_analysis_text, 'delete', "1.0", tk.END)
        safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, 
                               f"Running 10-Q study for {ticker} by searching online...")
        self.root.update_idletasks()

        def study_thread():
            try:
                logging.info(f"Running 10-Q analysis for {ticker}")
                analysis_result = gemini_analyzer.analyze_10q_report(ticker)
                beautified_result = self._beautify_business_analysis(analysis_result, ticker)
                
                # Store the original text for filtering
                self.business_analysis_original_text = beautified_result
                
                # Reset filter when loading new content
                if threading.current_thread() is threading.main_thread():
                    self.business_analysis_filter_var.set("")
                else:
                    # Use thread-safe method to update StringVar
                    self.root.after(0, lambda: self.business_analysis_filter_var.set(""))
                
                # Update UI with results using thread-safe methods
                safe_update_text_widget(self.business_analysis_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, analysis_result)
                safe_update_text_widget(self.business_analysis_text, 'see', "1.0")
                
                # Save the analysis to a file
                try:
                    analysis_dir = os.path.join("stock_data", "business_analysis")
                    os.makedirs(analysis_dir, exist_ok=True)
                    analysis_file = os.path.join(analysis_dir, f"{ticker}_10Q_analysis.txt")
                    with open(analysis_file, "w", encoding="utf-8") as f:
                        f.write(analysis_result)
                    safe_update_status(self.status_var, f"Saved 10-Q analysis for {ticker}")
                except Exception as e:
                    logging.error(f"Could not save 10-Q analysis: {e}")
            except Exception as e:
                logging.error(f"Error in 10-Q analysis thread: {e}")
                safe_update_text_widget(self.business_analysis_text, 'delete', "1.0", tk.END)
                safe_update_text_widget(self.business_analysis_text, 'insert', tk.END, 
                                       f"Error during 10-Q analysis: {e}")

        # Run the study in a separate thread to avoid freezing the GUI
        study_thread_obj = threading.Thread(target=study_thread, daemon=True)
        study_thread_obj.name = f"10QAnalysis-{ticker}"
        study_thread_obj.start()
    
    def _open_10k_report(self):
        """Opens the downloaded 10-K report file."""
        # This button is now disabled as we are fetching from the web
        messagebox.showinfo("Info", "10-K reports are now fetched directly from the web and not saved locally.")

    def _toggle_mock_data(self):
        """Toggle between mock and real SEC API"""
        use_mock = self.use_mock_data_var.get()
        sec_api_wrapper.use_mock_sec_api(use_mock)
        
        if use_mock:
            self.sec_api_status_var.set("Using MOCK SEC data (for testing only)")
        else:
            self.sec_api_status_var.set("Using real SEC API with caching (recommended for production)")
    
    def _clear_sec_cache(self):
        """Clear the SEC API cache"""
        try:
            import shutil
            from pathlib import Path
            
            cache_dir = Path("sec_cache")
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
                self.sec_api_status_var.set("SEC cache cleared successfully")
                messagebox.showinfo("Cache Cleared", "SEC API cache has been cleared successfully.")
            else:
                self.sec_api_status_var.set("No SEC cache to clear")
                messagebox.showinfo("No Cache", "No SEC cache directory found.")
        except Exception as e:
            error_msg = f"Error clearing SEC cache: {str(e)}"
            self.sec_api_status_var.set(error_msg)
            messagebox.showerror("Error", error_msg)
            logging.error(error_msg)
    
    def _extract_sec_filing(self, form_type):
        """Extract tables from SEC filings (10-K or 10-Q) for the selected ticker
        
        Args:
            form_type (str): Either '10-K' or '10-Q'
        """
        # Get selected ticker
        selected_tickers = self._get_selected_tickers(show_warning=True)
        if not selected_tickers:
            # Check watch list if no ticker is selected in the main list
            selected_indices = self.watch_listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("No Selection", "Please select a ticker from the 'Available Tickers' or 'Watch List'.")
                return
            selected_tickers = [self.watch_listbox.get(i).strip() for i in selected_indices]
        
        ticker = selected_tickers[0]
        
        # Update status
        self.status_var.set(f"Extracting {form_type} tables for {ticker}...")
        self.root.update_idletasks()
        
        # Create output directory
        output_dir = os.path.join("sec_filings", ticker)
        os.makedirs(output_dir, exist_ok=True)
        
        # Store ticker and form type in the SEC tab
        self.sec_ticker_var.set(ticker)
        self.sec_form_type_var.set(form_type)
        
        # Switch to SEC filings tab
        self.chart_notebook.select(self.sec_filings_frame)
        # Set active tab to SEC filings
        self.active_tab = "sec_filings"
        logging.info("Switched to SEC filings tab")
        
        # Run extraction in a separate thread to avoid freezing the GUI
        def extraction_thread():
            try:
                # Get SEC API wrapper (will use mock or real based on checkbox)
                api = sec_api_wrapper.sec_api
                using_mock = self.use_mock_data_var.get()
                
                # Update business analysis text area with status
                self.business_analysis_text.delete("1.0", tk.END)
                self.business_analysis_text.insert(tk.END, f"Extracting {form_type} tables for {ticker}...\n\n")
                if using_mock:
                    self.business_analysis_text.insert(tk.END, "Using MOCK SEC data for testing\n\n")
                else:
                    self.business_analysis_text.insert(tk.END, "Using real SEC API with caching and rate limiting\n\n")
                    
                self.business_analysis_text.insert(tk.END, "Step 1: Getting company CIK...\n")
                self.root.update_idletasks()
                
                # Update SEC tab status
                self.sec_status_var.set(f"Step 1: Getting company CIK for {ticker}...")
                self.root.update_idletasks()
                
                # Get company CIK using the wrapper
                start_time = time.time()
                cik = api.get_company_cik(ticker)
                elapsed = time.time() - start_time
                
                if not cik:
                    self.business_analysis_text.insert(tk.END, f"Error: Could not find CIK for {ticker}\n")
                    self.sec_status_var.set(f"Error: Could not find CIK for {ticker}")
                    self.status_var.set(f"Error: Could not find CIK for {ticker}")
                    return
                
                self.business_analysis_text.insert(tk.END, f"Found CIK: {cik} (took {elapsed:.2f}s)\n\n")
                self.business_analysis_text.insert(tk.END, f"Step 2: Getting latest {form_type} filing info...\n")
                self.sec_status_var.set(f"Step 2: Getting latest {form_type} filing info...")
                self.root.update_idletasks()
                
                # Get latest filing info using the wrapper
                start_time = time.time()
                filing_info = api.get_latest_filing_info(cik, form_type)
                elapsed = time.time() - start_time
                
                if not filing_info:
                    self.business_analysis_text.insert(tk.END, f"Error: Could not find {form_type} filing for {ticker}\n")
                    self.sec_status_var.set(f"Error: Could not find {form_type} filing for {ticker}")
                    self.status_var.set(f"Error: Could not find {form_type} filing for {ticker}")
                    return
                
                self.business_analysis_text.insert(tk.END, f"Found {form_type} filing from {filing_info['filingDate']} (took {elapsed:.2f}s)\n")
                self.business_analysis_text.insert(tk.END, f"Filing URL: {filing_info['detailUrl']}\n\n")
                self.business_analysis_text.insert(tk.END, "Step 3: Downloading filing...\n")
                self.sec_status_var.set(f"Step 3: Downloading {form_type} filing from {filing_info['filingDate']}...")
                self.root.update_idletasks()
                
                # Download filing using the wrapper
                start_time = time.time()
                html_content = api.download_filing(filing_info)
                elapsed = time.time() - start_time
                
                if not html_content:
                    self.business_analysis_text.insert(tk.END, "Error: Failed to download filing\n")
                    self.sec_status_var.set("Error: Failed to download filing")
                    self.status_var.set("Error: Failed to download filing")
                    return
                
                self.business_analysis_text.insert(tk.END, f"Successfully downloaded {len(html_content)} bytes (took {elapsed:.2f}s)\n\n")
                self.business_analysis_text.insert(tk.END, "Step 4: Extracting tables...\n")
                self.sec_status_var.set("Step 4: Extracting tables...")
                self.root.update_idletasks()
                
                # Extract tables
                start_time = time.time()
                tables = sec_filing_extractor.extract_tables(html_content)
                elapsed = time.time() - start_time
                
                if not tables:
                    self.business_analysis_text.insert(tk.END, "Error: No tables found in filing\n")
                    self.sec_status_var.set("Error: No tables found in filing")
                    self.status_var.set("Error: No tables found in filing")
                    return
                
                self.business_analysis_text.insert(tk.END, f"Found {len(tables)} tables (took {elapsed:.2f}s)\n\n")
                self.business_analysis_text.insert(tk.END, "Step 5: Identifying financial tables...\n")
                self.sec_status_var.set("Step 5: Identifying financial tables...")
                self.root.update_idletasks()
                
                # Identify financial tables
                start_time = time.time()
                financial_tables = sec_filing_extractor.identify_financial_tables(tables)
                elapsed = time.time() - start_time
                
                # Count identified tables
                identified_count = sum(1 for table in financial_tables.values() if table is not None)
                self.business_analysis_text.insert(tk.END, f"Identified {identified_count} financial tables (took {elapsed:.2f}s)\n\n")
                self.business_analysis_text.insert(tk.END, "Step 6: Saving tables to Excel...\n")
                self.sec_status_var.set("Step 6: Saving tables to Excel...")
                self.root.update_idletasks()
                
                # Save tables to Excel
                start_time = time.time()
                success = sec_filing_extractor.save_tables_to_excel(financial_tables, tables, ticker, output_dir)
                elapsed = time.time() - start_time
                
                if success:
                    self.business_analysis_text.insert(tk.END, f"\nSuccessfully extracted and saved tables for {ticker} (took {elapsed:.2f}s)\n")
                    self.business_analysis_text.insert(tk.END, f"\nFiles saved to: {os.path.abspath(output_dir)}\n")
                    
                    # Store tables for display in SEC tab
                    self.sec_tables = tables
                    self.sec_financial_tables = financial_tables
                    self.sec_output_dir = output_dir
                    
                    # Update SEC tab with available tables
                    self._update_sec_table_list()
                    
                    self.sec_status_var.set(f"Successfully extracted {len(tables)} tables ({identified_count} financial tables)")
                    self.status_var.set(f"Successfully extracted {form_type} tables for {ticker}")
                else:
                    self.business_analysis_text.insert(tk.END, "\nError: Failed to save tables to Excel\n")
                    self.sec_status_var.set("Error: Failed to save tables to Excel")
                    self.status_var.set("Error: Failed to save tables to Excel")
                    
            except Exception as e:
                error_msg = f"Error extracting {form_type} tables: {str(e)}"
                self.business_analysis_text.insert(tk.END, f"\n{error_msg}\n")
                self.sec_status_var.set(error_msg)
                self.status_var.set(error_msg)
                logging.error(error_msg)
                logging.error(traceback.format_exc())
        
        # Start the extraction thread
        threading.Thread(target=extraction_thread, daemon=True).start()
        
    def _extract_sec_tables_from_tab(self):
        """Extract SEC tables from the SEC filings tab"""
        ticker = self.sec_ticker_var.get().strip().upper()
        if not ticker:
            messagebox.showwarning("No Ticker", "Please enter a ticker symbol.")
            return
            
        form_type = self.sec_form_type_var.get()
        self._extract_sec_filing(form_type)
        
    def _open_sec_output_folder(self):
        """Open the SEC output folder for the current ticker"""
        ticker = self.sec_ticker_var.get().strip().upper()
        if not ticker:
            messagebox.showwarning("No Ticker", "Please enter a ticker symbol.")
            return
            
        output_dir = os.path.join("sec_filings", ticker)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            messagebox.showinfo("Directory Created", f"Created new directory for {ticker} SEC filings.")
            
        os.startfile(os.path.abspath(output_dir))
        
    def _update_sec_table_list(self):
        """Update the SEC table list in the SEC filings tab"""
        # Clear existing items
        self.sec_table_listbox.delete(0, tk.END)
        
        if not hasattr(self, 'sec_tables') or not self.sec_tables:
            return
            
        # Add financial tables first with a prefix
        if hasattr(self, 'sec_financial_tables') and self.sec_financial_tables:
            for table_type, table in self.sec_financial_tables.items():
                if table is not None:
                    self.sec_table_listbox.insert(tk.END, f"[Financial] {table_type}")
        
        # Add all tables
        for i, table in enumerate(self.sec_tables):
            # Skip tables that are already in financial tables
            is_financial = False
            if hasattr(self, 'sec_financial_tables') and self.sec_financial_tables:
                for table_type, fin_table in self.sec_financial_tables.items():
                    if fin_table is not None and fin_table.equals(table):
                        is_financial = True
                        break
                        
            if not is_financial:
                self.sec_table_listbox.insert(tk.END, f"Table {i+1}")
        
    def _on_sec_table_selected(self, event):
        """Handle selection of a table in the SEC filings tab"""
        if not hasattr(self, 'sec_tables') or not self.sec_tables:
            return
            
        selected_indices = self.sec_table_listbox.curselection()
        if not selected_indices:
            return
            
        selected_item = self.sec_table_listbox.get(selected_indices[0])
        
        # Determine which table to display
        if selected_item.startswith("[Financial]"): 
            # This is a financial table
            table_type = selected_item.replace("[Financial] ", "")
            if hasattr(self, 'sec_financial_tables') and self.sec_financial_tables and table_type in self.sec_financial_tables:
                table = self.sec_financial_tables[table_type]
                self._display_sec_table(table, f"{self.sec_ticker_var.get()} - {table_type}")
        else:
            # This is a regular table
            table_index = int(selected_item.replace("Table ", "")) - 1
            if 0 <= table_index < len(self.sec_tables):
                table = self.sec_tables[table_index]
                self._display_sec_table(table, f"{self.sec_ticker_var.get()} - {selected_item}")
    
    def _display_sec_table(self, table, title):
        """Display a SEC table in the treeview"""
        # Clear existing columns and items
        for col in self.sec_table_tree['columns']:
            self.sec_table_tree.heading(col, text="")
            
        self.sec_table_tree['columns'] = ()
        for item in self.sec_table_tree.get_children():
            self.sec_table_tree.delete(item)
            
        if table is None or table.empty:
            return
            
        # Configure columns
        columns = list(table.columns)
        self.sec_table_tree['columns'] = columns
        
        # Configure column headings
        for col in columns:
            self.sec_table_tree.heading(col, text=str(col))
            # Set a reasonable width based on content
            max_width = max(len(str(col)), table[col].astype(str).str.len().max() * 10)
            self.sec_table_tree.column(col, width=min(max_width, 300))
            
        # Add data rows
        for i, row in table.iterrows():
            values = [str(row[col]) for col in columns]
            self.sec_table_tree.insert('', 'end', values=values)
            
        # Update the table frame title
        self.sec_table_tree.master.master['text'] = f"Table Content: {title}"
        
    def _export_sec_table_to_excel(self):
        """Export the currently selected SEC table to Excel"""
        if not hasattr(self, 'sec_tables') or not self.sec_tables:
            messagebox.showwarning("No Data", "No tables available to export.")
            return
            
        selected_indices = self.sec_table_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select a table to export.")
            return
            
        selected_item = self.sec_table_listbox.get(selected_indices[0])
        ticker = self.sec_ticker_var.get().strip().upper()
        
        # Determine which table to export
        if selected_item.startswith("[Financial]"):
            # This is a financial table
            table_type = selected_item.replace("[Financial] ", "")
            if hasattr(self, 'sec_financial_tables') and self.sec_financial_tables and table_type in self.sec_financial_tables:
                table = self.sec_financial_tables[table_type]
                filename = f"{ticker}_{table_type.replace(' ', '_')}.xlsx"
        else:
            # This is a regular table
            table_index = int(selected_item.replace("Table ", "")) - 1
            if 0 <= table_index < len(self.sec_tables):
                table = self.sec_tables[table_index]
                filename = f"{ticker}_table_{table_index+1}.xlsx"
            else:
                messagebox.showerror("Error", "Invalid table index.")
                return
                
        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile=filename
        )
        
        if not file_path:
            return  # User cancelled
            
        try:
            table.to_excel(file_path, index=False)
            messagebox.showinfo("Success", f"Table exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export table: {str(e)}")
            
    def _copy_sec_table_to_clipboard(self):
        """Copy the currently selected SEC table to clipboard"""
        if not hasattr(self, 'sec_tables') or not self.sec_tables:
            messagebox.showwarning("No Data", "No tables available to copy.")
            return
            
        selected_indices = self.sec_table_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select a table to copy.")
            return
            
        selected_item = self.sec_table_listbox.get(selected_indices[0])
        
        # Determine which table to copy
        if selected_item.startswith("[Financial]"):
            # This is a financial table
            table_type = selected_item.replace("[Financial] ", "")
            if hasattr(self, 'sec_financial_tables') and self.sec_financial_tables and table_type in self.sec_financial_tables:
                table = self.sec_financial_tables[table_type]
        else:
            # This is a regular table
            table_index = int(selected_item.replace("Table ", "")) - 1
            if 0 <= table_index < len(self.sec_tables):
                table = self.sec_tables[table_index]
            else:
                messagebox.showerror("Error", "Invalid table index.")
                return
                
        try:
            table.to_clipboard(excel=True, index=False)
            messagebox.showinfo("Success", "Table copied to clipboard")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy table: {str(e)}")
