import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from pandastable import Table
import traceback
import difflib

class CSVViewer:
    def __init__(self, app, parent_frame):
        self.app = app
        self.parent_frame = parent_frame
        self.csv_table = None
        self.original_csv_df = None
        self.filtered_csv_df = None
        self.highlighted_columns = {}
        self.last_searched_column = None
        self.visible_columns = None
        self.csv_view_container = None
        self.csv_filter_frame = None
        self.column_search_entry = None
        self.move_to_start_btn = None
        self.save_to_csv_button = None
        self.column_filter_frame = None
        self.column_filter_entry = None
        self.reset_column_filter_btn = None
        self.save_filter_button = None
        self.load_filter_button = None
        self.reset_all_filters_button = None
        self.csv_table_frame = None

        self.setup_csv_viewer()

    def setup_csv_viewer(self):
        """Setup the CSV viewer panel"""
        try:
            if self.csv_view_container:
                self.csv_view_container.destroy()
            self.csv_view_container = ttk.Frame(self.parent_frame)
            self.csv_view_container.pack(fill="both", expand=True)

            self.csv_filter_frame = ttk.Frame(self.csv_view_container)
            self.csv_filter_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
            self.csv_filter_frame.columnconfigure(1, weight=1)
            self.csv_filter_frame.columnconfigure(3, weight=1)

            ttk.Label(self.csv_filter_frame, text="Filter CSV:").grid(row=0, column=0, padx=(0,5))
            self.csv_filter_entry = ttk.Entry(self.csv_filter_frame, textvariable=self.app.csv_filter_text)
            self.csv_filter_entry.grid(row=0, column=1, sticky='ew')

            ttk.Label(self.csv_filter_frame, text="Search Column:").grid(row=0, column=2, padx=(10,5))
            self.column_search_entry = ttk.Entry(self.csv_filter_frame, textvariable=self.app.column_search_var)
            self.column_search_entry.grid(row=0, column=3, sticky='ew')
            self.column_search_entry.bind("<Button-3>", self.show_column_search_menu)
            self.column_search_entry.bind("<Return>", self.show_column_search_menu)

            self.move_to_start_btn = ttk.Button(self.csv_filter_frame, text="Move to Start", command=self.move_searched_column_to_start, width=12)
            self.move_to_start_btn.grid(row=0, column=4, padx=(5,0))

            self.save_to_csv_button = ttk.Button(self.csv_filter_frame, text="Save to CSV", command=self.save_to_filtered_csv, width=12)
            self.save_to_csv_button.grid(row=0, column=5, padx=(5,0))

            self.column_filter_frame = ttk.Frame(self.csv_view_container)
            self.column_filter_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=(0,5))
            self.column_filter_frame.columnconfigure(1, weight=1)

            ttk.Label(self.column_filter_frame, text="Column Filter:").grid(row=0, column=0, padx=(0,5))
            self.column_filter_entry = ttk.Entry(self.column_filter_frame, textvariable=self.app.column_filter_var)
            self.column_filter_entry.grid(row=0, column=1, sticky='ew')

            self.reset_column_filter_btn = ttk.Button(self.column_filter_frame, text="Reset Columns", command=self.reset_column_filter, width=12)
            self.reset_column_filter_btn.grid(row=0, column=2, padx=(5,0))

            self.save_filter_button = ttk.Button(self.column_filter_frame, text="Save Filter", command=self.app.save_current_filter, width=12)
            self.save_filter_button.grid(row=0, column=3, padx=(5,0))

            self.load_filter_button = ttk.Button(self.column_filter_frame, text="Load Filter", command=self.app.show_saved_filters, width=12)
            self.load_filter_button.grid(row=0, column=4, padx=(5,0))

            self.reset_all_filters_button = ttk.Button(self.column_filter_.frame, text="Reset All Filters", command=self.app.reset_all_filters, width=12)
            self.reset_all_filters_button.grid(row=0, column=5, padx=(5,0))

            self.csv_table_frame = ttk.Frame(self.csv_view_container)
            self.csv_table_frame.grid(row=2, column=0, sticky='nsew', padx=5, pady=5)

            self.csv_view_container.rowconfigure(2, weight=1)
            self.csv_view_container.columnconfigure(0, weight=1)

            empty_df = pd.DataFrame()
            self.csv_table = Table(self.csv_table_frame, dataframe=empty_df, showtoolbar=True, showstatusbar=True)
            self.csv_table.show()

            self.setup_csv_filter_context_menu()
            self.setup_column_search_menu()
            self.setup_column_filter_context_menu()

        except Exception as e:
            print(f"Error in setup_csv_viewer: {e}")
            traceback.print_exc()

    def load_csv_file(self, file_path):
        """Load and display the selected CSV file with comprehensive error handling and diagnostics"""
        print(f"\n=== Loading CSV file: {file_path} ===")

        try:
            self.app.current_csv_file = file_path
            df = self.app.actions.advanced_file_read(file_path)

            if df is None or df.empty:
                messagebox.showerror("Error", f"Failed to load or empty file: {file_path}")
                return

            self.original_csv_df = df.copy()

            has_row_filter = self.app.csv_filter_text.get().strip() != ""
            has_column_filter = self.visible_columns is not None

            if has_row_filter or has_column_filter:
                self.filter_csv_content()
            else:
                self.csv_table.model.df = df
                self.csv_table.redraw()

            self.adjust_column_widths()

            filename = os.path.basename(file_path)
            self.app.title(f"CSV Browser - {filename}")

        except Exception as e:
            print(f"Error loading CSV file: {e}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load CSV file:\n{str(e)}")

    def filter_csv_content(self, *args):
        """Advanced CSV content filtering"""
        if self.original_csv_df is None:
            return

        try:
            filter_text = self.app.csv_filter_text.get().strip()

            if filter_text:
                query_part, contains_part = (filter_text.split('@', 1) + [''])[:2]
                filtered_df = self.original_csv_df.copy()

                if query_part:
                    try:
                        filtered_df = filtered_df.query(query_part)
                    except Exception:
                        str_df = filtered_df.astype(str)
                        mask = str_df.apply(lambda x: x.str.contains(query_part, case=False, na=False, regex=False)).any(axis=1)
                        filtered_df = filtered_df[mask]

                if contains_part:
                    str_df = filtered_df.astype(str)
                    mask = pd.Series([True] * len(str_df), index=str_df.index)
                    for term in contains_part.replace('&', '+').split('+'):
                        term = term.strip()
                        if not term: continue
                        if term.startswith('!'):
                            mask &= ~str_df.apply(lambda x: x.str.contains(term[1:], case=False, na=False, regex=False)).any(axis=1)
                        else:
                            mask &= str_df.apply(lambda x: x.str.contains(term, case=False, na=False, regex=False)).any(axis=1)
                    filtered_df = filtered_df[mask]
            else:
                filtered_df = self.original_csv_df.copy()

            self.filtered_csv_df = filtered_df.copy()

            if self.visible_columns:
                self._apply_column_filter_to_filtered_data()
            else:
                self.csv_table.model.df = filtered_df
                self.csv_table.redraw()
        except Exception as e:
            print(f"Error in filter_csv_content: {e}")
            traceback.print_exc()

    def search_columns(self, *args):
        """Search for column names matching the search text"""
        # This will be refactored to use a better search/autocomplete later
        pass

    def show_column_search_menu(self, event=None):
        """Show the column search menu with matching columns"""
        # This will be populated with search results
        pass

    def move_searched_column_to_start(self):
        """Move the last searched column to the start of the dataframe"""
        pass

    def save_to_filtered_csv(self):
        """Save the currently filtered CSV data to a new CSV file"""
        pass

    def reset_column_filter(self):
        """Reset the column filter to show all columns"""
        self.visible_columns = None
        self.app.column_filter_var.set("")
        self.filter_csv_content()

    def setup_csv_filter_context_menu(self):
        """Create a context menu for CSV filter with instructions and examples"""
        pass

    def setup_column_search_menu(self):
        """Set up the right-click menu for column search"""
        pass

    def setup_column_filter_context_menu(self):
        """Create a context menu for column filter with instructions and examples"""
        pass

    def adjust_column_widths(self):
        """Adjust column widths to fit content"""
        if self.csv_table.model.df is None or self.csv_table.model.df.empty:
            return

        df = self.csv_table.model.df
        for col in df.columns:
            col_name_length = len(str(col))
            sample_data = pd.concat([df[col].head(20), df[col].tail(20)])
            try:
                max_data_length = sample_data.astype(str).str.len().max()
            except:
                max_data_length = 0

            pixel_width = max(min((max(col_name_length, max_data_length) * 8), 300), 50)
            self.csv_table.columnwidths[col] = pixel_width
        self.csv_table.redraw()

    def _apply_column_filter_to_filtered_data(self):
        """Apply the stored column filter to the row-filtered data"""
        if self.filtered_csv_df is not None and self.visible_columns:
            valid_columns = [col for col in self.visible_columns if col in self.filtered_csv_df.columns]
            if valid_columns:
                self.csv_table.model.df = self.filtered_csv_df[valid_columns].copy()
            else:
                self.csv_table.model.df = self.filtered_csv_df.copy()
            self.csv_table.redraw()
            self.adjust_column_widths()

    def filter_columns(self, *args):
        """Filter the CSV table to show only columns matching the filter text."""
        if self.original_csv_df is None:
            return

        filter_text = self.app.column_filter_var.get().strip()
        if not filter_text:
            self.reset_column_filter()
            return

        all_columns = self.original_csv_df.columns.tolist()
        filter_terms = [term.strip() for term in filter_text.replace(',', ' ').replace(';', ' ').split() if term]

        include_remaining = "*" in filter_terms
        if include_remaining: filter_terms.remove("*")

        matching_columns = []
        for term in filter_terms:
            for col in all_columns:
                if term.lower() in col.lower() and col not in matching_columns:
                    matching_columns.append(col)

        if include_remaining:
            matching_columns.extend([col for col in all_columns if col not in matching_columns])

        if not matching_columns:
            return

        self.visible_columns = matching_columns

        if self.filtered_csv_df is not None:
            self._apply_column_filter_to_filtered_data()
        else:
            self._apply_column_filter()

    def _apply_column_filter(self):
        """Apply the stored column filter to the current CSV data"""
        if self.original_csv_df is not None and self.visible_columns:
            valid_columns = [col for col in self.visible_columns if col in self.original_csv_df.columns]
            if valid_columns:
                self.csv_table.model.df = self.original_csv_df[valid_columns].copy()
            else:
                self.csv_table.model.df = self.original_csv_df.copy()
            self.csv_table.redraw()
            self.adjust_column_widths()
