"""
CSV Browser Application with Excel-like File List

A tkinter-based CSV browser that displays CSV files in a dual-pane layout with Excel-like tables.
### Fixed
- Fixed issue with filtered table where row selection would load incorrect CSV files

Changelog:
### [2025-01-19]- add button to rename all files based on the field values
### [2025-01-17]Changed
- Added auto column width adjustment for file list table
- Set default sort order to sort by date modified (newest first)

## [2025-01-14] - CSV Browser Update
### Added
- Converted from Image Browser to CSV Browser
- Added CSV file viewing functionality in second panel
- Added Move Files functionality for CSV files
- Added Delete Files functionality for CSV files

### Changed
- Modified file filtering to only show CSV files
- Replaced image grid with pandastable for CSV viewing
- Updated UI layout for CSV file viewing
- Removed image-specific functionality
- Simplified toolbar to remove image-specific controls

Features:
- File browser with:
  - Sorting by name, date, size, and filename fields
  - Dynamic columns based on filename structure
  - Filter functionality across all fields
  - Horizontal scrolling for many fields
- CSV file preview in second panel
- Vertical and horizontal layout options
- Directory selection
- File operations (move, delete)
"""
# Import os and sys first
import os
import sys

# Add custom pandastable path to sys.path BEFORE importing pandastable
custom_pandastable_path = r"C:\Users\juesh\OneDrive\Documents\windsurf\pandastable\pandastable"
# custom_pandastable_path = r"C:\Users\JueShi\OneDrive - Astera Labs, Inc\Documents\windsurf\pandastable\pandastable"
if os.path.exists(custom_pandastable_path):
    # Insert at the beginning of sys.path to prioritize this version
    sys.path.insert(0, custom_pandastable_path)
    print(f"Using custom pandastable from: {custom_pandastable_path}")
    
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import traceback
import difflib
import warnings
import shutil
import subprocess
import json
from datetime import datetime
from pandastable import Table, TableModel
import time
import pyperclip
import copy
import types

warnings.filterwarnings('ignore', category=FutureWarning)

class CSVBrowser(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("CSV Browser")
        self.state('zoomed')
        
        # Suppress specific warnings globally
        import warnings
        warnings.filterwarnings('ignore', message='.*color.*and.*colormap.*cannot be used simultaneously.*')
        warnings.filterwarnings('ignore', message='.*Tight layout not applied.*')
        
        # Initialize variables
        self.current_file = None
        self.is_horizontal = False  # Start with vertical layout
        self.filter_text = tk.StringVar()
        self.filter_text.trace_add("write", self.filter_files)
        self.last_clicked_row = None
        self.csv_frame = None
        self.include_subfolders = tk.BooleanVar(value=False)
        
        # Initialize CSV filter variables
        self.current_csv_filter = ""
        self.csv_filter_text = tk.StringVar(value="")
        self.csv_filter_text.trace_add("write", self.filter_csv_content)
        
        # Initialize column search variable
        self.column_search_var = tk.StringVar(value="")
        self.column_search_var.trace_add("write", self.search_columns)
        
        # Initialize column filter variable
        self.column_filter_var = tk.StringVar(value="")
        self.column_filter_var.trace_add("write", self.filter_columns)
        self.visible_columns = None  # Store currently visible columns
        
        # Store saved filters (row and column)
        self.saved_filters = []  # List to store saved filter configurations
        
        # Store saved file filters
        self.saved_file_filters = []  # List to store saved file filter configurations
        
        # Set maximum number of recent directories to store
        self.max_recent_directories = 5
        
        # Initialize recent directories list
        self.recent_directories = []
        
        # File to store settings (recent directories and filters)
        self.settings_file = os.path.join(os.path.expanduser("~"), "csv_browser_settings.json")
        print(f"Settings file path: {self.settings_file}")

        # Load settings from file
        self.load_settings()
        
        # Track highlighted columns
        self.highlighted_columns = {}
        
        # Initialize last_searched_column
        self.last_searched_column = None
        
        # Initialize plot settings storage
        self.saved_plot_settings = None
        
        # Initialize table settings storage
        self.saved_table_settings = None
        
        # # Initialize frame attributes
        self.pt_frame = ttk.Frame(self)
        self.csv_frame = ttk.Frame(self)
        
        # Set default directory
        # self.current_directory = os.path.dirname(os.path.abspath(__file__))
        # self.current_directory = r"C:\Users\JueShi\Astera Labs, Inc\Silicon Engineering - T3_MPW_Rx_C2M_Test_Results"
        # self.current_directory = r"C:\Users\juesh\OneDrive\Documents\windsurf\stock_data_"
        self.current_directory = r"D:\hp-jue\downloads"
        # self.current_directory = r"C:\Users\JueShi\OneDrive - Astera Labs, Inc\Documents\taurus-sdk-python\validation_Rx_results\100G_KR"        
        
        # Add current directory to recent directories
        self.add_to_recent_directories(self.current_directory)
        
        # Get initial list of CSV files
        self.update_file_list()
        
        # Calculate max fields for the current directory
        self.max_fields = self.get_max_fields()
        
        # Create main container
        self.main_container = ttk.Frame(self)
        self.main_container.pack(fill="both", expand=True)
        
        # Create and pack the toolbar
        self.toolbar = ttk.Frame(self.main_container)
        self.toolbar.pack(fill="x", padx=5, pady=5)
        self.setup_toolbar()
        
        # Create and pack the panels
        self.setup_panels()
        
        # Create the file browser and CSV viewer
        self.setup_file_browser()
        self.setup_csv_viewer()
        
        # Setup column search menu
        self.setup_column_search_menu()
        
        # Set up keyboard shortcuts
        self.bind("<Control-f>", self.focus_column_search)
        
        # Create the application menu
        self.create_menu()

    def normalize_long_path(self, path):
        """Normalize path and add long path prefix if needed"""
        # Normalize the path
        normalized_path = os.path.normpath(os.path.abspath(path))
        
        # If path is longer than 250 characters, add the long path prefix
        if len(normalized_path) > 250 and not normalized_path.startswith('\\\\?\\'):
            # Add Windows long path prefix
            normalized_path = '\\\\?\\' + normalized_path
            print(f"Using long path format: {normalized_path}")
        
        return normalized_path
        
    def format_size(self, size):
        # Convert size to human readable format
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def format_date(self, timestamp):
        # Convert timestamp to readable format
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M")

    def setup_panels(self):
        """Setup the main panels with proper orientation"""
        # Create main paned window with proper orientation
        orient = tk.HORIZONTAL if self.is_horizontal else tk.VERTICAL
        self.paned = ttk.PanedWindow(self.main_container, orient=orient)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Create file browser frame
        self.file_frame = ttk.Frame(self.paned)
        self.paned.add(self.file_frame, weight=1)

        # Create CSV viewer container frame
        self.csv_container = ttk.Frame(self.paned)
        self.paned.add(self.csv_container, weight=2)

        # Set minimum sizes to prevent collapse
        if self.is_horizontal:
            self.file_frame.configure(width=400)
            self.csv_container.configure(width=800)
        else:
            self.file_frame.configure(height=300)
            self.csv_container.configure(height=500)

        # Force geometry update
        self.update_idletasks()
        
        # Set initial sash position
        if self.is_horizontal:
            self.paned.sashpos(0, 400)  # 400 pixels from left
        else:
            self.paned.sashpos(0, 300)  # 300 pixels from top

    def setup_file_browser(self):
        """Setup the file browser panel with pandastable"""
        print("\n=== Setting up file browser ===")  # Debug print
        
        # Create frame for pandastable
        if hasattr(self, 'pt_frame'):
            self.pt_frame.destroy()
        self.pt_frame = ttk.Frame(self.file_frame)
        self.pt_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add filter frame
        filter_frame = ttk.Frame(self.pt_frame)
        filter_frame.pack(fill="x", padx=5, pady=5)
        
        # Add filter label and entry
        ttk.Label(filter_frame, text="Filter Files:").pack(side="left", padx=(0,5))
        filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_text)
        filter_entry.pack(side="left", fill="x", expand=True)
        
        # Add save file filter button
        save_file_filter_btn = ttk.Button(filter_frame, text="Save Filter", command=self.save_file_filter)
        save_file_filter_btn.pack(side="left", padx=5)
        
        # Add load file filter button
        load_file_filter_btn = ttk.Button(filter_frame, text="Load Filter", command=self.show_saved_file_filters)
        load_file_filter_btn.pack(side="left", padx=5)
        
        # Create right-click context menu for filtering instructions
        filter_menu = tk.Menu(filter_entry, tearoff=0)
        filter_menu.add_command(label="Filtering Instructions", state='disabled', font=('Arial', 10, 'bold'))
        filter_menu.add_separator()
        filter_menu.add_command(label="Basic Search: Enter any text to match", state='disabled')
        filter_menu.add_command(label="Multiple Terms: Use space ' 'to combine", state='disabled')
        filter_menu.add_command(label="Exclude Terms: Use '!' prefix", state='disabled')
        filter_menu.add_separator()
        filter_menu.add_command(label="Examples:", state='disabled', font=('Arial', 10, 'bold'))
        filter_menu.add_command(label="'csv': Show files with 'csv'", state='disabled')
        filter_menu.add_command(label="'2024 report': Files with both terms", state='disabled')
        filter_menu.add_command(label="'!temp': Exclude files with 'temp'", state='disabled')
        filter_menu.add_command(label="'csv !old': CSV files, not old", state='disabled')
        
        def show_filter_menu(event):
            filter_menu.tk_popup(event.x_root, event.y_root)
        
        filter_entry.bind('<Button-3>', show_filter_menu)  # Right-click
        
        # Create table frame
        table_frame = ttk.Frame(self.pt_frame)
        table_frame.pack(fill="both", expand=True)
        
        # Create DataFrame for files
        self.update_file_dataframe()
        
        # Create pandastable with editable cells
        self.table = Table(table_frame, dataframe=self.df,
                        showtoolbar=True, showstatusbar=True)
        
        # Enable editing and bind to key events
        self.table.editable = True
        self.table.bind('<Key>', self.on_key_press)
        self.table.bind('<Return>', self.on_return_press)
        self.table.bind('<FocusOut>', self.on_focus_out)
        
        self.table.show()
        
        # Configure table options
        self.table.autoResizeColumns()
        self.table.columnwidths['Name'] = 50
        self.table.columnwidths['File_Path'] = 90

        for col in self.df.columns:
            if col not in ['Name', 'File_Path']:
                # Check if DataFrame is not empty before calculating max width
                if not self.df.empty and len(self.df[col]) > 0:
                    max_width = max(len(str(x)) for x in self.df[col].head(20))
                    self.table.columnwidths[col] = max(min(max_width * 10, 200), 50)
                else:
                    # Set a default width if DataFrame is empty
                    self.table.columnwidths[col] = 100

        self.table.redraw()
        
        # Bind selection event
        self.table.bind('<ButtonRelease-1>', self.on_table_select)
        self.table.bind('<Up>', self.on_key_press)
        self.table.bind('<Down>', self.on_key_press)

    def merge_selected_csv_files(self):
        """Merge selected CSV files with comprehensive error handling and column preservation"""
        try:
            # Check if table exists
            if not hasattr(self, 'table'):
                messagebox.showwarning("Merge CSV", "File table not initialized.")
                return
    
            # Get selected rows indices
            selected_rows = self.table.multiplerowlist
            if not selected_rows or len(selected_rows) < 2:
                messagebox.showwarning("Merge CSV", "Please select at least two CSV files to merge.")
                return
            
            # Get full paths of selected files from the filtered/displayed DataFrame
            displayed_df = self.table.model.df
            selected_files = [displayed_df.iloc[idx]['File_Path'] for idx in selected_rows]
            
            # Read the first CSV as the base file with error handling
            try:
                base_df = pd.read_csv(selected_files[0], index_col=False)
                base_filename = os.path.basename(selected_files[0])
            except UnicodeDecodeError:
                # Try different encodings
                encodings = ['utf-8', 'latin1', 'cp1252']
                for encoding in encodings:
                    try:
                        base_df = pd.read_csv(selected_files[0], encoding=encoding, index_col=False)
                        base_filename = os.path.basename(selected_files[0])
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise Exception(f"Failed to read {selected_files[0]} with any encoding")
            
            # Merge subsequent CSVs
            for file_path in selected_files[1:]:
                try:
                    # Try reading with default encoding
                    next_df = pd.read_csv(file_path, index_col=False)
                except UnicodeDecodeError:
                    # Try different encodings
                    encodings = ['utf-8', 'latin1', 'cp1252']
                    for encoding in encodings:
                        try:
                            next_df = pd.read_csv(file_path, encoding=encoding, index_col=False)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        raise Exception(f"Failed to read {file_path} with any encoding")
                
                # Concatenate with outer join to preserve all columns
                base_df = pd.concat([base_df, next_df], ignore_index=True, join='outer')
            
            # Create merged filename
            merged_filename = os.path.splitext(base_filename)[0] + "_merged.csv"
            merged_filepath = os.path.join(os.path.dirname(selected_files[0]), merged_filename)
            
            # Save merged CSV
            base_df.to_csv(merged_filepath, index=False)
            
            messagebox.showinfo("Merge CSV", f"Files merged successfully!\nMerged file: {merged_filename}")
            
            # Refresh file list to show the new merged file
            self.update_file_list()
        
        except Exception as e:
            messagebox.showerror("Merge CSV Error", f"An error occurred while merging files:\n{str(e)}")

    def filter_files(self, *args):
        """Filter files based on the filter text"""
        if hasattr(self, 'table'):
            try:
                # Get filter text and remove any quotes
                filter_text = self.filter_text.get().lower().strip('"\'')
                print(f"\n=== Filtering files with: '{filter_text}' ===")  # Debug print
                
                if filter_text:
                    # Split filter text by AND (both & and +)
                    filter_terms = [term.strip() for term in filter_text.split()]
                    print(f"Searching for terms: {filter_terms}")  # Debug print
                    
                    # Start with all rows
                    mask = pd.Series([True] * len(self.df), index=self.df.index)
                    
                    # Apply each filter term with AND logic
                    for term in filter_terms:
                        term = term.strip()
                        if term:  # Skip empty terms
                            if term.startswith('!'):  # Exclusion term
                                exclude_term = term[1:].strip()  # Remove ! and whitespace
                                if exclude_term:  # Only if there's something to exclude
                                    term_mask = ~self.df['Name'].str.contains(exclude_term, case=False, na=False)
                                    print(f"Excluding rows with name containing: '{exclude_term}'")  # Debug print
                            else:  # Inclusion term
                                term_mask = self.df['Name'].str.contains(term, case=False, na=False)
                                print(f"Including rows with name containing: '{term}'")  # Debug print
                            mask = mask & term_mask
                    
                    # Debug print matches
                    print("\nMatching results:")
                    for idx, row in self.df[mask].iterrows():
                        print(f"Match found in row {idx}: {row['Name']}")
                    
                    filtered_df = self.df[mask].copy()
                    print(f"\nFound {len(filtered_df)} matching files")  # Debug print
                else:
                    filtered_df = self.df.copy()
                    print("No filter applied, showing all files")  # Debug print
                
                # Update table
                self.table.model.df = filtered_df
                
                # Only update column widths if we have data
                if not filtered_df.empty:
                    # Preserve column widths
                    for col in filtered_df.columns:
                        if col in self.table.columnwidths:
                            width = max(
                                len(str(filtered_df[col].max())),
                                len(str(filtered_df[col].min())),
                                len(col),
                                self.table.columnwidths[col]
                            )
                            self.table.columnwidths[col] = width
                
                # Redraw the table
                self.table.redraw()
                
            except Exception as e:
                print(f"Error in filter_files: {str(e)}")
                traceback.print_exc()  # Print the full traceback for debugging
                self.update_file_dataframe()
    def on_key_press(self, event):
        """Handle key press events"""
        try:
            if event.keysym in ('Up', 'Down'):
                # Handle arrow key navigation
                current_row = self.table.getSelectedRow()
                num_rows = len(self.table.model.df)
                
                if event.keysym == 'Up' and current_row > 0:
                    new_row = current_row - 1
                elif event.keysym == 'Down' and current_row < num_rows - 1:
                    new_row = current_row + 1
                else:
                    return
                
                # Select the new row and ensure it's visible
                self.table.setSelectedRow(new_row)
                self.table.redraw()  # Ensure table updates
                
                # Get filename and path directly from the displayed DataFrame
                displayed_df = self.table.model.df
                if new_row < 0 or new_row >= len(displayed_df):  # Fixed variable name from 'row' to 'new_row'
                    print(f"Row index {new_row} out of bounds for displayed_df of length {len(displayed_df)}")
                    return
                file_path = str(displayed_df.iloc[new_row]['File_Path'])
    
                # Load the CSV file
                self.load_csv_file(file_path)   
            elif event.char and event.char.isprintable():
                row = self.table.getSelectedRow()
                col = self.table.getSelectedColumn()
                if row is not None and col is not None:
                    # Then check for changes
                    self.check_for_changes(row, col)
        except Exception as e:
            print(f"Error handling key press: {e}")
            traceback.print_exc()

    def on_return_press(self, event):
        """Handle return key press"""
        try:
            row = self.table.getSelectedRow()
            col = self.table.getSelectedColumn()
            if row is not None and col is not None:
                self.check_for_changes(row, col)
        except Exception as e:
            print(f"Error handling return press: {e}")
            traceback.print_exc()

    def on_focus_out(self, event):
        """Handle focus out events"""
        try:
            row = self.table.getSelectedRow()
            col = self.table.getSelectedColumn()
            if row is not None and col is not None:
                self.check_for_changes(row, col)
        except Exception as e:
            print(f"Error handling focus out: {e}")
            traceback.print_exc()
    
    def check_for_changes(self, row, col):
        """Check for changes in the cell and handle file renaming"""
        # try:
        #     # Get the displayed DataFrame before any filter operations
        #     displayed_df = self.table.model.df
        #     if row < len(displayed_df):
        #         # Get current filename and path from displayed DataFrame
        #         file_path = os.path.normpath(str(displayed_df.iloc[row]['File_Path']))
        #         current_name = displayed_df.iloc[row]['Name']
                
        #         # Reconstruct filename from Field_ columns
        #         new_filename = self.reconstruct_filename(displayed_df.iloc[row])
                
        #         # If filename is different, rename the file
        #         if new_filename != current_name:
        #             print(f"Renaming file from {current_name} to {new_filename}")  # Debug print
        #             new_filepath = self.rename_csv_file(file_path, new_filename)
                    
        #             if new_filepath != file_path:  # Only update if rename was successful
        #                 # Update the displayed DataFrame
        #                 displayed_df.loc[row, 'Name'] = new_filename
        #                 displayed_df.loc[row, 'File_Path'] = new_filepath
                        
        #                 # Find and update the corresponding row in the original DataFrame
        #                 orig_idx = self.df[self.df['File_Path'] == file_path].index
        #                 if len(orig_idx) > 0:
        #                     # Update all Field_ columns in the original DataFrame
        #                     for col in self.df.columns:
        #                         if col.startswith('Field_'):
        #                             self.df.loc[orig_idx[0], col] = displayed_df.iloc[row][col]
        #                     # Update Name and File_Path
        #                     self.df.loc[orig_idx[0], 'Name'] = new_filename
        #                     self.df.loc[orig_idx[0], 'File_Path'] = new_filepath
                        
        #                 # Store current filter text
        #                 filter_text = self.filter_text.get()
                        
        #                 # Temporarily disable filter to update the model
        #                 if filter_text:
        #                     self.filter_text.set('')
        #                     self.table.model.df = displayed_df
        #                     self.table.redraw()
                        
        #                 # Reapply the filter if it was active
        #                 if filter_text:
        #                     self.filter_text.set(filter_text)
                        
        #                 # Show confirmation
        #                 messagebox.showinfo("File Renamed", 
        #                                 f"File has been renamed from:\n{current_name}\nto:\n{new_filename}")
                        
        #         else:
        #             print("No changes detected")  # Debug print
            
        # except Exception as e:
        #     print(f"Error checking for changes: {e}")
        #     traceback.print_exc()
        pass
    
    def setup_csv_viewer(self):
        """Setup the CSV viewer panel"""
        try:
            # Create frame for CSV viewer
            if hasattr(self, 'csv_frame') and self.csv_frame is not None:
                self.csv_frame.destroy()
            self.csv_frame = ttk.Frame(self.csv_container)
            self.csv_frame.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Create a container frame that will use grid
            self.csv_view_container = ttk.Frame(self.csv_frame)
            self.csv_view_container.pack(fill="both", expand=True)
            
            # Add filter entry for CSV viewer using grid
            self.csv_filter_frame = ttk.Frame(self.csv_view_container)
            self.csv_filter_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
            self.csv_filter_frame.columnconfigure(1, weight=1)  # Make the entry expand
            self.csv_filter_frame.columnconfigure(3, weight=1)  # Make the column search entry expand
            
            ttk.Label(self.csv_filter_frame, text="Filter CSV:").grid(row=0, column=0, padx=(0,5))
            
            # Use the existing StringVar - don't add another trace
            print(f"Current filter value: '{self.csv_filter_text.get()}'")  # Debug print
            
            # Create and set up the entry widget
            self.csv_filter_entry = ttk.Entry(self.csv_filter_frame, textvariable=self.csv_filter_text)
            self.csv_filter_entry.grid(row=0, column=1, sticky='ew')
            
            # Add column search box
            ttk.Label(self.csv_filter_frame, text="Search Column:").grid(row=0, column=2, padx=(10,5))
            self.column_search_entry = ttk.Entry(self.csv_filter_frame, textvariable=self.column_search_var)
            self.column_search_entry.grid(row=0, column=3, sticky='ew')
            self.column_search_entry.bind("<Button-3>", self.show_column_search_menu)
            self.column_search_entry.bind("<Return>", self.show_column_search_menu)
            
            # Add Move to Start button
            self.move_to_start_btn = ttk.Button(
                self.csv_filter_frame, 
                text="Move to Start", 
                command=self.move_searched_column_to_start,
                width=12
            )
            self.move_to_start_btn.grid(row=0, column=4, padx=(5,0))
            
            # Add Save to CSV button
            self.save_to_csv_button = ttk.Button(
                self.csv_filter_frame, 
                text="Save to CSV",
                command=self.save_to_filtered_csv,
                width=12
            )
            self.save_to_csv_button.grid(row=0, column=5, padx=(5,0))
            
            # Add column filter row
            self.column_filter_frame = ttk.Frame(self.csv_view_container)
            self.column_filter_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=(0,5))
            self.column_filter_frame.columnconfigure(1, weight=1)  # Make the entry expand
            
            ttk.Label(self.column_filter_frame, text="Column Filter:").grid(row=0, column=0, padx=(0,5))
            self.column_filter_entry = ttk.Entry(self.column_filter_frame, textvariable=self.column_filter_var)
            self.column_filter_entry.grid(row=0, column=1, sticky='ew')
            
            # Add reset button for column filter
            self.reset_column_filter_btn = ttk.Button(
                self.column_filter_frame,
                text="Reset Columns",
                command=self.reset_column_filter,
                width=12
            )
            self.reset_column_filter_btn.grid(row=0, column=2, padx=(5,0))
            
            # Add save filter button
            self.save_filter_button = ttk.Button(
                self.column_filter_frame,
                text="Save Filter",
                command=self.save_current_filter,
                width=12
            )
            self.save_filter_button.grid(row=0, column=3, padx=(5,0))
            
            # Add load filter button
            self.load_filter_button = ttk.Button(
                self.column_filter_frame,
                text="Load Filter",
                command=self.show_saved_filters,
                width=12
            )
            self.load_filter_button.grid(row=0, column=4, padx=(5,0))
            
            # Add reset all filters button
            self.reset_all_filters_button = ttk.Button(
                self.column_filter_frame,
                text="Reset All Filters",
                command=self.reset_all_filters,
                width=12
            )
            self.reset_all_filters_button.grid(row=0, column=5, padx=(5,0))
            
            # Create frame for pandastable using grid
            self.csv_table_frame = ttk.Frame(self.csv_view_container)
            self.csv_table_frame.grid(row=2, column=0, sticky='nsew', padx=5, pady=5)
            
            # Configure grid weights
            self.csv_view_container.rowconfigure(2, weight=1)
            self.csv_view_container.columnconfigure(0, weight=1)
            
            # Create empty pandastable for CSV viewing
            empty_df = pd.DataFrame()
            self.csv_table = Table(self.csv_table_frame, dataframe=empty_df,
                                showtoolbar=True, showstatusbar=True)
            self.csv_table.show()
            
            # Store original data
            self.original_csv_df = None
            
            # Reset highlighted columns
            self.highlighted_columns = {}
            
            # Setup filter context menu
            self.setup_csv_filter_context_menu()
            
            # Set up the column search menu in row 1
            self.setup_column_search_menu()
            
            # Set up the column filter context menu
            self.setup_column_filter_context_menu()
            
        except Exception as e:
            print(f"Error in setup_csv_viewer: {e}")
            traceback.print_exc()

    def filter_csv_content(self, *args):
        """
        Advanced CSV content filtering with support for:
        1. Pandas query filtering
        2. Text-based contains search
        3. Mixing query and contains search using '@' separator
        4. Column name filtering
        """
        print("\n=== Advanced filter_csv_content called! ===")
        
        if hasattr(self, 'csv_table') and self.original_csv_df is not None:
            try:
                filter_text = self.csv_filter_text.get().strip()
                print(f"Filter text: '{filter_text}'")
                
                # Store the current filter
                self.current_csv_filter = filter_text
                
                # Always start with a fresh copy of the original data
                if filter_text:
                    # Split filter into query and contains parts
                    query_part = ''
                    contains_part = ''
                    
                    # Check for '@' separator
                    if '@' in filter_text:
                        query_part, contains_part = [part.strip() for part in filter_text.split('@', 1)]
                    else:
                        # If no '@', try to use as query first
                        query_part = filter_text
                    
                    # Start with original DataFrame
                    filtered_df = self.original_csv_df.copy()
                    
                    # Apply query filtering if query part exists
                    query_successful = False
                    if query_part:
                        try:
                            # First, check if query part matches a column name exactly or partially
                            column_matches = [col for col in filtered_df.columns if query_part.lower() in col.lower()]
                            
                            if column_matches:
                                # If column matches found, filter to those columns
                                print(f"Column matches found: {column_matches}")
                                filtered_df = filtered_df[column_matches]
                                query_successful = True
                            else:
                                # Try pandas query
                                try:
                                    print(f"Attempting pandas query: '{query_part}'")
                                    filtered_df = filtered_df.query(query_part)
                                    print(f"Query successful, rows after query: {len(filtered_df)}")
                                    query_successful = True
                                except Exception as query_error:
                                    print(f"Query failed: {query_error}")
                                    query_successful = False
                        
                        except Exception as e:
                            print(f"Error in filtering: {e}")
                            filtered_df = self.original_csv_df.copy()
                            query_successful = False
                    
                    # If query was not successful or no query part, perform contains search
                    if not query_successful:
                        print(f"Performing contains search for: '{query_part}'")
                        str_df = filtered_df.astype(str)
                        mask = str_df.apply(
                            lambda x: x.str.contains(query_part, case=False, na=False, regex=False)
                        ).any(axis=1)
                        filtered_df = filtered_df[mask].copy()
                    
                    # Apply contains search if contains part exists
                    if contains_part:
                        # Split contains terms
                        filter_terms = [term.strip() for term in contains_part.split()]
                        print(f"Contains search terms: {filter_terms}")
                        
                        # Convert to string DataFrame for searching
                        str_df = filtered_df.astype(str)
                        
                        # Start with all rows
                        mask = pd.Series([True] * len(str_df), index=str_df.index)
                        
                        # Apply each filter term with AND logic
                        for term in filter_terms:
                            term = term.strip()
                            if term:  # Skip empty terms
                                if term.startswith('!'):  # Exclusion term
                                    exclude_term = term[1:].strip()
                                    if exclude_term:
                                        term_mask = ~str_df.apply(
                                            lambda x: x.str.contains(exclude_term, case=False, na=False, regex=False)
                                        ).any(axis=1)
                                        print(f"Excluding rows containing: '{exclude_term}'")
                                else:  # Inclusion term
                                    term_mask = str_df.apply(
                                        lambda x: x.str.contains(term, case=False, na=False, regex=False)
                                    ).any(axis=1)
                                    print(f"Including rows containing: '{term}'")
                                
                                mask = mask & term_mask
                        
                        # Apply the mask to filtered DataFrame
                        filtered_df = filtered_df[mask].copy()
                    
                    print(f"Final filtered rows: {len(filtered_df)}")
                else:
                    # No filter, use original DataFrame
                    filtered_df = self.original_csv_df.copy()
                    print("No filter text, using original DataFrame")
                
                # Store the row-filtered dataframe for column filtering
                self.filtered_csv_df = filtered_df.copy()
                
                # Apply any active column filter to the row-filtered data
                if hasattr(self, 'visible_columns') and self.visible_columns:
                    self._apply_column_filter_to_filtered_data()
                else:
                    # Update the table with just the row-filtered data
                    self.csv_table.model.df = filtered_df
                    self.csv_table.redraw()
                
            except Exception as e:
                print(f"Error in advanced filter_csv_content: {e}")
                traceback.print_exc()
        else:
            print("No CSV table or original data available")

    def search_columns(self, *args):
        """Search for column names matching the search text"""
        try:
            # First check if we have a valid dataframe to search in
            if not hasattr(self, 'original_csv_df') or self.original_csv_df is None:
                print("No CSV file loaded or dataframe is None")
                return
                
            # Check if we have a valid table
            if not hasattr(self, 'csv_table') or not hasattr(self.csv_table, 'model'):
                print("CSV table not initialized")
                return
                
            search_text = self.column_search_var.get().lower().strip()
            
            if search_text:
                print(f"\n=== Searching columns with: '{search_text}' ===")
            else:
                # If search text is empty, remove any column highlighting
                if hasattr(self.csv_table, 'columncolors'):
                    self.csv_table.columncolors = {}
                    self.csv_table.redraw()
                return
                
            # Check if the entered text matches a column exactly
            all_columns = self.original_csv_df.columns.tolist()
            
            # Find matching columns
            matching_column = None
            partial_matches = []
            
            for col in all_columns:
                if col.lower() == search_text:
                    matching_column = col
                    break
                elif search_text in col.lower():
                    partial_matches.append(col)
            
            if matching_column:
                # Highlight the matching column
                # self.highlight_column(matching_column)
                pass
            elif partial_matches:
                # If no exact match but we have partial matches, show a tooltip or status message
                partial_match_str = ", ".join(partial_matches[:5])
                if len(partial_matches) > 5:
                    partial_match_str += f"... ({len(partial_matches) - 5} more)"
                
                print(f"Partial matches found: {partial_match_str}")
                
                # If there's only one partial match, highlight it
                if len(partial_matches) == 1:
                    # self.highlight_column(partial_matches[0])
                    pass
                else:
                    # Show a tooltip or message about right-clicking
                    self.column_search_entry.config(background='#FFFFCC')  # Light yellow background
                    self.after(1500, lambda: self.column_search_entry.config(background='white'))  # Reset after 1.5 seconds
                    
                    # Create a tooltip-like message
                    tooltip = tk.Toplevel(self)
                    tooltip.wm_overrideredirect(True)
                    tooltip.wm_geometry(f"+{self.winfo_pointerx()+10}+{self.winfo_pointery()+10}")
                    label = tk.Label(tooltip, text=f"Found {len(partial_matches)} matches.\n\nRight-click to select.", 
                                  justify=tk.LEFT, background="#ffffe0", relief=tk.SOLID, borderwidth=1)
                    label.pack()
                    
                    # Auto-destroy after 3 seconds
                    self.after(3000, tooltip.destroy)
            else:
                # No matches found
                print(f"No columns matching '{search_text}' found")
                self.column_search_entry.config(background='#FFCCCC')  # Light red background
                self.after(1500, lambda: self.column_search_entry.config(background='white'))  # Reset after 1.5 seconds
                
        except Exception as e:
            print(f"Error in search_columns: {e}")
            traceback.print_exc()
    
    def on_table_select(self, event):
        """Handle table row selection"""
        try:
            # Get the row that was clicked
            row = self.table.get_row_clicked(event)
            print(f"Clicked row: {row}")  # Debug print
            if self.last_clicked_row == row:
                # Get the current column as well
                col = self.table.get_col_clicked(event)
                if col is not None:
                    self.table.drawCellEntry(row, col)  # Changed from createCellEntry to drawCellEntry
            elif row is not None:            
                # Get the actual data from the filtered/sorted view                
                displayed_df = self.table.model.df
                if row < len(displayed_df):
                    # Get filename and path directly from the displayed DataFrame
                    if row < 0 or row >= len(displayed_df): # Add this check
                        print(f"Row index {row} out of bounds for displayed_df of length {len(displayed_df)}")
                        return
                    file_path = str(displayed_df.iloc[row]['File_Path'])

                    # Load the CSV file
                    self.load_csv_file(file_path)   
                self.last_clicked_row = row
        except Exception as e:
            print(f"Error in table selection: {e}")
            traceback.print_exc()

    def load_csv_file(self, file_path):
        """Load and display the selected CSV file with comprehensive error handling and diagnostics"""
        print(f"\n=== Loading CSV file: {file_path} ===")
        
        try:
            # Save plot settings before loading a new file
            if hasattr(self, 'csv_table') and hasattr(self.csv_table, 'pf'):
                self.save_plot_settings()
            
            # Save table settings before loading a new file
            if hasattr(self, 'csv_table'):
                self.save_table_settings()
            
            # Store the current file path
            self.current_csv_file = file_path
            
            # Try to read the file with advanced error handling
            df = self._advanced_file_read(file_path)
            print(f"DEBUG: Original DataFrame index type: {type(df.index)}")
            print(f"DEBUG: Original DataFrame index name: {df.index.name}")
            print(f"DEBUG: Original DataFrame index sample: {df.index[:5]}")            
            if df is None or df.empty:
                messagebox.showerror("Error", f"Failed to load or empty file: {file_path}")
                return
                
            # Store the original data
            self.original_csv_df = df.copy()

            if hasattr(self, 'saved_table_settings') and self.saved_table_settings is not None:
                if 'index_column' in self.saved_table_settings and self.saved_table_settings['index_column'] is not None:
                    saved_index_column = self.saved_table_settings['index_column']
                    print(f"Found saved index column: {saved_index_column}")
                    
                    # Check if this column exists in the new DataFrame
                    if saved_index_column in df.columns:
                        print(f"Setting {saved_index_column} as index column")
                        # Set this column as the index before applying any filters
                        df = df.set_index(saved_index_column)
                        # Update the original DataFrame as well
                        self.original_csv_df = df.copy()
                                    
            # Restore table settings if available
            self.restore_table_settings()
                                                
            # Check if we have active filters
            has_row_filter = self.csv_filter_text.get().strip() != ""
            has_column_filter = hasattr(self, 'visible_columns') and self.visible_columns is not None
            
            if has_row_filter and has_column_filter:
                # Apply both row and column filtering
                print("Applying both row and column filters")
                # First apply row filter (this will store filtered_csv_df)
                self.filter_csv_content()
                # Column filter will be applied by filter_csv_content
            elif has_row_filter:
                # Apply only row filter
                print("Applying only row filter")
                self.filter_csv_content()
            elif has_column_filter:
                # Apply only column filter
                print("Applying only column filter")
                self._apply_column_filter()
                
                # Ensure the index is properly preserved after column filtering
                self._safe_preserve_index()
            else:
                # No filters, just show the data
                print("No filters active, showing all data")
                self.csv_table.model.df = df
                self.csv_table.redraw()

            print(f"DEBUG: After filters - DataFrame index type: {type(self.csv_table.model.df.index)}")
            print(f"DEBUG: After filters - DataFrame index name: {self.csv_table.model.df.index.name}")
            print(f"DEBUG: After filters - DataFrame index sample: {self.csv_table.model.df.index[:5]}")

            # Ensure the index is properly preserved regardless of which filter was applied
            self._safe_preserve_index()     

            # Adjust column widths to fit content
            self.adjust_column_widths()
            
            # Update window title with filename
            filename = os.path.basename(file_path)
            self.title(f"CSV Browser - {filename}")
            
            # Restore plot settings if available
            self.restore_plot_settings()

            print(f"DEBUG: After restore_table_settings - DataFrame index type: {type(self.csv_table.model.df.index)}")
            print(f"DEBUG: After restore_table_settings - DataFrame index name: {self.csv_table.model.df.index.name}")
            print(f"DEBUG: After restore_table_settings - DataFrame index sample: {self.csv_table.model.df.index[:5]}")

            print(f"Successfully loaded CSV file with {len(df)} rows and {len(df.columns)} columns")
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load CSV file:\n{str(e)}")

    def apply_saved_filter(self, filter_config):
        """Apply a saved filter configuration"""
        try:
            # Get the filter settings
            row_filter = filter_config.get("row_filter", "")
            column_filter = filter_config.get("column_filter", "")
            
            # Apply the row filter
            if row_filter:
                self.csv_filter_text.set(row_filter)
                # Note: filter_csv_content will be called automatically due to the trace
            else:
                # Clear the row filter if empty
                self.csv_filter_text.set("")
            
            # Apply the column filter
            if column_filter:
                self.column_filter_var.set(column_filter)
                # Note: filter_columns will be called automatically due to the trace
            else:
                # Clear the column filter if empty
                self.column_filter_var.set("")
                
            # Show a confirmation message
            filter_name = filter_config.get("name", "Selected")
            messagebox.showinfo("Filter Applied", f"Filter '{filter_name}' has been applied")
            
        except Exception as e:
            print(f"Error applying filter: {e}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to apply filter:\n{str(e)}")
            
    def restore_filter(self, filter_text):
        """Restore the filter text and apply it"""
        try:
            print("\n=== Restoring filter ===")  # Debug print
            print(f"Filter text to restore: '{filter_text}'")  # Debug print
            
            # Only proceed if we have data to filter
            if hasattr(self, 'csv_table') and self.original_csv_df is not None:
                print("CSV table and original data available")  # Debug print
                
                # Update the stored filter
                self.current_csv_filter = filter_text
                
                # Set the filter text - this will trigger filter_csv_content via the trace
                print("Setting filter text...")  # Debug print
                self.csv_filter_text.set(filter_text)
                
                print(f"Current filter text value: '{self.csv_filter_text.get()}'")  # Debug print
                
                # Force a filter operation if the trace didn't trigger
                print("Forcing filter operation...")  # Debug print
                self.filter_csv_content()
                
                # Force a redraw
                print("Forcing table redraw...")  # Debug print
                self.csv_table.redraw()
                
                print("Filter restoration complete")  # Debug print
            else:
                print("No CSV table or original data available")  # Debug print
                
        except Exception as e:
            print(f"Error restoring filter: {e}")
            traceback.print_exc()
    
    def _highlight_column_after_filter(self, column_name):
        """Highlight a column after the filter has been cleared"""
        try:
            print(f"Checking if column '{column_name}' is now visible after clearing filter")
            
            # Clean the column name
            clean_column_name = self._clean_column_name(column_name)
            
            # Debug: Print all available columns for troubleshooting
            current_columns = list(self.csv_table.model.df.columns)
            clean_current_columns = [self._clean_column_name(col) for col in current_columns]
            print(f"Available columns after clearing filter ({len(current_columns)}): {current_columns[:10]}...")
            
            # Check if the column is now visible
            if clean_column_name in clean_current_columns:
                print(f"Column '{clean_column_name}' is now visible after clearing filter")
                # Find the actual column name
                idx = clean_current_columns.index(clean_column_name)
                actual_column = current_columns[idx]
                # self.highlight_column(actual_column)
            else:
                print(f"Column '{clean_column_name}' still not visible after clearing filter")
                
                # Try to find a close match
                closest_matches = difflib.get_close_matches(clean_column_name, clean_current_columns, n=3, cutoff=0.6)
                if closest_matches:
                    print(f"Closest matches after clearing filter: {closest_matches}")
                    
                    # Map back to actual column names
                    actual_matches = []
                    for match in closest_matches:
                        for col, clean_col in zip(current_columns, clean_current_columns):
                            if clean_col == match:
                                actual_matches.append(col)
                                break
                    
                    # Ask user if they want to use a close match
                    msg = f"Column '{clean_column_name}' still not found after clearing filters. Did you mean one of these?\n\n"
                    for i, match in enumerate(actual_matches):
                        msg += f"{i+1}. {self._clean_column_name(match)}\n"
                    
                    response = messagebox.askyesno("Column Not Found", 
                                                 msg + "\nWould you like to use the first match?")
                    if response:
                        column_name = actual_matches[0]
                        print(f"Using closest match after clearing filter: '{column_name}'")
                        # self.highlight_column(column_name)
                    else:
                        messagebox.showinfo("Column Not Found", 
                                           f"Column '{clean_column_name}' could not be found even after clearing filters.")
        except Exception as e:
            print(f"Error in _highlight_column_after_filter: {e}")
            traceback.print_exc()
    
    def _clean_column_name(self, column_name):
        """Remove any decorative elements from column names"""
        # Remove arrow indicators and any other decorative elements
        cleaned = str(column_name)
        cleaned = cleaned.replace('→', '').replace('←', '').strip()
        # Remove any other decorative characters that might be added
        return cleaned

    def setup_csv_filter_context_menu(self):
        """Create a context menu for CSV filter with instructions and examples"""
        # Create context menu
        self.csv_filter_context_menu = tk.Menu(self, tearoff=0)
        
        # Add instructions and examples
        self.csv_filter_context_menu.add_command(
            label="Filtering Instructions", 
            state='disabled', 
            font=('Arial', 10, 'bold')
        )
        
        # Add separator
        self.csv_filter_context_menu.add_separator()
        
        # Detailed examples
        examples = [
            ("Column Name", "Filter columns containing 'name'"),
            ("Pandas Query", "age > 30 and salary < 50000"),
            ("Contains Search", "apple & banana"),
            ("Exclusion", "fruit & !apple"),
            ("Mixed Query", "age > 30 @ apple & banana"),
        ]
        
        for example, description in examples:
            self.csv_filter_context_menu.add_command(
                label=f"{example}: {description}",
                command=lambda ex=example, desc=description: self.show_filter_example(ex, desc)
            )
        
        # Bind right-click to show context menu
        self.csv_filter_entry.bind("<Button-3>", self.show_csv_filter_context_menu)
    
    def show_csv_filter_context_menu(self, event):
        """Display the context menu for CSV filter"""
        try:
            self.csv_filter_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.csv_filter_context_menu.grab_release()
    
    def show_filter_example(self, example, description):
        """Show a detailed tooltip with filter example"""
        # Create a custom tooltip window
        tooltip = tk.Toplevel(self)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{self.winfo_pointerx()+10}+{self.winfo_pointery()+10}")
        
        # Create frame for better styling
        frame = tk.Frame(tooltip, borderwidth=1, relief=tk.SOLID)
        frame.pack(expand=True, fill=tk.BOTH)
        
        # Title
        title_label = tk.Label(frame, text=example, font=('Arial', 10, 'bold'))
        title_label.pack(pady=(5,0))
        
        # Description
        desc_label = tk.Label(frame, text=description, font=('Arial', 9))
        desc_label.pack(pady=(0,5), padx=10)
        
        # Detailed explanation based on example type
        explanations = {
            "Column Name": "Filters columns that contain the specified text.\nCase-insensitive.",
            "Pandas Query": "Use standard pandas query syntax.\nSupports comparisons and logical operators.",
            "Contains Search": "Find rows containing ALL specified terms.\n'&' acts as AND operator.",
            "Exclusion": "Use '!' to exclude rows containing a term.\nCombine with other search terms.",
            "Mixed Query": "Combine pandas query and contains search using '@'.\nQuery before '@', contains after."
        }
        
        explanation_label = tk.Label(frame, text=explanations.get(example, ""), 
                                     font=('Arial', 8), fg='dark gray')
        explanation_label.pack(pady=(0,5), padx=10)
        
        # Auto-destroy after 5 seconds
        self.after(5000, tooltip.destroy)
    
    def setup_toolbar(self):
        """Setup the toolbar with necessary controls"""
        # Add browse folder button
        ttk.Button(self.toolbar, text="Browse Folder", 
                command=self.browse_folder).pack(side="left", padx=5)
            
        # Add load subfolders button
        ttk.Button(self.toolbar, text="Load Subfolders",
                command=self.load_subfolders).pack(side="left", padx=5)

        # Add move files button
        ttk.Button(self.toolbar, text="Move Files", 
                command=self.move_selected_files).pack(side="left", padx=5)

        # Add copy files button
        ttk.Button(self.toolbar, text="Copy Files",
                command=self.copy_selected_files).pack(side="left", padx=5)
                
        # Add delete files button
        ttk.Button(self.toolbar, text="Delete Files", 
                command=self.delete_selected_files).pack(side="left", padx=5)

        # Add rename all files button
        ttk.Button(self.toolbar, text="Rename All Files",
                command=self.rename_all_files).pack(side="left", padx=5)

        # Add reveal in explorer button
        ttk.Button(self.toolbar, text="Reveal in Explorer", 
                command=self.reveal_in_explorer).pack(side="left", padx=5)                

        # Add Merge CSV button
        self.merge_csv_button = ttk.Button(
            self.toolbar, 
            text="Merge CSV", 
            command=self.merge_selected_csv_files
        )
        self.merge_csv_button.pack(side="left", padx=2)

        # Add open in Excel button
        ttk.Button(self.toolbar, text="Open in Excel",
                command=self.open_in_excel).pack(side="left", padx=5)

        # Add open in Spotfire button
        ttk.Button(self.toolbar, text="Open in Spotfire",
                command=self.open_in_spotfire).pack(side="left", padx=5)

        # Add correlation analysis button
        ttk.Button(self.toolbar, text="Correlation Analysis",
                command=self.save_correlation_analysis).pack(side="left", padx=5)

        # Add refresh button  
        ttk.Button(self.toolbar, text="Refresh", 
                    command=self.refresh_file_list).pack(side="left", padx=5)  # Changed from browse_directory to refresh_file_list

        # Add save plot settings button
        ttk.Button(self.toolbar, text="Save Plot Settings",
                command=self.save_plot_settings_to_file).pack(side="right", padx=5)
                
        # Add load plot settings button
        ttk.Button(self.toolbar, text="Load Plot Settings",
                command=self.load_plot_settings_from_file).pack(side="right", padx=5)
        
        # Add save table settings button
        ttk.Button(self.toolbar, text="Save Table Settings",
                command=self.save_table_settings).pack(side="right", padx=5)

    def browse_folder(self):
        """Open a directory chooser dialog and update the file list"""
        print("\n=== Browse folder called ===")  # Debug print
        directory = filedialog.askdirectory(
            initialdir=self.current_directory
        )
        if directory:
            print(f"Selected directory: {directory}")  # Debug print
            self.current_directory = directory
            # Add to recent directories list
            self.add_to_recent_directories(directory)
            # Force explicit menu update
            self.update_recent_directories_menu()
            print(f"Recent directories menu updated with {len(self.recent_directories)} items")
            # Save settings explicitly
            print("Explicitly saving settings after directory change")
            self.save_settings()
            
            self.include_subfolders.set(False)  # Reset to not include subfolders
            
            # Update file list
            self.update_file_list()
            
            # Update max fields
            old_max = self.max_fields
            self.max_fields = self.get_max_fields()
            print(f"Max fields changed from {old_max} to {self.max_fields}")  # Debug print
            
            # Update file browser
            self.setup_file_browser()
            self.setup_csv_viewer()

    def update_file_list(self):
        """Update the list of CSV and TSV files"""
        print("\n=== Updating file list ===")  # Debug print
        print(f"Current directory: {self.current_directory}")
        print(f"Include subfolders: {self.include_subfolders.get()}")
        
        # Normalize the current directory path
        try:
            normalized_directory = self.normalize_long_path(self.current_directory)
            print(f"Normalized directory: {normalized_directory}")
        except Exception as e:
            print(f"Error normalizing directory path: {e}")
            normalized_directory = self.current_directory
        
        if self.include_subfolders.get():
            # Walk through all subdirectories with error handling
            self.csv_files = []
            try:
                for root, _, files in os.walk(normalized_directory):
                    try:
                        # Normalize the root path for each iteration
                        normalized_root = self.normalize_long_path(root)
                        print(f"Searching in subdirectory: {normalized_root}")
                        
                        for file in files:
                            if file.lower().endswith(('.csv', '.tsv')):
                                # Use normpath to ensure consistent path separators
                                full_path = os.path.normpath(os.path.join(normalized_root, file))
                                
                                # Additional path validation
                                if os.path.exists(full_path):
                                    self.csv_files.append(full_path)
                                    print(f"Found file: {full_path}")
                                else:
                                    print(f"File not accessible: {full_path}")
                    except PermissionError:
                        print(f"Permission denied accessing directory: {normalized_root}")
                    except Exception as dir_error:
                        print(f"Error searching directory {normalized_root}: {dir_error}")
                
                print(f"Found {len(self.csv_files)} CSV/TSV files in all subfolders")
            except Exception as walk_error:
                print(f"Critical error during file walk: {walk_error}")
                self.csv_files = []
        else:
            # Only get files in current directory
            try:
                files = os.listdir(normalized_directory)
                # Use normpath for consistent path separators
                self.csv_files = [
                    os.path.normpath(os.path.join(normalized_directory, f)) 
                    for f in files 
                    if f.lower().endswith(('.csv', '.tsv')) and os.path.isfile(os.path.join(normalized_directory, f))
                ]
                print(f"Found {len(self.csv_files)} CSV/TSV files in current directory")
            except PermissionError:
                print(f"Permission denied accessing directory: {normalized_directory}")
                self.csv_files = []
            except Exception as list_error:
                print(f"Error listing directory: {list_error}")
                self.csv_files = []
        
        # Print out all found files for verification
        for file in self.csv_files:
            print(f"Discovered file: {file}")
        
        if not self.csv_files:
            print("WARNING: No CSV or TSV files found. Check directory path and permissions.")
        
    def get_max_fields(self):
        """Get the maximum number of underscore-separated fields in filenames"""
        max_fields = 0
        for file_path in self.csv_files:
            # Get just the filename without path
            file_name = os.path.basename(file_path)
            # Remove extension and split by underscore
            name_without_ext = os.path.splitext(file_name)[0]
            fields = name_without_ext.split('_')
            
            # Add all field columns at once
            for i in range(len(fields)):
                field_name = f'Field_{i+1}'
                max_fields = max(max_fields, i+1)
        
        # Ensure at least 25 fields even if directory is empty
        max_fields = max(max_fields, 25)
        print(f"Max fields found: {max_fields}")  # Debug print
        return max_fields

    def open_in_excel(self):
        """Open the selected CSV file in Excel"""
        selected_rows = self.table.multiplerowlist
        if not selected_rows:
            messagebox.showinfo("Info", "Please select a file to open in Excel")
            return
    
        try:
            # Get the filtered/displayed DataFrame
            displayed_df = self.table.model.df
            
            for row in selected_rows:
                if row < len(displayed_df):
                    file_path = displayed_df.iloc[row]['File_Path']
                    os.startfile(file_path)  # This will open the file with its default application (Excel for CSV)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file in Excel:\n{str(e)}")

    def open_in_spotfire(self):
        """Open the first selected CSV file in Spotfire and copy remaining paths to clipboard"""
        selected_rows = self.table.multiplerowlist
        if not selected_rows:
            messagebox.showinfo("Info", "Please select file(s) to open in Spotfire")
            return
    
        # Create a temporary directory for files
        temp_dir = os.path.join(os.environ.get('TEMP', os.path.expanduser('~')), 'CSVBrowser_temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Clear any existing files in the temp directory
        for f in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, f))
            except:
                pass
        
        try:
            spotfire_path = r"C:\Users\JueShi\AppData\Local\Spotfire Cloud\14.4.0\Spotfire.Dxp.exe"
            
            # Get all selected file paths and process them
            file_paths = []
            temp_paths = []  # To store any temporary copies
            
            # Get the filtered/displayed DataFrame
            displayed_df = self.table.model.df
            
            for row in selected_rows:
                if row < len(displayed_df):
                    original_path = displayed_df.iloc[row]['File_Path']
                    file_path = self.normalize_long_path(original_path)
                    
                    # If path is too long, create a temporary copy with a shorter name
                    if len(file_path) > 200:  # Conservative limit
                        file_name = os.path.basename(file_path)
                        # Create a shorter name by using index and hash
                        short_name = f"{row}_{hash(file_path) % 1000000}_{file_name[-50:]}" if len(file_name) > 50 else file_name
                        temp_path = os.path.join(temp_dir, short_name)
                        
                        # Copy the file to temp location
                        try:
                            shutil.copy2(file_path, temp_path)
                            print(f"Created temporary copy at: {temp_path}")
                            file_paths.append(temp_path)
                            temp_paths.append(temp_path)  # Remember to track this for cleanup
                        except Exception as copy_error:
                            print(f"Error creating temporary copy: {str(copy_error)}")
                            # If copying fails, try with original path
                            file_paths.append(file_path)
                    else:
                        file_paths.append(file_path)
            
            if file_paths:
                num_files = len(file_paths)
                
                # Show info about temporary files if any were created
                if temp_paths:
                    messagebox.showinfo("Long Paths Found", 
                        f"Created {len(temp_paths)} temporary copies with shorter paths in {temp_dir}.\n"
                        "These files will remain until the application is closed.")
                
                # If there's only one file, just open it
                if num_files == 1:
                    subprocess.Popen([spotfire_path, file_paths[0]])
                    print(f"Opened single file in Spotfire: {file_paths[0]}")
                else:
                    # Open first file in Spotfire
                    first_file = file_paths[0]
                    subprocess.Popen([spotfire_path, first_file])
                    
                    # Copy remaining file paths to clipboard with double quotes
                    remaining_files = file_paths[1:]
                    quoted_paths = [f'"{path}"' for path in remaining_files]
                    clipboard_text = " ".join(quoted_paths)
                    
                    # Copy to clipboard
                    pyperclip.copy(clipboard_text)
                    
                    # Show simple notification
                    messagebox.showinfo("Multiple Files", 
                        f"Opened first file in Spotfire.\n\n"
                        f"Remaining {len(remaining_files)} file paths have been copied to clipboard with quotes.\n\n"
                        f"Use File > Add Data Table... in Spotfire to add the additional files.")
                    
                    print(f"Opened first file in Spotfire and copied {len(remaining_files)} paths to clipboard")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file(s) in Spotfire:\n{str(e)}")


    def get_top_correlated_columns(self, data, target_column, top_n=10):
        """
        Get the top N most correlated columns to a specified target column in a DataFrame.
        Excludes constant columns (columns with zero variance).

        Parameters:
        - data (pd.DataFrame): The input DataFrame.
        - target_column (str): The name of the target column.
        - top_n (int): The number of top correlated columns to return.

        Returns:
        - tuple: (correlation_df, data_df) where:
            - correlation_df: DataFrame with correlation information
            - data_df: DataFrame with all columns, but correlated columns moved to the front
        """
        # Store original column order for non-correlated columns
        original_columns = list(data.columns)
        
        # Ensure the target column exists in the dataset
        if target_column not in data.columns:
            raise ValueError(f"The specified target column '{target_column}' does not exist in the dataset.")
        
        # Convert all possible columns to numeric
        numeric_data = data.apply(pd.to_numeric, errors='coerce')
        
        # Remove columns with all NaN values (non-numeric columns)
        numeric_data = numeric_data.dropna(axis=1, how='all')
        
        # Ensure the target column is still in the dataset
        if target_column not in numeric_data.columns:
            raise ValueError(f"The target column '{target_column}' could not be converted to numeric type.")
        
        # Remove constant columns (columns with zero variance)
        non_constant_data = numeric_data.loc[:, numeric_data.nunique() > 1]
        
        # Check if the target column still exists after filtering
        if target_column not in non_constant_data.columns:
            raise ValueError(f"The target column '{target_column}' is constant (has zero variance).")
        
        # Calculate the correlation matrix
        cor_matrix = non_constant_data.corr()
        
        # Extract correlations for the target column
        cor_target = cor_matrix[target_column]
        
        # Create a DataFrame for sorting
        cor_target_df = pd.DataFrame({
            'Column': cor_target.index,
            'Correlation': cor_target.values
        })
        
        # Sort by absolute correlation and get the top N (excluding the target column itself)
        top_correlated = cor_target_df[cor_target_df['Column'] != target_column] \
            .sort_values(by='Correlation', key=abs, ascending=False) \
            .head(top_n)
        
        # Get the column names for the top correlated columns
        selected_columns = list(top_correlated['Column'])
        
        # Create the final column order:
        # 1. Target column
        # 2. Top correlated columns
        # 3. Remaining columns (in original order)
        final_columns = [target_column] + selected_columns
        remaining_columns = [col for col in original_columns if col not in final_columns]
        final_columns.extend(remaining_columns)
        
        # Create a DataFrame with all data in the new column order
        data_df = data[final_columns].copy()
        
        return top_correlated, data_df

    def save_correlation_analysis(self):
        """Save correlation analysis for the selected CSV file"""
        if self.current_csv_file is None or not hasattr(self, 'csv_table'):
            messagebox.showinfo("Info", "Please load a CSV file first")
            return

        # Get the current DataFrame
        df = self.csv_table.model.df
        
        # Get numeric columns that have variation
        numeric_columns = self.get_numeric_varying_columns(df)
        
        if not numeric_columns:
            messagebox.showerror("Error", "No numeric columns with variation found in the dataset")
            return
        
        # Create a dialog to get user input
        dialog = tk.Toplevel(self)
        dialog.title("Correlation Analysis Settings")
        dialog.geometry("400x200")
        
        # Make dialog modal
        dialog.transient(self)
        dialog.grab_set()
        
        # Create and pack widgets
        ttk.Label(dialog, text="Target Column:").pack(pady=5)
        target_var = tk.StringVar(value="")
        target_combo = ttk.Combobox(dialog, textvariable=target_var)
        # Set only numeric columns with variation in the dropdown
        target_combo['values'] = numeric_columns
        # Set the first numeric column as default if available
        if numeric_columns:
            target_combo.set(numeric_columns[0])
        target_combo.pack(pady=5)
        
        ttk.Label(dialog, text="Number of top correlated columns:").pack(pady=5)
        # Default to number of numeric columns - 1 (excluding target column)
        default_n = len(numeric_columns) - 1
        num_cols_var = tk.StringVar(value=str(default_n))
        num_cols_entry = ttk.Entry(dialog, textvariable=num_cols_var)
        # Add a note about the maximum available columns
        ttk.Label(dialog, text=f"(Maximum available: {default_n} columns)").pack(pady=2)
        num_cols_entry.pack(pady=5)
        
        def on_ok():
            target_col = target_var.get()
            if not target_col:
                messagebox.showerror("Error", "Please select a target column")
                return
                
            try:
                num_cols = int(num_cols_var.get())
                if num_cols <= 0:
                    raise ValueError("Number of columns must be positive")
                
                # Limit number of columns to available numeric columns - 1
                max_cols = len(numeric_columns) - 1
                if num_cols > max_cols:
                    if not messagebox.askyesno("Warning", 
                        f"Requested {num_cols} columns but only {max_cols} numeric columns available (excluding target).\n"
                        f"Continue with {max_cols} columns?"):
                        return
                    num_cols = max_cols
                
                # Get correlation analysis
                try:
                    corr_df, data_df = self.get_top_correlated_columns(df, target_col, num_cols)
                    
                    # Ask user where to save the results
                    base_path = filedialog.asksaveasfilename(
                        defaultextension=".csv",
                        filetypes=[("CSV files", "*.csv")],
                        initialdir=os.path.dirname(self.current_csv_file),
                        # Use a shorter filename by taking first 30 chars of the original name
                        initialfile=f"corr-{os.path.basename(self.current_csv_file)[:30]}"
                    )
                    
                    if base_path:
                        # Remove .csv extension if present
                        base_path = base_path.rsplit('.csv', 1)[0]
                        base_path = os.path.normpath(os.path.abspath(base_path))
                        base_path = self.normalize_long_path(base_path)

                        # Save correlation results
                        # Use shorter suffixes for output files
                        corr_path = f"{base_path}_corr.csv"
                        corr_df.to_csv(corr_path, index=False)
                        
                        # Save data with all columns
                        data_path = f"{base_path}_data.csv"
                        data_df.to_csv(data_path, index=False)
                        
                        # Create and save correlation plots
                        plots_dir = self.create_correlation_plots(data_df, target_col, corr_df, base_path)
                        
                        
                        success_message = (
                            f"Analysis saved to:\n"
                            f"Correlations: {os.path.basename(corr_path)}\n"
                            f"Data: {os.path.basename(data_path)}\n"
                        )
                        
                        if plots_dir:
                            success_message += (
                                f"\nCorrelation plots saved to:\n"
                                f"{os.path.basename(plots_dir)}/\n"
                                f"- Individual correlation plots\n"
                                f"- Combined correlation plot\n"
                            )
                        
                        success_message += "\nNote: In the data file, columns are ordered by correlation strength"
                        
                        messagebox.showinfo("Success", success_message)
                        dialog.destroy()
                        
                except Exception as e:
                    messagebox.showerror("Error", f"Error in correlation analysis:\n{str(e)}")
                
            except ValueError as e:
                messagebox.showerror("Error", "Please enter a valid positive number for top columns")
        
        def on_cancel():
            dialog.destroy()
        
        # Add OK and Cancel buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        # Center the dialog on the screen
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'{width}x{height}+{x}+{y}')

    def create_correlation_plots(self, data_df, target_column, corr_df, base_path):
        """
        Create scatter plots for each correlated column vs target column.
        
        Parameters:
        - data_df: DataFrame containing the data
        - target_column: Name of the target column
        - corr_df: DataFrame containing correlation information
        - base_path: Base path for saving plots
        """
        try:
            # Check for required packages
            try:
                import matplotlib.pyplot as plt
                from math import ceil
                import numpy as np
                import pandas as pd
            except ImportError as e:
                messagebox.showerror("Error", 
                    "Required plotting package not found. Please install matplotlib:\n"
                    "pip install matplotlib")
                return None
            
            # Ensure data types are numeric
            try:
                # Convert target column to numeric
                data_df[target_column] = pd.to_numeric(data_df[target_column], errors='coerce')
                
                # Convert correlated columns to numeric
                for _, row in corr_df.iterrows():
                    corr_col = row['Column']
                    data_df[corr_col] = pd.to_numeric(data_df[corr_col], errors='coerce')
                
                # Remove any rows with NaN values
                data_df = data_df.dropna(subset=[target_column] + corr_df['Column'].tolist())
                
                if len(data_df) == 0:
                    messagebox.showerror("Error", "No valid numeric data found after conversion")
                    return None
                    
            except Exception as e:
                messagebox.showerror("Error", f"Error converting data to numeric format:\n{str(e)}")
                return None
            
            # Create a directory for plots if it doesn't exist
            # plots_dir = os.path.normpath(f"{base_path}_plots")
            plots_dir = self.normalize_long_path(f"{base_path}_plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Set style for better visibility
            plt.style.use('default')
            
            # Create individual scatter plots
            for _, row in corr_df.iterrows():
                corr_col = row['Column']
                corr_val = row['Correlation']
                
                # Create figure with white background
                plt.figure(figsize=(10, 6), facecolor='white')
                
                # Get numeric data for plotting
                x_data = data_df[target_column].values  # Target column on x-axis
                y_data = data_df[corr_col].values      # Correlated column on y-axis
                
                # Create scatter plot with improved styling
                plt.scatter(x_data, y_data, alpha=0.5, c='#1f77b4', edgecolors='none')
                
                # Add trend line
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                x_range = np.linspace(x_data.min(), x_data.max(), 100)
                plt.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)
                
                # Add labels and title with improved styling
                plt.xlabel(target_column, fontsize=12, fontweight='bold')  # Target column on x-axis
                plt.ylabel(corr_col, fontsize=12, fontweight='bold')      # Correlated column on y-axis
                plt.title(f'Correlation: {corr_val:.3f}', fontsize=14, pad=15)
                
                # Add grid with lighter style
                plt.grid(True, alpha=0.3, linestyle='--')
                
                # Improve tick label visibility
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                
                # Add a light box around the plot
                plt.box(True)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save plot with white background
                plt.savefig(os.path.normpath(os.path.join(plots_dir, f"{corr_col}_vs_{target_column}.png")), 
                           dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
            
            # Create combined plot
            n_cols = min(len(corr_df), 3)
            n_rows = ceil(len(corr_df) / n_cols)
            
            # Create figure with white background
            fig, axes = plt.subplots(n_rows, n_cols, 
                                   figsize=(5*n_cols, 4*n_rows), 
                                   facecolor='white')
            fig.suptitle(f'Correlation Plots for {target_column}', 
                        fontsize=16, y=1.02, fontweight='bold')
            
            # Flatten axes for easier iteration
            if n_rows * n_cols > 1:
                axes_flat = axes.flatten()
            else:
                axes_flat = [axes]
            
            # Create subplots
            for idx, (_, row) in enumerate(corr_df.iterrows()):
                corr_col = row['Column']
                corr_val = row['Correlation']
                
                ax = axes_flat[idx]
                
                # Get numeric data for plotting
                x_data = data_df[target_column].values  # Target column on x-axis
                y_data = data_df[corr_col].values      # Correlated column on y-axis
                
                # Create scatter plot with improved styling
                ax.scatter(x_data, y_data, alpha=0.5, c='#1f77b4', edgecolors='none')
                
                # Add trend line
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                x_range = np.linspace(x_data.min(), x_data.max(), 100)
                ax.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)
                
                # Add labels and title with improved styling
                ax.set_xlabel(target_column, fontsize=10, fontweight='bold')  # Target column on x-axis
                ax.set_ylabel(corr_col, fontsize=10, fontweight='bold')      # Correlated column on y-axis
                ax.set_title(f'Correlation: {corr_val:.3f}', fontsize=12)
                
                # Add grid with lighter style
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Improve tick label visibility
                ax.tick_params(labelsize=8)
                
                # Add a light box around the plot
                ax.set_frame_on(True)
            
            # Remove empty subplots
            for idx in range(len(corr_df), len(axes_flat)):
                fig.delaxes(axes_flat[idx])
            
            # Adjust layout
            plt.tight_layout()
            
            # Save combined plot with white background
            plt.savefig(os.path.normpath(os.path.join(plots_dir, "combined_correlation_plots.png")), 
                        dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return plots_dir
            
        except Exception as e:
            error_msg = str(e)
            messagebox.showerror("Error", f"Error creating correlation plots:\n{error_msg}")
            return None

    def get_numeric_varying_columns(self, data):
        """
        Get list of columns that are numeric and have more than one unique value.
        
        Parameters:
        - data (pd.DataFrame): Input DataFrame
        
        Returns:
        - list: List of column names that are numeric and non-constant
        """
        numeric_columns = []
        for col in data.columns:
            # Try to convert to numeric
            numeric_series = pd.to_numeric(data[col], errors='coerce')
            # Check if conversion was successful (not all NaN) and has more than one unique value
            if not numeric_series.isna().all() and data[col].nunique() > 1:
                numeric_columns.append(col)
        return numeric_columns

    def reconstruct_filename(self, row):
        """Reconstruct filename from columns starting with 'Field_'"""
        # Find columns starting with 'Field_'
        f_columns = [col for col in row.index if col.startswith('Field_')]
        
        # Extract values from these columns, skipping None/empty values
        filename_parts = [str(row[col]) for col in f_columns if pd.notna(row[col]) and str(row[col]).strip()]
        
        # Join with underscore, add .csv extension
        new_filename = '_'.join(filename_parts).replace('\n', '') + '.csv'
        
        return new_filename
    
    def rename_csv_file(self, old_filepath, new_filename):
        """Rename CSV file on disk"""
        try:
            # Convert to raw string and normalize path separators
            old_filepath = os.path.normpath(str(old_filepath).replace('/', os.sep))
            directory = os.path.dirname(old_filepath)
            new_filepath = os.path.normpath(os.path.join(directory, new_filename))
            
            print("\nRenaming file details:")  # Debug prints
            print(f"Original path: {old_filepath}")
            print(f"Directory: {directory}")
            print(f"New filename: {new_filename}")
            print(f"New full path: {new_filepath}")
            
            # Check if source file exists
            if not os.path.exists(old_filepath):
                print(f"Source file not found: {old_filepath}")  # Debug print
                raise FileNotFoundError(f"Source file not found: {old_filepath}")
            
            # Check if target directory exists
            if not os.path.exists(directory):
                raise FileNotFoundError(f"Target directory not found: {directory}")
            
            # Check if target file already exists
            if os.path.exists(new_filepath):
                raise FileExistsError(f"Target file already exists: {new_filepath}")
            
            # Rename the file
            print(f"Performing rename operation...")  # Debug print
            os.rename(old_filepath, new_filepath)
            print(f"Rename successful")  # Debug print
            return new_filepath
            
        except Exception as e:
            print(f"\nError renaming file:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Old path: {old_filepath}")
            print(f"New filename: {new_filename}")
            print(f"Directory: {directory}")
            print("\nFull traceback:")
            traceback.print_exc()
            return old_filepath

    def rename_all_files(self):
        """Rename all files where constructed name differs from current name"""
        try:
            renamed_count = 0
            for idx, row in self.df.iterrows():
                current_name = row['Name']
                new_name = self.reconstruct_filename(row)
                
                if new_name != current_name:
                    old_path = row['File_Path']
                    new_path = self.rename_csv_file(old_path, new_name)
                    
                    # Update the DataFrame
                    self.df.at[idx, 'Name'] = new_name
                    self.df.at[idx, 'File_Path'] = new_path
                    renamed_count += 1
            
            if renamed_count > 0:
                # Update the table display
                self.table.model.df = self.df
                self.table.redraw()
                messagebox.showinfo("Files Renamed", 
                                 f"Renamed {renamed_count} files based on field values")
            else:
                messagebox.showinfo("No Changes", 
                                 "All filenames already match their field values")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to rename files: {str(e)}")

    def refresh_file_list(self):
        """Refresh the file list and reload the currently selected file if any"""
        print("\n=== Refreshing file list ===")  # Debug print
        
        # Store the currently selected file path
        current_selected_file = None
        current_selected_row = None
        
        # Check if a file is currently selected
        if hasattr(self, 'table') and hasattr(self, 'df'):
            selected_rows = self.table.multiplerowlist
            if selected_rows and selected_rows[0] < len(self.df):
                current_selected_row = selected_rows[0]
                current_selected_file = self.df.iloc[current_selected_row]['File_Path']
        
        # Update the file list
        self.update_file_list()
        
        # Update max fields
        old_max = self.max_fields
        self.max_fields = self.get_max_fields()
        print(f"Max fields changed from {old_max} to {self.max_fields}")  # Debug print
        
        # Update file browser
        self.setup_file_browser()
        self.setup_csv_viewer()
        
        # Attempt to reload the previously selected file
        if current_selected_file and os.path.exists(current_selected_file):
            try:
                # Find the row of the previously selected file in the new dataframe
                selected_row = self.df[self.df['File_Path'] == current_selected_file].index
                
                if not selected_row.empty:
                    # Select the row in the table
                    self.table.setSelectedRow(selected_row[0])
                    
                    # Load the CSV file using the method that populates the CSV viewer
                    self.load_csv_file(current_selected_file)   
                self.last_clicked_row = selected_row[0]
            except Exception as e:
                print(f"Error reloading selected file: {e}")

    def toggle_layout(self):
        """Toggle between horizontal and vertical layouts"""
        # Toggle orientation
        self.is_horizontal = not self.is_horizontal
        
        # Update button text
        self.toggle_btn.configure(
            text="Switch to Vertical Layout" if self.is_horizontal else "Switch to Horizontal Layout"
        )

        # Remove old panes
        if hasattr(self, 'paned'):
            self.paned.pack_forget()
            
        # Create new layout
        self.setup_panels()
        
        # Restore file browser and CSV viewer
        self.setup_file_browser()
        self.setup_csv_viewer()
        
        # Force geometry update
        self.update_idletasks()
        
        # Set final sash position
        if self.is_horizontal:
            self.paned.sashpos(0, 400)
        else:
            self.paned.sashpos(0, 300)


    def copy_selected_files(self):
        """Copy selected files to another folder"""
        # Get selected rows from the table's multiplerowlist
        selected_rows = self.table.multiplerowlist
        if not selected_rows:
            messagebox.showinfo("Info", "Please select files to copy")
            return

        # Ask for destination directory once
        dest_dir = filedialog.askdirectory(
            title="Select Destination Folder",
            initialdir=os.path.dirname(self.current_directory)
        )
        if not dest_dir:
            return
            
        # Normalize the destination directory
        dest_dir = os.path.normpath(os.path.abspath(dest_dir))
        
        try:
            copied_files = []
            failed_files = []
            
            # Get the filtered/displayed DataFrame
            displayed_df = self.table.model.df
            
            for row in selected_rows:
                if row < len(displayed_df):
                    filename = displayed_df.iloc[row]['Name']
                    
                    # Use the stored file path from the displayed DataFrame
                    if 'File_Path' in displayed_df.columns and not pd.isna(displayed_df.iloc[row]['File_Path']):
                        src_path = displayed_df.iloc[row]['File_Path']
                    else:
                        # Fall back to constructing the path
                        src_path = os.path.join(self.current_directory, filename)
                    
                    # Normalize the source path
                    src_path = os.path.normpath(os.path.abspath(src_path))
                    
                    # Destination path
                    dst_path = os.path.join(dest_dir, filename)
                    
                    print(f"Copying file: {src_path} to {dst_path}")
                    
                    try:
                        # Check if source file exists
                        if not os.path.exists(src_path):
                            print(f"Source file not found: {src_path}")  # Debug print
                            # Try with forward slashes
                            alt_path = src_path.replace(os.sep, '/')
                            if os.path.exists(alt_path):
                                src_path = alt_path
                                print(f"Found file using alternative path: {alt_path}")  # Debug print
                            else:
                                failed_files.append((filename, "Source file not found"))
                                continue
                        
                        # Check if file already exists in destination
                        if os.path.exists(dst_path):
                            if not messagebox.askyesno("File Exists", 
                                f"File {filename} already exists in destination.\nDo you want to overwrite it?"):
                                continue
                        
                        # Copy the file
                        shutil.copy2(src_path, dst_path)  # copy2 preserves metadata
                        copied_files.append(filename)
                        
                        # If we're copying a CSV file that's currently loaded, update the path
                        if hasattr(self, 'current_csv_file') and self.current_csv_file == src_path:
                            self.current_csv_file = dst_path
                            
                    except PermissionError as pe:
                        failed_files.append((filename, f"Permission denied: {str(pe)}"))
                    except Exception as e:
                        print(f"Error copying file {filename}: {e}")
                        failed_files.append((filename, str(e)))
            
            # Show success message
            if copied_files:
                messagebox.showinfo("Success", f"Copied {len(copied_files)} files to:\n{dest_dir}")
            
            # Show errors if any files failed to copy
            if failed_files:
                error_message = "Failed to copy the following files:\n\n"
                for filename, reason in failed_files:
                    error_message += f"• {filename}: {reason}\n"
                
                messagebox.showerror("Copy Errors", error_message)
                
        except Exception as e:
            print(f"Error in copy_selected_files: {e}")
            messagebox.showerror("Error", f"Failed to copy files:\n{str(e)}")

    def move_selected_files(self):
        """Move selected files to a new directory"""
        try:
            # Get selected rows from the table
            selected_rows = self.table.multiplerowlist
            
            # Check if any files are selected
            if not selected_rows:
                messagebox.showinfo("Info", "Please select files to move")
                return
            
            # Open directory selection dialog
            destination_dir = filedialog.askdirectory(
                title="Select Destination Folder for Moving Files",
                initialdir=os.path.dirname(self.current_directory)
            )
            
            # Check if a destination was selected
            if not destination_dir:
                return
            
            # Track successful and failed moves
            moved_files = []
            failed_files = []
            
            # Move each selected file
            for row in selected_rows:
                try:
                    # Get the actual file path from the filtered DataFrame
                    file_path = self.table.model.df.iloc[row]['File_Path']
                    
                    # Generate destination file path
                    file_name = os.path.basename(file_path)
                    dest_path = os.path.join(destination_dir, file_name)
                    
                    # Ensure unique filename if file already exists
                    counter = 1
                    while os.path.exists(dest_path):
                        name, ext = os.path.splitext(file_name)
                        dest_path = os.path.join(destination_dir, f"{name}_{counter}{ext}")
                        counter += 1
                    
                    # Move the file
                    shutil.move(file_path, dest_path)
                    moved_files.append(file_path)
                    
                except Exception as e:
                    failed_files.append((file_path, str(e)))

            # Provide summary of move operation
            if moved_files:
                # Refresh file list after moving
                self.update_file_list()
                
                # Create summary message
                summary_msg = f"Moved {len(moved_files)} file(s) to {destination_dir}"
                if failed_files:
                    summary_msg += f"\n\nFailed to move {len(failed_files)} file(s):"
                    for file, error in failed_files:
                        summary_msg += f"\n- {os.path.basename(file)}: {error}"
                
                messagebox.showinfo("Move Files", summary_msg)
            else:
                messagebox.showwarning("Move Files", "No files were moved")
        
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while moving files:\n{str(e)}")

    def delete_selected_files(self):
        """Delete selected files"""
        # Get selected rows from the table's multiplerowlist
        selected_rows = self.table.multiplerowlist
        if not selected_rows:
            messagebox.showinfo("Info", "Please select a file to delete")
            return

        # Show confirmation dialog with count of files
        if not messagebox.askyesno("Confirm Delete", 
                                f"Are you sure you want to delete {len(selected_rows)} selected files?",
                                icon='warning'):
            return

        deleted_files = []
        failed_files = []
        try:
            # Get the filtered/displayed DataFrame
            displayed_df = self.table.model.df
            
            for row in selected_rows:
                if row < len(displayed_df):
                    file_path = displayed_df.iloc[row]['File_Path']
                    
                    # Normalize the path to handle potential issues
                    file_path = os.path.normpath(file_path)
                    
                    # Ensure the file path exists
                    if not os.path.exists(file_path):
                        messagebox.showerror("Error", f"File not found: {file_path}")
                        continue
                    
                    # Use Windows-specific command to open and select file
                    try:
                        # Check if file is in use or locked
                        try:
                            # Try to open the file in exclusive write mode to check if it's locked
                            with open(file_path, 'a+b') as test_file:
                                pass
                        except PermissionError:
                            # File is locked or in use
                            failed_files.append((file_path, "File is in use by another program"))
                            continue
                        
                        # Delete the file
                        print(f"Performing delete operation...")  # Debug print
                        os.remove(file_path)
                        deleted_files.append(file_path)
                        
                        # Clear CSV viewer if it was the deleted file
                        if hasattr(self, 'current_file') and self.current_file == file_path:
                            self.current_file = None
                            self.current_csv_file = None
                            if hasattr(self, 'csv_table') and hasattr(self.csv_table, 'model'):
                                self.csv_table.model.df = pd.DataFrame()
                                self.csv_table.redraw()
                                
                    except PermissionError as pe:
                        failed_files.append((file_path, f"Permission denied: {str(pe)}"))
                    except FileNotFoundError as fnf:
                        failed_files.append((file_path, f"File not found: {str(fnf)}"))
                    except Exception as e:
                        print(f"Exception during delete: {e}")
                        failed_files.append((file_path, str(e)))

            # Update the file list after deletion
            if deleted_files:
                # Refresh the file list instead of manually updating the DataFrame
                self.update_file_list()
                
                if len(deleted_files) == 1:
                    messagebox.showinfo("Success", f"Deleted file: {os.path.basename(deleted_files[0])}")
                else:
                    messagebox.showinfo("Success", f"Deleted {len(deleted_files)} files")
            
            # Show errors if any files failed to delete
            if failed_files:
                error_message = "Failed to delete the following files:\n\n"
                for filename, reason in failed_files:
                    error_message += f"• {os.path.basename(filename)}: {reason}\n"
                
                messagebox.showerror("Delete Errors", error_message)
                
        except Exception as e:
            print(f"Error in delete_selected_files: {e}")
            messagebox.showerror("Error", f"An error occurred during file deletion:\n{str(e)}")
            traceback.print_exc()

    def load_subfolders(self):
        """Load all CSV and TSV files from current directory and all subdirectories"""
        try:
            print("\n=== Loading files from subfolders ===")  # Debug print
            
            # Set include_subfolders to True
            self.include_subfolders.set(True)
            
            # Use standard directory selection
            directory = filedialog.askdirectory(
                initialdir=self.current_directory
            )
            
            # Validate the selected directory
            if directory:
                # Update current directory
                self.current_directory = directory
                
                # Add to recent directories list
                self.add_to_recent_directories(directory)
                # Force explicit menu update
                self.update_recent_directories_menu()
                print(f"Recent directories menu updated with {len(self.recent_directories)} items")
                # Save settings explicitly
                print("Explicitly saving settings after directory change")
                self.save_settings()
                
                # Update file list
                self.update_file_list()
                
                # Update file browser dataframe
                self.update_file_dataframe()
                
                # Refresh the file browser table
                if hasattr(self, 'table'):
                    self.table.model.df = self.df
                    self.table.redraw()
                
                # Show summary of files found
                messagebox.showinfo("Subfolder Search", 
                    f"Found {len(self.csv_files)} CSV/TSV files in {directory}")
            
        except Exception as e:
            print(f"Error in load_subfolders: {e}")
            messagebox.showerror("Error", "Failed to load subfolders")

    def update_file_dataframe(self):
        """Update the pandas DataFrame with file information"""
        print("\n=== Updating File DataFrame ===")
        
        # Ensure csv_files is a list
        if not hasattr(self, 'csv_files'):
            self.csv_files = []
        
        # If no files found, create an empty DataFrame with expected columns
        if not self.csv_files:
            print("WARNING: No CSV/TSV files found in the directory.")
            self.df = pd.DataFrame(columns=[
                'Name', 'File_Path', 'Size', 'Modified', 
                'Type'
            ] + [f'Field_{i+1}' for i in range(25)])
            return
        
        # Prepare lists to store file information
        names = []
        file_paths = []
        sizes = []
        modified_times = []
        types = []
        
        # Collect file information
        for file_path in self.csv_files:
            try:
                # Get file stats
                file_stat = os.stat(file_path)
                
                # Extract filename and path
                file_name = os.path.basename(file_path)
                
                # Collect basic file information
                names.append(file_name)
                file_paths.append(file_path)
                sizes.append(self.format_size(file_stat.st_size))
                modified_times.append(self.format_date(file_stat.st_mtime))
                types.append(os.path.splitext(file_name)[1][1:].upper())
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
        # Create initial DataFrame
        self.df = pd.DataFrame({
            'Name': names,
            'File_Path': file_paths,
            'Size': sizes,
            'Modified': modified_times,
            'Type': types
        })
        
        # Add dynamic Field columns based on filename
        for i in range(25):  # Ensure at least 25 fields
            field_name = f'Field_{i+1}'
            self.df[field_name] = self.df['Name'].apply(
                lambda x: self._extract_field(x, i) if '_' in x else ''
            )
        
        # Print DataFrame info for debugging
        print(f"Created DataFrame with {len(self.df)} rows")
        print("DataFrame Columns:", list(self.df.columns))
        
        # Optional: Sort by modified time (newest first)
        try:
            self.df['Modified_Datetime'] = pd.to_datetime(self.df['Modified'], errors='coerce')
            self.df = self.df.sort_values('Modified_Datetime', ascending=False)
            self.df = self.df.drop(columns=['Modified_Datetime'])
        except Exception as sort_error:
            print(f"Error sorting DataFrame: {sort_error}")
        
        # Reset index to ensure clean indexing
        self.df = self.df.reset_index(drop=True)
    
    def _extract_field(self, filename, field_index):
        """Extract a specific field from filename split by underscore"""
        try:
            # Remove extension
            name_without_ext = os.path.splitext(filename)[0]
            
            # Split by underscore
            fields = name_without_ext.split('_')
            
            # Return field if exists, otherwise empty string
            return fields[field_index] if field_index < len(fields) else ''
        except Exception as e:
            print(f"Error extracting field from {filename}: {e}")
            return ''

    def setup_column_search_menu(self):
        """Set up the right-click menu for column search"""
        self.column_search_menu = tk.Menu(self, tearoff=0)
        
        # The menu will be populated dynamically when shown

    def show_column_search_menu(self, event=None):
        """Show the column search menu with matching columns"""
        try:
            # First check if we have a valid dataframe to search in
            if not hasattr(self, 'original_csv_df') or self.original_csv_df is None:
                print("No CSV file loaded or dataframe is None")
                # Show a message to the user
                self.column_search_entry.config(background='#FFCCCC')  # Light red background
                self.after(1500, lambda: self.column_search_entry.config(background='white'))  # Reset after 1.5 seconds
                return
                
            # Check if we have a valid table
            if not hasattr(self, 'csv_table') or not hasattr(self.csv_table, 'model'):
                print("CSV table not initialized")
                return
                
            search_text = self.column_search_var.get().lower().strip()
            
            if not search_text:
                return
                
            print(f"\n=== Searching columns with: '{search_text}' ===")
            
            # Clear existing menu items
            self.column_search_menu.delete(0, tk.END)
            
            # Find matching columns
            all_columns = self.original_csv_df.columns.tolist()
            exact_matches = []
            partial_matches = []
            
            for col in all_columns:
                col_str = str(col).lower()
                if col_str == search_text:
                    exact_matches.append(col)
                elif search_text in col_str:
                    partial_matches.append(col)
            
            # If no exact or partial matches, try fuzzy matching
            if not exact_matches and not partial_matches and len(search_text) > 2:
                # Use difflib for fuzzy matching
                import difflib
                fuzzy_matches = difflib.get_close_matches(search_text, 
                                                         [str(col).lower() for col in all_columns], 
                                                         n=10, 
                                                         cutoff=0.6)
                
                # Get the original case for these matches
                fuzzy_matches = [col for col in all_columns 
                                if str(col).lower() in fuzzy_matches]
            else:
                fuzzy_matches = []
            
            # Sort matches for better usability
            exact_matches.sort(key=lambda x: str(x).lower())
            partial_matches.sort(key=lambda x: str(x).lower())
            fuzzy_matches.sort(key=lambda x: str(x).lower())
            
            # Helper function to create a callback for a specific column
            def make_callback(column):
                def callback():
                    self.column_search_var.set(column)
                    self.last_searched_column = column
                return callback
            
            # Add header for exact matches if any
            if exact_matches:
                self.column_search_menu.add_command(
                    label="=== Exact Matches ===", 
                    state=tk.DISABLED,
                    background='#E0E0E0'
                )
                
                for col in exact_matches:
                    self.column_search_menu.add_command(
                        label=col,
                        command=make_callback(col)
                    )
                
                # Add separator if we have both exact and partial matches
                if partial_matches or fuzzy_matches:
                    self.column_search_menu.add_separator()
            
            # Add header for partial matches if any
            if partial_matches:
                self.column_search_menu.add_command(
                    label="=== Partial Matches ===", 
                    state=tk.DISABLED,
                    background='#E0E0E0'
                )
                
                # Group partial matches if there are many
                if len(partial_matches) > 15:
                    # Create submenus for groups of columns
                    group_size = 10
                    for i in range(0, len(partial_matches), group_size):
                        group = partial_matches[i:i+group_size]
                        start_letter = str(group[0])[0].upper()
                        end_letter = str(group[-1])[0].upper()
                        
                        submenu = tk.Menu(self.column_search_menu, tearoff=0)
                        for col in group:
                            submenu.add_command(
                                label=col,
                                command=make_callback(col)
                            )
                        
                        self.column_search_menu.add_cascade(
                            label=f"{start_letter}-{end_letter} ({len(group)} columns)",
                            menu=submenu
                        )
                else:
                    # Add all partial matches directly to the menu
                    for col in partial_matches:
                        self.column_search_menu.add_command(
                            label=col,
                            command=make_callback(col)
                        )
                
                # Add separator if we have both partial and fuzzy matches
                if fuzzy_matches:
                    self.column_search_menu.add_separator()
            
            # Add header for fuzzy matches if any
            if fuzzy_matches:
                self.column_search_menu.add_command(
                    label="=== Similar Matches ===", 
                    state=tk.DISABLED,
                    background='#E0E0E0'
                )
                
                for col in fuzzy_matches:
                    self.column_search_menu.add_command(
                        label=col,
                        command=make_callback(col)
                    )
            
            # If no matches found, show a message
            if not exact_matches and not partial_matches and not fuzzy_matches:
                self.column_search_menu.add_command(
                    label=f"No columns matching '{search_text}'",
                    state=tk.DISABLED
                )
            
            # Add a separator and additional options
            self.column_search_menu.add_separator()
            
            # Add Move to Start option
            if self.last_searched_column:
                self.column_search_menu.add_command(
                    label=f"Move '{self.last_searched_column}' to Start",
                    command=self.move_searched_column_to_start
                )
                self.column_search_menu.add_separator()
            
            self.column_search_menu.add_command(
                label="Clear Search",
                command=lambda: self.column_search_var.set("")
            )
            
            # Show the menu at the event position
            if event:
                self.column_search_menu.post(event.x_root, event.y_root)
            else:
                # If called programmatically, show near the search entry
                x = self.column_search_entry.winfo_rootx()
                y = self.column_search_entry.winfo_rooty() + self.column_search_entry.winfo_height()
                self.column_search_menu.post(x, y)
            
        except Exception as e:
            print(f"Error in show_column_search_menu: {e}")
            traceback.print_exc()
    
    def focus_column_search(self, event=None):
        """Focus on the column search box"""
        if hasattr(self, 'column_search_entry'):
            self.column_search_entry.focus_set()
            # Select all text if there's any
            self.column_search_entry.select_range(0, tk.END)
            return "break"  # Prevent default behavior

    def create_menu(self):
        """Create the application menu"""
        menubar = tk.Menu(self)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Directory", command=self.browse_directory)  # Use the browse_directory method
        file_menu.add_command(label="Open CSV File", command=self.open_csv_file)
        
        # Add recent directories submenu
        self.recent_dirs_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent Directories", menu=self.recent_dirs_menu)
        self.update_recent_directories_menu()
        
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)  # Use self.quit instead of self.master.quit
        menubar.add_cascade(label="File", menu=file_menu)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Horizontal Layout", command=self.set_horizontal_layout)
        view_menu.add_command(label="Vertical Layout", command=self.set_vertical_layout)
        view_menu.add_separator()
        view_menu.add_command(label="Find Column (Ctrl+F)", command=self.focus_column_search)
        menubar.add_cascade(label="View", menu=view_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.config(menu=menubar)  # Use self.config instead of self.master.config

    def set_horizontal_layout(self):
        """Set the layout to horizontal (file browser on left, CSV viewer on right)"""
        if not self.is_horizontal:
            # Remove existing panes
            for widget in self.paned.panes():
                self.paned.forget(widget)
                
            # Change orientation
            self.paned.configure(orient=tk.HORIZONTAL)
            self.is_horizontal = True
            
            # Re-add the panes
            self.paned.add(self.file_browser_container, weight=1)
            self.paned.add(self.csv_view_container, weight=3)
            
            # Set sash position
            self.after(100, lambda: self.paned.sashpos(0, 400))
            
    def set_vertical_layout(self):
        """Set the layout to vertical (file browser on top, CSV viewer on bottom)"""
        if self.is_horizontal:
            # Remove existing panes
            for widget in self.paned.panes():
                self.paned.forget(widget)
                
            # Change orientation
            self.paned.configure(orient=tk.VERTICAL)
            self.is_horizontal = False
            
            # Re-add the panes
            self.paned.add(self.file_browser_container, weight=1)
            self.paned.add(self.csv_view_container, weight=3)
            
            # Set sash position
            self.after(100, lambda: self.paned.sashpos(0, 300))
    
    def show_about(self):
        """Show the about dialog"""
        about_text = """CSV Browser

A tool for browsing and viewing CSV files with advanced column search functionality.

Features:
- File browser with:
  - Sorting by name, date, size, and filename fields
  - Dynamic columns based on filename structure
  - Filter functionality across all fields
  - Horizontal scrolling for many fields
- CSV file preview in second panel
- Vertical and horizontal layout options
- Directory selection
- File operations (move, delete)
"""
        messagebox.showinfo("About CSV Browser", about_text)
    
    def browse_directory(self):
        """Open a directory chooser dialog and update the file list"""
        print("\n=== Browse directory called ===")  # Debug print
        directory = filedialog.askdirectory(
            initialdir=self.current_directory
        )
        if directory:
            print(f"Selected directory: {directory}")  # Debug print
            self.current_directory = directory
            
            # Add to recent directories list
            self.add_to_recent_directories(directory)
            # Force explicit menu update
            self.update_recent_directories_menu()
            print(f"Recent directories menu updated with {len(self.recent_directories)} items")
            # Save settings explicitly
            print("Explicitly saving settings after directory change")
            self.save_settings()
            
            self.include_subfolders.set(False)  # Reset to not include subfolders
            
            # Update file list
            self.update_file_list()
            
            # Update max fields
            old_max = self.max_fields
            self.max_fields = self.get_max_fields()
            print(f"Max fields changed from {old_max} to {self.max_fields}")  # Debug print
            

    def add_to_recent_directories(self, directory):
        """Add a directory to the recent directories list"""
        try:
            print(f"Adding to recent directories: {directory}")
            
            # Remove the directory if it already exists in the list
            if directory in self.recent_directories:
                self.recent_directories.remove(directory)
                
            # Add the directory to the beginning of the list
            self.recent_directories.insert(0, directory)
            
            # Trim the list to max_recent_directories
            if len(self.recent_directories) > self.max_recent_directories:
                self.recent_directories = self.recent_directories[:self.max_recent_directories]
                
            # Update the recent directories menu only if it exists
            if hasattr(self, 'recent_dirs_menu'):
                self.update_recent_directories_menu()
            else:
                print("Recent directories menu not yet created, skipping menu update")
            
            # Save recent directories to file
            self.save_settings()
            
            # Verify recent directories were updated
            print(f"Recent directories after update: {self.recent_directories}")
            
        except Exception as e:
            print(f"Error adding to recent directories: {e}")
            traceback.print_exc()

    def open_csv_file(self):
        """Open a CSV file chooser dialog and load the selected file"""
        print("\n=== Open CSV file called ===")  # Debug print
        file_path = filedialog.askopenfilename(
            initialdir=self.current_directory,
            title="Select CSV file",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if file_path:
            print(f"Selected CSV file: {file_path}")  # Debug print
            # Update current directory to the directory of the selected file
            self.current_directory = os.path.dirname(file_path)
            
            # Load the selected CSV file
            self.load_csv_file(file_path)

    def move_searched_column_to_start(self):
        """Move the last searched column to the start of the dataframe"""
        try:
            if not hasattr(self, 'last_searched_column') or not self.last_searched_column:
                print("No column was searched")
                return
                
            if not hasattr(self, 'original_csv_df') or self.original_csv_df is None:
                print("No CSV file loaded")
                return
                
            column = self.last_searched_column
            
            # Check if the column exists in the dataframe
            if column not in self.original_csv_df.columns:
                print(f"Column '{column}' not found in dataframe")
                return
                
            print(f"\n=== Moving column '{column}' to start ===")
            
            # Get all columns and move the specified one to the start
            cols = list(self.original_csv_df.columns)
            if column in cols:
                cols.remove(column)
                cols.insert(0, column)
                
                # Reorder the dataframe columns
                self.original_csv_df = self.original_csv_df[cols]
                
                # If we have a filtered dataframe, reorder that too
                if hasattr(self, 'filtered_csv_df') and self.filtered_csv_df is not None:
                    # Only reorder if the column exists in the filtered dataframe
                    if column in self.filtered_csv_df.columns:
                        filtered_cols = list(self.filtered_csv_df.columns)
                        filtered_cols.remove(column)
                        filtered_cols.insert(0, column)
                        self.filtered_csv_df = self.filtered_csv_df[filtered_cols]
                
                # Update the displayed dataframe
                self._update_csv_display()
                
                # Highlight the column
                self.highlight_column(column)
        except Exception as e:
            print(f"Error in move_searched_column_to_start: {e}")
            traceback.print_exc()
    
    def _update_csv_display(self):
        """Update the CSV display with the current dataframe"""
        if hasattr(self, 'csv_table') and hasattr(self.csv_table, 'model'):
            self.csv_table.model.df = self.original_csv_df
            self.csv_table.redraw()
    
    def highlight_column(self, column):
        """Highlight a column in the CSV table"""
        if hasattr(self, 'csv_table') and hasattr(self.csv_table, 'columncolors'):
            self.csv_table.columncolors[column] = '#C6F4D6'  # Light green
            self.csv_table.redraw()
    
    def _apply_column_filter_to_filtered_data(self):
        """Apply the stored column filter to the row-filtered data"""
        if self.filtered_csv_df is not None and self.visible_columns:
            try:
                # Get valid columns (in case we switched to a file with different columns)
                valid_columns = [col for col in self.visible_columns if col in self.filtered_csv_df.columns]
                
                if not valid_columns:
                    print("No valid columns for this file, showing all columns")
                    self.csv_table.model.df = self.filtered_csv_df
                else:
                    # Apply column filter
                    # Create a new DataFrame with the selected columns while preserving the index
                    filtered_df = self.filtered_csv_df[valid_columns].copy()
                    # The index is already preserved since we're using the filtered_csv_df
                    self.csv_table.model.df = filtered_df
                
                # Redraw the table
                self.csv_table.redraw()
                
                # Preserve the index
                self._safe_preserve_index()
                
                # Adjust column widths to fit content
                self.adjust_column_widths()
                
            except Exception as e:
                print(f"Error applying column filter: {e}")
                traceback.print_exc()
    
    def filter_columns(self, *args):
        """
        Filter the CSV table to show only columns matching the filter text.
        Multiple column names can be specified using comma, space, or semicolon as separators.
        Partial matches are included. The filter persists when switching files.
        Use "*" to include all remaining columns not explicitly specified.
        """
        print("\n=== Column filter called ===")
        
        if hasattr(self, 'csv_table') and self.original_csv_df is not None:
            try:
                # Store the current index for preservation
                current_index = None
                if hasattr(self.csv_table, 'model') and hasattr(self.csv_table.model, 'df'):
                    current_index = self.csv_table.model.df.index.copy()
                    print(f"Stored current index with {len(current_index)} rows for preservation")
                
                filter_text = self.column_filter_var.get().strip()
                print(f"Column filter text: '{filter_text}'")
                
                # If filter text is empty, show all columns
                if not filter_text:
                    self.reset_column_filter()
                    return
                
                # Split filter text by comma, space, or semicolon
                filter_terms = [term.strip() for term in filter_text.replace(',', ' ').replace(';', ' ').split()]
                filter_terms = [term for term in filter_terms if term]  # Remove empty terms
                
                if not filter_terms:
                    self.reset_column_filter()
                    return
                
                print(f"Column filter terms: {filter_terms}")
                
                # Get all column names
                all_columns = self.original_csv_df.columns.tolist()
                
                # Check if wildcard is present
                include_remaining = "*" in filter_terms
                if include_remaining:
                    filter_terms.remove("*")
                    print("Wildcard '*' found - will include all remaining columns")
                
                # Find matching columns (including partial matches)
                matching_columns = []
                for term in filter_terms:
                    term = term.strip()
                    for col in all_columns:
                        if term.lower() in col.lower() and col not in matching_columns:
                            matching_columns.append(col)
                            
                # If wildcard is present, include all remaining columns
                if include_remaining:
                    remaining_columns = [col for col in all_columns if col not in matching_columns]
                    matching_columns.extend(remaining_columns)
                    print(f"Added {len(remaining_columns)} remaining columns due to wildcard")

                if matching_columns:
                    # Store the visible columns for persistence
                    self.visible_columns = matching_columns
                    
                    # Apply the column filter
                    if hasattr(self, 'filtered_csv_df') and self.filtered_csv_df is not None:
                        self._apply_column_filter_to_filtered_data()
                    else:
                        self._apply_column_filter()
                else:
                    # No matches found, silently ignore and keep current view
                    print("No matching columns found, keeping current view")
                    return
                
                # Restore the index after filtering
                if current_index is not None:
                    self.csv_table.model.df.index = current_index
                    print(f"Restored index with {len(current_index)} rows after filtering")
                
                # Preserve the index
                self._safe_preserve_index()

            except Exception as e:
                print(f"Error in filter_columns: {e}")
                traceback.print_exc()

    def _safe_preserve_index(self):
        """
        Safely preserve the index after operations that might reset it,
        handling the case where the index name is already a column
        """
        if hasattr(self, 'csv_table') and hasattr(self.csv_table, 'model') and hasattr(self.csv_table.model, 'df'):
            try:
                # Get the current DataFrame
                df = self.csv_table.model.df
                
                print(f"DEBUG: _safe_preserve_index - Starting - DataFrame index type: {type(df.index)}")
                print(f"DEBUG: _safe_preserve_index - Starting - DataFrame index name: {df.index.name}")
                print(f"DEBUG: _safe_preserve_index - Starting - DataFrame index sample: {df.index[:5]}")
                
                # Store the current index
                current_index = df.index.copy()
                index_name = df.index.name
                
                # Check if the index name is already a column
                if index_name is not None and index_name in df.columns:
                    print(f"Index column '{index_name}' already exists in DataFrame")
                    
                    # Create a temporary name for the index
                    temp_name = f"{index_name}_index_temp"
                    while temp_name in df.columns:
                        temp_name = f"{temp_name}_temp"
                    
                    # Rename the index
                    df.index.name = temp_name
                
                # Force the table to show the index
                self.csv_table.showindex = True
                print(f"DEBUG: _safe_preserve_index - After setting showindex - Table showindex: {self.csv_table.showindex}")
                
                # Force a redraw of the row header
                if hasattr(self.csv_table, 'rowheader'):
                    self.csv_table.rowheader.redraw()
                    
                # Make sure the index is visible
                try:
                    self.csv_table.showIndex()
                    print(f"DEBUG: _safe_preserve_index - After showIndex() - DataFrame index type: {type(df.index)}")
                    print(f"DEBUG: _safe_preserve_index - After showIndex() - DataFrame index name: {df.index.name}")
                    print(f"DEBUG: _safe_preserve_index - After showIndex() - DataFrame index sample: {df.index[:5]}")
                except Exception as e:
                    print(f"Warning: Error showing index: {e}")
                    
                    # Try a more direct approach - reset and restore the index
                    try:
                        # Create a temporary column with the index values
                        temp_col = "__temp_index__"
                        while temp_col in df.columns:
                            temp_col = f"{temp_col}_temp"
                        
                        print(f"DEBUG: _safe_preserve_index - Using direct approach with temp column: {temp_col}")
                        
                        # Store index values in a temporary column
                        df[temp_col] = current_index
                        
                        # Reset the index (this will create a new default numeric index)
                        df.reset_index(drop=True, inplace=True)
                        
                        # Set the index back to the stored values
                        df.set_index(temp_col, inplace=True)
                        
                        # Rename the index back to its original name
                        if index_name is not None:
                            df.index.name = index_name
                        
                        # Remove the temporary column if it still exists
                        if temp_col in df.columns:
                            df.drop(columns=[temp_col], inplace=True)
                        
                        print(f"DEBUG: _safe_preserve_index - After direct approach - DataFrame index type: {type(df.index)}")
                        print(f"DEBUG: _safe_preserve_index - After direct approach - DataFrame index name: {df.index.name}")
                        print(f"DEBUG: _safe_preserve_index - After direct approach - DataFrame index sample: {df.index[:5]}")
                        
                        # Force a redraw
                        self.csv_table.redraw()
                    except Exception as e2:
                        print(f"Failed to restore index: {e2}")
                        
            except Exception as e:
                print(f"Warning: Error in _safe_preserve_index: {e}")

    def _preserve_index(self):
        """Ensure the index is preserved after operations that might reset it"""
        if hasattr(self, 'csv_table') and hasattr(self.csv_table, 'model') and hasattr(self.csv_table.model, 'df'):
            # Store the current index name
            index_name = self.csv_table.model.df.index.name
            
            # Make sure the index is visible
            self.csv_table.showIndex()
            
            # Restore the index name if it was lost
            if index_name is not None and self.csv_table.model.df.index.name is None:
                self.csv_table.model.df.index.name = index_name

    def _apply_column_filter(self):
        """Apply the stored column filter to the current CSV data"""
        if self.original_csv_df is not None and self.visible_columns:
            try:
                print(f"DEBUG: _apply_column_filter - Original DataFrame index type: {type(self.original_csv_df.index)}")
                print(f"DEBUG: _apply_column_filter - Original DataFrame index name: {self.original_csv_df.index.name}")
                print(f"DEBUG: _apply_column_filter - Original DataFrame index sample: {self.original_csv_df.index[:5]}")
                
                # Get valid columns (in case we switched to a file with different columns)
                valid_columns = [col for col in self.visible_columns if col in self.original_csv_df.columns]
                
                if not valid_columns:
                    print("No valid columns for this file, showing all columns")
                    self.csv_table.model.df = self.original_csv_df.copy()
                else:
                    # Apply column filter
                    # Use .loc to ensure the index is preserved
                    filtered_df = self.original_csv_df.loc[:, valid_columns].copy()
                    print(f"DEBUG: _apply_column_filter - Filtered DataFrame index type: {type(filtered_df.index)}")
                    print(f"DEBUG: _apply_column_filter - Filtered DataFrame index name: {filtered_df.index.name}")
                    print(f"DEBUG: _apply_column_filter - Filtered DataFrame index sample: {filtered_df.index[:5]}")
                    
                    self.csv_table.model.df = filtered_df
                
                # Redraw the table
                self.csv_table.redraw()
                
                print(f"DEBUG: _apply_column_filter - After redraw - DataFrame index type: {type(self.csv_table.model.df.index)}")
                print(f"DEBUG: _apply_column_filter - After redraw - DataFrame index name: {self.csv_table.model.df.index.name}")
                print(f"DEBUG: _apply_column_filter - After redraw - DataFrame index sample: {self.csv_table.model.df.index[:5]}")
                
                # Check if index was reset and restore it if needed
                if not self.csv_table.model.df.index.equals(filtered_df.index):
                    print("Warning: Index was reset during redraw, restoring...")
                    self.csv_table.model.df.index = filtered_df.index.copy()
                    print(f"DEBUG: _apply_column_filter - After index restoration - DataFrame index type: {type(self.csv_table.model.df.index)}")
                    print(f"DEBUG: _apply_column_filter - After index restoration - DataFrame index name: {self.csv_table.model.df.index.name}")
                    print(f"DEBUG: _apply_column_filter - After index restoration - DataFrame index sample: {self.csv_table.model.df.index[:5]}")
                
                # Preserve the index
                self._safe_preserve_index()
                
                # Adjust column widths to fit content
                self.adjust_column_widths()
                
            except Exception as e:
                print(f"Error applying column filter: {e}")
                traceback.print_exc()

    def adjust_column_widths(self):
        """Adjust column widths to fit content"""
        if hasattr(self, 'csv_table') and self.csv_table.model.df is not None:
            try:
                # Get the current dataframe
                df = self.csv_table.model.df
                
                if df.empty:
                    return
                
                # Iterate over columns and adjust widths
                for col in df.columns:
                    # Get column name length
                    col_name_length = len(str(col))
                    
                    # Sample data to determine maximum width needed
                    # Use head and tail to avoid processing the entire dataset for large files
                    sample_data = pd.concat([df[col].head(20), df[col].tail(20)])
                    
                    # Convert to string and get max length
                    try:
                        max_data_length = sample_data.astype(str).str.len().max()
                    except:
                        # Fallback if conversion fails
                        max_data_length = max([len(str(x)) for x in sample_data if x is not None and pd.notna(x)] or [0])
                    
                    # Use the maximum of column name length and data length
                    max_length = max(col_name_length, max_data_length)
                    
                    # Calculate pixel width (approximately 8 pixels per character)
                    # Set minimum width of 50 pixels and maximum of 300 pixels
                    pixel_width = max(min(max_length * 8, 300), 50)
                    
                    # Set the column width
                    self.csv_table.columnwidths[col] = pixel_width
                
                # Apply the new column widths
                self.csv_table.redraw()
                
                print("Column widths adjusted to fit content")
                
            except Exception as e:
                print(f"Error adjusting column widths: {e}")
                traceback.print_exc()
    
    def reset_column_filter(self):
        """Reset the column filter to show all columns"""
        try:
            self.visible_columns = None
            
            if hasattr(self, 'csv_table') and self.original_csv_df is not None:
                # Reset to original data (but keep any row filters)
                current_filter = self.csv_filter_text.get().strip()
                
                # Apply the current row filter to the full dataset
                if current_filter:
                    self.filter_csv_content()
                else:
                    self.csv_table.model.df = self.original_csv_df.copy()
                    self.csv_table.redraw()
                
                print("Column filter reset, showing all columns")
            
            # Clear the filter text
            self.column_filter_var.set("")
            
        except Exception as e:
            print(f"Error resetting column filter: {e}")
            traceback.print_exc()
    
    def setup_column_filter_context_menu(self):
        """Create a context menu for column filter with instructions and examples"""
        try:
            if hasattr(self, 'column_filter_entry'):
                # Create right-click context menu
                self.column_filter_menu = tk.Menu(self.column_filter_entry, tearoff=0)
                self.column_filter_menu.add_command(label="Column Filter Instructions", state='disabled', font=('Arial', 10, 'bold'))
                self.column_filter_menu.add_separator()
                self.column_filter_menu.add_command(label="Enter column names separated by spaces, commas, or semicolons", state='disabled')
                self.column_filter_menu.add_command(label="Partial matches are included", state='disabled')
                self.column_filter_menu.add_command(label="Filter persists when switching files", state='disabled')
                self.column_filter_menu.add_command(label="Leave empty to show all columns", state='disabled')
                self.column_filter_menu.add_separator()
                self.column_filter_menu.add_command(label="Examples:", state='disabled', font=('Arial', 10, 'bold'))
                self.column_filter_menu.add_command(label="'date price': Show date and price columns", state='disabled')
                self.column_filter_menu.add_command(label="'open,high,low,close': Show OHLC columns", state='disabled')
                self.column_filter_menu.add_command(label="'vol': Show all columns containing 'vol'", state='disabled')
                
                # Bind right-click to show menu
                self.column_filter_entry.bind('<Button-3>', self.show_column_filter_menu)
                
        except Exception as e:
            print(f"Error setting up column filter context menu: {e}")
            traceback.print_exc()
    
    def show_column_filter_menu(self, event):
        """Display the context menu for column filter"""
        try:
            if hasattr(self, 'column_filter_menu'):
                self.column_filter_menu.tk_popup(event.x_root, event.y_root)
        except Exception as e:
            print(f"Error showing column filter menu: {e}")
            traceback.print_exc()

    def save_to_filtered_csv(self):
        """Save the currently filtered CSV data to a new CSV file"""
        try:
            # Check if we have data to save
            if not hasattr(self, 'csv_table') or not hasattr(self.csv_table, 'model') or self.csv_table.model.df is None:
                messagebox.showinfo("No Data", "No CSV data available to save.")
                return
                
            # Get the current dataframe (filtered or not)
            current_df = self.csv_table.model.df
            
            if current_df.empty:
                messagebox.showerror("Empty Data", "The current data is empty and cannot be saved.")
                return
            
            # Generate filename based on original file
            if not hasattr(self, 'current_csv_file') or not self.current_csv_file:
                messagebox.showinfo("No Source File", "Cannot determine source file to create filtered filename.")
                return
                
            # Get the directory and filename from the current file
            file_dir = os.path.dirname(self.current_csv_file)
            if not os.path.exists(file_dir):
                file_dir = self.current_directory
            
            # Get the filename without extension
            base_filename = os.path.basename(self.current_csv_file)
            filename_no_ext, ext = os.path.splitext(base_filename)
            
            # Create new filename with "_filtered" suffix
            new_filename = f"{filename_no_ext}_filtered{ext}"
            file_path = os.path.join(file_dir, new_filename)
            
            # Check if file already exists and create a unique name if needed
            counter = 1
            original_path = file_path
            while os.path.exists(file_path):
                file_path = os.path.join(file_dir, f"{filename_no_ext}_filtered_{counter}{ext}")
                counter += 1
                
            # Save the dataframe to CSV
            try:
                current_df.to_csv(file_path, index=False)
                print(f"Successfully saved filtered data to: {file_path}")
                messagebox.showinfo("Save Successful", f"Data saved to:\n{file_path}")
                
                # Refresh the file list to show the new file
                self.refresh_file_list()
                
            except Exception as save_error:
                print(f"Error saving CSV: {save_error}")
                messagebox.showerror("Save Error", f"Failed to save CSV file:\n{str(save_error)}")
                
        except Exception as e:
            print(f"Error in save_to_filtered_csv: {e}")
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred while saving:\n{str(e)}")

    def _advanced_file_read(self, file_path):
        """Advanced file reading method with comprehensive diagnostics and long path handling"""
        try:
            print(f"\n=== Advanced file read for: {file_path} ===")
            
            # Normalize and handle long paths
            normalized_path = self.normalize_long_path(file_path)
            
            # Try to read the file with various encodings
            encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252', 'utf-16']
            
            # Try different reading methods
            for encoding in encodings:
                try:
                    print(f"Attempting to read with encoding: {encoding}")
                    df = pd.read_csv(normalized_path, encoding=encoding)
                    if df is not None and not df.empty:
                        print(f"Successfully read with encoding: {encoding}")
                        return df
                except Exception as e:
                    print(f"Reading with {encoding} failed: {e}")
            
            # Try with Python engine and error handling
            try:
                print("Attempting to read with Python engine and error handling")
                df = pd.read_csv(normalized_path, encoding='utf-8', engine='python', on_bad_lines='skip')
                if df is not None and not df.empty:
                    print("Successfully read with Python engine")
                    return df
            except Exception as e:
                print(f"Reading with Python engine failed: {e}")
            
            # Last resort: try to read as text and parse manually
            print("All standard reading methods failed, attempting manual parsing")
            return None
            
        except Exception as e:
            print(f"Advanced file read failed: {e}")
            traceback.print_exc()
            return None

    def _manual_csv_parse(self, file_path, encoding):
        """Manual CSV parsing with error handling"""
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                lines = f.readlines()
            
            # Remove problematic lines
            clean_lines = [line for line in lines if len(line.split(',')) > 1]
            
            # Create a DataFrame from the clean lines
            df = pd.DataFrame([line.strip().split(',') for line in clean_lines])
            
            # Try to convert columns to numeric types
            df = df.apply(pd.to_numeric, errors='coerce')
            
            return df
        
        except Exception as e:
            print(f"Error in manual CSV parsing: {e}")
            return None

    def reveal_in_explorer(self):
        """Reveal the selected file(s) in File Explorer"""
        selected_rows = self.table.multiplerowlist
        if not selected_rows:
            messagebox.showinfo("Info", "Please select a file to reveal in Explorer")
            return
    
        try:
            # Get the filtered/displayed DataFrame
            displayed_df = self.table.model.df
            
            for row in selected_rows:
                if row < len(displayed_df):
                    file_path = displayed_df.iloc[row]['File_Path']
                    
                    # Normalize the path to handle potential issues
                    file_path = os.path.normpath(file_path)
                    
                    # Ensure the file path exists
                    if not os.path.exists(file_path):
                        messagebox.showerror("Error", f"File not found: {file_path}")
                        continue
                    
                    # Use Windows-specific command to open and select file
                    subprocess.Popen(f'explorer /select,"{file_path}"', shell=True)
    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reveal file in Explorer:\n{str(e)}")

    def save_current_filter(self):
        """Save the current filter configuration"""
        try:
            # Get the current filter settings
            row_filter = self.csv_filter_text.get().strip()
            column_filter = self.column_filter_var.get().strip()
            
            # Skip if both filters are empty
            if not row_filter and not column_filter:
                messagebox.showinfo("Empty Filter", "No filter is currently active")
                return
            
            # Ask for a name for this filter configuration
            filter_name = simpledialog.askstring("Save Filter", "Enter a name for this filter:")
            if not filter_name:
                return  # User canceled
            
            # Create a filter configuration dictionary
            filter_config = {
                "name": filter_name,
                "row_filter": row_filter,
                "column_filter": column_filter
            }
            
            # Check if a filter with this name already exists
            for i, saved_filter in enumerate(self.saved_filters):
                if saved_filter.get("name") == filter_name:
                    # Ask if user wants to overwrite
                    if messagebox.askyesno("Filter Exists", f"A filter named '{filter_name}' already exists. Overwrite?"):
                        self.saved_filters[i] = filter_config
                        messagebox.showinfo("Filter Saved", f"Filter '{filter_name}' updated")
                    return
            
            # Add the new filter to the list
            self.saved_filters.append(filter_config)
            
            # Save settings to file
            self.save_settings()
            
            # Show a confirmation message
            messagebox.showinfo("Filter Saved", f"Filter '{filter_name}' saved")
            
        except Exception as e:
            print(f"Error saving filter: {e}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to save filter:\n{str(e)}")
            
    def show_saved_filters(self):
        """Show the saved filters and allow the user to select one"""
        try:
            # Check if there are any saved filters
            if not self.saved_filters:
                messagebox.showinfo("No Saved Filters", "You don't have any saved filters yet")
                return
                
            # Create a dialog to show the saved filters
            dialog = tk.Toplevel(self)
            dialog.title("Saved Filters")
            dialog.geometry("500x300")
            
            # Make dialog modal
            dialog.transient(self)
            dialog.grab_set()
            
            # Create a frame for the listbox and scrollbar
            list_frame = ttk.Frame(dialog)
            list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Add a scrollbar
            scrollbar = ttk.Scrollbar(list_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Create listbox with scrollbar
            filter_listbox = tk.Listbox(list_frame, width=50, height=10, yscrollcommand=scrollbar.set)
            filter_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=filter_listbox.yview)
            
            # Add column headers
            filter_listbox.insert(tk.END, "Name | Row Filter | Column Filter")
            filter_listbox.insert(tk.END, "-" * 60)
            
            # Add saved filters to the listbox
            for filter_config in self.saved_filters:
                filter_name = filter_config.get("name", "Unnamed")
                row_filter = filter_config.get("row_filter", "")
                column_filter = filter_config.get("column_filter", "")
                
                # Format the display string
                display_text = f"{filter_name} | {row_filter} | {column_filter}"
                filter_listbox.insert(tk.END, display_text)
            
            # Select the first item
            if len(self.saved_filters) > 0:
                filter_listbox.selection_set(2)  # Skip the headers
            
            def on_apply():
                # Get the selected filter
                selected_index = filter_listbox.curselection()
                if not selected_index or selected_index[0] < 2:  # Skip headers
                    messagebox.showinfo("No Selection", "Please select a filter to apply")
                    return
                    
                # Get the selected filter configuration
                filter_config = self.saved_filters[selected_index[0] - 2]  # Adjust for headers
                self.apply_saved_filter(filter_config)
                dialog.destroy()
            
            def on_delete():
                # Get the selected filter
                selected_index = filter_listbox.curselection()
                if not selected_index or selected_index[0] < 2:  # Skip headers
                    messagebox.showinfo("No Selection", "Please select a filter to delete")
                    return
                    
                # Confirm deletion
                filter_name = self.saved_filters[selected_index[0] - 2].get("name", "Unnamed")
                if messagebox.askyesno("Confirm Delete", f"Delete filter '{filter_name}'?"):
                    # Delete the filter
                    del self.saved_filters[selected_index[0] - 2]  # Adjust for headers
                    
                    # Save settings to file
                    self.save_settings()
                    
                    # Refresh the listbox
                    filter_listbox.delete(0, tk.END)
                    filter_listbox.insert(tk.END, "Name | Row Filter | Column Filter")
                    filter_listbox.insert(tk.END, "-" * 60)
                    
                    for filter_config in self.saved_filters:
                        filter_name = filter_config.get("name", "Unnamed")
                        row_filter = filter_config.get("row_filter", "")
                        column_filter = filter_config.get("column_filter", "")
                        display_text = f"{filter_name} | {row_filter} | {column_filter}"
                        filter_listbox.insert(tk.END, display_text)
            
            def on_cancel():
                dialog.destroy()
            
            # Add buttons
            button_frame = ttk.Frame(dialog)
            button_frame.pack(pady=10)
            ttk.Button(button_frame, text="Apply", command=on_apply).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Delete", command=on_delete).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
            
            # Center the dialog on the screen
            dialog.update_idletasks()
            width = dialog.winfo_width()
            height = dialog.winfo_height()
            x = (dialog.winfo_screenwidth() // 2) - (width // 2)
            y = (dialog.winfo_screenheight() // 2) - (height // 2)
            dialog.geometry(f'{width}x{height}+{x}+{y}')

        except Exception as e:
            print(f"Error showing saved filters: {e}")
            traceback.print_exc()
            
    def reset_all_filters(self):
        """Reset both row and column filters"""
        try:
            # Reset the row filter
            self.csv_filter_text.set("")
            
            # Reset the column filter
            self.reset_column_filter()
            
            # Show a confirmation message
            print("All filters have been reset")
            
        except Exception as e:
            print(f"Error resetting filters: {e}")
            traceback.print_exc()
    
    def save_file_filter(self):
        """Save the current file filter configuration"""
        try:
            # Get the current file filter
            file_filter = self.filter_text.get().strip()
            
            # Skip if filter is empty
            if not file_filter:
                messagebox.showinfo("Empty Filter", "No file filter is currently active")
                return
            
            # Ask for a name for this filter configuration
            filter_name = simpledialog.askstring("Save File Filter", "Enter a name for this file filter:")
            if not filter_name:
                return  # User canceled
            
            # Create a filter configuration dictionary
            filter_config = {
                "name": filter_name,
                "filter": file_filter
            }
            
            # Check if a filter with this name already exists
            for i, saved_filter in enumerate(self.saved_file_filters):
                if saved_filter.get("name") == filter_name:
                    # Ask if user wants to overwrite
                    if messagebox.askyesno("Filter Exists", f"A file filter named '{filter_name}' already exists. Overwrite?"):
                        self.saved_file_filters[i] = filter_config
                        messagebox.showinfo("Filter Saved", f"File filter '{filter_name}' updated")
                    return
            
            # Add the new filter to the list
            self.saved_file_filters.append(filter_config)
            
            # Save settings to file
            self.save_settings()
            
            # Show a confirmation message
            messagebox.showinfo("Filter Saved", f"File filter '{filter_name}' saved")
            
        except Exception as e:
            print(f"Error saving file filter: {e}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to save file filter:\n{str(e)}")
            
    def show_saved_file_filters(self):
        """Show the saved file filters and allow the user to select one"""
        try:
            # Check if there are any saved filters
            if not self.saved_file_filters:
                messagebox.showinfo("No Saved Filters", "You don't have any saved file filters yet")
                return
                
            # Create a dialog to show the saved filters
            dialog = tk.Toplevel(self)
            dialog.title("Saved File Filters")
            dialog.geometry("500x300")
            
            # Make dialog modal
            dialog.transient(self)
            dialog.grab_set()
            
            # Create a frame for the listbox and scrollbar
            list_frame = ttk.Frame(dialog)
            list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Add a scrollbar
            scrollbar = ttk.Scrollbar(list_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Create listbox with scrollbar
            filter_listbox = tk.Listbox(list_frame, width=50, height=10, yscrollcommand=scrollbar.set)
            filter_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=filter_listbox.yview)
            
            # Add column headers
            filter_listbox.insert(tk.END, "Name | Filter")
            filter_listbox.insert(tk.END, "-" * 60)
            
            # Add saved filters to the listbox
            for filter_config in self.saved_file_filters:
                filter_name = filter_config.get("name", "Unnamed")
                file_filter = filter_config.get("filter", "")
                
                # Format the display string
                display_text = f"{filter_name} | {file_filter}"
                filter_listbox.insert(tk.END, display_text)
            
            # Select the first item
            if len(self.saved_file_filters) > 0:
                filter_listbox.selection_set(2)  # Skip the headers
            
            def on_apply():
                # Get the selected filter
                selected_index = filter_listbox.curselection()
                if not selected_index or selected_index[0] < 2:  # Skip headers
                    messagebox.showinfo("No Selection", "Please select a filter to apply")
                    return
                    
                # Get the selected filter configuration
                filter_config = self.saved_file_filters[selected_index[0] - 2]  # Adjust for headers
                self.apply_saved_file_filter(filter_config)
                dialog.destroy()
            
            def on_delete():
                # Get the selected filter
                selected_index = filter_listbox.curselection()
                if not selected_index or selected_index[0] < 2:  # Skip headers
                    messagebox.showinfo("No Selection", "Please select a filter to delete")
                    return
                    
                # Confirm deletion
                filter_name = self.saved_file_filters[selected_index[0] - 2].get("name", "Unnamed")
                if messagebox.askyesno("Confirm Delete", f"Delete file filter '{filter_name}'?"):
                    # Delete the filter
                    del self.saved_file_filters[selected_index[0] - 2]  # Adjust for headers
                    
                    # Save settings to file
                    self.save_settings()
                    
                    # Refresh the listbox
                    filter_listbox.delete(0, tk.END)
                    filter_listbox.insert(tk.END, "Name | Filter")
                    filter_listbox.insert(tk.END, "-" * 60)
                    
                    for filter_config in self.saved_file_filters:
                        filter_name = filter_config.get("name", "Unnamed")
                        file_filter = filter_config.get("filter", "")
                        display_text = f"{filter_name} | {file_filter}"
                        filter_listbox.insert(tk.END, display_text)
            
            def on_cancel():
                dialog.destroy()
            
            # Add buttons
            button_frame = ttk.Frame(dialog)
            button_frame.pack(pady=10)
            
            ttk.Button(button_frame, text="Apply", command=on_apply).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Delete", command=on_delete).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
            
            # Center the dialog on the screen
            dialog.update_idletasks()
            width = dialog.winfo_width()
            height = dialog.winfo_height()
            x = (dialog.winfo_screenwidth() // 2) - (width // 2)
            y = (dialog.winfo_screenheight() // 2) - (height // 2)
            dialog.geometry(f'{width}x{height}+{x}+{y}')

        except Exception as e:
            print(f"Error showing saved file filters: {e}")
            traceback.print_exc()
            
    def apply_saved_file_filter(self, filter_config):
        """Apply a saved file filter configuration"""
        try:
            # Get the filter setting
            file_filter = filter_config.get("filter", "")
            
            # Apply the file filter
            if file_filter:
                self.filter_text.set(file_filter)
                # Note: filter_files will be called automatically due to the trace
            else:
                # Clear the filter if empty
                self.filter_text.set("")
                
            # Show a confirmation message
            filter_name = filter_config.get("name", "Selected")
            messagebox.showinfo("Filter Applied", f"File filter '{filter_name}' has been applied")
            
        except Exception as e:
            print(f"Error applying saved file filter: {e}")
            traceback.print_exc()
            
    def update_recent_directories_menu(self):
        """Update the recent directories menu with current list of directories"""
        try:
            # First check if the menu attribute exists
            if not hasattr(self, 'recent_dirs_menu'):
                print("Recent directories menu not yet created")
                return
                
            print(f"Updating recent dirs menu")
            
            # Clear existing menu items
            self.recent_dirs_menu.delete(0, tk.END)
            
            # Force update of the menu widget
            self.recent_dirs_menu.update()
            
            # Add each recent directory to the menu
            if self.recent_directories and len(self.recent_directories) > 0:
                for directory in self.recent_directories:
                    # Skip empty or None entries
                    if not directory:
                        continue
                        
                    # Format directory for display (shorten if too long)
                    display_name = directory
                    if len(display_name) > 50:
                        display_name = "..." + display_name[-47:]
                    
                    print(f"Adding menu item: {display_name}")
                    
                    # Use a lambda with a default argument to avoid late binding issues
                    self.recent_dirs_menu.add_command(
                        label=display_name, 
                        command=lambda dir=directory: self.open_recent_directory(dir)
                    )
                    
                    # Add a debug print to verify the command is correctly set
                    print(f"Added menu item for directory: {directory}")
            else:
                # If no recent directories, add a disabled item
                print("No recent directories to add to menu")
                self.recent_dirs_menu.add_command(label="No recent directories", state="disabled")
                
        except Exception as e:
            print(f"Error updating recent directories menu: {e}")
            traceback.print_exc()
            
    def open_recent_directory(self, directory):
        """Open a directory from the recent directories list"""
        try:
            print(f"Opening recent directory: {directory}")
            
            # Check if directory exists
            if not os.path.isdir(directory):
                messagebox.showerror("Directory Not Found", f"The directory no longer exists:\n{directory}")
                # Remove from recent directories
                if directory in self.recent_directories:
                    self.recent_directories.remove(directory)
                    self.update_recent_directories_menu()
                    self.save_settings()  # Save after removing invalid directory
                return
                
            # Set as current directory
            self.current_directory = directory
            
            # Move to top of recent list
            self.add_to_recent_directories(directory)
            
            # Update file list
            self.update_file_list()

            # Update the file dataframe and refresh the browser
            self.update_file_dataframe()
            self.setup_file_browser()
            
            # Update max fields
            old_max = self.max_fields
            self.max_fields = self.get_max_fields()
            print(f"Max fields changed from {old_max} to {self.max_fields}")
            
        except Exception as e:
            print(f"Error opening recent directory: {e}")
            traceback.print_exc()

    def load_settings(self):
        """Load settings from file"""
        try:
            print(f"Attempting to load settings from: {self.settings_file}")
            
            if os.path.exists(self.settings_file):
                print(f"File exists, reading content...")
                with open(self.settings_file, 'r') as f:
                    file_content = f.read()
                    print(f"Raw file content: {file_content}")
                    
                    if file_content.strip():
                        settings = json.loads(file_content)
                        print(f"Loaded settings: {settings}")
                        
                        # Load recent directories
                        self.recent_directories = settings.get("recent_directories", [])
                        print(f"Loaded {len(self.recent_directories)} recent directories")
                        if hasattr(self, 'recent_dirs_menu'):
                            self.update_recent_directories_menu()
                            print("Updated recent directories menu after loading settings")
                        else:
                            print("Recent directories menu not yet created, will update later")
                        # Load saved filters
                        self.saved_filters = settings.get("saved_filters", [])
                        print(f"Loaded {len(self.saved_filters)} saved filters")
                        
                        # Load saved file filters
                        self.saved_file_filters = settings.get("saved_file_filters", [])
                        print(f"Loaded {len(self.saved_file_filters)} saved file filters")
                        
                        # Load plot settings
                        self.saved_plot_settings = settings.get("plot_settings", None)
                        print(f"Loaded plot settings: {self.saved_plot_settings}")
                    else:
                        print("File exists but is empty")
            else:
                print(f"No settings file found at {self.settings_file}")
        except Exception as e:
            print(f"Error loading settings: {e}")
            traceback.print_exc()
            # Initialize with empty lists in case of error
            self.recent_directories = []
            self.saved_filters = []
            self.saved_file_filters = []
            self.saved_plot_settings = None

    def _safe_replot_with_index_preservation(self, pf):
        """
        Safely replot with proper index preservation when column filtering is active
        """
        try:
            # Validate input
            if pf is None:
                print("Warning: Cannot replot - plot viewer is None")
                return False
                
            # Check if we have a saved index column that should be set
            if hasattr(self, 'saved_table_settings') and self.saved_table_settings is not None:
                if 'index_column' in self.saved_table_settings and self.saved_table_settings['index_column'] is not None:
                    saved_index_column = self.saved_table_settings['index_column']
                    
                    # Check if this column exists in the current DataFrame
                    df = self.csv_table.model.df
                    if saved_index_column in df.columns:
                        print(f"Setting {saved_index_column} as index column before replot")
                        # Get the column index
                        col_idx = list(df.columns).index(saved_index_column)
                        # Set the index using pandastable's method
                        self.csv_table.model.setindex([col_idx])
                        # Make sure the index is shown
                        self.csv_table.showIndex()
                    else:
                        print(f"Index column {saved_index_column} not found, using first column for plotting")
                        # Set useindex to False in plot options
                        if hasattr(pf, 'mplopts') and hasattr(pf.mplopts, 'kwds'):
                            pf.mplopts.kwds['useindex'] = False
                            if hasattr(pf.mplopts, 'useindexvar'):
                                pf.mplopts.useindexvar.set(0)
                            
            # Check if we have column filtering active
            has_column_filter = hasattr(self, 'visible_columns') and self.visible_columns is not None
            
            if has_column_filter:
                # When column filtering is active, ensure the index is preserved
                try:
                    # First, ensure the index is properly preserved
                    self._safe_preserve_index()
                    
                    # Make sure the selected columns are valid for the current DataFrame
                    if hasattr(self.csv_table, 'multiplecollist') and self.csv_table.multiplecollist:
                        # Get the current DataFrame
                        df = self.csv_table.model.df
                        
                        # Verify that all selected columns exist in the filtered DataFrame
                        valid_cols = []
                        for col_idx in self.csv_table.multiplecollist:
                            if col_idx < len(df.columns):
                                valid_cols.append(col_idx)
                        
                        # Update the selected columns list
                        if valid_cols:
                            self.csv_table.multiplecollist = valid_cols
                            self.csv_table.drawSelectedCol()
                        else:
                            print("Warning: No valid columns selected for plotting")
                            return False
                    
                    # Ensure the index is properly preserved
                    try:
                        # Force the table to show the index
                        self.csv_table.showindex = True
                        if hasattr(self.csv_table, 'rowheader'):
                            self.csv_table.rowheader.redraw()
                        
                        # Make sure the index is visible
                        try:
                            self.csv_table.showIndex()
                        except Exception as e:
                            print(f"Warning: Error showing index before replot: {e}")
                    except Exception as e:
                        print(f"Warning: Error preserving index before replot: {e}")
                    
                    # Now try to replot with the validated columns and preserved index
                    try:
                        # Try the standard replot first
                        pf.replot()
                    except Exception as e:
                        print(f"Standard replot failed, trying alternative approach: {e}")
                        # If that fails, try a more direct approach
                        try:
                            pf.plotCurrent()
                        except Exception as e2:
                            print(f"Alternative replot also failed: {e2}")
                            return False
                    
                    return True
                except Exception as e:
                    print(f"Warning: Error during safe replot with column filter: {e}")
                    traceback.print_exc()
                    
                    # Try an even safer fallback approach
                    try:
                        # Reset any problematic state in the plot viewer
                        if hasattr(pf, 'local_labels'):
                            delattr(pf, 'local_labels')
                        if hasattr(pf, 'total_curves'):
                            delattr(pf, 'total_curves')
                        
                        # Ensure index is visible
                        self.csv_table.showindex = True
                        if hasattr(self.csv_table, 'rowheader'):
                            self.csv_table.rowheader.redraw()
                        
                        # Try to replot with minimal settings
                        pf.plotCurrent()
                        return True
                    except Exception as e2:
                        print(f"Warning: Fallback replot also failed: {e2}")
                        return False
            else:
                # Standard replot when no column filtering is active
                try:
                    pf.replot()
                    return True
                except Exception as e:
                    print(f"Standard replot failed: {e}")
                    try:
                        # Try alternative approach
                        pf.plotCurrent()
                        return True
                    except Exception as e2:
                        print(f"Alternative replot also failed: {e2}")
                        return False
        except Exception as e:
            print(f"Error during safe replot: {e}")
            traceback.print_exc()
            return False
    
    def convert_numpy_types(self, obj):
        """Convert NumPy types to native Python types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value)for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_numpy_types(item) for item in obj)
        else:
            return obj    


    def save_settings(self):
        """Save settings to file"""
        print(f"Settings file absolute path: {os.path.abspath(self.settings_file)}")
        try:
            # Create settings directory if it doesn't exist
            directory = os.path.dirname(self.settings_file)
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            
            # Ensure recent_directories is a list
            if not hasattr(self, 'recent_directories') or self.recent_directories is None:
                self.recent_directories = []
                
            settings = {
                "recent_directories": self.recent_directories,
                "saved_filters": self.saved_filters,
                "saved_file_filters": self.saved_file_filters,
                "plot_settings": self.saved_plot_settings,
                "table_settings": self.saved_table_settings
            }
            
            # Convert NumPy types to native Python types
            settings = self.convert_numpy_types(settings)
            
            # Debug print before saving
            print(f"Saving settings: {settings}")
            # Add this after line 3954 (inside the save_settings method)
            print(f"Writing settings to file: {self.settings_file}")
            print(f"File exists before writing: {os.path.exists(self.settings_file)}")
            print(f"Directory exists: {os.path.exists(os.path.dirname(self.settings_file))}")
            print(f"Directory is writable: {os.access(os.path.dirname(self.settings_file), os.W_OK)}")
            
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f)
            
            # Verify the file was created and has content
            if os.path.exists(self.settings_file):
                file_size = os.path.getsize(self.settings_file)
                print(f"Settings file saved successfully. Size: {file_size} bytes")
            else:
                print(f"Error: Settings file was not created at {self.settings_file}")
                
        except Exception as e:
            print(f"Error saving settings: {e}")
            traceback.print_exc()

    def quit(self):
        """Save settings and quit the application"""
        try:
            # Save settings to file
            self.save_settings()
            
            # Destroy the window and exit
            self.destroy()
            
        except Exception as e:
            print(f"Error during quit: {e}")
            traceback.print_exc()
            # Still try to exit even if there was an error
            self.destroy()

    def save_table_settings(self):
        """Save the current table settings including index column and selected columns"""
        try:
            # Check if the CSV table exists
            if not hasattr(self, 'csv_table'):
                return
            
            # Create a dictionary to store table settings
            settings = {}
            
            # Save index column information
            if hasattr(self.csv_table, 'model') and hasattr(self.csv_table.model, 'df'):
                # Save the current index name
                index_name = self.csv_table.model.df.index.name
                if index_name is not None:
                    # Check if this is a temporary index name (ends with _index_temp or similar)
                    original_name = index_name
                    if "_index_temp" in index_name:
                        # Extract the original column name
                        original_name = index_name.split("_index_temp")[0]
                        print(f"Detected temporary index name, using original name: {original_name}")
                    
                    # Save both names
                    settings['index_column'] = original_name
                    settings['temp_index_column'] = index_name
                    print(f"Saved index column: {original_name}")
            
            # Save visible columns
            if hasattr(self, 'visible_columns') and self.visible_columns is not None:
                settings['visible_columns'] = list(self.visible_columns)
            
            # Save column order if available
            if hasattr(self.csv_table, 'columnorder'):
                settings['column_order'] = self.csv_table.columnorder
            
            # Save column widths if available
            if hasattr(self.csv_table, 'columnwidths'):
                # Convert to a serializable format (dict of string keys)
                col_widths = {}
                for col, width in self.csv_table.columnwidths.items():
                    col_widths[str(col)] = width
                settings['column_widths'] = col_widths
            
            # Save current sort column and order if available
            if hasattr(self.csv_table, 'sortkey'):
                settings['sort_key'] = self.csv_table.sortkey
            if hasattr(self.csv_table, 'sortOrder'):
                settings['sort_order'] = self.csv_table.sortOrder
            
            # Save selected columns for plotting
            if hasattr(self.csv_table, 'multiplecollist'):
                # Save the column names instead of indices for better persistence
                if self.csv_table.multiplecollist and hasattr(self.csv_table.model, 'df'):
                    try:
                        # Convert column indices to column names
                        col_indices = self.csv_table.multiplecollist
                        col_names = [self.csv_table.model.df.columns[i] for i in col_indices]
                        settings['plot_columns'] = col_names
                        print(f"Saved plot columns: {col_names}")
                    except Exception as e:
                        print(f"Error saving plot columns: {e}")
                        traceback.print_exc()
            
            # Store the settings
            self.saved_table_settings = settings
            print("Table settings saved successfully")
            
        except Exception as e:
            print(f"Error saving table settings: {e}")
            traceback.print_exc()

    def restore_table_settings(self):
        """Restore previously saved table settings to the current table"""
        try:
            # Check if we have saved settings and if the table exists
            if not hasattr(self, 'saved_table_settings') or self.saved_table_settings is None or not hasattr(self, 'csv_table'):
                return
            
            settings = self.saved_table_settings
            
            print(f"DEBUG: restore_table_settings - Starting - DataFrame index type: {type(self.csv_table.model.df.index)}")
            print(f"DEBUG: restore_table_settings - Starting - DataFrame index name: {self.csv_table.model.df.index.name}")
            print(f"DEBUG: restore_table_settings - Starting - DataFrame index sample: {self.csv_table.model.df.index[:5]}")
            
            # Restore index column if specified
            if 'index_column' in settings and settings['index_column'] is not None:
                try:
                    # Check if the column exists in the current dataframe
                    index_column = settings['index_column']
                    print(f"DEBUG: restore_table_settings - Trying to restore index column: {index_column}")
                    
                    if index_column in self.csv_table.model.df.columns:
                        # Get the column index
                        col_idx = list(self.csv_table.model.df.columns).index(index_column)
                        print(f"DEBUG: restore_table_settings - Found index column at position: {col_idx}")
                        
                        # Set the index using the column index
                        self.csv_table.model.setindex([col_idx])
                        
                        # Make sure the index is shown
                        self.csv_table.showindex = True
                        self.csv_table.showIndex()
                        
                        print(f"DEBUG: restore_table_settings - After restoring index - DataFrame index type: {type(self.csv_table.model.df.index)}")
                        print(f"DEBUG: restore_table_settings - After restoring index - DataFrame index name: {self.csv_table.model.df.index.name}")
                        print(f"DEBUG: restore_table_settings - After restoring index - DataFrame index sample: {self.csv_table.model.df.index[:5]}")
                        
                        print(f"Restored index column: {index_column}")
                    else:
                        print(f"DEBUG: restore_table_settings - Index column {index_column} not found in current DataFrame")
                except Exception as e:
                    print(f"Error restoring index column: {e}")
                    traceback.print_exc()
            
            # Restore visible columns if specified
            if 'visible_columns' in settings and settings['visible_columns'] is not None:
                try:
                    # Store the visible columns
                    self.visible_columns = settings['visible_columns']
                    
                    # Apply the column filter
                    self._apply_column_filter()
                    print(f"Restored visible columns: {len(self.visible_columns)} columns")
                except Exception as e:
                    print(f"Error restoring visible columns: {e}")
                    traceback.print_exc()
            
            # Restore column order if specified
            if 'column_order' in settings and settings['column_order'] is not None:
                try:
                    self.csv_table.columnorder = settings['column_order']
                    print(f"Restored column order")
                except Exception as e:
                    print(f"Error restoring column order: {e}")
                    traceback.print_exc()
            
            # Restore column widths if specified
            if 'column_widths' in settings and settings['column_widths'] is not None:
                try:
                    # Convert back from string keys to original format
                    col_widths = {}
                    for col_str, width in settings['column_widths'].items():
                        try:
                            # Try to convert to int if it's a numeric string
                            col = int(col_str)
                        except ValueError:
                            # Otherwise keep as string
                            col = col_str
                        col_widths[col] = width
                    
                    self.csv_table.columnwidths = col_widths
                    print(f"Restored column widths")
                except Exception as e:
                    print(f"Error restoring column widths: {e}")
                    traceback.print_exc()
            
            # Restore sort settings if specified
            if 'sort_key' in settings and settings['sort_key'] is not None:
                try:
                    self.csv_table.sortkey = settings['sort_key']
                    print(f"Restored sort key: {settings['sort_key']}")
                except Exception as e:
                    print(f"Error restoring sort key: {e}")
                    traceback.print_exc()
                    
            if 'sort_order' in settings and settings['sort_order'] is not None:
                try:
                    self.csv_table.sortOrder = settings['sort_order']
                    print(f"Restored sort order: {settings['sort_order']}")
                except Exception as e:
                    print(f"Error restoring sort order: {e}")
                    traceback.print_exc()
            
            # Restore selected columns for plotting
            if 'plot_columns' in settings and settings['plot_columns'] is not None:
                try:
                    # Convert column names to indices
                    col_names = settings['plot_columns']
                    col_indices = []
                    
                    for col_name in col_names:
                        if col_name in self.csv_table.model.df.columns:
                            col_idx = list(self.csv_table.model.df.columns).index(col_name)
                            col_indices.append(col_idx)
                    
                    if col_indices:
                        self.csv_table.multiplecollist = col_indices
                        # Draw the selected columns in the UI
                        self.csv_table.drawSelectedCol()
                        print(f"Restored plot columns: {col_names}")
                except Exception as e:
                    print(f"Error restoring plot columns: {e}")
                    traceback.print_exc()
            
            # Redraw the table to apply all settings
            self.csv_table.redraw()
            
            # At the end, after redraw
            print(f"DEBUG: restore_table_settings - End - DataFrame index type: {type(self.csv_table.model.df.index)}")
            print(f"DEBUG: restore_table_settings - End - DataFrame index name: {self.csv_table.model.df.index.name}")
            print(f"DEBUG: restore_table_settings - End - DataFrame index sample: {self.csv_table.model.df.index[:5]}")
            
        except Exception as e:
            print(f"Error restoring table settings: {e}")
            traceback.print_exc()

    def save_plot_settings_to_file(self):
        """Save the current plot settings to a file"""
        try:
            # Check if the plot frame exists and has been initialized
            if not hasattr(self, 'csv_table') or not hasattr(self.csv_table, 'pf'):
                return
            
            # Get the PlotViewer instance
            pf = self.csv_table.pf
            
            # Create a dictionary to store plot settings
            settings = {}
            
            # Save global options
            if hasattr(pf, 'globalopts'):
                settings['globalopts'] = copy.deepcopy(pf.globalopts)
            
            # Save matplotlib options
            if hasattr(pf, 'mplopts') and hasattr(pf.mplopts, 'kwds'):
                # Create a copy to avoid modifying the original
                mpl_options = copy.deepcopy(pf.mplopts.kwds)
                
                # Fix for 'color' and 'colormap' conflict - only save one of them
                if 'color' in mpl_options and 'colormap' in mpl_options:
                    # Prioritize 'color' over 'colormap'
                    del mpl_options['colormap']
                
                settings['mplopts'] = mpl_options
            
            # Save layout options
            if hasattr(pf, 'layoutopts'):
                settings['layout'] = {
                    'rows': pf.layoutopts.rows,
                    'cols': pf.layoutopts.cols
                }
                # Save mode if available
                if hasattr(pf.layoutopts, 'modevar'):
                    settings['layout']['mode'] = pf.layoutopts.modevar.get()
            
            # Save 3D options
            if hasattr(pf, 'mplopts3d') and hasattr(pf.mplopts3d, 'kwds'):
                # Create a copy to avoid modifying the original
                mpl_3d_options = copy.deepcopy(pf.mplopts3d.kwds)
                
                # Fix for 'color' and 'colormap' conflict in 3D options
                if 'color' in mpl_3d_options and 'colormap' in mpl_3d_options:
                    del mpl_3d_options['colormap']
                
                settings['mplopts3d'] = mpl_3d_options
            
            # Save label options
            if hasattr(pf, 'labelopts') and hasattr(pf.labelopts, 'kwds'):
                settings['labelopts'] = copy.deepcopy(pf.labelopts.kwds)
            
            # Ask user where to save the settings
            base_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
                initialdir=os.path.dirname(self.current_file) if self.current_file else None,
                initialfile=f"plot_settings-{os.path.basename(self.current_file) if self.current_file else 'default'}"
            )
            
            if base_path:
                # Remove .json extension if present
                base_path = base_path.rsplit('.json', 1)[0]
                base_path = os.path.normpath(os.path.abspath(base_path))
                base_path = self.normalize_long_path(base_path)

                # Save the settings to file
                with open(base_path + '.json', 'w') as f:
                    json.dump(settings, f)
                
                print(f"Plot settings saved to: {base_path}.json")
                messagebox.showinfo("Save Successful", f"Plot settings saved to:\n{base_path}.json")
                
        except Exception as e:
            print(f"Error saving plot settings: {e}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to save plot settings:\n{str(e)}")
            
    def load_plot_settings_from_file(self):
        """Load plot settings from a file and apply them to the current plot"""
        try:
            # Ask user to select a settings file
            file_path = filedialog.askopenfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
                initialdir=os.path.dirname(self.current_file) if self.current_file else None,
                title="Select Plot Settings File"
            )
            
            if not file_path:
                return
                
            # Load settings from the file
            with open(file_path, 'r') as f:
                settings = json.load(f)
                
            # Store the settings
            self.saved_plot_settings = settings
            
            # Apply settings if a plot is currently open
            if hasattr(self, 'csv_table'):
                # Store the settings in the table for later use when plot is shown
                self.csv_table.custom_plot_settings = settings
                
                # If the plot viewer is already open, apply settings immediately
                if hasattr(self.csv_table, 'pf'):
                    pf = self.csv_table.pf
                    
                    # Apply global options
                    if 'globalopts' in settings and hasattr(pf, 'globalopts'):
                        for k, v in settings['globalopts'].items():
                            pf.globalopts[k] = v
                            if hasattr(pf, 'globalvars') and k in pf.globalvars:
                                pf.globalvars[k].set(v)
                    
                    # Apply matplotlib options with conflict resolution
                    if 'mplopts' in settings and hasattr(pf, 'mplopts'):
                        # Create a copy to avoid modifying the original
                        mpl_options = settings['mplopts'].copy()
                        
                        # Fix for 'color' and 'colormap' conflict
                        if 'color' in mpl_options and 'colormap' in mpl_options:
                            # Prioritize 'color' over 'colormap'
                            del mpl_options['colormap']
                        
                        pf.mplopts.kwds.update(mpl_options)
                        if hasattr(pf.mplopts, 'updateFromDict'):
                            pf.mplopts.updateFromDict(mpl_options)
                    
                    # Apply layout options
                    if 'layout' in settings and hasattr(pf, 'layoutopts'):
                        if 'rows' in settings['layout']:
                            pf.layoutopts.rows = settings['layout']['rows']
                        if 'cols' in settings['layout']:
                            pf.layoutopts.cols = settings['layout']['cols']
                        if 'mode' in settings['layout'] and hasattr(pf.layoutopts, 'modevar'):
                            pf.layoutopts.modevar.set(settings['layout']['mode'])
                    
                    # Apply 3D options with conflict resolution
                    if 'mplopts3d' in settings and hasattr(pf, 'mplopts3d'):
                        # Create a copy to avoid modifying the original
                        mpl_3d_options = settings['mplopts3d'].copy()
                        
                        # Fix for 'color' and 'colormap' conflict in 3D options
                        if 'color' in mpl_3d_options and 'colormap' in mpl_3d_options:
                            del mpl_3d_options['colormap']
                        
                        pf.mplopts3d.kwds.update(mpl_3d_options)
                        if hasattr(pf.mplopts3d, 'updateFromDict'):
                            pf.mplopts3d.updateFromDict(mpl_3d_options)
                    
                    # Apply label options
                    if 'labelopts' in settings and hasattr(pf, 'labelopts'):
                        pf.labelopts.kwds.update(settings['labelopts'])
                        if hasattr(pf.labelopts, 'updateFromDict'):
                            pf.labelopts.updateFromDict(settings['labelopts'])
                    
                    # Suppress warnings globally for this session
                    import warnings
                    warnings.filterwarnings('ignore', message='.*color.*and.*colormap.*cannot be used simultaneously.*')
                    warnings.filterwarnings('ignore', message='.*Tight layout not applied.*')
                    
                    # Replot with the new settings
                    if hasattr(pf, 'replot') and callable(pf.replot):
                        try:
                            pf.replot()
                        except Exception as e:
                            print(f"Warning: Error during replot: {e}")
                
                # If plot viewer is not open, patch the showPlot method to apply settings when opened
                else:
                    # Monkey patch the showPlot method to apply our settings
                    original_showPlot = self.csv_table.showPlot
                    
                    def patched_showPlot(self, *args, **kwargs):
                        # Call the original method first
                        result = original_showPlot(*args, **kwargs)
                        
                        # Now apply our saved settings
                        if hasattr(self, 'custom_plot_settings') and hasattr(self, 'pf'):
                            pf = self.pf
                            settings = self.custom_plot_settings
                            
                            # Apply global options
                            if 'globalopts' in settings and hasattr(pf, 'globalopts'):
                                for k, v in settings['globalopts'].items():
                                    pf.globalopts[k] = v
                                    if hasattr(pf, 'globalvars') and k in pf.globalvars:
                                        pf.globalvars[k].set(v)
                            
                            # Apply matplotlib options with conflict resolution
                            if 'mplopts' in settings and hasattr(pf, 'mplopts'):
                                # Create a copy to avoid modifying the original
                                mpl_options = settings['mplopts'].copy()
                                
                                # Fix for 'color' and 'colormap' conflict
                                if 'color' in mpl_options and 'colormap' in mpl_options:
                                    # Prioritize 'color' over 'colormap'
                                    del mpl_options['colormap']
                                
                                pf.mplopts.kwds.update(mpl_options)
                                if hasattr(pf.mplopts, 'updateFromDict'):
                                    pf.mplopts.updateFromDict(mpl_options)
                            
                            # Apply layout options
                            if 'layout' in settings and hasattr(pf, 'layoutopts'):
                                if 'rows' in settings['layout']:
                                    pf.layoutopts.rows = settings['layout']['rows']
                                if 'cols' in settings['layout']:
                                    pf.layoutopts.cols = settings['layout']['cols']
                                if 'mode' in settings['layout'] and hasattr(pf.layoutopts, 'modevar'):
                                    pf.layoutopts.modevar.set(settings['layout']['mode'])
                            
                            # Apply 3D options with conflict resolution
                            if 'mplopts3d' in settings and hasattr(pf, 'mplopts3d'):
                                # Create a copy to avoid modifying the original
                                mpl_3d_options = settings['mplopts3d'].copy()
                                
                                # Fix for 'color' and 'colormap' conflict in 3D options
                                if 'color' in mpl_3d_options and 'colormap' in mpl_3d_options:
                                    del mpl_3d_options['colormap']
                                
                                pf.mplopts3d.kwds.update(mpl_3d_options)
                                if hasattr(pf.mplopts3d, 'updateFromDict'):
                                    pf.mplopts3d.updateFromDict(mpl_3d_options)
                            
                            # Apply label options
                            if 'labelopts' in settings and hasattr(pf, 'labelopts'):
                                pf.labelopts.kwds.update(settings['labelopts'])
                                if hasattr(pf.labelopts, 'updateFromDict'):
                                    pf.labelopts.updateFromDict(settings['labelopts'])
                            
                            # Suppress warnings globally for this session
                            import warnings
                            warnings.filterwarnings('ignore', message='.*color.*and.*colormap.*cannot be used simultaneously.*')
                            warnings.filterwarnings('ignore', message='.*Tight layout not applied.*')
                            
                            # Replot with the new settings
                            if hasattr(pf, 'replot') and callable(pf.replot):
                                try:
                                    pf.replot()
                                except Exception as e:
                                    print(f"Warning: Error during replot: {e}")
                        
                        return result
                    
                    # Replace the original method with our patched version
                    self.csv_table.showPlot = types.MethodType(patched_showPlot, self.csv_table)
            
            # Save to application settings
            self.save_settings()
            
            print(f"Plot settings loaded from: {file_path}")
            messagebox.showinfo("Load Successful", f"Plot settings loaded from:\n{file_path}")
            
        except Exception as e:
            print(f"Error loading plot settings: {e}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load plot settings:\n{str(e)}")
            
    def save_plot_settings(self):
        """Save the current plot settings from the pandastable PlotViewer"""
        try:
            # Check if the plot frame exists and has been initialized
            if not hasattr(self, 'csv_table') or not hasattr(self.csv_table, 'pf'):
                return
            
            # Get the PlotViewer instance
            pf = self.csv_table.pf
            
            # Create a dictionary to store plot settings
            settings = {}
            
            # Save global options
            if hasattr(pf, 'globalopts'):
                settings['globalopts'] = copy.deepcopy(pf.globalopts)
            
            # Save matplotlib options
            if hasattr(pf, 'mplopts') and hasattr(pf.mplopts, 'kwds'):
                # Create a copy to avoid modifying the original
                mpl_options = copy.deepcopy(pf.mplopts.kwds)
                
                # Fix for 'color' and 'colormap' conflict - only save one of them
                if 'color' in mpl_options and 'colormap' in mpl_options:
                    # Prioritize 'color' over 'colormap'
                    del mpl_options['colormap']
                
                settings['mplopts'] = mpl_options
            
            # Save layout options
            if hasattr(pf, 'layoutopts'):
                settings['layout'] = {
                    'rows': pf.layoutopts.rows,
                    'cols': pf.layoutopts.cols
                }
                # Save mode if available
                if hasattr(pf.layoutopts, 'modevar'):
                    settings['layout']['mode'] = pf.layoutopts.modevar.get()
            
            # Save 3D options
            if hasattr(pf, 'mplopts3d') and hasattr(pf.mplopts3d, 'kwds'):
                # Create a copy to avoid modifying the original
                mpl_3d_options = copy.deepcopy(pf.mplopts3d.kwds)
                
                # Fix for 'color' and 'colormap' conflict in 3D options
                if 'color' in mpl_3d_options and 'colormap' in mpl_3d_options:
                    del mpl_3d_options['colormap']
                
                settings['mplopts3d'] = mpl_3d_options
            
            # Save label options
            if hasattr(pf, 'labelopts') and hasattr(pf.labelopts, 'kwds'):
                settings['labelopts'] = copy.deepcopy(pf.labelopts.kwds)
            
            # Store the settings
            self.saved_plot_settings = settings
            
            # Save settings to file
            self.save_settings()
            
            print("Plot settings saved successfully")
            
        except Exception as e:
            print(f"Error saving plot settings: {e}")
            traceback.print_exc()
            
    def restore_plot_settings(self):
        """Restore previously saved plot settings to the pandastable PlotViewer"""
        try:
            # Check if plot settings exist
            if not hasattr(self, 'saved_plot_settings') or not self.saved_plot_settings:
                print("No plot settings to restore")
                return False
                
            # Check if csv_table exists
            if not hasattr(self, 'csv_table'):
                print("No table to apply plot settings to")
                return False
                
            # IMPORTANT: Check if plot viewer exists before proceeding
            if not hasattr(self.csv_table, 'pf') or self.csv_table.pf is None:
                print("Plot viewer not open, skipping plot settings restoration")
                return False            

            # The PlotViewer is created on demand when showPlot is called
            # We'll add a custom method to the csv_table that will apply our settings when the plot is shown
            self.csv_table.custom_plot_settings = self.saved_plot_settings
            
            # Check if plot viewer is already open
            if hasattr(self.csv_table, 'pf') and self.csv_table.pf is not None:
                pf = self.csv_table.pf
                settings = self.saved_plot_settings
                
                # Apply settings to the open plot viewer
                if not self._apply_plot_settings_to_viewer(pf, settings):
                    print("Warning: Failed to apply plot settings to viewer")
                    return False
                
                # Automatically replot with the selected columns
                if hasattr(self.csv_table, 'multiplecollist') and self.csv_table.multiplecollist:
                    try:
                        # Draw the selected columns in the UI
                        self.csv_table.drawSelectedCol()
                        
                        # Use our safe replot method that ensures index preservation
                        if self._safe_replot_with_index_preservation(pf):
                            print("Automatically replotted with the selected columns")
                        else:
                            print("Warning: Could not automatically replot")
                    except Exception as e:
                        print(f"Warning: Error during automatic replot: {e}")
                        traceback.print_exc()
                
                return True
            else:
                # Monkey patch the showPlot method to apply our settings
                try:
                    original_showPlot = self.csv_table.showPlot
                    
                    def patched_showPlot(self, *args, **kwargs):
                        # Call the original method first
                        result = original_showPlot(*args, **kwargs)
                        
                        # Now apply our saved settings
                        if hasattr(self, 'custom_plot_settings') and hasattr(self, 'pf') and self.pf is not None:
                            pf = self.pf
                            settings = self.custom_plot_settings
                            
                            # Apply settings to the newly opened plot viewer
                            app = self.table.app if hasattr(self.table, 'app') else None
                            if app and hasattr(app, '_apply_plot_settings_to_viewer'):
                                app._apply_plot_settings_to_viewer(pf, settings)
                        
                        return result
                    
                    # Replace the original method with our patched version
                    self.csv_table.showPlot = types.MethodType(patched_showPlot, self.csv_table)
                    return True
                except Exception as e:
                    print(f"Warning: Error patching showPlot method: {e}")
                    traceback.print_exc()
                    return False
            
        except Exception as e:
            print(f"Error restoring plot settings: {e}")
            traceback.print_exc()
            return False    

    def _apply_plot_settings_to_viewer(self, pf, settings):
        """Apply the given settings to a plot viewer instance"""
        try:
            # Validate inputs first
            if pf is None or settings is None:
                print("Warning: Cannot apply plot settings - plot viewer or settings are None")
                return False
                
            # Apply global options
            if 'globalopts' in settings and hasattr(pf, 'globalopts'):
                for k, v in settings['globalopts'].items():
                    pf.globalopts[k] = v
                    if hasattr(pf, 'globalvars') and k in pf.globalvars:
                        pf.globalvars[k].set(v)
            
            # Apply matplotlib options with conflict resolution
            if 'mplopts' in settings and hasattr(pf, 'mplopts'):
                # Create a copy to avoid modifying the original
                mpl_options = settings['mplopts'].copy()
                
                # Fix for 'color' and 'colormap' conflict
                if 'color' in mpl_options and 'colormap' in mpl_options:
                    # Prioritize 'color' over 'colormap'
                    del mpl_options['colormap']
                
                pf.mplopts.kwds.update(mpl_options)
                if hasattr(pf.mplopts, 'updateFromDict'):
                    pf.mplopts.updateFromDict(mpl_options)
            
            # Apply layout options
            if 'layout' in settings and hasattr(pf, 'layoutopts'):
                if 'rows' in settings['layout']:
                    pf.layoutopts.rows = settings['layout']['rows']
                if 'cols' in settings['layout']:
                    pf.layoutopts.cols = settings['layout']['cols']
                if 'mode' in settings['layout'] and hasattr(pf.layoutopts, 'modevar'):
                    pf.layoutopts.modevar.set(settings['layout']['mode'])
            
            # Apply 3D options with conflict resolution
            if 'mplopts3d' in settings and hasattr(pf, 'mplopts3d'):
                # Create a copy to avoid modifying the original
                mpl_3d_options = settings['mplopts3d'].copy()
                
                # Fix for 'color' and 'colormap' conflict in 3D options
                if 'color' in mpl_3d_options and 'colormap' in mpl_3d_options:
                    del mpl_3d_options['colormap']
                
                pf.mplopts3d.kwds.update(mpl_3d_options)
                if hasattr(pf.mplopts3d, 'updateFromDict'):
                    pf.mplopts3d.updateFromDict(mpl_3d_options)
            
            # Apply label options - Add extra validation here
            if 'labelopts' in settings and hasattr(pf, 'labelopts') and hasattr(pf.labelopts, 'kwds'):
                try:
                    pf.labelopts.kwds.update(settings['labelopts'])
                    if hasattr(pf.labelopts, 'updateFromDict'):
                        pf.labelopts.updateFromDict(settings['labelopts'])
                except Exception as e:
                    print(f"Warning: Error applying label options: {e}")
            
            # Suppress warnings globally for this session
            import warnings
            warnings.filterwarnings('ignore', message='.*color.*and.*colormap.*cannot be used simultaneously.*')
            warnings.filterwarnings('ignore', message='.*Tight layout not applied.*')
            
            return True
            
        except Exception as e:
            print(f"Error applying plot settings: {e}")
            traceback.print_exc()
            return False

if __name__ == "__main__":
    app = CSVBrowser()
    app.mainloop()
