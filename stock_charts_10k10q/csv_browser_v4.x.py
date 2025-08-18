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
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from pandastable import Table, TableModel
from datetime import datetime
import shutil
import traceback
import subprocess
import numpy as np

class CSVBrowser(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("CSV Browser")
        self.state('zoomed')
        
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

        # # Initialize frame attributes
        # self.pt_frame = ttk.Frame(self)
        # self.csv_frame = ttk.Frame(self)
        
        # Set default directory
        # self.current_directory = os.path.dirname(os.path.abspath(__file__))
        self.current_directory = r"C:\Users\juesh\OneDrive\Documents\windsurf\stock_data_"
        # self.current_directory = r"C:\Users\JueShi\Astera Labs, Inc\Silicon Engineering - T3_MPW_Rx_C2M_Test_Results"
        
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
        ttk.Label(filter_frame, text="Filter Files:").pack(side="left", padx=(0, 5))
        filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_text)
        filter_entry.pack(side="left", fill="x", expand=True)
        
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
                max_width = max(len(str(x)) for x in self.df[col].head(20))
                self.table.columnwidths[col] = max(min(max_width * 10, 200), 50)

        self.table.redraw()
        
        # Bind selection event
        self.table.bind('<ButtonRelease-1>', self.on_table_select)
        self.table.bind('<Up>', self.on_key_press)
        self.table.bind('<Down>', self.on_key_press)

    def filter_files(self, *args):
        """Filter files based on the filter text"""
        if hasattr(self, 'table'):
            try:
                # Get filter text and remove any quotes
                filter_text = self.filter_text.get().lower().strip('"\'')
                print(f"\n=== Filtering files with: '{filter_text}' ===")  # Debug print
                
                if filter_text:
                    # Split filter text by AND (both & and +)
                    filter_terms = [term.strip() for term in filter_text.replace('&', '+').split('+')]
                    print(f"Searching for terms: {filter_terms}")  # Debug print
                    
                    # Convert all columns to string
                    str_df = self.df.astype(str)
                    
                    # Start with all rows
                    mask = pd.Series([True] * len(str_df), index=str_df.index)
                    
                    # Apply each filter term with AND logic
                    for term in filter_terms:
                        term = term.strip()
                        if term:  # Skip empty terms
                            if term.startswith('!'):  # Exclusion term
                                exclude_term = term[1:].strip()  # Remove ! and whitespace
                                if exclude_term:  # Only if there's something to exclude
                                    term_mask = ~str_df.apply(
                                        lambda x: x.str.contains(exclude_term, case=False, na=False, regex=False)
                                    ).any(axis=1)
                                    print(f"Excluding rows containing: '{exclude_term}'")  # Debug print
                            else:  # Inclusion term
                                term_mask = str_df.apply(
                                    lambda x: x.str.contains(term, case=False, na=False, regex=False)
                                ).any(axis=1)
                                print(f"Including rows containing: '{term}'")  # Debug print
                            mask = mask & term_mask
                    
                    # Debug print matches
                    print("\nMatching results:")
                    for idx, row in str_df[mask].iterrows():
                        print(f"Match found in row {idx}:")
                        for col in str_df.columns:
                            matches = []
                            col_value = str(row[col]).lower()
                            for term in filter_terms:
                                term = term.strip()
                                if term.startswith('!'):
                                    continue  # Skip exclusion terms in match display
                                if term in col_value:
                                    matches.append(term)
                            if matches:
                                print(f"  Column '{col}': {row[col]} (matched terms: {matches})")
                    
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
                if row < 0 or row >= len(displayed_df): # Add this check
                    print(f"Row index {row} out of bounds for displayed_df of length {len(displayed_df)}")
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
            
            ttk.Label(self.csv_filter_frame, text="Filter CSV:").grid(row=0, column=0, padx=(0,5))
            
            # Use the existing StringVar - don't add another trace
            print(f"Current filter value: '{self.csv_filter_text.get()}'")  # Debug print
            
            # Create and set up the entry widget
            filter_entry = ttk.Entry(self.csv_filter_frame, textvariable=self.csv_filter_text)
            filter_entry.grid(row=0, column=1, sticky='ew')
            
            # Create frame for pandastable using grid
            self.csv_table_frame = ttk.Frame(self.csv_view_container)
            self.csv_table_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
            
            # Configure grid weights
            self.csv_view_container.rowconfigure(1, weight=1)
            self.csv_view_container.columnconfigure(0, weight=1)
            
            # Create empty pandastable for CSV viewing
            empty_df = pd.DataFrame()
            self.csv_table = Table(self.csv_table_frame, dataframe=empty_df,
                                showtoolbar=True, showstatusbar=True)
            self.csv_table.show()
            
            # Store original data
            self.original_csv_df = None
            
        except Exception as e:
            print(f"Error in setup_csv_viewer: {e}")
            traceback.print_exc()


    def filter_csv_content(self, *args):
        """Filter the CSV content based on the filter text"""
        print("\n=== filter_csv_content called! ===")  # Debug print
        if hasattr(self, 'csv_table') and self.original_csv_df is not None:
            try:
                filter_text = self.csv_filter_text.get()
                print(f"Filter text: '{filter_text}'")  # Debug print
                print(f"Original DataFrame shape: {self.original_csv_df.shape}")  # Debug print
                
                # Store the current filter
                self.current_csv_filter = filter_text
                
                # Always start with a fresh copy of the original data
                if filter_text:
                    try:
                        # First try to evaluate as a pandas query/expression
                        print("Attempting pandas query...")  # Debug print
                        filtered_df = self.original_csv_df.query(filter_text).copy()
                        print(f"Query successful, filtered shape: {filtered_df.shape}")  # Debug print
                    except Exception as e:
                        print(f"Query failed ({str(e)}), falling back to contains search")  # Debug print
                        # Fall back to contains search if query fails
                        # Split filter text by AND (both & and +)
                        filter_terms = [term.strip() for term in filter_text.replace('&', '+').split('+')]
                        print(f"Text search with terms: {filter_terms}")  # Debug print
                        
                        # Convert all columns to string for searching
                        str_df = self.original_csv_df.astype(str)
                        
                        # Start with all rows
                        mask = pd.Series([True] * len(str_df), index=str_df.index)
                        
                        # Apply each filter term with AND logic
                        for term in filter_terms:
                            term = term.strip()
                            if term:  # Skip empty terms
                                if term.startswith('!'):  # Exclusion term
                                    exclude_term = term[1:].strip()  # Remove ! and whitespace
                                    if exclude_term:  # Only if there's something to exclude
                                        term_mask = ~str_df.apply(
                                            lambda x: x.str.contains(exclude_term, case=False, na=False, regex=False)
                                        ).any(axis=1)
                                        print(f"Excluding rows containing: '{exclude_term}'")  # Debug print
                                else:  # Inclusion term
                                    term_mask = str_df.apply(
                                        lambda x: x.str.contains(term, case=False, na=False, regex=False)
                                    ).any(axis=1)
                                    print(f"Including rows containing: '{term}'")  # Debug print
                                mask = mask & term_mask
                                
                                # Debug print matches for this term
                                if not term.startswith('!'):  # Only show matches for inclusion terms
                                    matching_rows = str_df[term_mask]
                                    print(f"\nRows matching term '{term}': {len(matching_rows)}")
                                    for idx, row in matching_rows.head(5).iterrows():  # Show first 5 matches
                                        print(f"Row {idx}:")
                                        for col in str_df.columns:
                                            if term.lower() in str(row[col]).lower():
                                                print(f"  {col}: {row[col]}")
                        
                        filtered_df = self.original_csv_df[mask].copy()
                        print(f"\nFinal results after combining terms: {len(filtered_df)} rows")
                else:
                    filtered_df = self.original_csv_df.copy()
                    print("No filter text, using original DataFrame")  # Debug print
                
                # Update the table with the filtered data
                print("Updating table with filtered data...")  # Debug print
                self.csv_table.model.df = filtered_df
                
                # Ensure column widths are maintained
                print("Adjusting column widths...")  # Debug print
                for col in filtered_df.columns:
                    if col in self.csv_table.columnwidths:
                        width = max(
                            len(str(x)) for x in filtered_df[col].head(20)
                        ) * 10
                        self.csv_table.columnwidths[col] = max(min(width, 250), 30)
                
                # Redraw the table
                print("Redrawing table...")  # Debug print
                self.csv_table.redraw()
                print("Filter operation complete")  # Debug print
                
            except Exception as e:
                print(f"Error in filter_csv_content: {e}")
                traceback.print_exc()
        else:
            print("No CSV table or original data available")  # Debug print

    def update_file_dataframe(self):
        """Update the pandas DataFrame with file information"""
        print("\n=== Updating file DataFrame ===")  # Debug print
        
        # Prepare all data at once
        data_list = []
        columns = ['Name', 'File_Path', 'Date_Modified', 'Size_(KB)']
        columns.extend([f'Field_{i+1}' for i in range(self.max_fields)])
        
        for file_path in self.csv_files:
            try:
                file_stat = os.stat(file_path)
                file_name = os.path.basename(file_path)
                
                # Get basic file info
                file_info = {
                    'Name': file_name,
                    'File_Path': file_path,
                    'Date_Modified': pd.to_datetime(datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M')),
                    'Size_(KB)': round(file_stat.st_size / 1024, 2)
                }
                
                # Add fields from filename
                name_without_ext = os.path.splitext(file_name)[0]
                fields = name_without_ext.split('_')
                
                # Add all field columns at once
                for i in range(self.max_fields):
                    field_name = f'Field_{i+1}'
                    file_info[field_name] = fields[i] if i < len(fields) else ''
                
                data_list.append(file_info)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
        
        # Create DataFrame all at once with predefined columns
        if data_list:
            self.df = pd.DataFrame(data_list, columns=columns)
            # Sort by date modified (newest first)
            self.df.sort_values(by='Date_Modified', ascending=False, inplace=True)
        else:
            # Create empty DataFrame with correct columns if no files found
            self.df = pd.DataFrame(columns=columns)

    def reconstruct_filename(self, row):
        """Reconstruct filename from columns starting with 'F_'"""
        # Find columns starting with 'F_'
        f_columns = [col for col in row.index if col.startswith('Field_')]
        
        # Sort the columns to maintain order, in case the user changed the column order in the table
        # f_columns.sort(key=lambda x: int(x.split('_')[1]))
        
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
                # Try with forward slashes
                alt_path = old_filepath.replace(os.sep, '/')
                if os.path.exists(alt_path):
                    old_filepath = alt_path
                    print(f"Found file using alternative path: {alt_path}")  # Debug print
                else:
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
                    file_path = os.path.normpath(str(displayed_df.iloc[row]['File_Path']))

                    # Load the CSV file
                    self.load_csv_file(file_path)
                self.last_clicked_row = row
        except Exception as e:
            print(f"Error in table selection: {e}")
            traceback.print_exc()

    def load_csv_file(self, file_path):
        """Load and display the selected CSV file"""
        try:
            print(f"\n=== Loading CSV file: {file_path} ===")  # Debug print
            
            # Save current filter before recreating the viewer
            saved_filter = self.csv_filter_text.get()
            print(f"Saved current filter: '{saved_filter}'")  # Debug print
            
            # Store current column widths before recreating the viewer
            saved_widths = {}
            if hasattr(self, 'csv_table'):
                saved_widths = self.csv_table.columnwidths.copy()
                print("Saved column widths")  # Debug print
            
            # Recreate the CSV viewer
            self.setup_csv_viewer()
            
            # Try multiple encodings including Windows-specific ones
            encodings = [
                'utf-8', 'latin1', 'ISO-8859-1', 
                'cp1252', 'utf-16', 'cp850', 'cp437',
                'mbcs', 'ascii', 'utf-16-le', 'utf-16-be',
                'cp1250', 'cp1251', 'cp1253', 'cp1254', 
                'cp1255', 'cp1256', 'cp1257', 'cp1258',
                'cp932', 'cp936', 'cp949', 'cp950'
            ]

            csv_df = None
            successful_encoding = None
            
            # Try each encoding
            for encoding in encodings:
                try:
                    csv_df = pd.read_csv(file_path, encoding=encoding)
                    successful_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Error with encoding {encoding}: {str(e)}")
                    continue
            
            # If standard encodings failed, try chardet
            if csv_df is None:
                try:
                    with open(file_path, 'rb') as f:
                        raw_data = f.read()
                    
                    import chardet
                    detected = chardet.detect(raw_data)
                    encoding = detected['encoding']
                    confidence = detected['confidence']
                    
                    if confidence > 0.7:  # Only use if confident
                        csv_df = pd.read_csv(file_path, encoding=encoding)
                        successful_encoding = f"{encoding} (detected with {confidence*100:.1f}% confidence)"
                except Exception as e:
                    print(f"Failed to detect encoding: {str(e)}")
            
            # Last resort - try reading with error handling
            if csv_df is None:
                try:
                    csv_df = pd.read_csv(file_path, encoding='utf-8', 
                                    engine='python', 
                                    error_bad_lines=False,
                                    warn_bad_lines=True)
                    successful_encoding = "utf-8 with error recovery"
                except Exception as e:
                    messagebox.showerror("Encoding Error", 
                        f"Could not read {file_path} with any encoding.\n"
                        f"Last error: {str(e)}\n"
                        "Please check the file encoding manually.")
                    return
            
            if csv_df is not None:
                print(f"Successfully read file with encoding: {successful_encoding}")
                
                # Store current file path
                self.current_file = file_path
                
                # Store original data and convert to object type immediately
                self.original_csv_df = csv_df.astype(object).copy()
                
                # Set the initial view to the full dataset
                self.csv_table.model.df = self.original_csv_df.copy()
                
                # Configure table options
                self.csv_table.autoResizeColumns()
                
                # Restore saved column widths or calculate new ones
                for col in self.csv_table.model.df.columns:
                    if col in saved_widths:
                        self.csv_table.columnwidths[col] = saved_widths[col]
                    else:
                        max_width = max(len(str(x)) for x in self.csv_table.model.df[col].head(20))
                        self.csv_table.columnwidths[col] = max(min(max_width * 10 + 10, 250), 30)
                
                # Update window title
                self.title(f"CSV Browser - {os.path.basename(file_path)}")
                
                # Apply the saved filter after a short delay
                if saved_filter:
                    print(f"Scheduling filter restoration: '{saved_filter}'")  # Debug print
                    self.after(100, lambda: self.apply_saved_filter(saved_filter))
                
                # Force a complete redraw
                self.csv_table.redraw()
                
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            traceback.print_exc()

    def apply_saved_filter(self, filter_text):
        """Apply a saved filter to the current CSV data"""
        try:
            print(f"\n=== Applying saved filter: '{filter_text}' ===")  # Debug print
            
            if not hasattr(self, 'csv_table') or self.original_csv_df is None:
                print("No CSV table or data available")  # Debug print
                return
                
            # Set the filter text (this will trigger filter_csv_content via the trace)
            self.csv_filter_text.set(filter_text)
            
            # Force an immediate filter operation
            self.filter_csv_content()
            
            # Ensure the table is redrawn
            self.csv_table.redraw()
            
            print("Filter applied successfully")  # Debug print
            
        except Exception as e:
            print(f"Error applying saved filter: {e}")
            traceback.print_exc()

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

    # def filter_files(self, *args):
    #     """Filter files based on the filter text"""
    #     if hasattr(self, 'table'):
    #         filter_text = self.filter_text.get().lower()
    #         if filter_text:
    #             # Apply filter across all columns
    #             mask = self.df.apply(lambda x: x.astype(str).str.contains(filter_text, case=False)).any(axis=1)
    #             filtered_df = self.df[mask].copy()  # Make a copy to ensure we have a new DataFrame
    #         else:
    #             filtered_df = self.df.copy()  # Make a copy of the original DataFrame
            
    #         # Update table
    #         self.table.model.df = filtered_df
    #         self.table.redraw()
            
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
        ttk.Button(self.toolbar, text="Refresh", command=self.refresh_file_list).pack(side="left", padx=5)

    def browse_folder(self):
        """Open a directory chooser dialog and update the file list"""
        print("\n=== Browse folder called ===")  # Debug print
        directory = filedialog.askdirectory(
            initialdir=self.current_directory
        )
        if directory:
            print(f"Selected directory: {directory}")  # Debug print
            self.current_directory = directory
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
        """Update the list of CSV files"""
        print("\n=== Updating file list ===")  # Debug print
        if self.include_subfolders.get():
            # Walk through all subdirectories
            self.csv_files = []
            for root, _, files in os.walk(self.current_directory):
                for file in files:
                    if file.lower().endswith(('.csv', '.tsv')):
                        # Use normpath to ensure consistent path separators
                        full_path = os.path.normpath(os.path.join(root, file))
                        self.csv_files.append(full_path)
            print(f"Found {len(self.csv_files)} CSV/TSV files in all subfolders")  # Debug print
        else:
            # Only get files in current directory
            files = os.listdir(self.current_directory)
            # Use normpath for consistent path separators
            self.csv_files = [os.path.normpath(os.path.join(self.current_directory, f)) 
                            for f in files if f.lower().endswith(('.csv', '.tsv'))]
            print(f"Found {len(self.csv_files)} CSV/TSV files in current directory")  # Debug print

    def get_max_fields(self):
        """Get the maximum number of underscore-separated fields in filenames"""
        max_fields = 0
        for file_path in self.csv_files:
            # Get just the filename without path
            file_name = os.path.basename(file_path)
            # Remove extension and split by underscore
            name_without_ext = os.path.splitext(file_name)[0]
            fields = name_without_ext.split('_')
            max_fields = max(max_fields, len(fields))
        print(f"Max fields found: {max_fields}")  # Debug print
        return max_fields

    def open_in_excel(self):
        """Open the selected CSV file in Excel"""
        selected_rows = self.table.multiplerowlist
        if not selected_rows:
            messagebox.showinfo("Info", "Please select a file to open in Excel")
            return

        try:
            for row in selected_rows:
                if row < len(self.df):
                    file_path = self.df.iloc[row]['File_Path']
                    os.startfile(file_path)  # This will open the file with its default application (Excel for CSV)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file in Excel:\n{str(e)}")

    def open_in_spotfire(self):
        """Open the selected CSV file in Spotfire"""
        selected_rows = self.table.multiplerowlist
        if not selected_rows:
            messagebox.showinfo("Info", "Please select a file to open in Spotfire")
            return

        try:
            spotfire_path = r"C:\Users\JueShi\AppData\Local\Spotfire Cloud\14.4.0\Spotfire.Dxp.exe"
            for row in selected_rows:
                if row < len(self.df):
                    file_path = self.df.iloc[row]['File_Path']
                    # Launch Spotfire with the selected file
                    subprocess.Popen([spotfire_path, file_path])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file in Spotfire:\n{str(e)}")


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
        if self.current_file is None or not hasattr(self, 'csv_table'):
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
                        initialdir=os.path.dirname(self.current_file),
                        initialfile=f"corr-ana-{os.path.basename(self.current_file)}"
                    )
                    
                    if base_path:
                        # Remove .csv extension if present
                        base_path = base_path.rsplit('.csv', 1)[0]
                        
                        # Save correlation results
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
            plots_dir = f"{base_path}_plots"
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
                plt.savefig(os.path.join(plots_dir, f"{corr_col}_vs_{target_column}.png"), 
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
            plt.savefig(os.path.join(plots_dir, "combined_correlation_plots.png"), 
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
        print(f"current directory: {self.current_directory}")  # Debug print
        
        # Update file list
        self.update_file_list()
        
        # Update file browser
        self.setup_file_browser()
        self.setup_csv_viewer()        

    def toggle_layout(self):
        """Toggle between horizontal and vertical layouts"""
        # Toggle orientation
        self.is_horizontal = not self.is_horizontal
        
        # Update button text
        self.toggle_btn.configure(
            text="Switch to Vertical Layout" if self.is_horizontal else "Switch to Horizontal Layout"
        )

        # Remove old paned window and its children
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

        # Ask for destination directory
        dest_dir = filedialog.askdirectory(
            title="Select Destination Folder",
            initialdir=os.path.dirname(self.current_directory)
        )
        
        if not dest_dir:
            return

        try:
            copied_files = []
            for row in selected_rows:
                if row < len(self.df):
                    filename = self.df.iloc[row]['Name']
                    src_path = os.path.join(self.current_directory, filename)
                    dst_path = os.path.join(dest_dir, filename)
                    
                    # Check if file already exists in destination
                    if os.path.exists(dst_path):
                        if not messagebox.askyesno("File Exists", 
                            f"File {filename} already exists in destination.\nDo you want to overwrite it?"):
                            continue
                    
                    # Copy the file
                    shutil.copy2(src_path, dst_path)  # copy2 preserves metadata
                    copied_files.append(filename)
            
            if copied_files:
                messagebox.showinfo("Success", f"Copied {len(copied_files)} files to:\n{dest_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy files:\n{str(e)}")

    def move_selected_files(self):
        """Move selected files to another folder"""
        # Get selected rows from the table's multiplerowlist
        selected_rows = self.table.multiplerowlist
        if not selected_rows:
            messagebox.showinfo("Info", "Please select files to move")
            return

        # Ask for destination directory
        dest_dir = filedialog.askdirectory(
            title="Select Destination Folder",
            initialdir=os.path.dirname(self.current_directory)
        )
        
        if not dest_dir:
            return

        try:
            moved_files = []
            for row in selected_rows:
                if row < len(self.df):
                    filename = self.df.iloc[row]['Name']
                    src_path = os.path.join(self.current_directory, filename)
                    dst_path = os.path.join(dest_dir, filename)
                    
                    # Check if file already exists in destination
                    if os.path.exists(dst_path):
                        if not messagebox.askyesno("File Exists", 
                            f"File {filename} already exists in destination.\nDo you want to overwrite it?"):
                            continue
                    
                    # Move the file
                    shutil.move(src_path, dst_path)
                    moved_files.append(filename)
                    
                    # Clear CSV viewer if it was the moved file
                    if self.current_file == src_path:
                        self.current_file = None
                        self.csv_table.model.df = pd.DataFrame()
                        self.csv_table.redraw()

            # Update the DataFrame and table
            if moved_files:
                self.df = self.df[~self.df['Name'].isin(moved_files)]
                self.table.model.df = self.df
                self.table.redraw()
                
                # Update file list
                self.csv_files = [f for f in self.csv_files if f not in moved_files]
                
                messagebox.showinfo("Success", f"Moved {len(moved_files)} files to:\n{dest_dir}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to move files:\n{str(e)}")

    def delete_selected_files(self):
        """Delete selected files"""
        # Get selected rows from the table's multiplerowlist
        selected_rows = self.table.multiplerowlist
        if not selected_rows:
            messagebox.showinfo("Info", "Please select files to delete")
            return

        # Show confirmation dialog with count of files
        if not messagebox.askyesno("Confirm Delete", 
                                 f"Are you sure you want to delete {len(selected_rows)} selected files?",
                                 icon='warning'):
            return

        deleted_files = []
        try:
            for row in selected_rows:
                if row < len(self.df):
                    filename = self.df.iloc[row]['Name']
                    filepath = os.path.join(self.current_directory, filename)

                    try:
                        # Delete the file
                        os.remove(filepath)
                        deleted_files.append(filename)
                        
                        # Clear CSV viewer if it was the deleted file
                        if self.current_file == filepath:
                            self.current_file = None
                            self.csv_table.model.df = pd.DataFrame()
                            self.csv_table.redraw()
                                
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to delete file {filename}:\n{str(e)}")

            # Update the DataFrame and table
            if deleted_files:
                self.df = self.df[~self.df['Name'].isin(deleted_files)]
                self.table.model.df = self.df
                self.table.redraw()
                
                # Update file list
                self.csv_files = [f for f in self.csv_files if f not in deleted_files]
                
                messagebox.showinfo("Success", f"Deleted {len(deleted_files)} files")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error during deletion:\n{str(e)}")

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


    def load_subfolders(self):
        """Load all CSV files from current directory and all subdirectories"""
        try:
            print("\n=== Loading files from subfolders ===")  # Debug print
            directory = filedialog.askdirectory(
                initialdir=self.current_directory
            )
            if directory:
                print(f"Selected directory: {directory}")  # Debug print
                self.current_directory = directory
                self.include_subfolders.set(True)
                
                # Update file list
                self.update_file_list()
                
                # Update max fields
                old_max = self.max_fields
                self.max_fields = self.get_max_fields()
                print(f"Max fields changed from {old_max} to {self.max_fields}")  # Debug print
                
                # Update file browser
                self.setup_file_browser()
                self.setup_csv_viewer()
                print("Completed loading files from subfolders")  # Debug print
        except Exception as e:
            print(f"Error loading subfolders: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    app = CSVBrowser()
    app.mainloop()
