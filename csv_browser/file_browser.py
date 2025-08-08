import os
import pandas as pd
import tkinter as tk
from tkinter import ttk
from pandastable import Table
import traceback

class FileBrowser:
    def __init__(self, app, parent_frame):
        self.app = app
        self.parent_frame = parent_frame
        self.table = None
        self.df = pd.DataFrame()
        self.last_clicked_row = None
        self.pt_frame = None

        self.setup_file_browser()

    def setup_file_browser(self):
        """Setup the file browser panel with pandastable"""
        print("\n=== Setting up file browser ===")

        if self.pt_frame:
            self.pt_frame.destroy()
        self.pt_frame = ttk.Frame(self.parent_frame)
        self.pt_frame.pack(fill="both", expand=True, padx=5, pady=5)

        filter_frame = ttk.Frame(self.pt_frame)
        filter_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(filter_frame, text="Filter Files:").pack(side="left", padx=(0,5))
        filter_entry = ttk.Entry(filter_frame, textvariable=self.app.filter_text)
        filter_entry.pack(side="left", fill="x", expand=True)

        ttk.Button(filter_frame, text="Save Filter", command=self.app.save_file_filter).pack(side="left", padx=5)
        ttk.Button(filter_frame, text="Load Filter", command=self.app.show_saved_file_filters).pack(side="left", padx=5)

        self.create_filter_context_menu(filter_entry)

        table_frame = ttk.Frame(self.pt_frame)
        table_frame.pack(fill="both", expand=True)

        self.app.update_file_dataframe()

        self.table = Table(table_frame, dataframe=self.app.df, showtoolbar=True, showstatusbar=True)
        self.table.editable = True
        self.table.bind('<Key>', self.on_key_press)
        self.table.bind('<Return>', self.on_return_press)
        self.table.bind('<FocusOut>', self.on_focus_out)
        self.table.show()

        self.table.autoResizeColumns()
        self.table.columnwidths['Name'] = 50
        self.table.columnwidths['File_Path'] = 90

        if not self.app.df.empty:
            for col in self.app.df.columns:
                if col not in ['Name', 'File_Path']:
                    if len(self.app.df[col]) > 0:
                        max_width = max(len(str(x)) for x in self.app.df[col].head(20))
                        self.table.columnwidths[col] = max(min(max_width * 10, 200), 50)
                    else:
                        self.table.columnwidths[col] = 100

        self.table.redraw()

        self.table.bind('<ButtonRelease-1>', self.on_table_select)
        self.table.bind('<Up>', self.on_key_press)
        self.table.bind('<Down>', self.on_key_press)

    def create_filter_context_menu(self, filter_entry):
        filter_menu = tk.Menu(filter_entry, tearoff=0)
        filter_menu.add_command(label="Filtering Instructions", state='disabled', font=('Arial', 10, 'bold'))
        filter_menu.add_separator()
        filter_menu.add_command(label="Basic Search: Enter any text to match", state='disabled')
        filter_menu.add_command(label="Multiple Terms: Use '+' or '&' to combine", state='disabled')
        filter_menu.add_command(label="Exclude Terms: Use '!' prefix", state='disabled')
        filter_menu.add_separator()
        filter_menu.add_command(label="Examples:", state='disabled', font=('Arial', 10, 'bold'))
        filter_menu.add_command(label="'csv': Show files with 'csv'", state='disabled')
        filter_menu.add_command(label="'2024 + report': Files with both terms", state='disabled')
        filter_menu.add_command(label="'!temp': Exclude files with 'temp'", state='disabled')
        filter_menu.add_command(label="'csv + !old': CSV files, not old", state='disabled')

        def show_filter_menu(event):
            filter_menu.tk_popup(event.x_root, event.y_root)

        filter_entry.bind('<Button-3>', show_filter_menu)

    def on_key_press(self, event):
        """Handle key press events"""
        try:
            if event.keysym in ('Up', 'Down'):
                current_row = self.table.getSelectedRow()
                num_rows = len(self.table.model.df)

                if event.keysym == 'Up' and current_row > 0:
                    new_row = current_row - 1
                elif event.keysym == 'Down' and current_row < num_rows - 1:
                    new_row = current_row + 1
                else:
                    return

                self.table.setSelectedRow(new_row)
                self.table.redraw()

                displayed_df = self.table.model.df
                if new_row < 0 or new_row >= len(displayed_df):
                    return
                file_path = str(displayed_df.iloc[new_row]['File_Path'])
                self.app.csv_viewer.load_csv_file(file_path)
            elif event.char and event.char.isprintable():
                row = self.table.getSelectedRow()
                col = self.table.getSelectedColumn()
                if row is not None and col is not None:
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
        pass

    def on_table_select(self, event):
        """Handle table row selection"""
        try:
            row = self.table.get_row_clicked(event)
            if self.last_clicked_row == row:
                col = self.table.get_col_clicked(event)
                if col is not None:
                    self.table.drawCellEntry(row, col)
            elif row is not None:
                displayed_df = self.table.model.df
                if 0 <= row < len(displayed_df):
                    file_path = str(displayed_df.iloc[row]['File_Path'])
                    self.app.csv_viewer.load_csv_file(file_path)
                self.last_clicked_row = row
        except Exception as e:
            print(f"Error in table selection: {e}")
            traceback.print_exc()

    def filter_files(self, *args):
        """Filter files based on the filter text"""
        if hasattr(self, 'table'):
            try:
                filter_text = self.app.filter_text.get().lower().strip('"\'')
                print(f"\n=== Filtering files with: '{filter_text}' ===")

                if filter_text:
                    filter_terms = [term.strip() for term in filter_text.replace('&', '+').split('+')]
                    mask = pd.Series([True] * len(self.app.df), index=self.app.df.index)

                    for term in filter_terms:
                        if term:
                            if term.startswith('!'):
                                exclude_term = term[1:].strip()
                                if exclude_term:
                                    term_mask = ~self.app.df['Name'].str.contains(exclude_term, case=False, na=False)
                            else:
                                term_mask = self.app.df['Name'].str.contains(term, case=False, na=False)
                            mask = mask & term_mask

                    filtered_df = self.app.df[mask].copy()
                else:
                    filtered_df = self.app.df.copy()

                self.table.model.df = filtered_df

                if not filtered_df.empty:
                    for col in filtered_df.columns:
                        if col in self.table.columnwidths:
                            width = max(
                                len(str(filtered_df[col].max())),
                                len(str(filtered_df[col].min())),
                                len(col),
                                self.table.columnwidths[col]
                            )
                            self.table.columnwidths[col] = width

                self.table.redraw()

            except Exception as e:
                print(f"Error in filter_files: {str(e)}")
                traceback.print_exc()
                self.app.update_file_dataframe()
