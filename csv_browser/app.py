import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import warnings
import traceback

from .file_browser import FileBrowser
from .csv_viewer import CSVViewer
from . import settings
from . import actions
from . import utils

warnings.filterwarnings('ignore', category=FutureWarning)

class CSVBrowser(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("CSV Browser")
        self.state('zoomed')

        # Initialize variables
        self.current_file = None
        self.is_horizontal = False
        self.filter_text = tk.StringVar()
        self.filter_text.trace_add("write", self.filter_files)
        self.include_subfolders = tk.BooleanVar(value=False)
        self.csv_filter_text = tk.StringVar()
        self.csv_filter_text.trace_add("write", self.filter_csv_content)
        self.column_search_var = tk.StringVar()
        self.column_search_var.trace_add("write", self.search_columns)
        self.column_filter_var = tk.StringVar()
        self.column_filter_var.trace_add("write", self.filter_columns)

        self.recent_directories = []
        self.saved_filters = []
        self.saved_file_filters = []
        self.max_recent_directories = 5

        settings.load_settings(self)

        self.current_directory = self.recent_directories[0] if self.recent_directories else os.getcwd()
        self.add_to_recent_directories(self.current_directory)

        self.csv_files = []
        self.df = pd.DataFrame()
        self.max_fields = 0

        self.update_file_list()
        self.max_fields = self.get_max_fields()

        self.main_container = ttk.Frame(self)
        self.main_container.pack(fill="both", expand=True)

        self.toolbar = ttk.Frame(self.main_container)
        self.toolbar.pack(fill="x", padx=5, pady=5)
        self.setup_toolbar()

        self.paned = ttk.PanedWindow(self.main_container, orient=tk.VERTICAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        self.file_frame = ttk.Frame(self.paned)
        self.paned.add(self.file_frame, weight=1)

        self.csv_container = ttk.Frame(self.paned)
        self.paned.add(self.csv_container, weight=2)

        self.file_browser = FileBrowser(self, self.file_frame)
        self.csv_viewer = CSVViewer(self, self.csv_container)

        self.create_menu()
        self.bind("<Control-f>", self.focus_column_search)
        self.protocol("WM_DELETE_WINDOW", self.quit)

    def setup_toolbar(self):
        """Setup the toolbar with necessary controls"""
        ttk.Button(self.toolbar, text="Browse Folder", command=self.browse_folder).pack(side="left", padx=5)
        ttk.Button(self.toolbar, text="Load Subfolders", command=self.load_subfolders).pack(side="left", padx=5)
        ttk.Button(self.toolbar, text="Move Files", command=lambda: actions.move_selected_files(self)).pack(side="left", padx=5)
        ttk.Button(self.toolbar, text="Copy Files", command=lambda: actions.copy_selected_files(self)).pack(side="left", padx=5)
        ttk.Button(self.toolbar, text="Delete Files", command=lambda: actions.delete_selected_files(self)).pack(side="left", padx=5)
        ttk.Button(self.toolbar, text="Rename All Files", command=lambda: actions.rename_all_files(self)).pack(side="left", padx=5)
        ttk.Button(self.toolbar, text="Reveal in Explorer", command=lambda: actions.reveal_in_explorer(self)).pack(side="left", padx=5)
        ttk.Button(self.toolbar, text="Open in Excel", command=lambda: actions.open_in_excel(self)).pack(side="left", padx=5)
        ttk.Button(self.toolbar, text="Open in Spotfire", command=lambda: actions.open_in_spotfire(self)).pack(side="left", padx=5)
        ttk.Button(self.toolbar, text="Correlation Analysis", command=lambda: actions.save_correlation_analysis(self)).pack(side="left", padx=5)
        ttk.Button(self.toolbar, text="Refresh", command=self.refresh_file_list).pack(side="left", padx=5)

    def create_menu(self):
        """Create the application menu"""
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Directory", command=self.browse_folder)
        self.recent_dirs_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent Directories", menu=self.recent_dirs_menu)
        self.update_recent_directories_menu()
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Find Column (Ctrl+F)", command=self.focus_column_search)
        menubar.add_cascade(label="View", menu=view_menu)

        self.config(menu=menubar)

    def browse_folder(self):
        """Open a directory chooser dialog and update the file list"""
        directory = filedialog.askdirectory(initialdir=self.current_directory)
        if directory:
            self.current_directory = directory
            self.include_subfolders.set(False)
            self.refresh_file_list()
            self.add_to_recent_directories(directory)

    def load_subfolders(self):
        """Load all CSV and TSV files from current directory and all subdirectories"""
        directory = filedialog.askdirectory(initialdir=self.current_directory)
        if directory:
            self.current_directory = directory
            self.include_subfolders.set(True)
            self.refresh_file_list()

    def refresh_file_list(self):
        """Refresh the file list and reload the currently selected file if any"""
        self.update_file_list()
        self.max_fields = self.get_max_fields()
        self.file_browser.setup_file_browser()
        self.csv_viewer.setup_csv_viewer()

    def update_file_list(self):
        """Update the list of CSV and TSV files"""
        normalized_directory = utils.normalize_long_path(self.current_directory)
        self.csv_files = []
        if self.include_subfolders.get():
            for root, _, files in os.walk(normalized_directory):
                for file in files:
                    if file.lower().endswith(('.csv', '.tsv')):
                        self.csv_files.append(os.path.join(root, file))
        else:
            try:
                files = os.listdir(normalized_directory)
                self.csv_files = [os.path.join(normalized_directory, f) for f in files if f.lower().endswith(('.csv', '.tsv')) and os.path.isfile(os.path.join(normalized_directory, f))]
            except Exception as e:
                print(f"Error listing directory: {e}")

    def get_max_fields(self):
        """Get the maximum number of underscore-separated fields in filenames"""
        max_fields = 0
        for file_path in self.csv_files:
            file_name = os.path.basename(file_path)
            name_without_ext = os.path.splitext(file_name)[0]
            fields = name_without_ext.split('_')
            max_fields = max(max_fields, len(fields))
        return max(max_fields, 25)

    def update_file_dataframe(self):
        """Update the pandas DataFrame with file information"""
        if not self.csv_files:
            self.df = pd.DataFrame(columns=['Name', 'File_Path', 'Size', 'Modified', 'Type'] + [f'Field_{i+1}' for i in range(25)])
            return

        file_info = []
        for file_path in self.csv_files:
            try:
                file_stat = os.stat(file_path)
                file_name = os.path.basename(file_path)
                file_info.append({
                    'Name': file_name,
                    'File_Path': file_path,
                    'Size': utils.format_size(file_stat.st_size),
                    'Modified': utils.format_date(file_stat.st_mtime),
                    'Type': os.path.splitext(file_name)[1][1:].upper()
                })
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

        self.df = pd.DataFrame(file_info)

        for i in range(self.max_fields):
            field_name = f'Field_{i+1}'
            self.df[field_name] = self.df['Name'].apply(lambda x: self._extract_field(x, i))

        if 'Modified' in self.df.columns:
            self.df['Modified_dt'] = pd.to_datetime(self.df['Modified'])
            self.df = self.df.sort_values(by='Modified_dt', ascending=False).drop(columns=['Modified_dt'])

        self.df = self.df.reset_index(drop=True)

    def _extract_field(self, filename, field_index):
        """Extract a specific field from filename split by underscore"""
        name_without_ext = os.path.splitext(filename)[0]
        fields = name_without_ext.split('_')
        return fields[field_index] if field_index < len(fields) else ''

    def add_to_recent_directories(self, directory):
        """Add a directory to the recent directories list"""
        if directory in self.recent_directories:
            self.recent_directories.remove(directory)
        self.recent_directories.insert(0, directory)
        if len(self.recent_directories) > self.max_recent_directories:
            self.recent_directories = self.recent_directories[:self.max_recent_directories]
        self.update_recent_directories_menu()
        settings.save_settings(self)

    def update_recent_directories_menu(self):
        """Update the recent directories menu"""
        self.recent_dirs_menu.delete(0, tk.END)
        for directory in self.recent_directories:
            self.recent_dirs_menu.add_command(label=directory, command=lambda d=directory: self.open_recent_directory(d))

    def open_recent_directory(self, directory):
        """Open a recently used directory"""
        if os.path.isdir(directory):
            self.current_directory = directory
            self.include_subfolders.set(False)
            self.refresh_file_list()
            self.add_to_recent_directories(directory)
        else:
            messagebox.showerror("Error", "Directory not found.")
            self.recent_directories.remove(directory)
            self.update_recent_directories_menu()

    def filter_files(self, *args):
        self.file_browser.filter_files(*args)

    def filter_csv_content(self, *args):
        self.csv_viewer.filter_csv_content(*args)

    def search_columns(self, *args):
        self.csv_viewer.search_columns(*args)

    def filter_columns(self, *args):
        self.csv_viewer.filter_columns(*args)

    def focus_column_search(self, event=None):
        self.csv_viewer.column_search_entry.focus_set()

    def save_file_filter(self):
        """Save the current file filter configuration"""
        file_filter = self.filter_text.get().strip()
        if not file_filter:
            messagebox.showinfo("Empty Filter", "No file filter is currently active")
            return
        filter_name = simpledialog.askstring("Save File Filter", "Enter a name for this file filter:")
        if not filter_name:
            return
        filter_config = {"name": filter_name, "filter": file_filter}
        for i, saved_filter in enumerate(self.saved_file_filters):
            if saved_filter.get("name") == filter_name:
                if messagebox.askyesno("Filter Exists", f"A file filter named '{filter_name}' already exists. Overwrite?"):
                    self.saved_file_filters[i] = filter_config
                    messagebox.showinfo("Filter Saved", f"File filter '{filter_name}' updated")
                return
        self.saved_file_filters.append(filter_config)
        settings.save_settings(self)
        messagebox.showinfo("Filter Saved", f"File filter '{filter_name}' saved")

    def show_saved_file_filters(self):
        """Show the saved file filters and allow the user to select one"""
        if not self.saved_file_filters:
            messagebox.showinfo("No Saved Filters", "You don't have any saved file filters yet")
            return

        dialog = tk.Toplevel(self)
        dialog.title("Saved File Filters")
        listbox = tk.Listbox(dialog, width=50, height=10)
        listbox.pack(padx=10, pady=10)

        for item in self.saved_file_filters:
            listbox.insert(tk.END, f"{item['name']}: {item['filter']}")

        def on_apply():
            selected_index = listbox.curselection()
            if selected_index:
                self.apply_saved_file_filter(self.saved_file_filters[selected_index[0]])
                dialog.destroy()

        ttk.Button(dialog, text="Apply", command=on_apply).pack(pady=5)

    def apply_saved_file_filter(self, filter_config):
        """Apply a saved file filter configuration"""
        self.filter_text.set(filter_config.get("filter", ""))

    def save_current_filter(self):
        """Save the current filter configuration with both row and column filters"""
        row_filter = self.csv_filter_text.get().strip()
        column_filter = self.column_filter_var.get().strip()
        if not row_filter and not column_filter:
            messagebox.showinfo("Empty Filter", "No filter is currently active")
            return
        filter_name = simpledialog.askstring("Save Filter", "Enter a name for this filter:")
        if not filter_name:
            return
        filter_config = {"name": filter_name, "row_filter": row_filter, "column_filter": column_filter}
        for i, saved_filter in enumerate(self.saved_filters):
            if saved_filter.get("name") == filter_name:
                if messagebox.askyesno("Filter Exists", f"A filter named '{filter_name}' already exists. Overwrite?"):
                    self.saved_filters[i] = filter_config
                    messagebox.showinfo("Filter Saved", f"Filter '{filter_name}' updated")
                return
        self.saved_filters.append(filter_config)
        settings.save_settings(self)
        messagebox.showinfo("Filter Saved", f"Filter '{filter_name}' saved")

    def show_saved_filters(self):
        """Show the saved filters and allow the user to select one"""
        if not self.saved_filters:
            messagebox.showinfo("No Saved Filters", "You don't have any saved filters yet")
            return

        dialog = tk.Toplevel(self)
        dialog.title("Saved Filters")
        listbox = tk.Listbox(dialog, width=50, height=10)
        listbox.pack(padx=10, pady=10)

        for item in self.saved_filters:
            listbox.insert(tk.END, f"{item['name']}: Row='{item['row_filter']}', Col='{item['column_filter']}'")

        def on_apply():
            selected_index = listbox.curselection()
            if selected_index:
                self.apply_saved_filter(self.saved_filters[selected_index[0]])
                dialog.destroy()

        ttk.Button(dialog, text="Apply", command=on_apply).pack(pady=5)

    def apply_saved_filter(self, filter_config):
        """Apply a saved filter configuration"""
        self.csv_filter_text.set(filter_config.get("row_filter", ""))
        self.column_filter_var.set(filter_config.get("column_filter", ""))

    def reset_all_filters(self):
        """Reset both row and column filters"""
        self.csv_filter_text.set("")
        self.column_filter_var.set("")
        self.csv_viewer.reset_column_filter()

    def quit(self):
        """Save settings and quit the application"""
        settings.save_settings(self)
        self.destroy()
