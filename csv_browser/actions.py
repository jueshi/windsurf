import os
import shutil
import subprocess
import pandas as pd
from tkinter import messagebox, filedialog, simpledialog
import pyperclip
import matplotlib.pyplot as plt
from math import ceil
import numpy as np

from .utils import normalize_long_path

def open_in_excel(app):
    """Open the selected CSV file in Excel"""
    selected_rows = app.file_browser.table.multiplerowlist
    if not selected_rows:
        messagebox.showinfo("Info", "Please select a file to open in Excel")
        return

    try:
        for row in selected_rows:
            if row < len(app.df):
                file_path = app.df.iloc[row]['File_Path']
                os.startfile(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open file in Excel:\n{str(e)}")

def open_in_spotfire(app):
    """Open the first selected CSV file in Spotfire and copy remaining paths to clipboard"""
    selected_rows = app.file_browser.table.multiplerowlist
    if not selected_rows:
        messagebox.showinfo("Info", "Please select file(s) to open in Spotfire")
        return

    temp_dir = os.path.join(os.environ.get('TEMP', os.path.expanduser('~')), 'CSVBrowser_temp')
    os.makedirs(temp_dir, exist_ok=True)

    for f in os.listdir(temp_dir):
        try:
            os.remove(os.path.join(temp_dir, f))
        except:
            pass

    try:
        spotfire_path = r"C:\Users\JueShi\AppData\Local\Spotfire Cloud\14.4.0\Spotfire.Dxp.exe"
        if not os.path.exists(spotfire_path):
             spotfire_path = filedialog.askopenfilename(title="Select Spotfire Executable", filetypes=[("Executable files", "*.exe")])
             if not spotfire_path:
                 return

        file_paths = []
        for row in selected_rows:
            if row < len(app.df):
                file_paths.append(app.df.iloc[row]['File_Path'])

        if file_paths:
            if len(file_paths) == 1:
                subprocess.Popen([spotfire_path, file_paths[0]])
            else:
                subprocess.Popen([spotfire_path, file_paths[0]])
                remaining_files = [f'"{path}"' for path in file_paths[1:]]
                pyperclip.copy(" ".join(remaining_files))
                messagebox.showinfo("Multiple Files",
                    f"Opened first file in Spotfire.\n\n"
                    f"Remaining {len(remaining_files)} file paths have been copied to clipboard with quotes.")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to open file(s) in Spotfire:\n{str(e)}")

def reveal_in_explorer(app):
    """Reveal the selected file(s) in File Explorer"""
    selected_rows = app.file_browser.table.multiplerowlist
    if not selected_rows:
        messagebox.showinfo("Info", "Please select a file to reveal in Explorer")
        return

    try:
        for row in selected_rows:
            if row < len(app.df):
                file_path = app.df.iloc[row]['File_Path']
                subprocess.Popen(f'explorer /select,"{os.path.normpath(file_path)}"')
    except Exception as e:
        messagebox.showerror("Error", f"Failed to reveal file in Explorer:\n{str(e)}")

def advanced_file_read(file_path):
    """Advanced file reading method with comprehensive diagnostics and long path handling"""
    normalized_path = normalize_long_path(file_path)
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252', 'utf-16']
    for encoding in encodings:
        try:
            return pd.read_csv(normalized_path, encoding=encoding)
        except Exception:
            continue
    try:
        return pd.read_csv(normalized_path, encoding='utf-8', engine='python', on_bad_lines='skip')
    except Exception as e:
        print(f"Advanced file read failed: {e}")
        return None

def copy_selected_files(app):
    """Copy selected files to another folder"""
    selected_rows = app.file_browser.table.multiplerowlist
    if not selected_rows:
        messagebox.showinfo("Info", "Please select files to copy")
        return

    dest_dir = filedialog.askdirectory(title="Select Destination Folder")
    if not dest_dir:
        return

    for row in selected_rows:
        if row < len(app.df):
            src_path = app.df.iloc[row]['File_Path']
            try:
                shutil.copy2(src_path, dest_dir)
            except Exception as e:
                messagebox.showerror("Copy Error", f"Failed to copy {os.path.basename(src_path)}:\n{e}")

    messagebox.showinfo("Success", f"Copied {len(selected_rows)} files to:\n{dest_dir}")

def move_selected_files(app):
    """Move selected files to a new directory"""
    selected_rows = app.file_browser.table.multiplerowlist
    if not selected_rows:
        messagebox.showinfo("Info", "Please select files to move")
        return

    destination_dir = filedialog.askdirectory(title="Select Destination Folder")
    if not destination_dir:
        return

    for row in selected_rows:
        if row < len(app.df):
            file_path = app.df.iloc[row]['File_Path']
            try:
                shutil.move(file_path, destination_dir)
            except Exception as e:
                messagebox.showerror("Move Error", f"Failed to move {os.path.basename(file_path)}:\n{e}")

    app.refresh_file_list()
    messagebox.showinfo("Success", f"Moved {len(selected_rows)} files.")

def delete_selected_files(app):
    """Delete selected files"""
    selected_rows = app.file_browser.table.multiplerowlist
    if not selected_rows:
        messagebox.showinfo("Info", "Please select a file to delete")
        return

    if not messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {len(selected_rows)} selected files?"):
        return

    for row in selected_rows:
        if row < len(app.df):
            file_path = app.df.iloc[row]['File_Path']
            try:
                os.remove(file_path)
                if app.csv_viewer.original_csv_df is not None and app.current_csv_file == file_path:
                    app.csv_viewer.original_csv_df = None
                    app.csv_viewer.csv_table.model.df = pd.DataFrame()
                    app.csv_viewer.csv_table.redraw()
            except Exception as e:
                messagebox.showerror("Delete Error", f"Failed to delete {os.path.basename(file_path)}:\n{e}")

    app.refresh_file_list()
    messagebox.showinfo("Success", f"Deleted {len(selected_rows)} files.")

def reconstruct_filename(row):
    """Reconstruct filename from columns starting with 'Field_'"""
    f_columns = [col for col in row.index if col.startswith('Field_')]
    filename_parts = [str(row[col]) for col in f_columns if pd.notna(row[col]) and str(row[col]).strip()]
    return '_'.join(filename_parts).replace('\n', '') + '.csv'

def rename_csv_file(old_filepath, new_filename):
    """Rename CSV file on disk"""
    directory = os.path.dirname(old_filepath)
    new_filepath = os.path.join(directory, new_filename)
    try:
        os.rename(old_filepath, new_filepath)
        return new_filepath
    except Exception as e:
        messagebox.showerror("Rename Error", f"Failed to rename {os.path.basename(old_filepath)}:\n{e}")
        return old_filepath

def rename_all_files(app):
    """Rename all files where constructed name differs from current name"""
    renamed_count = 0
    for idx, row in app.df.iterrows():
        current_name = row['Name']
        new_name = reconstruct_filename(row)

        if new_name != current_name:
            old_path = row['File_Path']
            new_path = rename_csv_file(old_path, new_name)

            if new_path != old_path:
                app.df.at[idx, 'Name'] = new_name
                app.df.at[idx, 'File_Path'] = new_path
                renamed_count += 1

    if renamed_count > 0:
        app.file_browser.table.model.df = app.df
        app.file_browser.table.redraw()
        messagebox.showinfo("Files Renamed", f"Renamed {renamed_count} files.")
    else:
        messagebox.showinfo("No Changes", "All filenames already match their field values.")

def save_correlation_analysis(app):
    """Save correlation analysis for the selected CSV file"""
    if app.csv_viewer.original_csv_df is None:
        messagebox.showinfo("Info", "Please load a CSV file first")
        return

    df = app.csv_viewer.original_csv_df
    numeric_columns = get_numeric_varying_columns(df)

    if not numeric_columns:
        messagebox.showerror("Error", "No numeric columns with variation found.")
        return

    target_col = simpledialog.askstring("Target Column", "Enter the target column for correlation analysis:", initialvalue=numeric_columns[0])
    if not target_col or target_col not in numeric_columns:
        messagebox.showerror("Error", "Invalid target column.")
        return

    top_n = simpledialog.askinteger("Top N", "Enter the number of top correlated columns to analyze:", initialvalue=10)
    if not top_n or top_n <= 0:
        return

    try:
        corr_df, data_df = get_top_correlated_columns(df, target_col, top_n)

        base_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not base_path:
            return

        corr_path = f"{os.path.splitext(base_path)[0]}_corr.csv"
        data_path = f"{os.path.splitext(base_path)[0]}_data.csv"

        corr_df.to_csv(corr_path, index=False)
        data_df.to_csv(data_path, index=False)

        plots_dir = create_correlation_plots(data_df, target_col, corr_df, base_path)

        messagebox.showinfo("Success", f"Analysis saved to:\nCorrelations: {corr_path}\nData: {data_path}\nPlots: {plots_dir}")

    except Exception as e:
        messagebox.showerror("Error", f"Correlation analysis failed:\n{e}")

def get_numeric_varying_columns(data):
    """Get list of columns that are numeric and have more than one unique value."""
    numeric_columns = []
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]) and data[col].nunique() > 1:
            numeric_columns.append(col)
    return numeric_columns

def get_top_correlated_columns(data, target_column, top_n=10):
    """Get the top N most correlated columns to a specified target column."""
    numeric_data = data.select_dtypes(include=np.number)
    if target_column not in numeric_data.columns:
        raise ValueError(f"Target column '{target_column}' is not numeric.")

    cor_matrix = numeric_data.corr()
    cor_target = abs(cor_matrix[target_column])
    top_correlated = cor_target.sort_values(ascending=False).head(top_n + 1)

    corr_df = top_correlated.reset_index()
    corr_df.columns = ['Column', 'Correlation']

    sorted_cols = corr_df['Column'].tolist()
    remaining_cols = [col for col in data.columns if col not in sorted_cols]
    data_df = data[sorted_cols + remaining_cols]

    return corr_df, data_df

def create_correlation_plots(data_df, target_column, corr_df, base_path):
    """Create scatter plots for each correlated column vs target column."""
    plots_dir = f"{os.path.splitext(base_path)[0]}_plots"
    os.makedirs(plots_dir, exist_ok=True)

    for _, row in corr_df.iterrows():
        corr_col = row['Column']
        if corr_col == target_column:
            continue

        plt.figure(figsize=(10, 6))
        plt.scatter(data_df[target_column], data_df[corr_col], alpha=0.5)
        plt.title(f'{target_column} vs {corr_col} (Corr: {row["Correlation"]:.2f})')
        plt.xlabel(target_column)
        plt.ylabel(corr_col)
        plt.savefig(os.path.join(plots_dir, f"{corr_col}_vs_{target_column}.png"))
        plt.close()

    return plots_dir
