import os
import pandas as pd
from . import utils
from . import actions

def get_files_in_directory(directory, include_subfolders=False):
    normalized_directory = utils.normalize_long_path(directory)
    csv_files = []
    if include_subfolders:
        for root, _, files in os.walk(normalized_directory):
            for file in files:
                if file.lower().endswith(('.csv', '.tsv')):
                    csv_files.append(os.path.join(root, file))
    else:
        try:
            files = os.listdir(normalized_directory)
            csv_files = [os.path.join(normalized_directory, f) for f in files if f.lower().endswith(('.csv', '.tsv')) and os.path.isfile(os.path.join(normalized_directory, f))]
        except Exception as e:
            print(f"Error listing directory: {e}")
    return csv_files

def _extract_field(filename, field_index):
    """Extract a specific field from filename split by underscore"""
    name_without_ext = os.path.splitext(filename)[0]
    fields = name_without_ext.split('_')
    return fields[field_index] if field_index < len(fields) else ''

def get_max_fields(csv_files):
    max_fields = 0
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(file_name)[0]
        fields = name_without_ext.split('_')
        max_fields = max(max_fields, len(fields))
    return max(max_fields, 25)

def create_file_dataframe(csv_files):
    if not csv_files:
        return pd.DataFrame(columns=['Name', 'File_Path', 'Size', 'Modified', 'Type'] + [f'Field_{i+1}' for i in range(25)])

    file_info = []
    for file_path in csv_files:
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

    df = pd.DataFrame(file_info)

    max_fields = get_max_fields(csv_files)

    for i in range(max_fields):
        field_name = f'Field_{i+1}'
        df[field_name] = df['Name'].apply(lambda x: _extract_field(x, i))

    if 'Modified' in df.columns:
        df['Modified_dt'] = pd.to_datetime(df['Modified'])
        df = df.sort_values(by='Modified_dt', ascending=False).drop(columns=['Modified_dt'])

    df = df.reset_index(drop=True)
    return df

def filter_file_dataframe(df, filter_text):
    if df.empty or not filter_text:
        return df

    filter_text = filter_text.lower().strip('"\'')
    if not filter_text:
        return df

    filter_terms = [term.strip() for term in filter_text.replace('&', '+').split('+')]
    mask = pd.Series([True] * len(df), index=df.index)

    for term in filter_terms:
        if term:
            if term.startswith('!'):
                exclude_term = term[1:].strip()
                if exclude_term:
                    term_mask = ~df['Name'].str.contains(exclude_term, case=False, na=False)
            else:
                term_mask = df['Name'].str.contains(term, case=False, na=False)
            mask = mask & term_mask

    return df[mask].copy()

def read_csv(file_path):
    return actions.advanced_file_read(file_path)

def filter_csv_dataframe(df, row_filter_text, column_filter_text):
    if df is None:
        return None

    filtered_df = df.copy()

    # Row filtering
    if row_filter_text:
        query_part, contains_part = (row_filter_text.split('@', 1) + [''])[:2]

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

    # Column filtering
    if column_filter_text:
        all_columns = df.columns.tolist()
        filter_terms = [term.strip() for term in column_filter_text.replace(',', ' ').replace(';', ' ').split() if term]

        include_remaining = "*" in filter_terms
        if include_remaining: filter_terms.remove("*")

        matching_columns = []
        for term in filter_terms:
            for col in all_columns:
                if term.lower() in col.lower() and col not in matching_columns:
                    matching_columns.append(col)

        if include_remaining:
            matching_columns.extend([col for col in all_columns if col not in matching_columns])

        if matching_columns:
            valid_columns = [col for col in matching_columns if col in filtered_df.columns]
            if valid_columns:
                filtered_df = filtered_df[valid_columns]

    return filtered_df
