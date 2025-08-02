import pandas as pd
import pyperclip
import os

def parse_clipboard_data(data):
    """
    Parse clipboard data with specific formatting
    
    Args:
        data (str): Raw clipboard data
    
    Returns:
        pd.DataFrame: Parsed DataFrame
    """
    # Split data into lines
    lines = data.split('\n')
    
    # Extract header
    header = lines[0].split('\t')
    
    # Prepare data rows
    data_rows = []
    current_row = {}
    header_index = 0
    
    for line in lines[1:]:
        # If line is a number (row number), start a new row
        if line.isdigit():
            if current_row:
                data_rows.append(current_row)
            current_row = {}
            header_index = 0
        else:
            # Add the line to the current row
            if header_index < len(header):
                current_row[header[header_index]] = line
                header_index += 1
    
    # Add the last row
    if current_row:
        data_rows.append(current_row)
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    return df

def first_capture(filename='clipboard_data.csv'):
    """
    First capture of clipboard data, creating a new CSV file
    
    Args:
        filename (str, optional): Name of the CSV file to save. Defaults to 'clipboard_data.csv'.
    
    Returns:
        pd.DataFrame: Captured DataFrame
    """
    # Get data from clipboard
    data = pyperclip.paste()

    # Parse clipboard data
    df = parse_clipboard_data(data)

    # Save dataframe to csv file
    df.to_csv(filename, index=False)
    
    # Print the DataFrame
    print("DataFrame contents:")
    print(df)
    
    return df

def additional_capture(filename='clipboard_data.csv'):
    """
    Append additional clipboard data to an existing CSV file
    
    Args:
        filename (str, optional): Name of the CSV file to append to. Defaults to 'clipboard_data.csv'.
    
    Returns:
        pd.DataFrame: Updated DataFrame
    """
    # Check if the file exists, if not, use first_capture
    if not os.path.exists(filename):
        return first_capture(filename)

    # Get data from clipboard
    data = pyperclip.paste()

    # Parse clipboard data
    new_df = parse_clipboard_data(data)

    # Read existing data
    existing_df = pd.read_csv(filename)

    # Ensure column names match
    if list(new_df.columns) != list(existing_df.columns):
        print("Warning: New data does not match existing data columns. Skipping append.")
        return existing_df

    # Combine dataframes
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Save updated dataframe
    updated_df.to_csv(filename, index=False)
    
    # Print the updated DataFrame
    print("Updated DataFrame contents:")
    print(updated_df)
    
    return updated_df

def main():
    """
    Main function to demonstrate usage
    """
    # Uncomment the function you want to use
    # first_capture()
    additional_capture()

if __name__ == '__main__':
    main()
