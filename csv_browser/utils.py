import os
from datetime import datetime

def normalize_long_path(path):
    """Normalize path and add long path prefix if needed"""
    # Normalize the path
    normalized_path = os.path.normpath(os.path.abspath(path))

    # If path is longer than 250 characters, add the long path prefix
    if len(normalized_path) > 250 and not normalized_path.startswith('\\\\?\\'):
        # Add Windows long path prefix
        normalized_path = '\\\\?\\' + normalized_path
        print(f"Using long path format: {normalized_path}")

    return normalized_path

def format_size(size):
    # Convert size to human readable format
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

def format_date(timestamp):
    # Convert timestamp to readable format
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M")
