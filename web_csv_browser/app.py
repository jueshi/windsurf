import os
import sys
import json
from flask import Flask, render_template, request, jsonify

# Add parent directory to sys.path to access csv_browser
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csv_browser import core
from csv_browser import settings
from csv_browser import actions

app = Flask(__name__)

# Mock app object for settings
class MockApp:
    def __init__(self):
        self.recent_directories = []
        self.saved_filters = []
        self.saved_file_filters = []

mock_app = MockApp()
settings.load_settings(mock_app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    if request.method == 'POST':
        data = request.json
        mock_app.recent_directories = data.get('recent_directories', mock_app.recent_directories)
        mock_app.saved_filters = data.get('saved_filters', mock_app.saved_filters)
        mock_app.saved_file_filters = data.get('saved_file_filters', mock_app.saved_file_filters)
        settings.save_settings(mock_app)
        return jsonify({"status": "success"})
    else:
        return jsonify({
            "recent_directories": mock_app.recent_directories,
            "saved_filters": mock_app.saved_filters,
            "saved_file_filters": mock_app.saved_file_filters
        })

@app.route('/api/files', methods=['POST'])
def list_files():
    data = request.json
    directory = data.get('directory', os.getcwd())
    include_subfolders = data.get('include_subfolders', False)
    file_filter = data.get('filter', '')

    if not os.path.exists(directory):
         return jsonify({"error": "Directory does not exist", "files": []})

    # Update recent directories
    if directory not in mock_app.recent_directories:
        mock_app.recent_directories.insert(0, directory)
        mock_app.recent_directories = mock_app.recent_directories[:5]
        settings.save_settings(mock_app)

    files = core.get_files_in_directory(directory, include_subfolders)
    df = core.create_file_dataframe(files)
    df = core.filter_file_dataframe(df, file_filter)

    # Convert to list of dicts
    files_list = df.to_dict(orient='records')
    return jsonify({"files": files_list, "current_directory": directory})

@app.route('/api/csv', methods=['POST'])
def get_csv():
    data = request.json
    filepath = data.get('filepath')
    row_filter = data.get('row_filter', '')
    col_filter = data.get('col_filter', '')

    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "File not found"})

    try:
        df = core.read_csv(filepath)
        if df is None:
             return jsonify({"error": "Failed to read CSV"})

        filtered_df = core.filter_csv_dataframe(df, row_filter, col_filter)

        # Limit rows for performance if needed, but for now let's send all (or first 1000)
        # Web browsers might choke on huge tables. Let's cap at 5000 rows for display
        total_rows = len(filtered_df)
        truncated = False
        if total_rows > 5000:
            filtered_df = filtered_df.head(5000)
            truncated = True

        return jsonify({
            "data": filtered_df.to_dict(orient='records'),
            "columns": filtered_df.columns.tolist(),
            "total_rows": total_rows,
            "truncated": truncated
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/action', methods=['POST'])
def perform_action():
    data = request.json
    action_type = data.get('type')
    files = data.get('files', []) # List of file paths
    target_dir = data.get('target', '')

    if not files:
        return jsonify({"error": "No files selected"})

    try:
        import shutil
        if action_type == 'delete':
            for f in files:
                if os.path.exists(f):
                    os.remove(f)
            return jsonify({"status": "success", "message": f"Deleted {len(files)} files"})

        elif action_type == 'copy':
            if not target_dir: return jsonify({"error": "Target directory required"})
            for f in files:
                shutil.copy2(f, target_dir)
            return jsonify({"status": "success", "message": f"Copied {len(files)} files"})

        elif action_type == 'move':
            if not target_dir: return jsonify({"error": "Target directory required"})
            for f in files:
                shutil.move(f, target_dir)
            return jsonify({"status": "success", "message": f"Moved {len(files)} files"})

        else:
            return jsonify({"error": "Unknown action"})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
