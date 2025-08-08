import os
import json
import traceback

def get_settings_path():
    """Determines the path for the settings file."""
    # We'll place the settings file in the directory containing the csv_browser package.
    package_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(package_dir)
    return os.path.join(parent_dir, "csv_browser_settings.json")

def load_settings(app):
    """Load settings from the JSON file and apply them to the app."""
    try:
        settings_file = get_settings_path()
        print(f"Attempting to load settings from: {settings_file}")

        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                file_content = f.read()
                if file_content.strip():
                    settings = json.loads(file_content)
                    app.recent_directories = settings.get("recent_directories", [])
                    app.saved_filters = settings.get("saved_filters", [])
                    app.saved_file_filters = settings.get("saved_file_filters", [])
                    print("Settings loaded successfully.")
                else:
                    print("Settings file is empty.")
        else:
            print(f"No settings file found at {settings_file}. Using default settings.")
            app.recent_directories = []
            app.saved_filters = []
            app.saved_file_filters = []

    except Exception as e:
        print(f"Error loading settings: {e}")
        traceback.print_exc()
        # Initialize with empty lists in case of error
        app.recent_directories = []
        app.saved_filters = []
        app.saved_file_filters = []

def save_settings(app):
    """Save the application's settings to the JSON file."""
    try:
        settings_file = get_settings_path()
        print(f"Saving settings to: {settings_file}")

        settings = {
            "recent_directories": app.recent_directories,
            "saved_filters": app.saved_filters,
            "saved_file_filters": app.saved_file_filters
        }

        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=4)

        print("Settings saved successfully.")

    except Exception as e:
        print(f"Error saving settings: {e}")
        traceback.print_exc()
