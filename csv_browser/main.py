import os
import sys
from .app import CSVBrowser

def main():
    app = CSVBrowser()
    app.mainloop()

if __name__ == "__main__":
    # This allows the script to be run directly
    # and also as a module.
    # The following is for when it's run as a module.
    if __package__ is None:
        # If run as a script, add the parent directory to sys.path
        # to allow for relative imports
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from csv_browser.app import CSVBrowser
        app = CSVBrowser()
        app.mainloop()
    else:
        # If run as a module
        app = CSVBrowser()
        app.mainloop()
