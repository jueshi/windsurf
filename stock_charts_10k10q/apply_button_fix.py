"""
Apply the button visibility fix to the stock chart application.
This script imports and applies the fix from button_visibility_fix.py.
"""

import logging
from button_visibility_fix import apply_button_visibility_fix

def apply_fixes(gui_instance):
    """
    Apply all fixes to the GUI instance.
    
    Args:
        gui_instance: The StockDataGUI instance to patch
    """
    # Apply the button visibility fix
    apply_button_visibility_fix(gui_instance)
    
    logging.info("All GUI fixes applied successfully")
