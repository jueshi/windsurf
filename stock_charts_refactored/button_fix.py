"""
Fix for button visibility issue in the stock chart GUI.
This patch ensures that bottom buttons remain visible when displaying charts.
"""

import logging
import tkinter as tk
from functools import wraps

def apply_button_visibility_fix(gui_instance):
    """
    Apply the button visibility fix to the GUI instance.
    
    Args:
        gui_instance: The StockDataGUI instance to patch
    """
    # Store original method reference
    original_display_plotly = gui_instance._display_plotly_chart
    
    # Define the patched method
    @wraps(original_display_plotly)
    def patched_display_plotly_chart(fig, tab="individual"):
        """
        Patched version of _display_plotly_chart that ensures bottom buttons remain visible.
        """
        try:
            # Call the original method
            original_display_plotly(fig, tab)
            
            # After displaying the chart, ensure the bottom frame is visible and properly packed
            if hasattr(gui_instance, 'root') and gui_instance.root.winfo_exists():
                # Force update of the UI to ensure proper layout
                gui_instance.root.update_idletasks()
                
                # Ensure all frames are properly packed
                for widget in gui_instance.root.winfo_children():
                    if isinstance(widget, tk.Frame) or isinstance(widget, tk.LabelFrame):
                        widget.update_idletasks()
                
                # Log success
                logging.info(f"Button visibility fix applied for {tab} chart display")
        except Exception as e:
            logging.error(f"Error in button visibility fix: {str(e)}")
    
    # Replace the original method with our patched version
    gui_instance._display_plotly_chart = patched_display_plotly_chart
    logging.info("Button visibility fix applied to GUI")
