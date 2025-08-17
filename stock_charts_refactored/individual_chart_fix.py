"""
Direct fix for the button visibility issue in the individual chart tab.
This patch specifically targets the _display_chart method to ensure buttons remain visible.
"""

import logging
import types
import tkinter as tk
from tkinter import ttk

def apply_individual_chart_fix(gui_instance):
    """
    Apply a direct fix to ensure buttons remain visible when displaying individual charts.
    
    Args:
        gui_instance: The StockDataGUI instance to patch
    """
    logging.info("Applying individual chart button visibility fix...")
    
    # Store the original method
    original_display_chart = gui_instance._display_chart
    
    def fixed_display_chart(self, ticker_or_path):
        """
        Fixed version of _display_chart that ensures bottom buttons remain visible.
        
        Args:
            ticker_or_path (str): Either a ticker symbol or a full path to an image file
        """
        try:
            # First check if the root window and widgets still exist
            if not hasattr(self, 'root') or not self.root.winfo_exists():
                logging.warning(f"Cannot display chart for {ticker_or_path}: root window no longer exists")
                return
                
            # Save reference to bottom frame before chart display
            has_bottom_frame = hasattr(self, 'bottom_frame') and self.bottom_frame.winfo_exists()
            
            # Call the original method to display the chart
            original_display_chart(ticker_or_path)
            
            # After chart display, ensure bottom frame is visible
            if has_bottom_frame and hasattr(self, 'bottom_frame') and self.bottom_frame.winfo_exists():
                # Force update and lift bottom frame to ensure it's visible
                self.bottom_frame.update_idletasks()
                self.bottom_frame.lift()
                
                # Force the root window to update its layout
                self.root.update_idletasks()
                
                # Ensure proper packing order - make sure bottom frame is at the bottom
                if hasattr(self, 'chart_notebook') and self.chart_notebook.winfo_exists():
                    self.chart_notebook.pack_forget()
                    self.bottom_frame.pack_forget()
                    
                    # Re-pack in correct order
                    self.chart_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
                    self.bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
                
                logging.info("Individual chart button visibility fix applied")
                
        except Exception as e:
            logging.error(f"Error in fixed_display_chart: {str(e)}")
            # Try to call original method as fallback
            try:
                original_display_chart(ticker_or_path)
            except Exception as inner_e:
                logging.error(f"Error in original_display_chart fallback: {str(inner_e)}")
    
    # Replace the original method with our fixed version
    gui_instance._display_chart = types.MethodType(fixed_display_chart, gui_instance)
    logging.info("Individual chart button visibility fix applied successfully")

def apply_fixes(gui_instance):
    """
    Apply all fixes to the GUI instance.
    
    Args:
        gui_instance: The StockDataGUI instance to patch
    """
    # Apply the individual chart button visibility fix
    apply_individual_chart_fix(gui_instance)
    
    logging.info("All individual chart fixes applied successfully")
