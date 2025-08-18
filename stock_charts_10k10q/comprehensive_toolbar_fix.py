"""
Comprehensive fix for both navigation toolbar and bottom button visibility issues.
This patch combines approaches from toolbar_fix.py and direct_chart_fix.py.
"""

import logging
import types
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def apply_comprehensive_toolbar_fix(gui_instance):
    """
    Apply a comprehensive fix to ensure both the navigation toolbar and bottom buttons remain visible.
    
    Args:
        gui_instance: The StockDataGUI instance to patch
    """
    logging.info("Applying comprehensive toolbar and button visibility fix...")
    
    # Store the original _display_chart method
    if hasattr(gui_instance, '_display_chart'):
        original_display_chart = gui_instance._display_chart
        
        def fixed_display_chart(self, ticker_or_path):
            """
            Fixed version of _display_chart that ensures both toolbar and bottom buttons remain visible.
            
            Args:
                ticker_or_path (str): Either a ticker symbol or a full path to an image file
            """
            try:
                # Store reference to bottom frame for later use
                bottom_frame_exists = hasattr(self, 'bottom_frame') and self.bottom_frame.winfo_exists()
                if bottom_frame_exists:
                    bottom_frame = self.bottom_frame
                    logging.info("Found bottom_frame reference")
                
                # Call the original method to handle initial chart display logic
                result = original_display_chart(ticker_or_path)
                
                # After chart display, ensure bottom frame is visible
                if bottom_frame_exists and hasattr(self, 'bottom_frame') and self.bottom_frame.winfo_exists():
                    # Force update and lift bottom frame to ensure it's visible
                    self.bottom_frame.update_idletasks()
                    self.bottom_frame.lift()
                    logging.info("Re-lifted bottom_frame to ensure visibility")
                
                return result
                
            except Exception as e:
                logging.error(f"Error in fixed_display_chart: {str(e)}")
                return False
        
        # Replace the original method with our fixed version
        gui_instance._display_chart = types.MethodType(fixed_display_chart, gui_instance)
        logging.info("Applied fix for _display_chart method")
    
    # Also patch the direct chart display method if it exists
    if hasattr(gui_instance, '_direct_display_chart'):
        original_direct_display_chart = gui_instance._direct_display_chart
        
        def fixed_direct_display_chart(self, ticker, frame):
            """
            Fixed version of _direct_display_chart that properly handles toolbar visibility.
            
            Args:
                ticker: Ticker symbol to display chart for
                frame: Frame to display the chart in
            """
            try:
                # Store reference to bottom frame for later use
                bottom_frame_exists = hasattr(self, 'bottom_frame') and self.bottom_frame.winfo_exists()
                if bottom_frame_exists:
                    bottom_frame = self.bottom_frame
                    logging.info("Found bottom_frame reference for direct chart")
                
                # Check if root window still exists before proceeding
                if not hasattr(self, 'root') or not self.root.winfo_exists():
                    logging.warning(f"Cannot display direct chart for {ticker}: root window no longer exists")
                    return False
                    
                # Check if frame still exists
                if not frame.winfo_exists():
                    logging.warning(f"Cannot display direct chart for {ticker}: frame no longer exists")
                    return False
                
                # Clear existing widgets in the frame
                for widget in frame.winfo_children():
                    widget.destroy()
                
                # Get stock data using original method's logic
                result = original_direct_display_chart(ticker, frame)
                
                # After chart display, ensure bottom frame is visible
                if bottom_frame_exists and hasattr(self, 'bottom_frame') and self.bottom_frame.winfo_exists():
                    # Force update and lift bottom frame to ensure it's visible
                    self.bottom_frame.update_idletasks()
                    self.bottom_frame.lift()
                    logging.info("Re-lifted bottom_frame after direct chart display")
                
                return result
                
            except Exception as e:
                logging.error(f"Error in fixed_direct_display_chart: {str(e)}")
                return False
        
        # Replace the original method with our fixed version
        gui_instance._direct_display_chart = types.MethodType(fixed_direct_display_chart, gui_instance)
        logging.info("Applied fix for _direct_display_chart method")
    
    # Also patch the Plotly chart display method if it exists
    if hasattr(gui_instance, '_display_plotly_chart'):
        original_display_plotly_chart = gui_instance._display_plotly_chart
        
        def fixed_display_plotly_chart(self, fig, tab=None):
            """
            Fixed version of _display_plotly_chart that ensures bottom buttons remain visible.
            
            Args:
                fig: Plotly figure to display
                tab: Tab to display the chart in
            """
            try:
                # Store reference to bottom frame for later use
                bottom_frame_exists = hasattr(self, 'bottom_frame') and self.bottom_frame.winfo_exists()
                if bottom_frame_exists:
                    bottom_frame = self.bottom_frame
                    logging.info("Found bottom_frame reference for Plotly chart")
                
                # Create a fixed height container for the chart if possible
                frame = None
                
                # Check if chart_frames attribute exists and contains the tab
                if hasattr(self, 'chart_frames') and tab and tab in self.chart_frames:
                    frame = self.chart_frames[tab]
                # Check if chart_tabs attribute exists (alternative implementation)
                elif hasattr(self, 'chart_tabs') and tab and tab in self.chart_tabs:
                    frame = self.chart_tabs[tab]
                # Check if we can find the frame by tab name in chart_notebook
                elif hasattr(self, 'chart_notebook') and tab:
                    # Try to get the frame from the notebook by tab name
                    try:
                        tab_id = None
                        for i in range(self.chart_notebook.index('end')):
                            if self.chart_notebook.tab(i, 'text').lower() == tab.lower():
                                tab_id = i
                                break
                        
                        if tab_id is not None:
                            frame = self.chart_notebook.winfo_children()[tab_id]
                    except Exception as e:
                        logging.error(f"Error finding tab frame: {str(e)}")
                
                # If we found a valid frame, proceed with container creation
                if frame and frame.winfo_exists():
                    # Clear existing widgets
                    for widget in frame.winfo_children():
                        widget.destroy()
                    
                    # Create a fixed height container frame
                    container_frame = ttk.Frame(frame, height=400)
                    container_frame.pack(fill=tk.BOTH, expand=False)
                    container_frame.pack_propagate(False)  # Prevent automatic resizing
                    
                    # Set fixed size for the figure to prevent it from expanding too much
                    fig.update_layout(
                        autosize=False,
                        width=800,
                        height=380
                    )
                    
                    # Call original method with our container frame
                    result = original_display_plotly_chart(fig, tab)
                    
                    # After chart display, ensure bottom frame is visible
                    if bottom_frame_exists and hasattr(self, 'bottom_frame') and self.bottom_frame.winfo_exists():
                        # Force update and lift bottom frame to ensure it's visible
                        self.bottom_frame.update_idletasks()
                        self.bottom_frame.lift()
                        logging.info("Re-lifted bottom_frame after Plotly chart display")
                    
                    return result
                else:
                    # If no tab specified, use original method
                    return original_display_plotly_chart(fig, tab)
                
            except Exception as e:
                logging.error(f"Error in fixed_display_plotly_chart: {str(e)}")
                return False
        
        # Replace the original method with our fixed version
        gui_instance._display_plotly_chart = types.MethodType(fixed_display_plotly_chart, gui_instance)
        logging.info("Applied fix for _display_plotly_chart method")
    
    logging.info("Comprehensive toolbar and button visibility fix applied successfully")

def apply_fixes(gui_instance):
    """
    Apply all fixes to the GUI instance.
    
    Args:
        gui_instance: The StockDataGUI instance to patch
    """
    apply_comprehensive_toolbar_fix(gui_instance)
