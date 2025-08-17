"""
Fixed height chart solution to prevent bottom buttons from disappearing.
This patch modifies the chart display to use a fixed height container that won't expand.
"""

import logging
import types
import tkinter as tk
from tkinter import ttk

def apply_fixed_height_chart_fix(gui_instance):
    """
    Apply a fixed height chart solution to prevent bottom buttons from disappearing.
    
    Args:
        gui_instance: The StockDataGUI instance to patch
    """
    logging.info("Applying fixed height chart solution...")
    
    # Store the original methods
    original_display_plotly_chart = gui_instance._display_plotly_chart
    
    def fixed_display_plotly_chart(self, fig, tab="individual"):
        """
        Fixed version of _display_plotly_chart that uses a fixed height container.
        
        Args:
            fig: Plotly figure to display
            tab: Tab to display the chart in ("individual", "comparison", or "seasonality")
        """
        try:
            # Check if root window still exists before proceeding
            if not hasattr(self, 'root') or not self.root.winfo_exists():
                logging.warning(f"Cannot display Plotly chart: root window no longer exists")
                return
                
            # Create a temporary HTML file to display the chart
            import tempfile
            import os
            import webbrowser
            from plotly.offline import plot
            
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, f"stock_chart_{tab}.html")
            
            # Configure the figure for better display - CRITICAL: Set a fixed height
            fig.update_layout(
                autosize=False,  # Disable autosize
                width=800,       # Fixed width
                height=450,      # Fixed height - IMPORTANT: Keep this small enough to not push buttons off screen
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(rangeslider=dict(visible=False)),
                template="plotly_white"
            )
            
            # Save the figure to HTML with full HTML header for browser display
            plot(fig, filename=html_path, auto_open=False)
            
            # Get the appropriate chart frame
            chart_frame = None
            if tab == "individual":
                if not hasattr(self, 'individual_chart_frame') or not self.individual_chart_frame.winfo_exists():
                    logging.warning("Cannot update individual chart frame: widget no longer exists")
                    return
                chart_frame = self.individual_chart_frame
            elif tab == "comparison":
                if not hasattr(self, 'comparison_chart_frame') or not self.comparison_chart_frame.winfo_exists():
                    logging.warning("Cannot update comparison chart frame: widget no longer exists")
                    return
                chart_frame = self.comparison_chart_frame
            elif tab == "seasonality":
                if not hasattr(self, 'seasonality_chart_container') or not self.seasonality_chart_container.winfo_exists():
                    logging.warning("Cannot update seasonality chart container: widget no longer exists")
                    return
                chart_frame = self.seasonality_chart_container
            
            if chart_frame:
                try:
                    # Clear existing widgets
                    for widget in chart_frame.winfo_children():
                        widget.destroy()
                    
                    # Create a fixed height container frame
                    container_frame = ttk.Frame(chart_frame, height=450)
                    container_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    container_frame.pack_propagate(False)  # CRITICAL: Prevent container from expanding with contents
                    
                    # Create a button to open the chart in browser for better interaction
                    btn_frame = ttk.Frame(container_frame)
                    btn_frame.pack(fill=tk.X, pady=5)
                    ttk.Button(btn_frame, text="Open in Browser", 
                              command=lambda: webbrowser.open(f"file:///{html_path}")).pack()
                    
                    # Create a label to show that interactive chart is available
                    ticker_text = ""
                    if hasattr(self, 'current_chart_ticker') and self.current_chart_ticker:
                        ticker_text = f" for {self.current_chart_ticker}"
                    
                    ttk.Label(container_frame, text=f"Interactive {tab} chart{ticker_text}\nUse mouse to zoom/pan").pack()
                    
                except tk.TclError as e:
                    logging.error(f"TclError updating {tab} chart frame: {str(e)}")
                    return
            
            # Update status if status_var exists
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Displayed interactive {tab} chart")
            
            # IMPORTANT: Don't automatically open the browser
            
            # Force update of the UI to ensure buttons remain visible
            self.root.update_idletasks()
            
            # Ensure bottom frame is visible
            if hasattr(self, 'bottom_frame') and self.bottom_frame.winfo_exists():
                self.bottom_frame.update_idletasks()
                self.bottom_frame.lift()  # Bring to front
            
        except Exception as e:
            logging.error(f"Error displaying Plotly chart: {e}")
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Error displaying chart: {str(e)}")
    
    # Replace the original method with our fixed version
    gui_instance._display_plotly_chart = types.MethodType(fixed_display_plotly_chart, gui_instance)
    
    # Also modify the chart notebook to have a fixed height
    if hasattr(gui_instance, 'chart_notebook') and gui_instance.chart_notebook.winfo_exists():
        chart_frame = ttk.Frame(gui_instance.root, height=500)
        chart_frame.pack_propagate(False)  # Prevent automatic resizing
        
        # Move the chart notebook to the new fixed height frame
        gui_instance.chart_notebook.pack_forget()
        gui_instance.chart_notebook.master = chart_frame
        gui_instance.chart_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Place the fixed height frame in the correct position
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    logging.info("Fixed height chart solution applied successfully")

def apply_fixes(gui_instance):
    """
    Apply all fixes to the GUI instance.
    
    Args:
        gui_instance: The StockDataGUI instance to patch
    """
    # Apply the fixed height chart solution
    apply_fixed_height_chart_fix(gui_instance)
    
    logging.info("All fixed height chart fixes applied successfully")
