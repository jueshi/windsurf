"""
Enhanced fix for the button visibility issue in the stock chart GUI.
This patch ensures that bottom buttons remain visible when displaying charts,
with special handling for the individual chart tab.
"""

import logging
import types
import tkinter as tk
from tkinter import ttk

def apply_enhanced_button_fix(gui_instance):
    """
    Apply an enhanced button visibility fix to the GUI instance.
    
    Args:
        gui_instance: The StockDataGUI instance to patch
    """
    logging.info("Applying enhanced button visibility fix...")
    
    # Store the original method
    original_display_plotly_chart = gui_instance._display_plotly_chart
    
    def enhanced_display_plotly_chart(self, fig, tab="individual"):
        """
        Enhanced version of _display_plotly_chart that ensures bottom buttons remain visible.
        
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
            
            # Configure the figure for better display
            fig.update_layout(
                autosize=True,
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(rangeslider=dict(visible=False)),  # Hide default range slider
                template="plotly_white",
                # Set a fixed height to prevent chart from expanding too much
                height=500  # Limit chart height to ensure buttons remain visible
            )
            
            # Add range selector buttons for time periods
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            
            # Save the figure to HTML with full HTML header for browser display
            plot(fig, filename=html_path, auto_open=False)
            
            # Handle the specific chart frame based on tab
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
                    
                    # Create a container frame with fixed height to prevent expanding too much
                    container = ttk.Frame(chart_frame, height=500)
                    container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    container.pack_propagate(False)  # Prevent container from expanding with contents
                    
                    # Create a button to open the chart in browser for better interaction
                    btn_frame = ttk.Frame(container)
                    btn_frame.pack(fill=tk.X, pady=5)
                    ttk.Button(btn_frame, text="Open in Browser", 
                              command=lambda: webbrowser.open(f"file:///{html_path}")).pack()
                    
                    # Create a label to show that interactive chart is available
                    ticker_text = ""
                    if hasattr(self, 'current_chart_ticker') and self.current_chart_ticker:
                        ticker_text = f" for {self.current_chart_ticker}"
                    
                    ttk.Label(container, text=f"Interactive {tab} chart{ticker_text}\nUse mouse to zoom/pan").pack()
                    
                except tk.TclError as e:
                    logging.error(f"TclError updating {tab} chart frame: {str(e)}")
                    return
            
            # Update status if status_var exists
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Displayed interactive {tab} chart")
            
            # IMPORTANT: Don't automatically open the browser - this causes the main window to lose focus
            # and can lead to buttons disappearing
            
            # Force update of the UI to ensure buttons remain visible
            self.root.update_idletasks()
            
            # Ensure bottom frame is visible by updating its geometry
            if hasattr(self, 'bottom_frame') and self.bottom_frame.winfo_exists():
                self.bottom_frame.update_idletasks()
                self.bottom_frame.lift()  # Bring to front
            
            # Update all frames to ensure proper layout
            for widget in self.root.winfo_children():
                if isinstance(widget, tk.Frame) or isinstance(widget, ttk.Frame):
                    widget.update_idletasks()
            
        except Exception as e:
            logging.error(f"Error displaying Plotly chart: {e}")
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Error displaying chart: {str(e)}")
    
    # Replace the original method with our enhanced version
    gui_instance._display_plotly_chart = types.MethodType(enhanced_display_plotly_chart, gui_instance)
    logging.info("Enhanced button visibility fix applied successfully")

def apply_fixes(gui_instance):
    """
    Apply all fixes to the GUI instance.
    
    Args:
        gui_instance: The StockDataGUI instance to patch
    """
    # Apply the enhanced button visibility fix
    apply_enhanced_button_fix(gui_instance)
    
    logging.info("All enhanced GUI fixes applied successfully")
