"""
Fix for the disappearing buttons issue in the stock chart application.
This patch modifies the _display_plotly_chart method to prevent the buttons from disappearing.
"""

import logging
import types

def apply_button_visibility_fix(gui_instance):
    """
    Apply fix to prevent buttons from disappearing when charts are displayed.
    
    Args:
        gui_instance: The StockDataGUI instance to patch
    """
    logging.info("Applying button visibility fix...")
    
    # Store the original method
    original_display_plotly_chart = gui_instance._display_plotly_chart
    
    def fixed_display_plotly_chart(self, fig, tab="individual"):
        """
        Fixed version of _display_plotly_chart that prevents buttons from disappearing.
        
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
            import tkinter as tk
            from tkinter import ttk, messagebox
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
                template="plotly_white"
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
            
            # Create a frame to embed the HTML - check if widgets exist first
            if tab == "individual":
                if not hasattr(self, 'individual_chart_frame') or not self.individual_chart_frame.winfo_exists():
                    logging.warning("Cannot update individual chart frame: widget no longer exists")
                    return
                    
                try:
                    for widget in self.individual_chart_frame.winfo_children():
                        widget.destroy()
                        
                    # Create a button to open the chart in browser for better interaction
                    btn_frame = ttk.Frame(self.individual_chart_frame)
                    btn_frame.pack(fill=tk.X, pady=5)
                    ttk.Button(btn_frame, text="Open in Browser", 
                              command=lambda: webbrowser.open(f"file:///{html_path}")).pack()
                    
                    # Create a label to show that interactive chart is available
                    ttk.Label(self.individual_chart_frame, 
                             text=f"Interactive chart for {self.current_chart_ticker if hasattr(self, 'current_chart_ticker') else ''}\nUse mouse to zoom/pan").pack()
                except tk.TclError as e:
                    logging.error(f"TclError updating individual chart frame: {str(e)}")
                    return
                
            elif tab == "comparison":
                if not hasattr(self, 'comparison_chart_frame') or not self.comparison_chart_frame.winfo_exists():
                    logging.warning("Cannot update comparison chart frame: widget no longer exists")
                    return
                    
                try:
                    for widget in self.comparison_chart_frame.winfo_children():
                        widget.destroy()
                        
                    # Create a button to open the chart in browser for better interaction
                    btn_frame = ttk.Frame(self.comparison_chart_frame)
                    btn_frame.pack(fill=tk.X, pady=5)
                    ttk.Button(btn_frame, text="Open in Browser", 
                              command=lambda: webbrowser.open(f"file:///{html_path}")).pack()
                    
                    # Create a label to show that interactive chart is available
                    ttk.Label(self.comparison_chart_frame, 
                             text="Interactive comparison chart\nUse mouse to zoom/pan").pack()
                except tk.TclError as e:
                    logging.error(f"TclError updating comparison chart frame: {str(e)}")
                    return
                
            elif tab == "seasonality":
                if not hasattr(self, 'seasonality_chart_container') or not self.seasonality_chart_container.winfo_exists():
                    logging.warning("Cannot update seasonality chart container: widget no longer exists")
                    return
                    
                try:
                    # Keep the controls frame but remove other widgets
                    for widget in self.seasonality_chart_container.winfo_children():
                        widget.destroy()
                        
                    # Create a button to open the chart in browser for better interaction
                    btn_frame = ttk.Frame(self.seasonality_chart_container)
                    btn_frame.pack(fill=tk.X, pady=5)
                    ttk.Button(btn_frame, text="Open in Browser", 
                              command=lambda: webbrowser.open(f"file:///{html_path}")).pack()
                    
                    # Create a label to show that interactive chart is available
                    ttk.Label(self.seasonality_chart_container, 
                             text=f"Interactive seasonality chart for {self.current_chart_ticker if hasattr(self, 'current_chart_ticker') else ''}\nUse mouse to zoom/pan").pack()
                except tk.TclError as e:
                    logging.error(f"TclError updating seasonality chart container: {str(e)}")
                    return
            
            # Update status if status_var exists
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Displayed interactive {tab} chart")
            
            # IMPORTANT: Don't automatically open the browser - this causes the main window to lose focus
            # and can lead to buttons disappearing
            # Instead, let the user click the "Open in Browser" button if they want to view in browser
            
            # Force update of the UI to ensure buttons remain visible
            self.root.update_idletasks()
            
        except Exception as e:
            logging.error(f"Error displaying Plotly chart: {e}")
            messagebox.showerror("Error", f"Failed to display interactive chart: {e}")
    
    # Replace the original method with our fixed version
    gui_instance._display_plotly_chart = types.MethodType(fixed_display_plotly_chart, gui_instance)
    logging.info("Button visibility fix applied successfully")
