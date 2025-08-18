"""
Fix for the missing _display_plotly_chart method in the GUI.
This module adds the necessary method to properly display Plotly charts.
"""

import logging
import types
import os
import tkinter as tk
from tkinter import ttk

# Try to import Plotly modules
try:
    import plotly
    import plotly.graph_objects as go
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    logging.warning("Plotly modules not available, interactive charts will be disabled")
    PLOTLY_AVAILABLE = False

def apply_widget_fix(app):
    """
    Apply a fix for the missing _display_plotly_chart method in the GUI.
    This adds the necessary method to properly display Plotly charts.
    
    Args:
        app: The GUI application instance
    """
    logging.info("Applying widget fix for Plotly chart display")
    
    def _display_plotly_chart(self, fig, tab="individual"):
        """
        Display a Plotly figure in the appropriate chart frame.
        
        Args:
            fig: The Plotly figure to display
            tab: The tab to display the chart in ("individual", "comparison", or "seasonality")
        """
        logging.info(f"===== STARTING DISPLAY PLOTLY CHART FOR {tab} TAB =====")
        logging.info(f"Figure type: {type(fig).__name__}")
        logging.info(f"Figure data count: {len(fig.data) if hasattr(fig, 'data') else 'unknown'}")
        logging.info(f"Current active tab: {self.active_tab if hasattr(self, 'active_tab') else 'unknown'}")
        logging.info(f"Chart notebook exists: {hasattr(self, 'chart_notebook') and self.chart_notebook.winfo_exists()}")
        logging.info(f"Chart notebook tabs: {self.chart_notebook.tabs() if hasattr(self, 'chart_notebook') and self.chart_notebook.winfo_exists() else 'unknown'}")
        logging.info(f"Current chart notebook tab: {self.chart_notebook.select() if hasattr(self, 'chart_notebook') and self.chart_notebook.winfo_exists() else 'unknown'}")
        logging.info(f"Individual chart frame exists: {hasattr(self, 'individual_chart_frame') and self.individual_chart_frame.winfo_exists() if hasattr(self, 'individual_chart_frame') else False}")
        logging.info(f"Comparison chart frame exists: {hasattr(self, 'comparison_chart_frame') and self.comparison_chart_frame.winfo_exists() if hasattr(self, 'comparison_chart_frame') else False}")
        logging.info(f"Seasonality chart frame exists: {hasattr(self, 'seasonality_chart_frame') and self.seasonality_chart_frame.winfo_exists() if hasattr(self, 'seasonality_chart_frame') else False}")
        
        try:
            logging.info(f"Displaying Plotly chart in {tab} tab")
            
            # Determine which frame to use based on the tab
            if tab == "individual":
                frame_name = "individual_chart_frame"
            elif tab == "comparison":
                frame_name = "comparison_chart_frame"
            elif tab == "seasonality":
                frame_name = "seasonality_chart_frame"
            else:
                logging.error(f"Unknown tab: {tab}")
                return
            
            # Check if the frame exists
            if not hasattr(self, frame_name):
                logging.error(f"Frame {frame_name} does not exist")
                return
            
            frame = getattr(self, frame_name)
            
            # Check if the frame still exists in the Tkinter hierarchy
            if not frame.winfo_exists():
                logging.error(f"Frame {frame_name} no longer exists in the Tkinter hierarchy")
                return
            
            # Clear the frame
            for widget in frame.winfo_children():
                widget.destroy()
            
            # Create a temporary HTML file for the Plotly figure
            import tempfile
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, f"plotly_chart_{tab}.html")
            
            # Save the figure to the HTML file
            plot(fig, filename=html_path, auto_open=False)
            
            # Create a Tkinter frame to embed the HTML
            # Always use the direct approach for reliability
            message_frame = ttk.Frame(frame)
            message_frame.pack(fill=tk.BOTH, expand=True)
            
            # Add a label to show we're displaying a chart
            title_label = ttk.Label(
                message_frame,
                text=f"Interactive {tab.capitalize()} Chart",
                font=("Helvetica", 14, "bold")
            )
            title_label.pack(pady=10)
            
            # Create a direct image display if possible
            try:
                # Try to render a static preview image
                import tempfile
                import matplotlib.pyplot as plt
                
                # Create a static preview
                preview_path = os.path.join(tempfile.gettempdir(), f"preview_{tab}.png")
                
                # Save a static image of the figure if possible
                if hasattr(fig, 'write_image'):
                    try:
                        fig.write_image(preview_path)
                        has_preview = True
                    except Exception as e:
                        logging.error(f"Error saving preview image: {str(e)}")
                        has_preview = False
                else:
                    has_preview = False
                
                if has_preview:
                    # Display the preview image
                    from PIL import Image, ImageTk
                    
                    # Open the image
                    img = Image.open(preview_path)
                    
                    # Resize to fit the frame
                    img = img.resize((800, 500), Image.LANCZOS)
                    
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(img)
                    
                    # Create a label to display the image
                    img_label = ttk.Label(message_frame, image=photo)
                    img_label.image = photo  # Keep a reference to prevent garbage collection
                    img_label.pack(pady=10)
                    
                    logging.info(f"Displayed static preview image for {tab} chart")
            except Exception as e:
                logging.error(f"Error creating preview image: {str(e)}")
            
            # Always provide a button to open in browser for full interactivity
            def open_html():
                import webbrowser
                webbrowser.open(html_path)
            
            open_button = ttk.Button(
                message_frame,
                text="Open Interactive Chart in Browser",
                command=open_html
            )
            open_button.pack(pady=10)
                
            # Log success
            logging.info(f"Created button to open Plotly chart in browser for {tab} tab")
            
            # Store the HTML path for later reference
            if not hasattr(self, '_plotly_html_paths'):
                self._plotly_html_paths = {}
            self._plotly_html_paths[tab] = html_path
            
            logging.info(f"Successfully displayed Plotly chart in {tab} tab")
            
        except Exception as e:
            logging.error(f"Error displaying Plotly chart: {str(e)}")
            
            # Create a simple error message
            if tab == "individual":
                frame_name = "individual_chart_frame"
            elif tab == "comparison":
                frame_name = "comparison_chart_frame"
            elif tab == "seasonality":
                frame_name = "seasonality_chart_frame"
            else:
                return
            
            if hasattr(self, frame_name):
                frame = getattr(self, frame_name)
                
                if frame.winfo_exists():
                    # Clear the frame
                    for widget in frame.winfo_children():
                        widget.destroy()
                    
                    # Display error message
                    error_label = ttk.Label(
                        frame,
                        text=f"Error displaying chart: {str(e)}",
                        foreground="red"
                    )
                    error_label.pack(pady=20)
    
    # Add the method to the app
    app._display_plotly_chart = types.MethodType(_display_plotly_chart, app)
    logging.info("Added _display_plotly_chart method to the app")
    
    # Also check and create chart frames if they don't exist
    def ensure_chart_frames(self):
        """Ensure that all necessary chart frames exist"""
        try:
            # Check if chart_notebook exists
            if not hasattr(self, 'chart_notebook') or not self.chart_notebook.winfo_exists():
                logging.error("Chart notebook does not exist")
                return
            
            # Check and create individual chart frame
            if not hasattr(self, 'individual_chart_frame'):
                logging.info("Creating individual_chart_frame")
                self.individual_chart_frame = ttk.Frame(self.chart_notebook)
                self.chart_notebook.add(self.individual_chart_frame, text="Individual")
            
            # Check and create comparison chart frame
            if not hasattr(self, 'comparison_chart_frame'):
                logging.info("Creating comparison_chart_frame")
                self.comparison_chart_frame = ttk.Frame(self.chart_notebook)
                self.chart_notebook.add(self.comparison_chart_frame, text="Comparison")
            
            # Check and create seasonality chart frame
            if not hasattr(self, 'seasonality_chart_frame'):
                logging.info("Creating seasonality_chart_frame")
                self.seasonality_chart_frame = ttk.Frame(self.chart_notebook)
                self.chart_notebook.add(self.seasonality_chart_frame, text="Seasonality")
            
            logging.info("All chart frames are now available")
            
        except Exception as e:
            logging.error(f"Error ensuring chart frames: {str(e)}")
    
    # Add the method to the app
    app.ensure_chart_frames = types.MethodType(ensure_chart_frames, app)
    
    # Call the method to ensure all frames exist
    app.ensure_chart_frames()
    
    logging.info("Widget fix for Plotly chart display applied successfully")
