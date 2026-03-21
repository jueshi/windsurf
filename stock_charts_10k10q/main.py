import tkinter as tk
import logging
from data_manager import StockDataManager
from gui import StockDataGUI
from direct_fix import apply_direct_fixes
from plotly_fix import apply_plotly_fix
from emergency_fix import apply_emergency_fix
from widget_fix import apply_widget_fix
from direct_chart_fix import apply_direct_chart_fix
from apply_button_fix import apply_fixes as apply_button_fixes
from individual_chart_fix import apply_fixes as apply_individual_chart_fixes
from fixed_height_chart_fix import apply_fixes as apply_fixed_height_chart_fixes
from toolbar_fix import apply_fixes as apply_toolbar_fixes
from comprehensive_toolbar_fix import apply_fixes as apply_comprehensive_fixes
from timeframe_chart_fix import apply_timeframe_chart_fix
from embedded_timeframe_fix import apply_embedded_timeframe_fix
from simple_timeframe_fix import apply_simple_timeframe_fix # Keep for now, might be needed
from ticker_timeframe_fix import apply_ticker_timeframe_fix # Keep for now, might be needed
from final_timeframe_fix import apply_final_timeframe_fix # Import the new comprehensive fix

def suppress_tkinter_exit_errors():
    """Suppress tkinter cleanup exceptions on exit with enhanced error handling."""
    # Save original __del__ methods
    original_image_del = tk.Image.__del__
    original_var_del = tk.Variable.__del__
    original_photoimage_del = None
    if hasattr(tk, 'PhotoImage') and hasattr(tk.PhotoImage, '__del__'):
        original_photoimage_del = tk.PhotoImage.__del__

    # Define safe __del__ methods with comprehensive error handling
    def safe_image_del(self):
        try:
            original_image_del(self)
        except (RuntimeError, AttributeError, TypeError) as e:
            # More comprehensive error handling
            pass

    def safe_var_del(self):
        try:
            original_var_del(self)
        except (RuntimeError, AttributeError, TypeError) as e:
            # More comprehensive error handling
            pass

    def safe_photoimage_del(self):
        try:
            original_photoimage_del(self)
        except (RuntimeError, AttributeError, TypeError) as e:
            # More comprehensive error handling
            pass

    # Replace __del__ methods with safe versions
    tk.Image.__del__ = safe_image_del
    tk.Variable.__del__ = safe_var_del
    if original_photoimage_del:
        tk.PhotoImage.__del__ = safe_photoimage_del

    # Also patch Tcl async handlers
    try:
        # Import threading for thread identification
        import threading
        import _thread
        
        # Monkey patch the Tcl interpreter's async delete handler
        if hasattr(tk, '_tkinter') and hasattr(tk._tkinter, 'TclError'):
            # Store the main thread ID for comparison
            main_thread_id = threading.current_thread().ident
            
            # Patch the tk 'call' method on the base Mix-in (Misc) to handle async thread issues
            if hasattr(tk.Misc, 'call'):
                original_tkapp_call = tk.Misc.call

                def safe_tkapp_call(self, *args, **kwargs):
                    try:
                        if threading.current_thread().ident == main_thread_id:
                            return original_tkapp_call(self, *args, **kwargs)
                        else:
                            return None
                    except (RuntimeError, AttributeError, TypeError):
                        return None

                tk.Misc.call = safe_tkapp_call
    except Exception as e:
        # If patching fails, continue without it
        print(f"Failed to patch Tkinter thread handling: {e}")
        pass

def main():
    """Main function to launch the Stock Data Manager GUI."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ) 
    # Suppress tkinter cleanup exceptions on exit
    suppress_tkinter_exit_errors()

    # Suppress FutureWarning from yfinance
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

    root = tk.Tk()
    root.title("Personal AI Stock Assistant")

    # Maximize the window
    root.state('zoomed')  # Windows-specific command to maximize

    # Create the StockDataManager instance
    manager = StockDataManager()

    # Create the application
    app = StockDataGUI(root, manager)
    
    # Apply all fixes
    apply_direct_fixes(app, manager)
    apply_plotly_fix(app)
    apply_emergency_fix(app)
    apply_widget_fix(app)
    apply_direct_chart_fix(app)
    apply_button_fixes(app)  # Apply button visibility fix
    apply_individual_chart_fixes(app)  # Apply individual chart button visibility fix
    apply_fixed_height_chart_fixes(app)  # Apply fixed height chart solution
    apply_toolbar_fixes(app)  # Apply toolbar visibility fix
    apply_comprehensive_fixes(app)  # Apply comprehensive toolbar and button visibility fix
    # apply_timeframe_chart_fix(app)  # Original timeframe chart implementation (disabled)
    # apply_embedded_timeframe_fix(app)  # Previous embedded Plotly charts implementation (disabled)
    # apply_simple_timeframe_fix(app)  # Disabled in favor of the comprehensive fix
    # apply_ticker_timeframe_fix(app)  # Disabled in favor of the comprehensive fix
    # from tab_switching_fix import apply_tab_switching_fix # Disabled in favor of the comprehensive fix
    # apply_tab_switching_fix(app)  # Disabled in favor of the comprehensive fix
    apply_final_timeframe_fix(app) # Apply the new, consolidated fix for timeframe charts

    # Define the on_closing handler
    def on_closing():
        try:
            print("Cleaning up resources...")
            
            # First, stop any running threads in the app
            if hasattr(app, 'cleanup'):
                app.cleanup()
                
            # Wait for any background threads to finish
            import threading
            main_thread = threading.current_thread()
            for thread in threading.enumerate():
                if thread is not main_thread and thread.is_alive():
                    try:
                        print(f"Waiting for thread {thread.name} to finish...")
                        thread.join(timeout=1.0)  # Wait with timeout to avoid hanging
                    except Exception as e:
                        print(f"Error joining thread {thread.name}: {e}")
            
            # Explicitly delete all tkinter variables to prevent cleanup exceptions
            for widget in root.winfo_children():
                if hasattr(widget, 'destroy'):
                    try:
                        widget.destroy()
                    except Exception as e:
                        print(f"Error destroying widget: {e}")

            # Delete all images with error handling
            try:
                for name in list(root.tk.call('image', 'names')):
                    try:
                        root.tk.call('image', 'delete', name)
                    except Exception as e:
                        print(f"Error deleting image {name}: {e}")
            except Exception as e:
                print(f"Error listing images: {e}")

            # Force garbage collection
            import gc
            gc.collect()
            
            # Use after() to schedule the destruction after all events are processed
            root.after(100, lambda: safe_destroy(root))
            
        except Exception as e:
            print(f"Error during application shutdown: {str(e)}")
            # Try to destroy the root window anyway
            try:
                root.destroy()
            except:
                pass
    
    def safe_destroy(widget):
        """Safely destroy a widget with error handling"""
        try:
            widget.destroy()
        except Exception as e:
            print(f"Error during final destruction: {e}")
            # If normal destroy fails, try to force quit
            import os, sys
            try:
                # Only in extreme cases
                os._exit(0)
            except:
                sys.exit(0)

    # Set the protocol handler
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start the main loop
    root.mainloop()

if __name__ == '__main__':
    main()
