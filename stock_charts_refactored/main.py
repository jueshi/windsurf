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
from simple_timeframe_fix import apply_simple_timeframe_fix
from ticker_timeframe_fix import apply_ticker_timeframe_fix

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
        # Monkey patch the Tcl interpreter's async delete handler
        if hasattr(tk, '_tkinter') and hasattr(tk._tkinter, 'TclError'):
            original_tcl_async_hook = None
            if hasattr(tk.Tcl(), 'async_hook'):
                original_tcl_async_hook = tk.Tcl().async_hook

                def safe_async_hook(*args, **kwargs):
                    try:
                        if original_tcl_async_hook:
                            return original_tcl_async_hook(*args, **kwargs)
                    except Exception:
                        pass

                tk.Tcl().async_hook = safe_async_hook
    except Exception:
        # If patching fails, continue without it
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
    root.title("Stock Data Manager")

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
    apply_simple_timeframe_fix(app)  # Apply simple and direct timeframe chart implementation
    apply_ticker_timeframe_fix(app)  # Apply direct ticker selection handler for timeframe charts
    from tab_switching_fix import apply_tab_switching_fix
    apply_tab_switching_fix(app)  # Apply tab switching fix to prevent timeframe tab from switching back

    # Define the on_closing handler
    def on_closing():
        try:
            print("Cleaning up resources...")
            app.cleanup()

            # Explicitly delete all tkinter variables to prevent cleanup exceptions
            for widget in root.winfo_children():
                if hasattr(widget, 'destroy'):
                    widget.destroy()

            # Delete all images
            for name in list(root.tk.call('image', 'names')):
                root.tk.call('image', 'delete', name)

            # Force garbage collection
            import gc
            gc.collect()

            root.destroy()
        except Exception as e:
            print(f"Error during application shutdown: {str(e)}")

    # Set the protocol handler
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start the main loop
    root.mainloop()

if __name__ == '__main__':
    main()
