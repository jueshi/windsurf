"""
Fixed Enhanced Widget Safety for Tkinter Applications
This file contains enhanced safety mechanisms to prevent 'invalid command name' errors
when widgets are destroyed during callbacks or threading operations.
"""

import logging
import tkinter as tk
from tkinter import messagebox
import functools
import threading
import types

def patch_gui_with_enhanced_safety(gui_instance):
    """
    Apply enhanced widget safety to a GUI instance.
    
    Args:
        gui_instance: The GUI instance to patch
    """
    # Define widget check functions
    def check_root_exists():
        return hasattr(gui_instance, 'root') and gui_instance.root.winfo_exists()
    
    def check_notebook_exists():
        return (hasattr(gui_instance, 'root') and gui_instance.root.winfo_exists() and
                hasattr(gui_instance, 'chart_notebook') and gui_instance.chart_notebook.winfo_exists())
    
    def check_ticker_listbox_exists():
        return (hasattr(gui_instance, 'root') and gui_instance.root.winfo_exists() and
                hasattr(gui_instance, 'ticker_listbox') and gui_instance.ticker_listbox.winfo_exists())
    
    # Create safe method wrappers
    def create_safe_method(original_method, check_func):
        """Create a safe method wrapper that checks widget existence"""
        @functools.wraps(original_method)
        def safe_method(*args, **kwargs):
            try:
                # Check if widgets exist
                if not check_func():
                    method_name = original_method.__name__
                    logging.warning(f"Cannot execute {method_name}: required widgets no longer exist")
                    return None
                
                # Execute the original method
                return original_method(*args, **kwargs)
            except tk.TclError as e:
                method_name = original_method.__name__
                logging.error(f"TclError in {method_name}: {str(e)}")
                return None
        return safe_method
    
    # Patch specific methods with appropriate widget checks
    methods_to_patch = {
        '_on_tab_changed': check_notebook_exists,
        '_get_selected_tickers': check_ticker_listbox_exists,
        '_display_chart': check_notebook_exists,
        '_generate_chart_thread': check_root_exists,
        '_update_chart_after_download': check_root_exists,
        '_generate_seasonality_chart': check_notebook_exists
    }
    
    # Apply patches to specific methods
    for method_name, check_func in methods_to_patch.items():
        if hasattr(gui_instance, method_name):
            original_method = getattr(gui_instance, method_name)
            if callable(original_method):
                safe_method = create_safe_method(original_method, check_func)
                setattr(gui_instance, method_name, types.MethodType(safe_method, gui_instance))
                logging.info(f"Enhanced safety applied to {method_name}")
    
    # Add a global exception handler for all event handlers
    for attr_name in dir(gui_instance):
        if attr_name.startswith('_on_') or attr_name.startswith('_handle_'):
            if attr_name not in methods_to_patch and hasattr(gui_instance, attr_name):
                attr = getattr(gui_instance, attr_name)
                if callable(attr) and not isinstance(attr, types.BuiltinFunctionType):
                    # Create a safe wrapper for this event handler
                    @functools.wraps(attr)
                    def safe_handler(*args, **kwargs):
                        try:
                            # Check if root exists before proceeding
                            if not check_root_exists():
                                logging.warning(f"Cannot execute {attr.__name__}: root window no longer exists")
                                return None
                            
                            return attr(*args, **kwargs)
                        except tk.TclError as e:
                            if "invalid command name" in str(e):
                                logging.warning(f"Caught invalid command name error in {attr.__name__}: {str(e)}")
                                return None
                            else:
                                logging.error(f"Uncaught TclError in {attr.__name__}: {str(e)}")
                                raise
                    
                    # Apply the safe handler
                    setattr(gui_instance, attr_name, types.MethodType(safe_handler, gui_instance))
                    logging.info(f"General safety applied to {attr_name}")
    
    logging.info("Enhanced widget safety mechanisms applied to GUI")
