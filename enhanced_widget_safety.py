"""
Enhanced Widget Safety for Tkinter Applications
This file contains enhanced safety mechanisms to prevent 'invalid command name' errors
when widgets are destroyed during callbacks or threading operations.
"""

import logging
import tkinter as tk
from tkinter import messagebox
import functools
import threading

def safe_widget_call(widget_check_func):
    """
    Decorator to ensure widget existence before executing a method.
    
    Args:
        widget_check_func: Function that checks if required widgets exist
        
    Returns:
        Decorated function that only executes if widgets exist
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                # Check if widgets exist
                if not widget_check_func(self):
                    logging.warning(f"Cannot execute {func.__name__}: required widgets no longer exist")
                    return None
                
                # Execute the original function
                return func(self, *args, **kwargs)
            except tk.TclError as e:
                logging.error(f"TclError in {func.__name__}: {str(e)}")
                return None
        return wrapper
    return decorator

def patch_gui_with_enhanced_safety(gui_instance):
    """
    Apply enhanced widget safety to a GUI instance.
    
    Args:
        gui_instance: The GUI instance to patch
    """
    # Define widget check functions for different scenarios
    def check_root_exists(instance):
        return hasattr(instance, 'root') and instance.root.winfo_exists()
    
    def check_notebook_exists(instance):
        return (hasattr(instance, 'root') and instance.root.winfo_exists() and
                hasattr(instance, 'chart_notebook') and instance.chart_notebook.winfo_exists())
    
    def check_ticker_listbox_exists(instance):
        return (hasattr(instance, 'root') and instance.root.winfo_exists() and
                hasattr(instance, 'ticker_listbox') and instance.ticker_listbox.winfo_exists())
    
    # Patch _on_tab_changed method if it exists
    if hasattr(gui_instance, '_on_tab_changed'):
        original_on_tab_changed = gui_instance._on_tab_changed
        gui_instance._on_tab_changed = safe_widget_call(check_notebook_exists)(original_on_tab_changed)
        logging.info("Enhanced safety applied to _on_tab_changed")
    
    # Patch _get_selected_tickers method if it exists
    if hasattr(gui_instance, '_get_selected_tickers'):
        original_get_selected_tickers = gui_instance._get_selected_tickers
        gui_instance._get_selected_tickers = safe_widget_call(check_ticker_listbox_exists)(original_get_selected_tickers)
        logging.info("Enhanced safety applied to _get_selected_tickers")
    
    # Add a global exception handler for Tkinter callbacks
    def global_tk_exception_handler(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except tk.TclError as e:
                if "invalid command name" in str(e):
                    logging.warning(f"Caught invalid command name error in {func.__name__}: {str(e)}")
                    return None
                else:
                    logging.error(f"Uncaught TclError in {func.__name__}: {str(e)}")
                    raise
        return wrapper
    
    # Apply global exception handler to all methods that might be callbacks
    for attr_name in dir(gui_instance):
        if attr_name.startswith('_on_') or attr_name.startswith('_handle_'):
            attr = getattr(gui_instance, attr_name)
            if callable(attr):
                setattr(gui_instance, attr_name, global_tk_exception_handler(attr))
                logging.info(f"Enhanced safety applied to {attr_name}")
    
    # Patch thread-related methods to ensure widget safety
    if hasattr(gui_instance, '_generate_chart_thread'):
        original_generate_chart_thread = gui_instance._generate_chart_thread
        
        @functools.wraps(original_generate_chart_thread)
        def safe_generate_chart_thread(*args, **kwargs):
            if not check_root_exists(gui_instance):
                logging.warning("Cannot generate chart: root window no longer exists")
                return None
            
            try:
                return original_generate_chart_thread(*args, **kwargs)
            except tk.TclError as e:
                logging.error(f"TclError in chart generation thread: {str(e)}")
                return None
        
        gui_instance._generate_chart_thread = safe_generate_chart_thread
        logging.info("Enhanced safety applied to _generate_chart_thread")
    
    logging.info("Enhanced widget safety mechanisms applied to GUI")
