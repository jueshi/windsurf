"""
Thread-safe Tkinter update utilities for the Stock Charts application.
This module provides functions to safely update Tkinter widgets from non-main threads.
"""

import tkinter as tk
import threading
import queue
import logging
import functools

# Queue for thread-safe Tkinter operations
tk_update_queue = queue.Queue()

# Bridge queue for root.after() calls made from worker threads. Calling
# after() from a non-main thread creates a Tcl timer from the wrong thread,
# which crashes with "Tcl_AsyncDelete: async handler deleted by the wrong
# thread". Bridged calls are re-scheduled on the main thread by the poller.
_after_bridge_queue = queue.Queue()
_original_after = None


def _patched_after(self, ms, func=None, *args):
    """Thread-safe replacement for tk.Misc.after.

    Main-thread calls behave exactly like the original (same timer id
    returned). Calls from worker threads with a callback are pushed onto the
    bridge queue; the main-thread poller re-schedules them with the real
    after(). (Timer ids from worker threads are not returned — legacy worker
    call sites never cancel them.)
    """
    if threading.current_thread() is threading.main_thread() or func is None:
        return _original_after(self, ms, func, *args)
    _after_bridge_queue.put((ms, func, args))
    return None

def setup_thread_safe_tkinter(root):
    """
    Set up thread-safe Tkinter updates by scheduling periodic queue processing.
    
    Args:
        root: The Tkinter root window
    """
    global _original_after
    # Patch after() once so any worker-thread call is bridged to the main
    # thread instead of creating a Tcl timer from the wrong thread.
    if _original_after is None:
        _original_after = tk.Misc.after
        tk.Misc.after = _patched_after

    def process_tk_queue():
        """Process all pending Tkinter update requests in the queue."""
        try:
            # Re-schedule any after() calls that arrived from worker threads
            for _ in range(_after_bridge_queue.qsize()):
                try:
                    ms, func, args = _after_bridge_queue.get_nowait()
                    _original_after(root, ms, func, *args)
                except queue.Empty:
                    break
                except Exception as e:
                    logging.error(f"Error bridging after() call: {e}")

            # Process all current items in the queue
            for _ in range(tk_update_queue.qsize()):
                try:
                    func, args, kwargs = tk_update_queue.get_nowait()
                    try:
                        # Log what we're about to do for debugging
                        func_name = getattr(func, '__name__', str(func))
                        logging.debug(f"Processing Tkinter queue item: {func_name} with args: {args}")
                        
                        # Check for text widget operations with string indices
                        if func_name in ('delete', 'insert') and args:
                            # For text widget operations, validate string indices
                            if isinstance(args[0], str) and '.' in args[0]:
                                try:
                                    line, char = args[0].split('.')
                                    int(line)  # Validate line is an integer
                                    int(char)  # Validate char is an integer
                                except (ValueError, IndexError) as e:
                                    logging.error(f"Invalid text position format in queue: {args[0]}, error: {e}")
                                    tk_update_queue.task_done()
                                    continue
                        
                        # Execute the function
                        func(*args, **kwargs)
                    except IndexError as e:
                        logging.error(f"String index error in Tkinter update: {e}, func: {getattr(func, '__name__', str(func))}, args: {args}")
                    except Exception as e:
                        logging.error(f"Error in Tkinter update: {e}, func: {getattr(func, '__name__', str(func))}")
                    finally:
                        tk_update_queue.task_done()
                except queue.Empty:
                    break
        except Exception as e:
            logging.error(f"Error processing Tkinter queue: {e}")
        
        # Schedule next queue check
        if root.winfo_exists():
            root.after(100, process_tk_queue)
    
    # Start the queue processing
    root.after(100, process_tk_queue)

def safe_update_text_widget(text_widget, operation, *args, **kwargs):
    """
    Safely update a Tkinter Text widget from any thread.
    
    Args:
        text_widget: The Tkinter Text widget to update
        operation: The operation to perform ('delete', 'insert', etc.)
        *args, **kwargs: Arguments to pass to the operation
    """
    # Check if widget exists
    try:
        if not text_widget or not hasattr(text_widget, 'winfo_exists') or not text_widget.winfo_exists():
            logging.warning(f"Attempted to update a non-existent text widget with operation: {operation}")
            return
    except Exception as e:
        logging.error(f"Error checking if widget exists: {e}")
        return
    
    # Validate string indices for text operations
    try:
        if operation == 'delete' or operation == 'insert':
            # Ensure we have at least one argument for position
            if not args:
                logging.error(f"Missing position argument for {operation} operation")
                return
            
            # For Tkinter text positions like "1.0", validate format
            if isinstance(args[0], str) and '.' in args[0]:
                try:
                    line, char = args[0].split('.')
                    int(line)  # Validate line is an integer
                    int(char)  # Validate char is an integer
                except (ValueError, IndexError):
                    logging.error(f"Invalid text position format: {args[0]}")
                    return
    except Exception as e:
        logging.error(f"Error validating arguments for {operation}: {e}")
        return
        
    if threading.current_thread() is threading.main_thread():
        # If we're in the main thread, perform the operation directly
        try:
            if operation == 'delete':
                text_widget.delete(*args, **kwargs)
            elif operation == 'insert':
                text_widget.insert(*args, **kwargs)
            elif operation == 'config':
                text_widget.config(*args, **kwargs)
            elif operation == 'see':
                text_widget.see(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in text widget {operation} operation: {e}")
    else:
        # If we're in a background thread, queue the operation
        try:
            if operation == 'delete':
                tk_update_queue.put((text_widget.delete, args, kwargs))
            elif operation == 'insert':
                tk_update_queue.put((text_widget.insert, args, kwargs))
            elif operation == 'config':
                tk_update_queue.put((text_widget.config, args, kwargs))
            elif operation == 'see':
                tk_update_queue.put((text_widget.see, args, kwargs))
        except Exception as e:
            logging.error(f"Error queuing text widget {operation} operation: {e}")

def safe_update_status(status_var, message):
    """
    Safely update a Tkinter StringVar (typically used for status messages) from any thread.
    
    Args:
        status_var: The Tkinter StringVar to update
        message: The message to set
    """
    if threading.current_thread() is threading.main_thread():
        status_var.set(message)
    else:
        tk_update_queue.put((status_var.set, (message,), {}))

def safe_show_message(message_type, title, message):
    """
    Safely show a message dialog from any thread.
    
    Args:
        message_type: The type of message ('error', 'info', 'warning')
        title: The dialog title
        message: The message to display
    """
    import tkinter.messagebox as messagebox
    
    if threading.current_thread() is threading.main_thread():
        if message_type == 'error':
            messagebox.showerror(title, message)
        elif message_type == 'info':
            messagebox.showinfo(title, message)
        elif message_type == 'warning':
            messagebox.showwarning(title, message)
    else:
        if message_type == 'error':
            tk_update_queue.put((messagebox.showerror, (title, message), {}))
        elif message_type == 'info':
            tk_update_queue.put((messagebox.showinfo, (title, message), {}))
        elif message_type == 'warning':
            tk_update_queue.put((messagebox.showwarning, (title, message), {}))

def thread_safe(widget_method):
    """
    Decorator to make a Tkinter widget method thread-safe.
    
    Args:
        widget_method: The method to make thread-safe
    
    Returns:
        A thread-safe version of the method
    """
    @functools.wraps(widget_method)
    def wrapper(self, *args, **kwargs):
        if threading.current_thread() is threading.main_thread():
            return widget_method(self, *args, **kwargs)
        else:
            tk_update_queue.put((widget_method, (self,) + args, kwargs))
    return wrapper
