import tkinter as tk
from tkinter import ttk
import calendar
import datetime

class CustomDateEntry(ttk.Frame):
    """A custom date entry widget that doesn't require tkcalendar package"""
    
    def __init__(self, master=None, textvariable=None, width=12, 
                 date_pattern='yyyy-mm-dd', background='darkblue', 
                 foreground='white', borderwidth=2, locale='en_US', **kw):
        super().__init__(master, **kw)
        
        # Store parameters
        self.date_pattern = date_pattern
        self.background = background
        self.foreground = foreground
        self.borderwidth = borderwidth
        self.locale = locale
        
        # Create a StringVar if one is not provided
        self.textvariable = textvariable if textvariable else tk.StringVar()
        
        # Set default date to today
        today = datetime.datetime.now()
        default_date = today.strftime('%Y-%m-%d')
        if not self.textvariable.get():
            self.textvariable.set(default_date)
        
        # Create entry widget
        self.entry = ttk.Entry(self, width=width, textvariable=self.textvariable)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Create button to open calendar popup
        self.calendar_button = ttk.Button(self, text="📅", width=3, 
                                         command=self._show_calendar)
        self.calendar_button.pack(side=tk.LEFT)
        
        # Calendar popup window
        self.top = None
        
    def _show_calendar(self):
        """Show a popup calendar for date selection"""
        if self.top and self.top.winfo_exists():
            return
        
        # Get current date from entry
        try:
            current_date = datetime.datetime.strptime(self.textvariable.get(), '%Y-%m-%d')
        except ValueError:
            current_date = datetime.datetime.now()
        
        year, month, day = current_date.year, current_date.month, current_date.day
        
        # Create popup window
        self.top = tk.Toplevel(self)
        self.top.title("Select Date")
        self.top.geometry("+%d+%d" % (self.winfo_rootx(), self.winfo_rooty() + 30))
        self.top.grab_set()
        self.top.transient(self)
        
        # Year and month selection
        nav_frame = ttk.Frame(self.top)
        nav_frame.pack(fill=tk.X)
        
        prev_year = ttk.Button(nav_frame, text="<<", width=5, 
                              command=lambda: self._change_year(-1))
        prev_year.pack(side=tk.LEFT)
        
        prev_month = ttk.Button(nav_frame, text="<", width=5,
                               command=lambda: self._change_month(-1))
        prev_month.pack(side=tk.LEFT)
        
        self.month_year_label = ttk.Label(nav_frame, width=15)
        self.month_year_label.pack(side=tk.LEFT, expand=True)
        
        next_month = ttk.Button(nav_frame, text=">", width=5,
                               command=lambda: self._change_month(1))
        next_month.pack(side=tk.LEFT)
        
        next_year = ttk.Button(nav_frame, text=">>", width=5,
                              command=lambda: self._change_year(1))
        next_year.pack(side=tk.LEFT)
        
        # Days of week
        days_frame = ttk.Frame(self.top)
        days_frame.pack(fill=tk.X)
        
        for i, day_name in enumerate(['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']):
            ttk.Label(days_frame, text=day_name, width=3).grid(row=0, column=i)
        
        # Calendar frame for days
        self.cal_frame = ttk.Frame(self.top)
        self.cal_frame.pack(fill=tk.BOTH, expand=True)
        
        # Store current displayed year and month
        self.displayed_year = year
        self.displayed_month = month
        
        # Draw the calendar
        self._draw_calendar(year, month, day)
        
    def _draw_calendar(self, year, month, selected_day=None):
        """Draw the calendar for the specified year and month"""
        # Update label
        month_name = calendar.month_name[month]
        self.month_year_label.config(text=f"{month_name} {year}")
        
        # Clear previous calendar
        for widget in self.cal_frame.winfo_children():
            widget.destroy()
        
        # Get calendar for month
        cal = calendar.monthcalendar(year, month)
        
        # Draw days
        for week_idx, week in enumerate(cal):
            for day_idx, day in enumerate(week):
                if day == 0:
                    # Empty cell for days not in this month
                    ttk.Label(self.cal_frame, text="", width=3).grid(row=week_idx+1, column=day_idx)
                else:
                    # Create button for each day
                    btn = ttk.Button(self.cal_frame, text=str(day), width=3,
                                    command=lambda d=day: self._select_date(year, month, d))
                    
                    # Highlight selected day
                    if day == selected_day:
                        btn.state(['pressed'])
                        
                    btn.grid(row=week_idx+1, column=day_idx)
    
    def _select_date(self, year, month, day):
        """Select a date and close the calendar"""
        selected_date = datetime.date(year, month, day)
        self.textvariable.set(selected_date.strftime('%Y-%m-%d'))
        if self.top:
            self.top.destroy()
            self.top = None
    
    def _change_month(self, delta):
        """Change the displayed month by delta"""
        month = self.displayed_month + delta
        year = self.displayed_year
        
        if month > 12:
            month = 1
            year += 1
        elif month < 1:
            month = 12
            year -= 1
            
        self.displayed_year = year
        self.displayed_month = month
        
        # Redraw calendar
        self._draw_calendar(year, month)
        
    def _change_year(self, delta):
        """Change the displayed year by delta"""
        self.displayed_year += delta
        self._draw_calendar(self.displayed_year, self.displayed_month)
        
    def get_date(self):
        """Return the selected date as a datetime.date object"""
        try:
            date_str = self.textvariable.get()
            return datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            return datetime.date.today()
