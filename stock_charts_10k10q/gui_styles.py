"""
GUI Styles Module - Centralized styling for Stock Analysis Tool

This module defines colors, fonts, spacing, and ttk styles for a modern,
professional appearance.
"""

import tkinter as tk
from tkinter import ttk
import sys


# =============================================================================
# COLOR PALETTE
# =============================================================================

# Theme state - can be 'light' or 'dark'
_current_theme = 'light'

class Colors:
    """Color constants for the application. Supports light and dark themes."""
    
    # Light theme colors
    _LIGHT = {
        'PRIMARY': "#1a365d",
        'PRIMARY_LIGHT': "#3b82f6",
        'PRIMARY_DARK': "#0f172a",
        'ACCENT': "#0d9488",
        'ACCENT_LIGHT': "#14b8a6",
        'SUCCESS': "#16a34a",
        'WARNING': "#f59e0b",
        'ERROR': "#dc2626",
        'INFO': "#3b82f6",
        'BACKGROUND': "#f9fafb",
        'BACKGROUND_ALT': "#f3f4f6",
        'SURFACE': "#ffffff",
        'BORDER': "#e5e7eb",
        'BORDER_DARK': "#d1d5db",
        'TEXT_PRIMARY': "#1f2937",
        'TEXT_SECONDARY': "#6b7280",
        'TEXT_MUTED': "#9ca3af",
        'TEXT_INVERSE': "#ffffff",
        'TAB_ACTIVE': "#1a365d",
        'TAB_INACTIVE': "#e5e7eb",
        'TAB_HOVER': "#dbeafe",
        'BTN_PRIMARY_BG': "#1a365d",
        'BTN_PRIMARY_FG': "#ffffff",
        'BTN_SECONDARY_BG': "#6b7280",
        'BTN_SECONDARY_FG': "#ffffff",
        'BTN_TERTIARY_BG': "#ffffff",
        'BTN_TERTIARY_FG': "#1a365d",
    }
    
    # Dark theme colors
    _DARK = {
        'PRIMARY': "#60a5fa",
        'PRIMARY_LIGHT': "#93c5fd",
        'PRIMARY_DARK': "#3b82f6",
        'ACCENT': "#2dd4bf",
        'ACCENT_LIGHT': "#5eead4",
        'SUCCESS': "#4ade80",
        'WARNING': "#fbbf24",
        'ERROR': "#f87171",
        'INFO': "#60a5fa",
        'BACKGROUND': "#1f2937",
        'BACKGROUND_ALT': "#374151",
        'SURFACE': "#111827",
        'BORDER': "#4b5563",
        'BORDER_DARK': "#6b7280",
        'TEXT_PRIMARY': "#f9fafb",
        'TEXT_SECONDARY': "#d1d5db",
        'TEXT_MUTED': "#9ca3af",
        'TEXT_INVERSE': "#1f2937",
        'TAB_ACTIVE': "#3b82f6",
        'TAB_INACTIVE': "#374151",
        'TAB_HOVER': "#4b5563",
        'BTN_PRIMARY_BG': "#3b82f6",
        'BTN_PRIMARY_FG': "#ffffff",
        'BTN_SECONDARY_BG': "#6b7280",
        'BTN_SECONDARY_FG': "#ffffff",
        'BTN_TERTIARY_BG': "#374151",
        'BTN_TERTIARY_FG': "#60a5fa",
    }
    
    # Primary colors
    PRIMARY = "#1a365d"          # Deep blue - main brand color
    PRIMARY_LIGHT = "#3b82f6"    # Light blue - hover states
    PRIMARY_DARK = "#0f172a"     # Darker blue - pressed states
    
    # Accent colors
    ACCENT = "#0d9488"           # Teal - success, positive indicators
    ACCENT_LIGHT = "#14b8a6"     # Light teal - hover
    
    # Semantic colors
    SUCCESS = "#16a34a"          # Green - positive values
    WARNING = "#f59e0b"          # Amber - warnings, alerts
    ERROR = "#dc2626"            # Red - errors, negative values
    INFO = "#3b82f6"             # Blue - informational
    
    # Neutral colors
    BACKGROUND = "#f9fafb"       # Off-white - main background
    BACKGROUND_ALT = "#f3f4f6"   # Slightly darker - alternate sections
    SURFACE = "#ffffff"          # White - cards, panels
    BORDER = "#e5e7eb"           # Light gray - borders
    BORDER_DARK = "#d1d5db"      # Darker border - focus states
    
    # Text colors
    TEXT_PRIMARY = "#1f2937"     # Dark gray - main text
    TEXT_SECONDARY = "#6b7280"   # Medium gray - secondary text
    TEXT_MUTED = "#9ca3af"       # Light gray - disabled, hints
    TEXT_INVERSE = "#ffffff"     # White - text on dark backgrounds
    
    # Tab colors
    TAB_ACTIVE = "#1a365d"       # Active tab background
    TAB_INACTIVE = "#e5e7eb"     # Inactive tab background
    TAB_HOVER = "#dbeafe"        # Tab hover state
    
    # Button colors
    BTN_PRIMARY_BG = "#1a365d"
    BTN_PRIMARY_FG = "#ffffff"
    BTN_SECONDARY_BG = "#6b7280"
    BTN_SECONDARY_FG = "#ffffff"
    BTN_TERTIARY_BG = "#ffffff"
    BTN_TERTIARY_FG = "#1a365d"
    
    @classmethod
    def apply_theme(cls, theme='light'):
        """Apply a color theme (light or dark).
        
        Args:
            theme: 'light' or 'dark'
        """
        global _current_theme
        _current_theme = theme
        colors = cls._LIGHT if theme == 'light' else cls._DARK
        
        for key, value in colors.items():
            setattr(cls, key, value)
    
    @classmethod
    def get_current_theme(cls):
        """Get the current theme name."""
        return _current_theme
    
    @classmethod
    def toggle_theme(cls):
        """Toggle between light and dark themes."""
        new_theme = 'dark' if _current_theme == 'light' else 'light'
        cls.apply_theme(new_theme)
        return new_theme


# =============================================================================
# TYPOGRAPHY
# =============================================================================

class Fonts:
    """Font configurations for the application."""
    
    # Font family - use system fonts for best rendering
    if sys.platform == "win32":
        FAMILY = "Segoe UI"
        MONO_FAMILY = "Consolas"
    elif sys.platform == "darwin":
        FAMILY = "SF Pro Text"
        MONO_FAMILY = "Menlo"
    else:
        FAMILY = "DejaVu Sans"
        MONO_FAMILY = "DejaVu Sans Mono"
    
    # Font sizes
    SIZE_H1 = 24        # Main titles
    SIZE_H2 = 18        # Section headers
    SIZE_H3 = 14        # Subsection headers
    SIZE_BODY = 12      # Body text
    SIZE_SMALL = 11     # Captions, labels
    SIZE_TINY = 10      # Very small text
    
    # Font configurations (family, size, weight)
    @classmethod
    def h1(cls):
        return (cls.FAMILY, cls.SIZE_H1, "bold")
    
    @classmethod
    def h2(cls):
        return (cls.FAMILY, cls.SIZE_H2, "bold")
    
    @classmethod
    def h3(cls):
        return (cls.FAMILY, cls.SIZE_H3, "bold")
    
    @classmethod
    def body(cls):
        return (cls.FAMILY, cls.SIZE_BODY)
    
    @classmethod
    def body_bold(cls):
        return (cls.FAMILY, cls.SIZE_BODY, "bold")
    
    @classmethod
    def small(cls):
        return (cls.FAMILY, cls.SIZE_SMALL)
    
    @classmethod
    def mono(cls):
        return (cls.MONO_FAMILY, cls.SIZE_BODY)


# =============================================================================
# SPACING
# =============================================================================

class Spacing:
    """Spacing constants for consistent layout."""
    
    # Base spacing unit (4px)
    UNIT = 4
    
    # Spacing scale
    XS = 4      # Extra small
    SM = 8      # Small
    MD = 12     # Medium
    LG = 16     # Large (default padding)
    XL = 24     # Extra large (section gaps)
    XXL = 32    # Double extra large
    
    # Specific use cases
    PADDING = 16            # Default container padding
    GAP = 8                 # Gap between related elements
    SECTION_GAP = 24        # Gap between sections
    BUTTON_PADDING_X = 12   # Horizontal button padding
    BUTTON_PADDING_Y = 6    # Vertical button padding
    
    # Touch targets
    MIN_TOUCH_TARGET = 40   # Minimum clickable area


# =============================================================================
# TTK STYLE CONFIGURATION
# =============================================================================

def configure_styles(root):
    """Configure ttk styles for the application.
    
    Args:
        root: The root Tk window
    """
    style = ttk.Style(root)
    
    # Try to use a modern theme as base
    available_themes = style.theme_names()
    if 'clam' in available_themes:
        style.theme_use('clam')
    elif 'vista' in available_themes:
        style.theme_use('vista')
    
    # -------------------------------------------------------------------------
    # Frame styles
    # -------------------------------------------------------------------------
    style.configure("TFrame",
                   background=Colors.BACKGROUND)
    
    style.configure("Card.TFrame",
                   background=Colors.SURFACE,
                   relief="flat")
    
    style.configure("Toolbar.TFrame",
                   background=Colors.SURFACE)
    
    # -------------------------------------------------------------------------
    # Label styles
    # -------------------------------------------------------------------------
    style.configure("TLabel",
                   background=Colors.BACKGROUND,
                   foreground=Colors.TEXT_PRIMARY,
                   font=Fonts.body())
    
    style.configure("H1.TLabel",
                   font=Fonts.h1(),
                   foreground=Colors.TEXT_PRIMARY)
    
    style.configure("H2.TLabel",
                   font=Fonts.h2(),
                   foreground=Colors.TEXT_PRIMARY)
    
    style.configure("H3.TLabel",
                   font=Fonts.h3(),
                   foreground=Colors.TEXT_PRIMARY)
    
    style.configure("Secondary.TLabel",
                   foreground=Colors.TEXT_SECONDARY,
                   font=Fonts.small())
    
    style.configure("Muted.TLabel",
                   foreground=Colors.TEXT_MUTED,
                   font=Fonts.small())
    
    style.configure("Success.TLabel",
                   foreground=Colors.SUCCESS)
    
    style.configure("Error.TLabel",
                   foreground=Colors.ERROR)
    
    style.configure("Warning.TLabel",
                   foreground=Colors.WARNING)
    
    # -------------------------------------------------------------------------
    # Button styles
    # -------------------------------------------------------------------------
    # Primary button (main actions)
    style.configure("Primary.TButton",
                   font=Fonts.body_bold(),
                   padding=(Spacing.BUTTON_PADDING_X, Spacing.BUTTON_PADDING_Y))
    style.map("Primary.TButton",
             background=[("active", Colors.PRIMARY_LIGHT),
                        ("!disabled", Colors.PRIMARY)],
             foreground=[("!disabled", Colors.TEXT_INVERSE)])
    
    # Secondary button
    style.configure("Secondary.TButton",
                   font=Fonts.body(),
                   padding=(Spacing.BUTTON_PADDING_X, Spacing.BUTTON_PADDING_Y))
    style.map("Secondary.TButton",
             background=[("active", Colors.TEXT_SECONDARY),
                        ("!disabled", Colors.BTN_SECONDARY_BG)],
             foreground=[("!disabled", Colors.TEXT_INVERSE)])
    
    # Tertiary/outline button
    style.configure("Tertiary.TButton",
                   font=Fonts.body(),
                   padding=(Spacing.BUTTON_PADDING_X, Spacing.BUTTON_PADDING_Y),
                   background=Colors.SURFACE)
    style.map("Tertiary.TButton",
             background=[("active", Colors.TAB_HOVER)],
             foreground=[("!disabled", Colors.PRIMARY)])
    
    # Toolbar button (compact)
    style.configure("Toolbar.TButton",
                   font=Fonts.body(),
                   padding=(Spacing.SM, Spacing.XS))
    
    # -------------------------------------------------------------------------
    # Entry styles
    # -------------------------------------------------------------------------
    style.configure("TEntry",
                   font=Fonts.body(),
                   padding=Spacing.SM,
                   fieldbackground=Colors.SURFACE)
    style.map("TEntry",
             fieldbackground=[("focus", Colors.SURFACE)],
             bordercolor=[("focus", Colors.PRIMARY_LIGHT)])
    
    # -------------------------------------------------------------------------
    # Combobox styles
    # -------------------------------------------------------------------------
    style.configure("TCombobox",
                   font=Fonts.body(),
                   padding=Spacing.XS,
                   fieldbackground=Colors.SURFACE)
    
    # -------------------------------------------------------------------------
    # Notebook (tabs) styles
    # -------------------------------------------------------------------------
    style.configure("TNotebook",
                   background=Colors.BACKGROUND,
                   tabmargins=[2, 5, 2, 0])
    
    style.configure("TNotebook.Tab",
                   font=Fonts.body(),
                   padding=[Spacing.MD, Spacing.SM],
                   background=Colors.TAB_INACTIVE)
    style.map("TNotebook.Tab",
             background=[("selected", Colors.PRIMARY),
                        ("active", Colors.TAB_HOVER)],
             foreground=[("selected", Colors.TEXT_INVERSE)],
             expand=[("selected", [1, 1, 1, 0])])
    
    # -------------------------------------------------------------------------
    # LabelFrame styles
    # -------------------------------------------------------------------------
    style.configure("TLabelframe",
                   background=Colors.BACKGROUND,
                   foreground=Colors.TEXT_PRIMARY,
                   font=Fonts.body_bold())
    style.configure("TLabelframe.Label",
                   font=Fonts.body_bold(),
                   foreground=Colors.PRIMARY)
    
    # Card-style labelframe
    style.configure("Card.TLabelframe",
                   background=Colors.SURFACE,
                   relief="flat",
                   borderwidth=1)
    
    # -------------------------------------------------------------------------
    # Treeview styles
    # -------------------------------------------------------------------------
    style.configure("Treeview",
                   font=Fonts.body(),
                   rowheight=28,
                   background=Colors.SURFACE,
                   fieldbackground=Colors.SURFACE,
                   foreground=Colors.TEXT_PRIMARY)
    style.configure("Treeview.Heading",
                   font=Fonts.body_bold(),
                   background=Colors.BACKGROUND_ALT,
                   foreground=Colors.TEXT_PRIMARY)
    style.map("Treeview",
             background=[("selected", Colors.PRIMARY_LIGHT)],
             foreground=[("selected", Colors.TEXT_INVERSE)])
    
    # Custom treeview for data display
    style.configure("Data.Treeview",
                   font=Fonts.body(),
                   rowheight=32)
    style.configure("Data.Treeview.Heading",
                   font=Fonts.h3())
    
    # -------------------------------------------------------------------------
    # Scrollbar styles
    # -------------------------------------------------------------------------
    style.configure("TScrollbar",
                   background=Colors.BACKGROUND_ALT,
                   troughcolor=Colors.BACKGROUND,
                   borderwidth=0)
    
    # -------------------------------------------------------------------------
    # Separator styles
    # -------------------------------------------------------------------------
    style.configure("TSeparator",
                   background=Colors.BORDER)
    
    # Vertical separator for toolbar
    style.configure("Toolbar.TSeparator",
                   background=Colors.BORDER_DARK)
    
    # -------------------------------------------------------------------------
    # Checkbutton styles
    # -------------------------------------------------------------------------
    style.configure("TCheckbutton",
                   font=Fonts.body(),
                   background=Colors.BACKGROUND,
                   foreground=Colors.TEXT_PRIMARY)
    
    # -------------------------------------------------------------------------
    # Menubutton styles
    # -------------------------------------------------------------------------
    style.configure("TMenubutton",
                   font=Fonts.body(),
                   padding=(Spacing.SM, Spacing.XS))
    
    # -------------------------------------------------------------------------
    # PanedWindow styles
    # -------------------------------------------------------------------------
    style.configure("TPanedwindow",
                   background=Colors.BORDER)
    
    # -------------------------------------------------------------------------
    # Progressbar styles
    # -------------------------------------------------------------------------
    style.configure("TProgressbar",
                   background=Colors.PRIMARY,
                   troughcolor=Colors.BACKGROUND_ALT)
    
    return style


# =============================================================================
# CUSTOM WIDGET STYLES
# =============================================================================

class TabButton(ttk.Frame):
    """Custom tab button with icon support and active state indicator."""
    
    def __init__(self, parent, text, icon="", command=None, active=False, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.command = command
        self._active = active
        
        # Configure frame
        self.configure(style="TFrame")
        
        # Create label for icon + text
        display_text = f"{icon} {text}" if icon else text
        self.label = ttk.Label(self, text=display_text, font=Fonts.body())
        self.label.pack(padx=Spacing.MD, pady=Spacing.SM)
        
        # Bind click events
        self.bind("<Button-1>", self._on_click)
        self.label.bind("<Button-1>", self._on_click)
        
        # Bind hover events
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.label.bind("<Enter>", self._on_enter)
        self.label.bind("<Leave>", self._on_leave)
        
        # Set initial state
        self._update_appearance()
    
    @property
    def active(self):
        return self._active
    
    @active.setter
    def active(self, value):
        self._active = value
        self._update_appearance()
    
    def _update_appearance(self):
        if self._active:
            self.configure(style="TFrame")
            # Use tk config for background since ttk doesn't support it directly
            for widget in [self, self.label]:
                widget.configure(style="TFrame")
            self.label.configure(foreground=Colors.PRIMARY, font=Fonts.body_bold())
        else:
            self.label.configure(foreground=Colors.TEXT_SECONDARY, font=Fonts.body())
    
    def _on_click(self, event):
        if self.command:
            self.command()
    
    def _on_enter(self, event):
        if not self._active:
            self.label.configure(foreground=Colors.PRIMARY_LIGHT)
    
    def _on_leave(self, event):
        self._update_appearance()


class MetricCard(ttk.Frame):
    """Card widget for displaying a metric with icon, label, and value."""
    
    def __init__(self, parent, icon, label, value, value_color=None, **kwargs):
        super().__init__(parent, style="Card.TFrame", **kwargs)
        
        # Icon
        icon_label = ttk.Label(self, text=icon, font=Fonts.h3())
        icon_label.grid(row=0, column=0, rowspan=2, padx=(Spacing.SM, Spacing.MD), pady=Spacing.SM)
        
        # Label
        label_widget = ttk.Label(self, text=label, style="Secondary.TLabel")
        label_widget.grid(row=0, column=1, sticky="w", padx=(0, Spacing.SM))
        
        # Value
        value_widget = ttk.Label(self, text=value, font=Fonts.body_bold())
        if value_color:
            value_widget.configure(foreground=value_color)
        value_widget.grid(row=1, column=1, sticky="w", padx=(0, Spacing.SM))
        
        self.columnconfigure(1, weight=1)


class StatusBar(ttk.Frame):
    """Status bar with message area and optional progress indicator."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.configure(style="TFrame")
        
        # Status icon
        self.icon_label = ttk.Label(self, text="✓", foreground=Colors.SUCCESS)
        self.icon_label.pack(side=tk.LEFT, padx=(Spacing.SM, Spacing.XS))
        
        # Status message
        self.message_var = tk.StringVar(value="Ready")
        self.message_label = ttk.Label(self, textvariable=self.message_var, style="Secondary.TLabel")
        self.message_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def set_message(self, message, status="info"):
        """Set status message with appropriate icon.
        
        Args:
            message: The status message text
            status: One of 'info', 'success', 'warning', 'error'
        """
        self.message_var.set(message)
        
        icons = {
            "info": ("ℹ", Colors.INFO),
            "success": ("✓", Colors.SUCCESS),
            "warning": ("⚠", Colors.WARNING),
            "error": ("✕", Colors.ERROR)
        }
        
        icon, color = icons.get(status, icons["info"])
        self.icon_label.configure(text=icon, foreground=color)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def apply_card_style(widget):
    """Apply card styling to a widget (white background, subtle border)."""
    widget.configure(style="Card.TFrame")


def create_separator(parent, orient="horizontal"):
    """Create a styled separator."""
    sep = ttk.Separator(parent, orient=orient)
    return sep


def create_toolbar_separator(parent):
    """Create a vertical separator for toolbars."""
    sep = ttk.Separator(parent, orient="vertical", style="Toolbar.TSeparator")
    return sep
