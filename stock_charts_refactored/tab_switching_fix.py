"""
Tab switching fix to prevent timeframe tab from switching back to individual chart tab.
This module patches the _display_chart method to respect the current active tab.
"""

import logging
import types

def apply_tab_switching_fix(app):
    """
    Apply the tab switching fix to prevent timeframe tab from switching back to individual chart tab.
    
    Args:
        app: The GUI application instance
    """
    logging.info("Applying tab switching fix...")
    
    try:
        # Store the original _display_chart method
        original_display_chart = app._display_chart
        
        # Define the enhanced _display_chart method
        def enhanced_display_chart(self, ticker_or_path):
            """Enhanced display chart method that respects the current active tab"""
            try:
                # Get the current active tab
                current_tab = getattr(self, 'active_tab', 'individual')
                logging.info(f"Current active tab before chart display: {current_tab}")
                
                # Get data for the ticker
                ticker = ticker_or_path
                if not ticker or not isinstance(ticker, str):
                    logging.error(f"Invalid ticker: {ticker}")
                    return
                
                df = None
                if hasattr(self.manager, 'get_data'):
                    df = self.manager.get_data(ticker)
                else:
                    df = self.manager.data.get(ticker)
                    
                if df is None or df.empty:
                    self.status_var.set(f"No data available for {ticker}")
                    return
                
                # Store the current chart ticker
                self.current_chart_ticker = ticker
                
                # Try to create an interactive Plotly chart
                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots
                    import tempfile
                    import os
                    
                    # Create a subplot with 2 rows for price and volume
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                       vertical_spacing=0.03, 
                                       row_heights=[0.7, 0.3],
                                       subplot_titles=(f"{ticker} Stock Price", "Volume"))
                    
                    # Add price candlestick chart
                    fig.add_trace(
                        go.Candlestick(
                            x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name=ticker
                        ),
                        row=1, col=1
                    )
                    
                    # Add volume bar chart
                    fig.add_trace(
                        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='rgba(0,0,255,0.5)'),
                        row=2, col=1
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=f'{ticker} Stock Price',
                        xaxis_title='Date',
                        yaxis_title='Price ($)',
                        yaxis2_title='Volume',
                        height=600,
                        xaxis_rangeslider_visible=False
                    )
                    
                    # Display the interactive chart - IMPORTANT: Use the current active tab, not hardcoded "individual"
                    self._display_plotly_chart(fig, tab=current_tab)
                    self.status_var.set(f"Generated interactive chart for {ticker}")
                    
                except Exception as e:
                    # If Plotly chart fails, fall back to static chart
                    logging.error(f"Error creating Plotly chart for {ticker}: {str(e)}")
                    self.status_var.set(f"Using static chart for {ticker} due to error: {str(e)}")
                    
                    # Generate static chart as fallback
                    plots_dir = self.manager.plot_save_path
                    chart_path = os.path.join(plots_dir, f"{ticker}_daily_weekly_monthly.png")
                    
                    if os.path.exists(chart_path):
                        self._display_static_chart(chart_path)
                    else:
                        self.status_var.set(f"Error: No chart available for {ticker}")
            
            except Exception as e:
                chart_name = ticker_or_path
                error_msg = f"Error displaying chart for {chart_name}: {str(e)}"
                self.status_var.set(error_msg)
                logging.error(error_msg)
                from tkinter import messagebox
                messagebox.showerror("Error", error_msg)
        
        # Replace the original method with the enhanced one
        app._display_chart = types.MethodType(enhanced_display_chart, app)
        
        logging.info("Tab switching fix applied successfully")
        return True
    except Exception as e:
        logging.error(f"Error applying tab switching fix: {str(e)}")
        return False
