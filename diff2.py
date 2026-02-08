    def _refresh_ticker_lists(self):
        """Refresh ticker lists from ticker_lists.py"""
        # Remember current selection
        current_selection = self.ticker_list_var.get()
        
        # Reload ticker lists from module
        self._load_ticker_lists_from_module()
        
        # Update the dropdown values
        self.ticker_list_combo['values'] = list(self.ticker_lists.keys())
        
        # Restore previous selection if it still exists
        if current_selection and current_selection in self.ticker_lists:
            self.ticker_list_var.set(current_selection)
        
        # Also refresh watch list if it exists in ticker_lists.py
        try:
            import ticker_lists
            if hasattr(ticker_lists, 'watch_list'):
                self.watch_list = ticker_lists.watch_list.copy()
                # Update watch list display
                self.watch_listbox.delete(0, tk.END)
                for ticker in self.watch_list:
                    self.watch_listbox.insert(tk.END, ticker)
                logging.info(f"Refreshed watch list with {len(self.watch_list)} tickers")
        except Exception as e:
            logging.error(f"Error refreshing watch list: {e}")
        
        # Update status
        self.status_var.set(f"Refreshed {len(self.ticker_lists)} ticker lists from ticker_lists.py")
    