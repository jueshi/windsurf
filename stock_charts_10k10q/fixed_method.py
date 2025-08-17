def _get_selected_tickers(self):
    """Get selected tickers from listbox"""
    try:
        # Check if root window and ticker_listbox still exist
        if not hasattr(self, 'root') or not self.root.winfo_exists():
            logging.warning("Cannot get selected tickers: root window no longer exists")
            return []
            
        if not hasattr(self, 'ticker_listbox') or not self.ticker_listbox.winfo_exists():
            logging.warning("Cannot get selected tickers: ticker_listbox no longer exists")
            return []
            
        selected_indices = self.ticker_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select at least one ticker.")
            return []

        selected_tickers = []
        for i in selected_indices:
            try:
                # Extract ticker symbol (it might include a comment after a dash)
                ticker_text = self.ticker_listbox.get(i)
                ticker = ticker_text.split(' - ')[0].strip()
                selected_tickers.append(ticker)
            except tk.TclError as e:
                logging.error(f"TclError accessing ticker at index {i}: {str(e)}")
                continue

        return selected_tickers
        
    except tk.TclError as e:
        logging.error(f"TclError getting selected tickers: {str(e)}")
        return []
