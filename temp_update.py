def _on_ticker_selected(self, event):
        """Handle ticker selection event from main ticker listbox
        
        Args:
            event: The selection event
        """
        try:
            # Get selected ticker indices
            selected_indices = self.ticker_listbox.curselection()
            if not selected_indices:
                return
                
            # Get selected tickers
            selected_tickers = []
            for i in selected_indices:
                ticker_text = self.ticker_listbox.get(i)
                # Extract ticker symbol (it might have a comment after it)
                ticker = ticker_text.split(' ')[0].strip()
                selected_tickers.append(ticker)
                
            logging.info(f"Selected tickers from main list: {selected_tickers}")
            
            # Update chart based on active tab
            if hasattr(self, 'active_tab') and self.active_tab == "comparison":
                # If comparison tab is active, update comparison chart regardless of ticker count
                self._compare_percentage_performance()
            else:
                # Otherwise update individual chart for the first selected ticker
                if selected_tickers:
                    self._display_chart(selected_tickers[0])
        except Exception as e:
            logging.error(f"Error handling ticker selection: {e}")
    
    def _on_watch_ticker_selected(self, event):
        """Handle ticker selection event from watch list
        
        Args:
            event: The selection event
        """
        try:
            # Get selected ticker indices
            selected_indices = self.watch_listbox.curselection()
            if not selected_indices:
                return
                
            # Get selected tickers
            selected_tickers = []
            for i in selected_indices:
                ticker = self.watch_listbox.get(i).strip()
                selected_tickers.append(ticker)
                
            logging.info(f"Selected tickers from watch list: {selected_tickers}")
            
            # Update chart based on active tab
            if hasattr(self, 'active_tab') and self.active_tab == "comparison":
                # If comparison tab is active, update comparison chart regardless of ticker count
                self._compare_percentage_performance()
            else:
                # Otherwise update individual chart for the first selected ticker
                if selected_tickers:
                    self._display_chart(selected_tickers[0])
        except Exception as e:
            logging.error(f"Error handling watch ticker selection: {e}")
