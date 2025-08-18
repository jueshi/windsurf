"""
Data Manager Patch for StockDataManager
This file contains a wrapper method for StockDataManager to handle missing get_data method calls.
"""

import logging
import pandas as pd

def patch_stock_data_manager(manager_instance):
    """
    Patch the StockDataManager instance with a get_data method.
    
    Args:
        manager_instance: Instance of StockDataManager to patch
    """
    def get_data(ticker, start_date=None, end_date=None):
        """
        Wrapper method to handle get_data calls by redirecting to appropriate methods.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str, optional): Start date for data retrieval
            end_date (str, optional): End date for data retrieval
            
        Returns:
            pd.DataFrame: Stock data for the specified ticker
        """
        logging.info(f"get_data wrapper called for {ticker}")
        
        try:
            # Try to use _load_stock_data if it exists
            if hasattr(manager_instance, '_load_stock_data'):
                logging.info(f"Redirecting to _load_stock_data for {ticker}")
                return manager_instance._load_stock_data(ticker)
                
            # Try to use download_stock_data if it exists
            elif hasattr(manager_instance, 'download_stock_data'):
                logging.info(f"Redirecting to download_stock_data for {ticker}")
                return manager_instance.download_stock_data(ticker, start_date, end_date)
                
            # If neither method exists, try to find any method that might return stock data
            else:
                for method_name in dir(manager_instance):
                    if method_name.startswith('_') and ('load' in method_name or 'get' in method_name) and 'data' in method_name:
                        method = getattr(manager_instance, method_name)
                        if callable(method):
                            try:
                                logging.info(f"Trying alternative method {method_name} for {ticker}")
                                result = method(ticker)
                                if isinstance(result, pd.DataFrame) and not result.empty:
                                    return result
                            except Exception as e:
                                logging.warning(f"Method {method_name} failed: {str(e)}")
                
                # If all else fails, raise an informative error
                raise AttributeError(f"Could not find a suitable method to get data for {ticker}")
                
        except Exception as e:
            logging.error(f"Error in get_data wrapper: {str(e)}")
            # Return empty DataFrame as fallback
            return pd.DataFrame()
    
    # Add the get_data method to the manager instance
    manager_instance.get_data = get_data
    
    logging.info("StockDataManager patched with get_data method wrapper")
