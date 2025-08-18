"""
Improved Data Manager Patch for StockDataManager
This file contains improved wrapper methods for StockDataManager to handle missing methods
and type conversion issues.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def patch_stock_data_manager(manager_instance):
    """
    Patch the StockDataManager instance with missing methods and improved type handling.
    
    Args:
        manager_instance: Instance of StockDataManager to patch
    """
    # Add get_data method
    def get_data(ticker, start_date=None, end_date=None):
        """
        Get stock data for a ticker with proper type handling.
        
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
                df = manager_instance._load_stock_data(ticker)
                
                # Ensure date column is properly formatted
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                
                # Ensure numeric columns are properly typed
                for col in df.columns:
                    if col != 'Date' and df[col].dtype == 'object':
                        try:
                            df[col] = pd.to_numeric(df[col])
                        except Exception as e:
                            logging.warning(f"Could not convert column {col} to numeric: {str(e)}")
                
                return df
                
            # Try to use download_stock_data if it exists
            elif hasattr(manager_instance, 'download_stock_data'):
                logging.info(f"Redirecting to download_stock_data for {ticker}")
                return manager_instance.download_stock_data(ticker, start_date, end_date)
                
            # If neither method exists, try to find any method that might return stock data
            else:
                for method_name in dir(manager_instance):
                    if ('load' in method_name or 'get' in method_name) and 'data' in method_name:
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
    
    # Add download_data method if it doesn't exist
    if not hasattr(manager_instance, 'download_data'):
        def download_data(ticker, start_date=None, end_date=None):
            """
            Download stock data for a ticker.
            
            Args:
                ticker (str): Stock ticker symbol
                start_date (str, optional): Start date for data retrieval
                end_date (str, optional): End date for data retrieval
                
            Returns:
                pd.DataFrame: Downloaded stock data
            """
            logging.info(f"download_data wrapper called for {ticker}")
            
            try:
                # Set default dates if not provided
                if not start_date:
                    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                if not end_date:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                
                # Try to use download_stock_data if it exists
                if hasattr(manager_instance, 'download_stock_data'):
                    logging.info(f"Redirecting to download_stock_data for {ticker}")
                    return manager_instance.download_stock_data(ticker, start_date, end_date)
                
                # Otherwise use yfinance directly
                logging.info(f"Using yfinance directly to download data for {ticker}")
                import yfinance as yf
                data = yf.download(ticker, start=start_date, end=end_date)
                
                # Reset index to make Date a column
                data.reset_index(inplace=True)
                
                return data
                
            except Exception as e:
                logging.error(f"Error in download_data wrapper: {str(e)}")
                # Return empty DataFrame as fallback
                return pd.DataFrame()
        
        manager_instance.download_data = download_data
        logging.info("Added download_data method to StockDataManager")
    
    # Add get_data method to the manager instance
    manager_instance.get_data = get_data
    logging.info("StockDataManager patched with improved get_data method wrapper")
    
    # Add type conversion utility method
    def ensure_numeric(df):
        """
        Ensure all non-date columns in a DataFrame are numeric.
        
        Args:
            df (pd.DataFrame): DataFrame to convert
            
        Returns:
            pd.DataFrame: DataFrame with numeric columns
        """
        if df is None or df.empty:
            return pd.DataFrame()
            
        result = df.copy()
        
        # Ensure date column is properly formatted
        if 'Date' in result.columns:
            result['Date'] = pd.to_datetime(result['Date'])
        
        # Convert all non-date columns to numeric
        for col in result.columns:
            if col != 'Date' and result[col].dtype == 'object':
                try:
                    result[col] = pd.to_numeric(result[col])
                except Exception as e:
                    logging.warning(f"Could not convert column {col} to numeric: {str(e)}")
        
        return result
    
    manager_instance.ensure_numeric = ensure_numeric
    logging.info("Added ensure_numeric utility method to StockDataManager")
