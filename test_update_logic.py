import logging
import sys
import pandas as pd
from datetime import date
from data_rechiever import StockDataManager

# Configure logging to display all messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def test_update_with_gaps():
    """Test the enhanced update logic that handles gaps in data"""
    
    # Create a stock data manager instance
    manager = StockDataManager()
    
    # Test tickers
    tickers = ['MSFT', 'AAPL', 'NVDA']
    
    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"Testing update logic for {ticker}")
        print(f"{'='*50}")
        
        # Load existing data
        existing_data = manager.load_data(ticker)
        if existing_data is not None:
            print(f"Existing data shape: {existing_data.shape}")
            print(f"Date range: {existing_data['Date'].min()} to {existing_data['Date'].max()}")
            
            # Check for gaps in the data
            existing_data['Date'] = pd.to_datetime(existing_data['Date'])
            date_range = pd.date_range(start=existing_data['Date'].min(), end=existing_data['Date'].max(), freq='B')
            existing_dates = set(existing_data['Date'].dt.date)
            all_business_dates = set(date.date() for date in date_range)
            missing_dates = all_business_dates - existing_dates
            
            if missing_dates:
                print(f"Found {len(missing_dates)} missing dates in the data")
                print(f"First 5 missing dates: {sorted(list(missing_dates))[:5]}")
            else:
                print("No missing dates found in the data")
        else:
            print(f"No existing data found for {ticker}")
        
        # Update the data
        print(f"\nUpdating data for {ticker}...")
        updated_data = manager.update_data(ticker)
        
        # Check the updated data
        if updated_data is not None:
            print(f"Updated data shape: {updated_data.shape}")
            if isinstance(updated_data, pd.DataFrame) and 'Date' in updated_data.columns:
                print(f"Date range: {updated_data['Date'].min()} to {updated_data['Date'].max()}")
                
                # Check for gaps in the updated data
                updated_data['Date'] = pd.to_datetime(updated_data['Date'])
                date_range = pd.date_range(start=updated_data['Date'].min(), end=updated_data['Date'].max(), freq='B')
                updated_dates = set(updated_data['Date'].dt.date)
                all_business_dates = set(date.date() for date in date_range)
                missing_dates = all_business_dates - updated_dates
                
                if missing_dates:
                    print(f"Found {len(missing_dates)} missing dates in the updated data")
                    print(f"First 5 missing dates: {sorted(list(missing_dates))[:5]}")
                else:
                    print("No missing dates found in the updated data")
            else:
                print("Could not analyze date range in updated data")
        else:
            print(f"Update failed for {ticker}")

if __name__ == "__main__":
    test_update_with_gaps()
