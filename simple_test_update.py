import logging
import sys
from data_rechiever import StockDataManager

# Configure logging to display all messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def test_update_logic():
    """Test the enhanced update logic that handles gaps in data"""
    
    # Create a stock data manager instance
    manager = StockDataManager()
    
    # Test ticker
    ticker = 'MSFT'
    
    print(f"\nTesting update logic for {ticker}...")
    
    # Update the data
    updated_data = manager.update_data(ticker)
    
    if updated_data is not None:
        print(f"Update successful for {ticker}")
        print(f"Data shape: {updated_data.shape}")
        if 'Date' in updated_data.columns:
            print(f"Date range: {updated_data['Date'].min()} to {updated_data['Date'].max()}")
    else:
        print(f"Update failed for {ticker}")

if __name__ == "__main__":
    test_update_logic()
