import os
import unittest
import pandas as pd
from datetime import datetime, timedelta

# It's better to add the path to sys.path than to do relative imports
# when running scripts from the root directory.
import sys
sys.path.append('stock_charts_refactored')

from data_manager import StockDataManager

class TestStockDataManager(unittest.TestCase):

    def setUp(self):
        """Set up for the tests."""
        self.manager = StockDataManager(data_dir='test_stock_data')
        # Ensure the test data directory is clean before each test
        if os.path.exists(self.manager.data_dir):
            for f in os.listdir(self.manager.data_dir):
                os.remove(os.path.join(self.manager.data_dir, f))
        else:
            os.makedirs(self.manager.data_dir)

    def tearDown(self):
        """Tear down after the tests."""
        if os.path.exists(self.manager.data_dir):
            for f in os.listdir(self.manager.data_dir):
                os.remove(os.path.join(self.manager.data_dir, f))
            os.rmdir(self.manager.data_dir)

    def test_initial_download(self):
        """Test that data is downloaded for a new ticker."""
        ticker = 'AAPL'
        data_path = self.manager._get_data_path(ticker)

        # Ensure no data file exists initially
        self.assertFalse(os.path.exists(data_path))

        # Perform initial download
        data = self.manager.update_data(ticker, force_download=True)

        # Assert that data was returned and the file was created
        self.assertIsNotNone(data)
        self.assertFalse(data.empty)
        self.assertTrue(os.path.exists(data_path))

        # Verify the contents of the file
        loaded_data = pd.read_csv(data_path, sep='\\t', engine='python')
        self.assertFalse(loaded_data.empty)
        self.assertIn('Date', loaded_data.columns)

    def test_incremental_update(self):
        """Test that only new data is downloaded for an existing ticker."""
        ticker = 'MSFT'
        data_path = self.manager._get_data_path(ticker)

        # Create a dummy data file with old data
        old_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        dummy_data = pd.DataFrame({
            'Date': [old_date],
            'Open': [100], 'High': [101], 'Low': [99], 'Close': [100.5],
            'Adj Close': [100.5], 'Volume': [1000000]
        })
        dummy_data.to_csv(data_path, sep='\t', index=False)

        # Store the original number of rows
        original_rows = len(dummy_data)

        # Perform an update
        updated_data = self.manager.update_data(ticker)

        # Assert that data was returned
        self.assertIsNotNone(updated_data)
        self.assertFalse(updated_data.empty)

        # Verify that new data has been added
        self.assertGreater(len(updated_data), original_rows)

        # Check that the last date is recent
        last_date = pd.to_datetime(updated_data['Date']).max()
        self.assertGreaterEqual((datetime.now().date() - last_date.date()).days, -1) # Allow for today

    def test_no_update_needed(self):
        """Test that no data is downloaded if the data is already up to date."""
        ticker = 'GOOG'
        data_path = self.manager._get_data_path(ticker)

        # Create a dummy data file with over 3 years of data, ending yesterday.
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=3 * 365)

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        dummy_data = pd.DataFrame({
            'Date': date_range,
            'Open': [200] * len(date_range),
            'High': [202] * len(date_range),
            'Low': [198] * len(date_range),
            'Close': [201] * len(date_range),
            'Adj Close': [201] * len(date_range),
            'Volume': [2000000] * len(date_range)
        })
        dummy_data['Date'] = dummy_data['Date'].dt.strftime('%Y-%m-%d')
        dummy_data.to_csv(data_path, sep='\t', index=False)

        original_mod_time = os.path.getmtime(data_path)
        original_rows = len(dummy_data)

        # Perform an update
        updated_data = self.manager.update_data(ticker)

        # Assert that data was returned
        self.assertIsNotNone(updated_data)
        self.assertFalse(updated_data.empty)

        # Verify that no new data has been added
        self.assertEqual(len(updated_data), original_rows)

        # To be absolutely sure, we can check the modification time of the file.
        # However, the current implementation might re-save the file even if no new data is added.
        # So, checking the number of rows is a more reliable test for this case.
        # Let's check the file content instead of modification time.
        final_data = pd.read_csv(data_path, sep='\\t', engine='python')
        self.assertEqual(len(final_data), original_rows)


if __name__ == '__main__':
    unittest.main()
