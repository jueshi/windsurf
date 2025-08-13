import os
import sys
import tempfile
sys.path.append('stock_charts_refactored')

from data_manager import StockDataManager
from final_timeframe_fix import apply_final_timeframe_fix

# Mock the GUI and its components to avoid needing a display
class MockTkWidget:
    def __init__(self):
        self._children = {}
    def winfo_children(self):
        return []
    def destroy(self):
        pass
    def pack(self, *args, **kwargs):
        pass
    def configure(self, *args, **kwargs):
        pass
    def tab(self, *args, **kwargs):
        return "Timeframe Charts" # Mock response
    def select(self, *args, **kwargs):
        pass

class MockApp:
    def __init__(self, manager):
        self.manager = manager
        self.chart_notebook = MockTkWidget()
        self.status_var = type("StringVar", (), {"set": lambda self, x: print(f"Status: {x}")})()

def verify_file_based_fix():
    """
    Test script to verify the file-based chart display fix without a real GUI.
    """
    try:
        # 1. Setup
        manager = StockDataManager(plot_save_path='test_charts')
        mock_app = MockApp(manager)

        # 2. Apply the patch
        print("Applying the final timeframe fix to mock app...")
        apply_final_timeframe_fix(mock_app)
        print("...fix applied.")

        # 3. Run the patched method
        ticker = "AAPL"
        print(f"Calling _generate_and_display_timeframe_charts for {ticker}...")
        # This will fail if the image generation or file writing fails
        mock_app._generate_and_display_timeframe_charts(ticker)
        print("...method called successfully.")

        # 4. Verify that the temporary image files were created
        temp_dir = tempfile.gettempdir()
        daily_chart_path = os.path.join(temp_dir, f"{ticker}_daily_image.png")
        weekly_chart_path = os.path.join(temp_dir, f"{ticker}_weekly_image.png")
        monthly_chart_path = os.path.join(temp_dir, f"{ticker}_monthly_image.png")

        charts_found = 0
        if os.path.exists(daily_chart_path):
            print(f"SUCCESS: Found daily chart temp file: {daily_chart_path}")
            charts_found += 1
            os.remove(daily_chart_path) # Clean up
        else:
            print(f"FAILURE: Did not find daily chart temp file: {daily_chart_path}")

        if os.path.exists(weekly_chart_path):
            print(f"SUCCESS: Found weekly chart temp file: {weekly_chart_path}")
            charts_found += 1
            os.remove(weekly_chart_path) # Clean up
        else:
            print(f"FAILURE: Did not find weekly chart temp file: {weekly_chart_path}")

        if os.path.exists(monthly_chart_path):
            print(f"SUCCESS: Found monthly chart temp file: {monthly_chart_path}")
            charts_found += 1
            os.remove(monthly_chart_path) # Clean up
        else:
            print(f"FAILURE: Did not find monthly chart temp file: {monthly_chart_path}")

        if charts_found == 3:
            print("\nVerification successful: All 3 temporary image files were created.")
        else:
            print(f"\nVerification failed: Only found {charts_found} out of 3 temporary image files.")

    except Exception as e:
        print(f"\nAn unexpected error occurred during verification: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    verify_file_based_fix()
