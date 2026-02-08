import yfinance as yf
import pandas as pd
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
TICKER_SYMBOL = "alab"  # Change to your desired stock
# ==========================================

def get_earnings_dates(symbol):
    print(f"Fetching earnings data for {symbol}...")
    
    try:
        ticker = yf.Ticker(symbol)
        
        # 1. Get the Calendar (Upcoming Earnings)
        # This usually returns a dictionary with dates and estimated EPS
        calendar = ticker.calendar
        
        print(f"\n--- Earnings Calendar for {symbol} ---")
        if calendar:
            # The structure of .calendar varies slightly by yfinance version
            # It is often a dictionary where keys are 'Earnings Date' or 'Earnings High', etc.
            # We convert it to a DataFrame for better display
            try:
                # Attempt to handle if it returns a dictionary of lists
                cal_df = pd.DataFrame(calendar).T
                print(cal_df)
                
                # specific extraction for the Date
                if 'Earnings Date' in calendar:
                    dates = calendar['Earnings Date']
                    print(f"\nNext Expected Earnings Date(s): {dates}")
            except Exception as e:
                # If dataframe conversion fails, just print raw
                print(calendar)
        else:
            print("No upcoming calendar data found.")

        # 2. Get Earnings Dates (Historical & Future Estimates)
        # This returns a DataFrame with 'Earnings Date' as the index
        print(f"\n--- Earnings History/Estimates for {symbol} ---")
        earnings_dates = ticker.earnings_dates
        
        if earnings_dates is not None and not earnings_dates.empty:
            # Sort by date descending (newest first)
            earnings_dates = earnings_dates.sort_index(ascending=False)
            
            # Show the next 2 and previous 4
            now = pd.Timestamp.now().tz_localize('UTC') # earnings_dates are usually UTC
            
            upcoming = earnings_dates[earnings_dates.index > now].tail(2)
            past = earnings_dates[earnings_dates.index <= now].head(4)
            
            print("\n[Upcoming/Recent]")
            combined = pd.concat([upcoming, past])
            print(combined[['EPS Estimate', 'Reported EPS', 'Surprise(%)']])
        else:
            print("No detailed earnings dates table found.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    get_earnings_dates(TICKER_SYMBOL)