import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
TICKER = "CRDO"  # Credo Technology Group
START_DATE = "2025-01-01"
END_DATE = "2025-12-20"

# Visual tuning
LINE_WIDTH_PT = 1.3
MIN_BUBBLE_SIZE = LINE_WIDTH_PT ** 2  # match the visual weight of the trend line
MAX_BUBBLE_SIZE = 60  # keep bubbles noticeably smaller than before

# ==========================================

def plot_stock_scatter(ticker_symbol):
    # 1. Fetch Data
    print(f"Fetching market data for {ticker_symbol}...")
    df = yf.download(ticker_symbol, start=START_DATE, end=END_DATE, progress=False)
    
    if df.empty:
        print("No data found. Check the ticker or date range.")
        return

    # Reset index to access Date
    df = df.reset_index()

    # 2. Prepare Data for Plotting
    dates = df['Date']
    close_prices = df['Close'].to_numpy().ravel()
    open_prices = df['Open'].to_numpy().ravel()
    volumes = df['Volume'].to_numpy().ravel()

    # 3. Determine Colors (Green if Close > Open, else Red)
    #    We use list comprehension for efficiency
    colors = ['green' if close > open_ else 'red' for close, open_ in zip(close_prices, open_prices)]

    # 4. Determine Marker Sizes (Proportional to Volume)
    #    Use interpolation so the minimum bubble matches the trend-line thickness
    vol_min, vol_max = volumes.min(), volumes.max()
    if vol_min == vol_max:
        sizes = np.full_like(volumes, MIN_BUBBLE_SIZE, dtype=float)
    else:
        sizes = np.interp(volumes, (vol_min, vol_max), (MIN_BUBBLE_SIZE, MAX_BUBBLE_SIZE))

    # 5. Plotting
    plt.figure(figsize=(14, 7))

    # Draw closing-price trend line underneath the bubbles for continuity
    plt.plot(
        dates,
        close_prices,
        color='midnightblue',
        linewidth=LINE_WIDTH_PT,
        alpha=0.85,
        label='Close trend'
    )

    plt.scatter(
        dates, 
        close_prices, 
        s=sizes,        # Size based on Volume
        c=colors,       # Color based on Price Action
        alpha=0.6,      # Transparency to see overlaps
        edgecolors='black', 
        linewidth=0.5
    )

    # 6. Formatting
    plt.title(f'{ticker_symbol} Stock Price (Close) & Volume Analysis', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Close Price ($)')
    plt.grid(True, linestyle=':', alpha=0.5)

    # Format Date Axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)

    # Add a dummy legend for size/color interpretation
    # (Creating empty handles for the legend)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Close > Open', markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Close < Open', markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Size ∝ Volume', markersize=10, markeredgecolor='k'),
    ]
    plt.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.show()

# Run the function
if __name__ == "__main__":
    plot_stock_scatter(TICKER)