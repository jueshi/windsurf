import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- 1. CONFIGURATION ---
TICKER = "MRVL"
START_DATE = "2024-01-01"

# --- 2. FETCH PRICE DATA ---
print(f"📥 Fetching price data for {TICKER}...")
df_price = yf.download(TICKER, start=START_DATE)

# yfinance can return MultiIndex columns even for a single ticker; flatten them
if isinstance(df_price.columns, pd.MultiIndex):
    df_price.columns = [col[0] or col[1] for col in df_price.columns]

# --- 3. CREATE INSTITUTIONAL OWNERSHIP DATA (SIMULATED) ---
# NOTE: Real historical ownership % is hard to get via free APIs. 
# We are using APPROXIMATE quarterly data points based on 13F trends for MRVL.
# You can update these values manually if you have exact FactSet/Bloomberg data.

ownership_data = {
    'Date': [
        '2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'
    ],
    'Inst_Ownership_Pct': [
        86.5,  # Q1
        87.2,  # Q2 (Accumulation begins)
        89.1,  # Q3 (Accelerated buying)
        91.3   # Q4 (Current High Conviction)
    ]
}

df_inst = pd.DataFrame(ownership_data)
df_inst['Date'] = pd.to_datetime(df_inst['Date'])
df_inst.set_index('Date', inplace=True)

# Resample ownership to daily to match stock price (forward fill the quarterly data)
# This creates a "Step" line showing when new 13F data hits the market.
df_price_reset = df_price.reset_index()
df_inst_reset = df_inst.reset_index()

df_combined = (
    pd.merge(df_price_reset, df_inst_reset, on='Date', how='outer')
      .sort_values('Date')
      .set_index('Date')
)
df_combined['Inst_Ownership_Pct'] = df_combined['Inst_Ownership_Pct'].ffill()
df_combined.dropna(subset=['Close'], inplace=True) # Drop future empty dates

# --- 4. GENERATE THE PLOT ---
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot A: Stock Price (Left Axis)
color_price = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Stock Price ($)', color=color_price, fontweight='bold')
ax1.plot(df_combined.index, df_combined['Close'], color=color_price, linewidth=1.5, label='Price')
ax1.tick_params(axis='y', labelcolor=color_price)
ax1.grid(True, which='major', linestyle='--', alpha=0.3)

# Plot B: Institutional Ownership (Right Axis)
ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
color_inst = 'tab:green'
ax2.set_ylabel('Institutional Ownership (%)', color=color_inst, fontweight='bold')
ax2.plot(df_combined.index, df_combined['Inst_Ownership_Pct'], color=color_inst, linewidth=3, linestyle='-', label='Smart Money %')
ax2.tick_params(axis='y', labelcolor=color_inst)
ax2.fill_between(df_combined.index, df_combined['Inst_Ownership_Pct'], 80, color='green', alpha=0.1) # Shading

# Title and Layout
plt.title(f"⚠️ SMART MONEY DIVERGENCE: {TICKER}\n(Price vs. Institutional Accumulation)", fontsize=14, fontweight='bold')
fig.tight_layout()

# Add Annotation (Example)
last_date = df_combined.index[-1]
last_own = df_combined['Inst_Ownership_Pct'].iloc[-1]
ax2.annotate(f'Current Ownership: {last_own}%', 
             xy=(last_date, last_own), 
             xytext=(last_date, last_own + 1),
             arrowprops=dict(facecolor='black', shrink=0.05))

print("📊 Generating Plot...")
plt.show()