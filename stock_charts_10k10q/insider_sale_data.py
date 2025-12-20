import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import time
from pathlib import Path
import requests
from io import StringIO
try:
    import mplcursors
except ImportError:
    mplcursors = None

# ==========================================
# CONFIGURATION
# ==========================================
TICKER = "ALAB"  # Change this to ALAB, NVDA, etc.
CACHE_DIR = Path(__file__).resolve().parent / "insider_cache"
MAX_RETRIES = 3
FINVIZ_URL = "https://finviz.com/quote.ashx?t={symbol}&p=d"
FINVIZ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def _cache_path(symbol: str) -> Path:
    safe_symbol = symbol.replace("/", "_").upper()
    return CACHE_DIR / f"{safe_symbol}_sales.csv"

def _load_cached_sales(symbol: str):
    path = _cache_path(symbol)
    if not path.exists():
        return None
    try:
        cached = pd.read_csv(path, parse_dates=['Date'])
        print(f"Loaded cached insider sales for {symbol} from {path}.")
        return cached
    except Exception as exc:
        print(f"Failed to load cached data for {symbol}: {exc}")
        return None

def _save_cached_sales(symbol: str, df: pd.DataFrame):
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        df.to_csv(_cache_path(symbol), index=False)
    except Exception as exc:
        print(f"Warning: unable to cache insider sales for {symbol}: {exc}")

def _attempt_finviz_fallback(symbol: str) -> pd.DataFrame | None:
    url = FINVIZ_URL.format(symbol=symbol.upper())
    try:
        resp = requests.get(url, headers=FINVIZ_HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"Finviz request failed: {exc}")
        return None
    try:
        tables = pd.read_html(StringIO(resp.text))
    except ValueError as exc:
        print(f"Unable to parse Finviz insider tables: {exc}")
        return None

    target = None
    for table in tables:
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = table.columns.get_level_values(-1)
        if 'Insider Trading' in table.columns:
            target = table
            break
    if target is None:
        print("Finviz insider table not found.")
        return None

    rename_map = {
        'Insider Trading': 'Who',
        'Relationship': 'Relationship',
        'Date': 'Date',
        'Transaction': 'TransactionText',
        'Cost': 'Cost',
        '#Shares': 'Shares',
        '# Shares': 'Shares',
        'Value ($)': 'Value'
    }
    df = target.rename(
        columns={k: v for k, v in rename_map.items() if k in target.columns}
    )
    required_cols = {'Date', 'Who', 'TransactionText', 'Shares', 'Value', 'Cost'}
    if not required_cols.issubset(df.columns):
        print(f"Finviz table missing required columns: {required_cols - set(df.columns)}")
        return None

    df['TransactionText'] = df['TransactionText'].astype(str)
    sales_df = df[df['TransactionText'].str.contains("Sale", case=False, na=False)].copy()
    if sales_df.empty:
        print("Finviz data returned but no Sale entries found.")
        return None

    sales_df['Date'] = pd.to_datetime(sales_df['Date'], format="%b %d '%y", errors='coerce')
    sales_df['Shares'] = (
        sales_df['Shares'].astype(str)
        .str.replace(',', '', regex=False)
        .astype(float)
    )
    sales_df['Value'] = (
        sales_df['Value'].astype(str)
        .str.replace(',', '', regex=False)
        .astype(float)
    )
    sales_df['Cost'] = pd.to_numeric(
        sales_df['Cost'].astype(str).str.replace(',', '', regex=False),
        errors='coerce'
    )
    sales_df.dropna(subset=['Date', 'Shares', 'Value', 'Cost'], inplace=True)
    sales_df = sales_df[sales_df['Shares'] > 0]
    if sales_df.empty:
        return None

    sales_df.rename(columns={'TransactionText': 'Transaction'}, inplace=True)
    sales_df['Who'] = sales_df['Who'].astype(str).str.strip()
    sales_df['Value'] = sales_df['Value'].astype(float)
    sales_df['Shares'] = sales_df['Shares'].astype(float)
    return sales_df[['Date', 'Who', 'Transaction', 'Cost', 'Shares', 'Value']].sort_values('Date')

# ==========================================

def get_insider_data(symbol):
    print(f"Fetching data for {symbol}...")
    try:
        df = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.insider_transactions
                break
            except Exception as exc:
                if "Too Many Requests" in str(exc) and attempt < MAX_RETRIES:
                    wait = 5 * attempt
                    print(f"Rate limited by Yahoo (429). Retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait)
                    continue
                raise
        
        if df is None or df.empty:
            print("No insider data found via yfinance. Trying Finviz fallback...")
            fallback_df = _attempt_finviz_fallback(symbol)
            if fallback_df is not None:
                _save_cached_sales(symbol, fallback_df)
                return fallback_df
            return _load_cached_sales(symbol)

        # 2. Reset index to get Date as a column (if it's the index)
        df = df.reset_index()
        
        # 3. Rename columns to match our plotting logic
        rename_map = {
            'Start Date': 'Date',
            'Insider': 'Who',
            'Transaction': 'TransactionType',
            'Text': 'TransactionText',
            'Value': 'TotalValue',  # Avoid conflict with plotting code
            '#Shares': 'Shares'
        }
        df.rename(
            columns={k: v for k, v in rename_map.items() if k in df.columns},
            inplace=True
        )
        
        # 4. Filter for Sales
        # We look for "Sale" in the Transaction text description
        transaction_cols = [
            col for col in ('TransactionType', 'TransactionText', 'Transaction')
            if col in df.columns
        ]
        if not transaction_cols:
            print("No transaction descriptor columns found.")
            fallback_df = _attempt_finviz_fallback(symbol)
            if fallback_df is not None:
                _save_cached_sales(symbol, fallback_df)
                return fallback_df
            cached = _load_cached_sales(symbol)
            if cached is not None:
                return cached
            return None
        
        sale_mask = pd.Series(False, index=df.index)
        for col in transaction_cols:
            sale_mask |= df[col].astype(str).str.contains("Sale", case=False, na=False)
        sales_df = df[sale_mask].copy()
        if sales_df.empty:
            print("Found insider data but no Sale entries. Trying Finviz fallback...")
            fallback_df = _attempt_finviz_fallback(symbol)
            if fallback_df is not None:
                _save_cached_sales(symbol, fallback_df)
                return fallback_df
            cached = _load_cached_sales(symbol)
            if cached is not None:
                return cached
            return None
        
        # 5. Clean Data Types
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        sales_df['Shares'] = pd.to_numeric(sales_df['Shares'], errors='coerce')
        sales_df['TotalValue'] = pd.to_numeric(sales_df['TotalValue'], errors='coerce')
        
        # 6. Calculate Share Price (Cost)
        # yfinance doesn't always give per-share price, but we can derive it
        sales_df = sales_df[sales_df['Shares'] > 0] # Avoid div by zero
        sales_df['Cost'] = sales_df['TotalValue'] / sales_df['Shares']
        
        # Rename TotalValue back to Value for the plotter
        sales_df.rename(columns={'TotalValue': 'Value'}, inplace=True)
        
        sales_df = sales_df.sort_values('Date')
        _save_cached_sales(symbol, sales_df)
        return sales_df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        fallback_df = _attempt_finviz_fallback(symbol)
        if fallback_df is not None:
            _save_cached_sales(symbol, fallback_df)
            return fallback_df
        cached = _load_cached_sales(symbol)
        if cached is not None:
            return cached
        return None

# --- Data Processing ---

sales_df = get_insider_data(TICKER)

if sales_df is not None and not sales_df.empty:
    
    # Function to clean up long job titles from names
    def clean_name(s):
        titles = [
            "Pres & Chief Executive Officer", "Chief Technology Officer", 
            "Chief Operating Officer", "Chief Financial Officer",
            "Chief Legal Officer, Secretary", "Director", "Officer", 
            "Former Director", "General Counsel"
        ]
        # Standardize formatting
        s = str(s).strip()
        
        # Custom mapping for known entities (Optional)
        if "ZHAN (BVI)" in s: return "ZHAN (BVI)"
        if "WALDEN" in s: return "WALDEN TECH"
        
        # Strip standard titles
        for t in titles:
            if s.endswith(t):
                return s[:-len(t)].strip()
        return s

    sales_df['Name'] = sales_df['Who'].apply(clean_name)

    # Group names: Keep top 6 most active, label rest as "Other"
    top_names = sales_df['Name'].value_counts().index[:6]
    sales_df['NameGroup'] = sales_df['Name'].apply(lambda x: x if x in top_names else 'Other')

    # Pick the best available transaction descriptor for tooltips
    transaction_fields = [
        col for col in ('Transaction', 'TransactionType', 'TransactionText')
        if col in sales_df.columns
    ]

    def _resolve_transaction(row):
        for col in transaction_fields:
            val = row.get(col)
            if pd.notnull(val) and str(val).strip():
                return str(val)
        return "Sale"

    if transaction_fields:
        sales_df['TransactionDisplay'] = sales_df.apply(_resolve_transaction, axis=1)
    else:
        sales_df['TransactionDisplay'] = "Sale"

    # Create a proxy for Stock Price (Average transaction cost per day)
    price_trend = sales_df.groupby('Date')['Cost'].mean().reset_index()

    # Ensure numeric Value before plotting/annotating
    sales_df['Value'] = pd.to_numeric(sales_df['Value'])

    # DataFrame used for plotting + tooltips
    plot_df = sales_df[['Date', 'Cost', 'Value', 'NameGroup', 'Name', 'Who', 'Shares', 'TransactionDisplay']].copy()
    plot_df.reset_index(drop=True, inplace=True)

    # --- Plotting ---

    plt.figure(figsize=(14, 7))

    # 1. Plot the Stock Price Line
    plt.plot(price_trend['Date'], price_trend['Cost'], 
             color='gray', alpha=0.4, linestyle='--', linewidth=1, label='Avg Transaction Price')

    # 2. Plot the Bubble Chart
    scatter = sns.scatterplot(
        data=plot_df, 
        x='Date', 
        y='Cost', 
        size='Value',       # Bubble size = Transaction Value
        hue='NameGroup',    # Color = Person
        sizes=(50, 1000),   # Range of bubble sizes
        alpha=0.8, 
        edgecolor='black',
        palette='deep'
    )

    # 3. Annotate the largest transactions
    top_transactions = sales_df.nlargest(4, 'Value')
    
    for idx, row in top_transactions.iterrows():
        # Safety check for nan
        if pd.notnull(row['Value']):
            label = f"${row['Value']/1e6:.1f}M"
            plt.annotate(
                label, 
                (row['Date'], row['Cost']),
                xytext=(0, 10), 
                textcoords='offset points',
                ha='center', 
                fontsize=9,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )

    # Optional interactive tooltips for sellers, if mplcursors is installed
    if mplcursors is not None:
        cursor = mplcursors.cursor(scatter.collections[0], hover=True)

        @cursor.connect("add")
        def _add_tooltip(sel):
            idx = sel.index
            if idx is None or idx >= len(plot_df):
                return
            row = plot_df.iloc[idx]
            shares = row['Shares']
            value = row['Value']
            sel.annotation.set_text(
                f"{row['Name']} ({row['TransactionDisplay']})\n"
                f"Official: {row['Who']}\n"
                f"Date: {row['Date']:%b %d, %Y}\n"
                f"Shares: {shares:,.0f}\n"
                f"Value: ${value:,.0f}\n"
                f"Price: ${row['Cost']:.2f}"
            )
            sel.annotation.get_bbox_patch().set(alpha=0.9)
    else:
        print("Install `mplcursors` to see hover tooltips identifying each seller.")

    # Formatting
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)
    plt.ylabel('Share Price ($)')
    plt.xlabel('Date')
    plt.title(f'Insider Sales Transactions: {TICKER}', fontsize=14)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='Insider')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    # Show the plot
    plt.show()
else:
    print("No sales data available to plot.")