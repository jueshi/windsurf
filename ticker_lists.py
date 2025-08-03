# -*- coding: utf-8 -*-
import sys
import io

# Set the default encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import json

mega_tickers0 = ["NVDA", "MSFT", "AAPL", "GOOG", "GOOGL", "AMZN", "META", "AVGO", "TSM", "BRK-A", "BRK-B", "TSLA", "JPM", "WMT", "LLY", "ORCL", "V", "MA", "NFLX", "XOM"]
mega_tickers1 = ['COST', 'JNJ', 'HD', 'PLTR', 'PG', 'ABBV', 'BAC', 'SAP', 'CVX', 'KO', 'GE', 'AMD', 'ASML', 'TMUS', 'CSCO', 'BABA', 'PM', 'WFC', 'CRM', 'TM']
mega_tickers2 = ['IBM', 'AZN', 'MS', 'ABT', 'NVS', 'GS', 'MCD', 'INTU', 'LIN', 'UNH', 'HSBC', 'SHEL', 'RTX', 'DIS', 'BX', 'AXP', 'CAT']
mega_tickers = mega_tickers0 + mega_tickers1 + mega_tickers2

index_tickers = ["SPX", "DJIA", "COMP", "RUT", "NYA", "INX", "DAX", "CAC", "^HSI"]

# Stock Ticker Lists: use _stocks or _tickers to tell main program to process them
tickers_comment_dict = {} #manually build a dictionary of tickers and comments


# List of tickers to process
Jues401k_stocks = ['ALAB','PSTR','QQQ', 'IWM', 'GLD','AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B','AVGO',
           'COST','MCD','BABA','AMD','NIO','AFRM','CQQQ','SPYX','SPYV','SPYU','CRM','ADI','TXN','AAOI','EWS','NKE',
           'AMZA','YINN','JD','BIDU','TNA','TECS','TECL','INTC','TSM','LRCX','MRVL','SPMO','WDC']
 
Jues401k_stocks1 = ['ALAB','PSTR','QQQ', 'IWM', 'GLD','AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B','AVGO']
Jues401k_stocks11 = ['spy','QQQ', 'IWM','dia']
Jues401k_stocks111 = ['PSTR','QQQ', 'IWM']
Jues401k_stocks112 = ['ALAB','PSTR', 'IWM']
Jues401k_stocks112 = [ 'ALAB','PSTR','QQQ']
Jues401k_stocks12 = [ 'GLD','AAPL', 'GOOGL', 'MSFT', 'AMZN']
Jues401k_stocks13 = ['TSLA', 'META', 'NVDA', 'BRK-B','AVGO']
Jues401k_stocks2 = ['COST','MCD','BABA','AMD','NIO','AFRM','CQQQ','SPYX','SPYV','SPYU','CRM','ADI','TXN','AAOI','EWS','NKE']
Jues401k_stocks3 = ['AMZA','YINN','JD','BIDU','TNA','TECS','TECL','INTC','TSM','LRCX','MRVL','SPMO','WDC']

new_highs1 = ["CSCO", "V", "MA", "AXP", "SAP", "TSM", "AMZN", "JPM", "NFLX", "GOOGL", "GOOG", "META", "AAPL", "WMT", "BAC", "AVGO", "MCD", "PG", "IBM", "BRK-B"]
new_highs2 = ["MS", "NOW", "BRK-A", "NVDA", "COST", "ACN", "WFC", "CRM", "DIS", "MSFT", "TMUS", "HD", "CVX", "ABBV", "BX", "JNJ", "XOM", "KO", "ORCL", "PEP"]        
# Combine and remove duplicates
new_highs_stocks = list(set(new_highs1 + new_highs2))

new_lows_stocks = ["RKLB", "AFRM", "SOFI", "HOOD", "NFLX", "TSLA", "COIN", "PTON", "DASH", "BYND"]

# List of top-performing stocks by sector
top_sectors_tickers = [
    # Energy sector
    "TPL",  # Texas Pacific Land Corp.
    "TRGP",  # Targa Resources
    "GPRK",  # Geopark
    "SUN",  # Sunoco LP

    # Utilities sector
    "XEL",  # Xcel Energy
    "WEC",  # WEC Energy Group
    "CMS",  # CMS Energy Corp.
    "AEP",  # American Electric Power
    "ES",   # Eversource Energy

    # Information Technology sector
    "NVDA",  # Nvidia
    "AVGO"   # Broadcom
]

# List of stocks with recent analyst upgrades
recent_analyst_upgrades_stocks = [
    "SQ",     # Block (Upgraded by Raymond James)
    "META",   # Meta Platforms (Upgraded by JPMorgan Chase)
    "AMZN",   # Amazon (Upgraded by UBS, JMP Securities, Tigress Financial)
    "GOOGL",  # Alphabet (Upgraded by JPMorgan Chase and Bank of America)
    "EQT",    # EQT Corporation (Upgraded by JPMorgan Chase and Truist Financial)
]

# Complete list of IBD 50 stocks
ibd_50_stocks = [
    "RKLB",    # Rocket Lab USA, Inc.
    "NTRA",    # Natera, Inc.
    "AGX",     # Argan, Inc.
    "ARIS",    # Aris Water Solutions, Inc.
    "ALAB",    # Astera Labs, Inc.
    "CLS",     # Celestica, Inc.
    "AXON",    # Axon Enterprise, Inc.
    "PLTR",    # Palantir Technologies Inc.
    "DOCS",    # Doximity, Inc.
    "HIMS",    # Hims & Hers Health, Inc.
    "MRX",     # Marex Group plc
    "LRN",     # Stride, Inc.
    "MMYT",    # MakeMyTrip Limited
    "NVDA",    # NVIDIA Corporation
    "DAVE",    # Dave Inc.
    "FTNT",    # Fortinet, Inc.
    "HOOD",    # Robinhood Markets, Inc.
    "LMND",    # Lemonade, Inc.
    "ATAT",    # Atour Lifestyle Holdings Limited
    "HUT",     # Hut 8 Corp. (TSX)
    "RELY",    # Remitly Global, Inc.
    "FOUR",    # Shift4 Payments, Inc.
    "IBKR",    # Interactive Brokers Group, Inc.
    "DECK",    # Deckers Outdoor Corporation
    "ARGX",    # argenx SE
    "SOFI",    # SoFi Technologies, Inc.
    "ANET",    # Arista Networks Inc
    "NFLX",    # Netflix, Inc.
    "KVYO",    # Klaviyo, Inc.
    "DUOL",    # Duolingo, Inc.
    "TKO",     # TKO Group Holdings, Inc.
    "WGS",     # GeneDx Holdings Corp.
    "HWM",     # Howmet Aerospace Inc.
    "TSM",     # Taiwan Semiconductor Manufacturing Company Limited
    "RCL",     # Royal Caribbean Cruises Ltd.
    "NOW",     # ServiceNow, Inc.
    "TOST",    # Toast, Inc.
    "AFRM",    # Affirm Holdings, Inc.
    "ZK",     # ZEEKR Intelligent Technology Holding Limited
    "RDDT",    # Reddit, Inc.
    "VIST",    # Vista Energy, S.A.B. de C.V.
    "GMED",    # Globus Medical, Inc.
    "GLBE",    # Global-E Online Ltd.
    "AVGO",    # Broadcom Limited
    "ONON",    # On Holding AG
    "EXLS",    # ExlService Holdings, Inc.
    "OWL",     # Blue Owl Capital Inc.
    "HUBS",    # HubSpot, Inc.
    "BROS",    # Dutch Bros Inc.
    "VITL"     # Vital Farms, Inc.
]

# List of Zacks Rank #1 (Strong Buy) stocks
zacks_rank_1_stocks = [
    "AAL",   # American Airlines
    "SKYW",  # SkyWest
    "UAA",   # Under Armour
    "BRBR",  # BellRing Brands
    "RBA",   # RB Global
    "SRDX",  # Surmodics
    "CTRA",  # Coterra Energy
    "ERIE",  # Erie Indemnity
    "DUOL",  # Duolingo
    "CART"   # Maplebear Inc.
]

# List of stocks with recent positive earnings surprises
positive_earnings_surprise_stocks = [
    "RBRK",  # Rubrik, Inc.
    "PSTG",  # Pure Storage, Inc.
    "HPE",   # Hewlett Packard Enterprise
    "C",     # Citigroup
    "JPM",   # JPMorgan Chase & Co.
    "WFC",   # Wells Fargo & Co.
    "META",  # Meta Platforms
    "JLL",   # Jones Lang LaSalle
    "RGA",   # Reinsurance Group of America
    "EME",   # EMCOR Group
    "COF",   # Capital One Financial
    "UBER"   # Uber Technologies
]

bitcoin_tickers = ["btc-usd",'ETH-USD','XRP-USD','SOL-USD']

canslim_tickers = ["APP", "FIX", "HWM", "NVDA", "TSM", "VRT"]

finvize_tickers = [
    "FCUV", "NITO", "CRNC", "KITT", "ACON", "TGL", "ATHE", "KLTR", "ATOM", "NVA", 
    "MBOT", "MFI", "SCPX", "SQ", "EMKR", "NYC", "PTLE", "CCM", "ALUR", "VRME", 
    "DOGZ", "HOLO", "NUKK", "EZGO", "STAI", "SISI", "LATG", "MMLP", "IIPR", 
    "ACAD", "NTRA", "PSEC", "ZYME", "INTU", "IOT", "BUXX", "ARMP", "GBIL", 
    "HSRT", "ARKG", "LJAN", "FLRN", "NVCT", "SGOV", "OCTZ", "FLJJ", "AORT", 
    "BNIX", "YIBO", "GCI", "CLIP", "CLGN", "MXE", "SAGE", "CTEC", "JPMO", 
    "ARTV", "LUX", "LSH", "EVSB", "TXSS", "PSFO", "PWP", "DWSH", "CCG", "FORD", 
    "OPER", "LEA", "DINO", "MUR", "BOWN", "AVIE", "GJUN", "RKLB", "FTNT", 
    "BUFD", "RM", "EXPI", "MFUT", "SUGP", "NOMD", "ELLO", "RES", "BVN", "PBT", 
    "MED", "AMBI", "NSA", "LU", "IDEC", "ASPC", "INLF", "RAIN", "ONEG", "WLAC", 
    "NTWO", "PHH", "HIT", "TDACU", "FACT", "RANGU", "TAVI", "YAAS", "GSRT", 
    "LSE", "NCEW"
]

newHigh_stock_tickers = [
    "FLJJ", "XMAR", "MAYW", "MARW", "DYCQ", "QCAP", "MLACU", "TJUL", "DFEB", "BALT",
    "XDAP", "CPNJ", "PBP", "GMAR", "UMAY", "PMAY", "GAPR", "XBAP", "EMPB", "DAPR"
]

newLow_tickers = ['KZIA', 'SID', 'SSTK', 'BF-A', 'ABEV', 'CCS', 'ADBE', 'BHP', 'BUD', 'KOF', 'ZROZ', 'AVY']

chinese_stocks_tickers = [
    "0700.HK",  # Tencent Holdings (Hong Kong)
    "1398.HK",  # Industrial and Commercial Bank of China (ICBC) (Hong Kong)
    "BABA",  # Alibaba Group (NYSE)
    "601857.SS",  # PetroChina (Shanghai)
    "0941.HK",  # China Mobile (Hong Kong)
    "600519.SS",  # Kweichow Moutai (Shanghai)
    "1288.HK",  # Agricultural Bank of China (Hong Kong)
    "601318.SS",  # Ping An Insurance (Shanghai)
    "0883.HK",  # CNOOC Limited (Hong Kong)
]
tickers_comment_dict['0700.HK'] = '腾讯'
tickers_comment_dict['0941.HK'] = '中移动'
tickers_comment_dict['1288.HK'] = '农行'
tickers_comment_dict['0883.HK'] = '中海油'
tickers_comment_dict['1398.HK'] = '工商银行'
tickers_comment_dict['600519.SS'] = '茅台'
tickers_comment_dict['601318.SS'] = '平安保险'
tickers_comment_dict['601857.SS'] = '中石油'

FUNDS_stocks = ['goog','aapl', 'meta', 'msft', 'amzn', 'nvda', 'tsla', 'brk-b']
China_FUNDS_stocks = ['baba', 'bidu', 'nio', 'jd', '0700.HK']


AI_ticker_extractor_tickers = [
    "ATOM",
    "JOBY",
    "KD",  
    "BBAR",
    "TSSI",
    "RDW", 
    "NEXT",
    "JBL", 
    "RGTI",
    "HTGC",
    "GRRR",
    "CRNC",
    "OUST",
    "RIVN",
    "NNOX",
    "SERV",
    "VSTM",
    "NUS", 
]

new_high_sector_tickers = [
    "NVDA",  # Nvidia
    "INTC",  # Intel
    "TSLA",  # Tesla
    "JETS",  # U.S. Global Jets ETF
    "DAL",  # Delta Air Lines
    "COIN",  # Coinbase
    "MSTR",  # MicroStrategy
    "GEO",  # GEO Group
]

tickers_comment_dict['JETS'] = 'U.S. Global Jets ETF'
tickers_comment_dict['DAL'] = 'Delta Air Lines'
tickers_comment_dict['COIN'] = 'Coinbase'
tickers_comment_dict['MSTR'] = 'MicroStrategy'
tickers_comment_dict['GEO'] = 'GEO Group'

daily_watch_tickers = [
    'WEC','XEL'
]

# Function to check if a ticker symbol is valid and suggest alternatives if not
def validate_ticker(ticker):
    import yfinance as yf
    import difflib
    
    # Special cases for indices and other symbols that yfinance handles differently
    special_cases = {
        'SPX': '^GSPC',  # S&P 500 index
        'DJIA': '^DJI',  # Dow Jones Industrial Average
        'COMP': '^IXIC', # NASDAQ Composite
        'RUT': '^RUT',   # Russell 2000
        'VIX': '^VIX',   # CBOE Volatility Index
        'NYA': '^NYA',   # NYSE Composite
        'INX': '^GSPC',  # Another symbol for S&P 500
    }
    
    # Check if it's a special case
    if ticker in special_cases:
        ticker = special_cases[ticker]
    
    try:
        # Try to get ticker info
        ticker_info = yf.Ticker(ticker).info
        
        # Check if we got valid data (yfinance returns empty dict for invalid tickers)
        if 'regularMarketPrice' in ticker_info or 'previousClose' in ticker_info:
            return True, None
        else:
            # If ticker is invalid, get a list of valid tickers to suggest alternatives
            # This is a simplified approach - in a real application, you might want to use a more comprehensive list
            common_tickers = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ', 
                             'WMT', 'PG', 'MA', 'UNH', 'HD', 'BAC', 'XOM', 'PFE', 'AVGO', 'COST', 'CSCO', 'LLY', 
                             'MRK', 'ADBE', 'NFLX', 'TMO', 'ABT', 'CRM', 'CMCSA', 'PEP', 'NKE', 'ACN', 'INTC', 
                             'VZ', 'QCOM', 'DIS', 'AMD', 'TXN', 'IBM', 'INTU', 'AMAT', 'GE', 'PYPL', 'SBUX',
                             'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'SLV', 'USO', 'EEM', 'XLF', 'XLE', 'XLK', 'XLV']
            
            # Find close matches
            matches = difflib.get_close_matches(ticker, common_tickers, n=3, cutoff=0.6)
            
            # Add special case suggestions
            if ticker in ['SPX', 'SP500', 'S&P500', 'S&P', 'SP']:
                matches.append('SPY')
            elif ticker in ['DOW', 'DOWJONES', 'DJ30']:
                matches.append('DIA')
            elif ticker == 'NASDAQ' or ticker == 'NDX':
                matches.append('QQQ')
            elif ticker == 'RUSSELL' or ticker == 'R2000':
                matches.append('IWM')
            # Check for common issues like missing dash in BRK-B vs BRKB
            elif ticker == 'BRKB':
                matches.append('BRK-B')
            elif ticker == 'BRKA':
                matches.append('BRK-A')
                
            return False, matches
    except Exception as e:
        return False, f"Error checking ticker: {str(e)}"

# Function to check all tickers in a list
def check_ticker_list(ticker_list, list_name):
    print(f"\nChecking tickers in {list_name}:")
    invalid_tickers = []
    valid_count = 0
    
    print(f"Processing {len(ticker_list)} tickers...")
    
    for i, ticker in enumerate(ticker_list):
        # Print progress for large lists
        if len(ticker_list) > 20 and i % 10 == 0 and i > 0:
            print(f"  Processed {i}/{len(ticker_list)} tickers...")
            
        is_valid, suggestions = validate_ticker(ticker)
        if is_valid:
            valid_count += 1
        else:
            invalid_tickers.append((ticker, suggestions))
    
    print(f"\nResults for {list_name}:")
    print(f"  - Valid tickers: {valid_count}/{len(ticker_list)}")
    
    if invalid_tickers:
        print(f"  - Invalid tickers: {len(invalid_tickers)}/{len(ticker_list)}")
        print("\nInvalid tickers:")
        for ticker, suggestions in invalid_tickers:
            print(f"  - {ticker}: Invalid", end="")
            if suggestions and isinstance(suggestions, list) and len(suggestions) > 0:
                print(f" (Did you mean: {', '.join(set(suggestions))}?)")
            else:
                print()
    else:
        print(f"\nAll tickers in {list_name} are valid!")
    
    return invalid_tickers

if __name__ == '__main__':
    import json
    print("\nStock Symbol Validator")
    print("====================\n")
    
    # Get all ticker lists from this module
    import sys
    current_module = sys.modules[__name__]
    all_ticker_lists = {}
    
    for name in dir(current_module):
        obj = getattr(current_module, name)
        # Find lists that contain 'ticker' or 'stock' in their name and are actually lists
        if (isinstance(obj, list) and 
            ('ticker' in name.lower() or 'stock' in name.lower()) and 
            len(obj) > 0 and 
            isinstance(obj[0], str)):
            all_ticker_lists[name] = obj
    
    # Sort lists by name
    sorted_names = sorted(all_ticker_lists.keys())
    
    # Ask user which list to check or check all
    print(f"Found {len(all_ticker_lists)} ticker lists in this module:")
    for i, name in enumerate(sorted_names, 1):
        print(f"{i}. {name} ({len(all_ticker_lists[name])} tickers)")
    
    print("\nOptions:")
    print("1-N: Check a specific list")
    print("a: Check all lists")
    print("c: Check custom tickers")
    print("q: Quit")
    
    try:
        choice = input("\nEnter your choice: ").strip().lower()
        
        if choice == 'q':
            print("Exiting...")
            sys.exit(0)
        elif choice == 'a':
            print("\nChecking all ticker lists. This may take some time...\n")
            all_invalid = {}
            for name in sorted_names:
                ticker_list = all_ticker_lists[name]
                invalid = check_ticker_list(ticker_list, name)
                if invalid:
                    all_invalid[name] = invalid
            
            if all_invalid:
                print(f"\nSummary: Found invalid tickers in {len(all_invalid)} lists.")
                print("Lists with invalid tickers:")
                for name in all_invalid:
                    print(f"  - {name}: {len(all_invalid[name])} invalid tickers")
            else:
                print("\nAll tickers in all lists are valid!")
        elif choice == 'c':
            custom_tickers = input("Enter tickers separated by commas: ").strip().split(',')
            custom_tickers = [t.strip().upper() for t in custom_tickers if t.strip()]
            if custom_tickers:
                check_ticker_list(custom_tickers, "custom list")
            else:
                print("No tickers entered.")
        elif choice.isdigit() and 1 <= int(choice) <= len(sorted_names):
            name = sorted_names[int(choice) - 1]
            check_ticker_list(all_ticker_lists[name], name)
        else:
            print("Invalid choice.")
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\n\nAn error occurred: {str(e)}")
    
    # Print tickers_comment_dict for debugging
    print("\nTicker Comments Dictionary:")
    print(json.dumps(tickers_comment_dict, indent=2, ensure_ascii=False))
