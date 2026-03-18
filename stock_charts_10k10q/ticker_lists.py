# -*- coding: utf-8 -*-
# Stock ticker lists for data_rechiever.py
# This file contains various stock ticker lists that can be imported and used
# Note: Do not modify sys.stdout or sys.stderr in this file as it causes issues with module reloading

import json

A_ping_401k_ping = ["AAPL", "AMD", "AMZN", "AMZU", "APP", "ASAN", "AVGO", "CRDO", "ENVA", "FRMI", "GGLL", "GS", "INTC", "JPM", "LRCX", "META", "MSFU", "MSTR", "MSTX", "MU", "NVDA", "NVDL", "ORCL", "PLTR", "PRCT", "QQQ", "RDDT", "SPMO", "SPY", "TECS", "TQQQ", "TSLA", "TSLL", "UEC", "V"]

A_jue_401k_2025 = ["AAOI", "AAPL", "ABBV", "ADI", "AFRM", "ALAB", "AMCR", "AMGN", "AMZU", "APP", "APPX", "ARMG", "ATO", "AVGO", "AVL", "AVPT", "AZN", "BABX", "BEN", "BIDU", "BRKB", "BRKU", "CELH", "CELT", "CQQQ", "CRCG", "CRCL", "CRDO", "CRDU", "CRM", "CRMG", "CRWD", "CRWL", "CRWV", "CTAS", "DBC", "DIS", "DOV", "DOXGX", "ED", "EDMCQ", "ESS", "FBL", "FBTC", "FIG", "FIGR", "FNGU", "FSSNX", "FXAIX", "GLD", "GS", "GSX", "GWW", "HD", "INTC", "JD", "JNJ", "LABX", "MA", "MCD", "MSFX", "MSTX", "NFXL", "ORCL", "ORCX", "PG", "PLTG", "PLTR", "PLTU", "QQQ", "SPY", "TECS", "THD", "TMO", "TNA", "TSLL", "UBRL", "VGT", "VIGIX", "YINN", "COSW", "NVDA", "AMD"]

A_magic_formula_12_22_25 = ["MO", "AMCX", "ATRA", "BTMD", "BBUC", "BMBL", "CRK", "CCSI", "CROX", "EGREF", "ESP", "EWCZ", "EVER", "FOXA", "FC", "GAMB", "GDEV", "GOT", "HRB", "HRMY", "HPQ", "IDT", "INVA", "IPG", "JILL", "LNTH", "LEVN", "NL", "OMC", "OMI", "MD", "PBI", "PLTK", "PTCT", "PRYI", "RCLD", "RIGL", "RMNI", "SBG", "SSTK", "SKYA", "SIRI", "TGNA", "AREN", "T2OO", "UIS", "WWW", "XPOF", "ANF"]

A_BTC_etfs = ["ARKB", "BITB", "IBIT", "FBTC", "EZBC", "GBTC", "BTCO", "HODL", "BRRR", "BTCW"]

A_btc_related = ["V", "MA", "PYPL", "CRCL", "COIN", "MSTR"]

mega_tickers0 = ["NVDA", "MSFT", "AAPL", "GOOG", "GOOGL", "AMZN", "META", "AVGO", "TSM", "BRK-A", "BRK-B", "TSLA", "JPM", "WMT", "LLY", "ORCL", "V", "MA", "NFLX", "XOM"]
mega_tickers1 = ['COST', 'JNJ', 'HD', 'PLTR', 'PG', 'ABBV', 'BAC', 'SAP', 'CVX', 'KO', 'GE', 'AMD', 'ASML', 'TMUS', 'CSCO', 'BABA', 'PM', 'WFC', 'CRM', 'TM']
mega_tickers2 = ['IBM', 'AZN', 'MS', 'ABT', 'NVS', 'GS', 'MCD', 'INTU', 'LIN', 'UNH', 'HSBC', 'SHEL', 'RTX', 'DIS', 'BX', 'AXP', 'CAT']
mega_tickers = mega_tickers0 + mega_tickers1 + mega_tickers2

index_tickers = ["SPX", "DJIA", "COMP", "RUT", "NYA", "INX", "DAX", "CAC", "^HSI"]

# Stock Ticker Lists: use _stocks or _tickers to tell main program to process them
tickers_comment_dict = {} #manually build a dictionary of tickers and comments

# List of tickers to process
Jues401k_stocks = ["AAOI", "AAPL", "ADI", "AFRM", "ALAB", "AMD", "AMZA", "AMZN", "AVGO", "BABA", "BIDU", "BRK-B", "COST", "CQQQ", "CRDO", "CRM", "EWS", "GLD", "GOOGL", "INTC", "IWM", "JD", "LRCX", "MCD", "META", "MRVL", "MSFT", "NIO", "NKE", "NVDA", "PLTR", "QQQ", "SOFI", "SPMO", "SPYU", "SPYV", "SPYX", "TECL", "TECS", "TNA", "TSLA", "TSM", "TXN", "WDC", "YINN"]

# new_highs1 = ["CSCO", "V", "MA", "AXP", "SAP", "TSM", "AMZN", "JPM", "NFLX", "GOOGL", "GOOG", "META", "AAPL", "WMT", "BAC", "AVGO", "MCD", "PG", "IBM", "BRK-B"]
# new_highs2 = ["MS", "NOW", "BRK-A", "NVDA", "COST", "ACN", "WFC", "CRM", "DIS", "MSFT", "TMUS", "HD", "CVX", "ABBV", "BX", "JNJ", "XOM", "KO", "ORCL", "PEP"]        
# # Combine and remove duplicates
# new_highs_stocks = list(set(new_highs1 + new_highs2))

# new_lows_stocks = ["RKLB", "AFRM", "SOFI", "HOOD", "NFLX", "TSLA", "COIN", "PTON", "DASH", "BYND"]

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

# # List of stocks with recent analyst upgrades
# recent_analyst_upgrades_stocks = [
#     "SQ",     # Block (Upgraded by Raymond James)
#     "META",   # Meta Platforms (Upgraded by JPMorgan Chase)
#     "AMZN",   # Amazon (Upgraded by UBS, JMP Securities, Tigress Financial)
#     "GOOGL",  # Alphabet (Upgraded by JPMorgan Chase and Bank of America)
#     "EQT",    # EQT Corporation (Upgraded by JPMorgan Chase and Truist Financial)
# ]

# # Complete list of IBD 50 stocks
# ibd_50_stocks = [
#     "RKLB",    # Rocket Lab USA, Inc.
#     "NTRA",    # Natera, Inc.
#     "AGX",     # Argan, Inc.
#     "ARIS",    # Aris Water Solutions, Inc.
#     "ALAB",    # Astera Labs, Inc.
#     "CLS",     # Celestica, Inc.
#     "AXON",    # Axon Enterprise, Inc.
#     "PLTR",    # Palantir Technologies Inc.
#     "DOCS",    # Doximity, Inc.
#     "HIMS",    # Hims & Hers Health, Inc.
#     "MRX",     # Marex Group plc
#     "LRN",     # Stride, Inc.
#     "MMYT",    # MakeMyTrip Limited
#     "NVDA",    # NVIDIA Corporation
#     "DAVE",    # Dave Inc.
#     "FTNT",    # Fortinet, Inc.
#     "HOOD",    # Robinhood Markets, Inc.
#     "LMND",    # Lemonade, Inc.
#     "ATAT",    # Atour Lifestyle Holdings Limited
#     "HUT",     # Hut 8 Corp. (TSX)
#     "RELY",    # Remitly Global, Inc.
#     "FOUR",    # Shift4 Payments, Inc.
#     "IBKR",    # Interactive Brokers Group, Inc.
#     "DECK",    # Deckers Outdoor Corporation
#     "ARGX",    # argenx SE
#     "SOFI",    # SoFi Technologies, Inc.
#     "ANET",    # Arista Networks Inc
#     "NFLX",    # Netflix, Inc.
#     "KVYO",    # Klaviyo, Inc.
#     "DUOL",    # Duolingo, Inc.
#     "TKO",     # TKO Group Holdings, Inc.
#     "WGS",     # GeneDx Holdings Corp.
#     "HWM",     # Howmet Aerospace Inc.
#     "TSM",     # Taiwan Semiconductor Manufacturing Company Limited
#     "RCL",     # Royal Caribbean Cruises Ltd.
#     "NOW",     # ServiceNow, Inc.
#     "TOST",    # Toast, Inc.
#     "AFRM",    # Affirm Holdings, Inc.
#     "ZK",     # ZEEKR Intelligent Technology Holding Limited
#     "RDDT",    # Reddit, Inc.
#     "VIST",    # Vista Energy, S.A.B. de C.V.
#     "GMED",    # Globus Medical, Inc.
#     "GLBE",    # Global-E Online Ltd.
#     "AVGO",    # Broadcom Limited
#     "ONON",    # On Holding AG
#     "EXLS",    # ExlService Holdings, Inc.
#     "OWL",     # Blue Owl Capital Inc.
#     "HUBS",    # HubSpot, Inc.
#     "BROS",    # Dutch Bros Inc.
#     "VITL"     # Vital Farms, Inc.
# ]

# # List of Zacks Rank #1 (Strong Buy) stocks
# zacks_rank_1_stocks = [
#     "AAL",   # American Airlines
#     "SKYW",  # SkyWest
#     "UAA",   # Under Armour
#     "BRBR",  # BellRing Brands
#     "RBA",   # RB Global
#     "SRDX",  # Surmodics
#     "CTRA",  # Coterra Energy
#     "ERIE",  # Erie Indemnity
#     "DUOL",  # Duolingo
#     "CART"   # Maplebear Inc.
# ]

# # List of stocks with recent positive earnings surprises
# positive_earnings_surprise_stocks = [
#     "RBRK",  # Rubrik, Inc.
#     "PSTG",  # Pure Storage, Inc.
#     "HPE",   # Hewlett Packard Enterprise
#     "C",     # Citigroup
#     "JPM",   # JPMorgan Chase & Co.
#     "WFC",   # Wells Fargo & Co.
#     "META",  # Meta Platforms
#     "JLL",   # Jones Lang LaSalle
#     "RGA",   # Reinsurance Group of America
#     "EME",   # EMCOR Group
#     "COF",   # Capital One Financial
#     "UBER"   # Uber Technologies
# ]

# bitcoin_tickers = ["btc-usd",'ETH-USD','XRP-USD','SOL-USD']

# canslim_tickers = ["APP", "FIX", "HWM", "NVDA", "TSM", "VRT"]

# finvize_tickers = [
#     "FCUV", "NITO", "CRNC", "KITT", "ACON", "TGL", "ATHE", "KLTR", "ATOM", "NVA", 
#     "MBOT", "MFI", "SCPX", "NYC", "PTLE", "CCM", "ALUR", "VRME", 
#     "DOGZ", "HOLO", "NUKK", "EZGO", "STAI", "SISI", "MMLP", "IIPR", 
#     "ACAD", "NTRA", "PSEC", "ZYME", "INTU", "IOT", "BUXX", "ARMP", "GBIL", 
#     "HSRT", "ARKG", "LJAN", "FLRN", "NVCT", "SGOV", "OCTZ", "FLJJ", "AORT", 
#     "BNIX", "YIBO", "GCI", "CLIP", "CLGN", "MXE", "SAGE", "CTEC", "JPMO", 
#     "ARTV", "LUX", "LSH", "EVSB", "TXSS", "PSFO", "PWP", "DWSH", "CCG", "FORD", 
#     "OPER", "LEA", "DINO", "MUR", "BOWN", "AVIE", "GJUN", "RKLB", "FTNT", 
#     "BUFD", "RM", "EXPI", "MFUT", "SUGP", "NOMD", "ELLO", "RES", "BVN", "PBT", 
#     "MED", "AMBI", "NSA", "LU", "IDEC", "ASPC", "INLF", "RAIN", "ONEG", "WLAC", 
#     "NTWO", "PHH", "HIT", "TDACU", "FACT", "RANGU", "TAVI", "YAAS", "GSRT", 
#     "LSE", "NCEW"
# ]

# newHigh_stock_tickers = [
#     "FLJJ", "XMAR", "MAYW", "MARW", "DYCQ", "QCAP", "MLACU", "TJUL", "DFEB", "BALT",
#     "XDAP", "CPNJ", "PBP", "GMAR", "UMAY", "PMAY", "GAPR", "XBAP", "EMPB", "DAPR"
# ]

# Temporarily commenting out this list to test dropdown refresh
# newLow_tickers = ['KZIA', 'SID', 'SSTK', 'BF-A', 'ABEV', 'CCS', 'ADBE', 'BHP', 'BUD', 'KOF', 'ZROZ', 'AVY']

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
# China_FUNDS_stocks = ['baba', 'bidu', 'nio', 'jd', '0700.HK']


AI_ticker_extractor_tickers = ["ATOM", "JOBY", "KD", "BBAR", "TSSI", "RDW", "NEXT", "JBL", "RGTI", "HTGC", "GRRR", "CRNC", "OUST", "RIVN", "NNOX", "SERV", "VSTM", "NUS", "LITE", "ALAB", "COHR", "CRDO", "VRT", "NVT", "MOD", "ATAT", "HTHT", "ANET", "APP", "PLTR", "CARR"]














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

index_etfs = [
    # Broad US Market
    'VTI',  # Vanguard Total Stock Market ETF
    'ITOT', # iShares Core S&P Total U.S. Stock Market ETF
    'SCHB', # Schwab U.S. Broad Market ETF

    # S&P 500
    'SPY',  # SPDR S&P 500 ETF Trust
    'IVV',  # iShares Core S&P 500 ETF
    'VOO',  # Vanguard S&P 500 ETF

    # Nasdaq-100
    'QQQ',  # Invesco QQQ Trust
    'QQQM', # Invesco NASDAQ 100 ETF

    # Dow Jones Industrial Average
    'DIA',  # SPDR Dow Jones Industrial Average ETF Trust

    # Small-Cap (Russell 2000)
    'IWM',  # iShares Russell 2000 ETF
    'VTWO', # Vanguard Russell 2000 ETF

    # International Developed Markets
    'VEA',  # Vanguard FTSE Developed Markets ETF
    'IEFA', # iShares Core MSCI EAFE ETF

    # International Emerging Markets
    'VWO',  # Vanguard FTSE Emerging Markets ETF
    'IEMG', # iShares Core MSCI Emerging Markets ETF

    # Global Market (US + International)
    'VT',   # Vanguard Total World Stock ETF
    'ACWI', # iShares MSCI ACWI ETF

    # Bond Market
    'BND',  # Vanguard Total Bond Market ETF
    'AGG'   # iShares Core U.S. Aggregate Bond ETF
]

eps_growth_stocks_2025_8_3 = ["NVDA", "AVGO", "AMD", "APH", "UBS", "NET", "CCJ", "GFI", "CLS", "ESLT", "KGC", "CELH", "EVR", "SRAD", "FYBR", "BE", "W", "LIF", "EGO", "OLO", "CDTX", "CCEC", "ATAI", "TBPH", "ISSC", "XGN", "ELTX", "ASMB"]
canslim_8_3_2025_stocks = ["APP", "AVGO", "CLS", "EVR", "GFI", "HIMS", "HWM", "NVDA", "ATLC", "DRD", "FUTU", "RCL"]

watch_list = ["0700.HK", "ALAB", "AMZN", "APH", "APP", "ARM", "ARMG", "ASML", "AVGO", "AVL", "BEN", "BRK-B", "BRKU", "BYD", "CCI", "CME", "COST", "CQQQ", "CRCL", "CRDO", "CRWV", "DBC", "DIA", "ESLT", "FBL", "FNGO", "FNGU", "GIS", "GLD", "GS", "HOOD", "IONQ", "IPG", "IWM", "JNJ", "KEY", "KIM", "KMI", "LRCX", "MAGS", "MO", "MRVL", "MSFX", "MU", "NFLX", "NTNX", "NVDA", "OKE", "OKTA", "OPEN", "ORCL", "ORCX", "PATH", "PLTR", "QCOM", "QQQ", "QQQM", "SOFI", "SPMO", "SPY", "TECL", "TFC", "TQQQ", "TSLL", "TSM", "UBER", "UBRL", "VGT", "VOO", "VZ", "WDC", "XMAR", "YINN", "RDDT", "UAMY", "CVX", "ED", "DOV", "ATOM"]





Jues401k_stocks_stocks = ["ALAB", "QQQ", "IWM", "GLD", "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "AVGO", "COST", "BABA", "AMD", "NIO", "AFRM", "CQQQ", "SPYX", "SPYV", "SPYU", "CRM", "ADI", "AAOI", "EWS", "YINN", "JD", "BIDU", "TNA", "TECS", "TECL", "INTC", "TSM", "LRCX", "MRVL", "SPMO", "WDC", "GS", "PLTR", "ORCL", "IVV", "IEMG", "PG", "JNJ", "KO", "PEP", "MCD", "WMT", "VT", "TLT", "IEF", "DBC", "VTI", "VOO", "CRWD", "CRDO", "SNDK", "AKBA", "ASND", "DVAX", "PRCT", "SNPS", "MAGS", "V", "MA", "AVGX", "AVL", "TSLL", "ORCX", "FNGU", "ARMG", "APED", "GGLL", "FNGO", "UBRL", "CRWL", "MSFX", "FBL", "BRKU", "LABX", "ENOR"]


nucleus_stock_stocks = ["BX", "CEG", "SMR", "NLR", "URA", "CCJ", "URNM", "URAN", "BWXT"]
sector_etfs = [
    # Information Technology
    "XLK",
    "VGT",
    "QQQ",

    # Health Care
    "XLV",
    "VHT",
    "IBB",

    # Financials
    "XLF",
    "VFH",
    "KRE",

    # Consumer Discretionary
    "XLY",
    "VCR",

    # Consumer Staples
    "XLP",
    "VDC",

    # Industrials
    "XLI",
    "VIS",

    # Energy
    "XLE",
    "VDE",
    "XOP",

    # Utilities
    "XLU",
    "VPU",

    # Real Estate
    "XLRE",
    "VNQ",

    # Materials
    "XLB",
    "VAW",

    # Communication Services
    "XLC",
    "VOX"
]

# --- 3x Leveraged BULL ETFs (Long Position) ---
# These aim for 3x the DAILY return of their underlying index.

bull_3x_etfs = [
    # Broad Market
    "TQQQ",  # ProShares UltraPro QQQ (Nasdaq-100)
    "UPRO",  # ProShares UltraPro S&P500
    "SPXL",  # Direxion Daily S&P 500 Bull 3X Shares
    "UDOW",  # ProShares UltraPro Dow30
    "TNA",   # Direxion Daily Small Cap Bull 3X Shares (Russell 2000)

    # Technology / Innovation
    "SOXL",  # Direxion Daily Semiconductor Bull 3X Shares
    "FNGU",  # MicroSectors FANG+ Index 3X Leveraged ETN
    "TECL",  # Direxion Daily Technology Bull 3X Shares

    # Other Sectors
    "FAS",   # Direxion Daily Financial Bull 3X Shares
    "LABU",  # Direxion Daily S&P Biotech Bull 3X Shares
    "GUSH",  # Direxion Daily S&P Oil & Gas Exp. & Prod. Bull 2X Shares (Note: GUSH is now a 2X ETF, was previously 3x)
    "DRN",   # Direxion Daily Real Estate Bull 3X Shares
]


# --- 3x Leveraged BEAR ETFs (Short Position) ---
# These aim for 3x the INVERSE DAILY return of their underlying index.

bear_3x_etfs = [
    # Broad Market
    "SQQQ",  # ProShares UltraPro Short QQQ (Nasdaq-100)
    "SPXU",  # ProShares UltraPro Short S&P500
    "SDOW",  # ProShares UltraPro Short Dow30
    "TZA",   # Direxion Daily Small Cap Bear 3X Shares (Russell 2000)

    # Technology / Innovation
    "SOXS",  # Direxion Daily Semiconductor Bear 3X Shares
    "FNGD",  # MicroSectors FANG+ Index -3X Inverse Leveraged ETN
    "TECS",  # Direxion Daily Technology Bear 3X Shares

    # Other Sectors
    "FAZ",   # Direxion Daily Financial Bear 3X Shares
    "LABD",  # Direxion Daily S&P Biotech Bear 3X Shares
    "DRIP",  # Direxion Daily S&P Oil & Gas Exp. & Prod. Bear 2X Shares (Note: DRIP is now a 2X ETF, was previously 3x)
    "DRV",   # Direxion Daily Real Estate Bear 3X Shares
]

# You can combine them into a single list if needed
all_3x_etfs = bull_3x_etfs + bear_3x_etfs

# --- 2x Leveraged BULL ETFs (Long Position) ---
# These aim for 2x the DAILY return of their underlying index.

bull_2x_etfs = ["SSO", "QLD", "DDM", "UWM", "ROM", "USD", "FINU", "UGE", "URE", "DIG", "GUSH", "UBT", "AMDL"]



# --- 2x Leveraged BEAR ETFs (Short Position) ---
# These aim for 2x the INVERSE DAILY return of their underlying index.

bear_2x_etfs = [
    # Broad Market
    "SDS",   # ProShares UltraShort S&P500
    "QID",   # ProShares UltraShort QQQ (Nasdaq-100)
    "DXD",   # ProShares UltraShort Dow30
    "TWM",   # ProShares UltraShort Russell2000

    # Sectors
    "REW",   # ProShares UltraShort Technology
    "SSG",   # ProShares UltraShort Semiconductors
    "SKF",   # ProShares UltraShort Financials
    "UREV",  # ProShares UltraShort Real Estate
    "DUG",   # ProShares UltraShort Oil & Gas
    "UBT",   # ProShares UltraShort 20+ Year Treasury (Bond Market)
]

all_2x_etfs = bull_2x_etfs + bear_2x_etfs

country_etfs = ["SPY", "IVV", "VTI", "EWC", "EWW", "EWG", "EWU", "EWQ", "EWL", "EWI", "EWP", "EWD", "EWN", "EWJ", "MCHI", "FXI", "INDA", "EWA", "EWY", "EWT", "EWH", "VNM", "EWZ", "EZA", "ECH", "TUR", "EPOL", "EWS"]

futures_etfs = [
    # Broad Commodities
    "DBC",  # Invesco DB Commodity Index Tracking Fund
    "BCI",  # abrdn Bloomberg All Commodity Strategy K-1 Free ETF

    # Single Commodities
    "USO",  # United States Oil Fund, LP
    "UNG",  # United States Natural Gas Fund, LP
    
    # Volatility
    "VIXY", # ProShares VIX Short-Term Futures ETF
    "VIXM", # ProShares VIX Mid-Term Futures ETF

    # Currencies
    "UUP",  # Invesco DB US Dollar Index Bullish Fund

    # Managed Futures (Multi-Asset Strategy)
    "DBMF", # iM DBi Managed Futures Strategy ETF
    "KMLM"  # KFA Mount Lucas Index Strategy ETF
]

us_crypto_futures_etfs = [
    "BITO", # ProShares Bitcoin Strategy ETF
    "EETH", # ProShares Ether Strategy ETF
    "XBTF", # VanEck Bitcoin Strategy ETF
    "BTF",  # Valkyrie Bitcoin and Ether Strategy ETF
]

Buffet_real_estate_stocks = ["DHI", "GLD", "HD", "SOFI", "LEN", "LOW", "SHW"]

Data_Center_REITs_stocks = ["EQIX", "AMT", "DLR", "IRM"]


Residential_REITs_stocks = ["AVB", "EQR", "MAA", "ESS", "UDR", "CPT", "INVH", "AMH"]

Industrial_REITs_stocks = ["PLD", "COLD", "STAG", "EGP", "FR"]

REITs_Retail_stocks = ["SPG", "O", "KIM", "REG", "FRT", "NNN"]

REITs_Healthcare_stocks = ["WELL", "VTR", "DOC"]

REITs_Self_Storage_stocks = ["PSA", "EXR", "CUBE"]

REITs_多户住宅_stocks = ["MAA", "EQR", "AVB", "CPT", "UDR", "ESS"]

eps_growth_8_20_2025_stocks = ["BE", "WGS", "LIF", "FIGS", "AVGO", "AVAH", "PAY", "ROAD", "BTSG", "SRAD", "ASLE", "TBPH", "NVDA", "TAK", "ESLT", "NFLX", "APH", "ESE", "INFU", "EVR", "GFI", "UBS", "FTLF", "INTR", "CWK", "KGC", "EGO", "ATLC", "RM", "VRNS", "CDTX", "COGT", "W", "XGN", "Z", "ELTX", "YEXT", "VSTM", "FYBR", "DBD", "OLO", "AAPG"]

# TL_support_stocks = ["THM", "WPM", "ROL", "GNTY", "GPAT", "SD", "PLTR", "WDC", "SNDK", "INTC", "XMAR", "ALAB"]

finviz_heat_map_stocks = ["MSFT", "ORCL", "PLTR", "RNRW", "SNPS", "CRWD", "FTNT", "NVDA", "AVGO", "AMD", "QCOM", "TXN", "INTC", "AAPL", "CRM", "NOW", "INTU", "UBER", "ADBE", "PAYX", "ADSK", "DOCU", "COMPUTE", "ANET", "CSCO", "IBM", "FIS", "ACN", "IT", "APH", "KLAC", "AMAT", "LRCX", "GPN", "TEL", "TRMB", "KFY", "FFIV", "DELL", "HPQ", "WDC", "STX", "AMZN", "EBAY", "GME", "TSLA", "GM", "F", "HD", "LOW", "MCD", "SBUX", "CMG", "YUM", "DRI", "TJX", "ROST", "AZO", "ORLY", "BKNG", "ABNB", "MAR", "HLT", "NKE", "LULU", "FSR", "RIVN", "LCID", "LVS", "LEN", "GOOG", "META", "NFLX", "DIS", "TKO", "FOXA", "TMUS", "T", "VZ", "CHTR", "BA", "LMT", "GE", "GD", "NOC", "RTX", "LHX", "TDG", "DE", "CAT", "PCAR", "UNP", "NSC", "CSX", "WM", "RSG", "DAL", "UAL", "LUV", "ITW", "CMI", "PH", "GWW", "OTIS", "IR", "HON", "IEX", "MMM", "URI", "EMR", "ETN", "ROK", "AME", "LLY", "JNJ", "ABT", "BSY", "DXCM", "GILD", "BMY", "AMGN", "MRK", "PFE", "ABBV", "UNH", "ELV", "CI", "HCA", "SYK", "MDT", "EW", "ISRG", "DHR", "TMO", "A", "VRTX", "REGN", "BIIB", "WMT", "COST", "TGT", "DG", "KO", "PEP", "K", "MDLZ", "GIS", "SYY", "PG", "EL", "CL", "MO", "PM", "JPM", "V", "BRK-B", "AXP", "BAC", "WFC", "C", "BK", "COF", "SYF", "BLK", "BX", "KKR", "STT", "MS", "GS", "SPGI", "ICE", "CME", "MCO", "CB", "ALL", "AIG", "TRV", "PGR", "MMC", "L", "MET", "PRU", "PNC", "USB", "TFC", "AON", "FITB", "RF", "AMT", "PLD", "CCI", "DLR", "IRM", "WY", "EQR", "O", "XOM", "CVX", "COP", "EOG", "OXY", "SLB", "EPD", "ET", "NEE", "DUK", "SO", "D", "ES", "AEP", "SRE", "FE", "ED", "LIN", "SHW", "LYB", "EMN", "ECL", "FCX", "NUE", "STLD", "VMC"]

high_profit_margin_stocks = ["NOW", "DDOG", "CRWD", "MPWR", "NVDA", "PANW"]

ticker_1_brokerage_link_stocks = ["AMZN", "AMZU", "ARMG", "AVL", "BRKU", "CQQQ", "CRM", "CRMG", "CRWL", "FBL", "MSFX", "ORCX", "THD", "UBRL"]

buffet_stocks = ["KO", "AXP", "MCO", "HSY", "UNP"]

value_stocks = ["AAPL", "KO", "AXP", "MCO", "JNJ", "PG", "V", "MA", "VRSN"]

mags_stocks = ["AAPL", "ALAB", "AMZN", "APLD", "AVGO", "CRDO", "DUST", "FNGO", "FNGU", "GLD", "GOOG", "META", "MRVL", "MSFT", "NUGT", "NVDA", "ORCL", "PLTR", "QQQ", "TSLA"]

long_10_13_25_stocks = ["PG", "CVX", "PM", "CRM", "ABT", "MCD", "DIS", "BX", "AXP", "GOOG", "TSM", "WMT", "KO", "V", "MA", "NFLX", "NFXL", "YEXT", "CCEC", "MCB", "ENVA", "SOFI"]

eps_growth_10_13_25_stocks = ["LSCC", "LIF", "PRVA", "FIGS", "CELH", "AMD", "AVAH", "BTSG", "MAMA", "AVGO", "WPM", "TBPH", "ESLT", "CLS", "NVDA", "OIS", "FNV", "NFLX", "APH", "AGI", "STRL", "TATT", "TSM", "RGLD", "PIPR", "SVM", "CTRE", "BANC", "MU", "SCHW", "ASX", "FTLF", "UBS", "GFI", "DRD", "KGC", "INTR", "FSM", "EGO", "NESR", "UVE", "MCB", "CCEC", "YEXT", "TGB", "EVLV", "FYBR", "ONDS", "COGT", "EBC", "CDTX", "COHR", "UNFI", "EXK", "XGN", "JMIA", "IRD"]


ticker_0_buy_stocks = ["ABBV", "GS", "JPM", "GSX", "CRDO", "ALAB", "APH", "META", "MSTR", "CRM", "PLTR"]


ticker_0_sell_stocks = ["BABA"]

ETFs_divident_stocks = [
    "BRW",
    "SABA",
    "HYBI",
    "QQQH",
    "QQQI",
    "NBXG",
    "NML",
    "NHS",
    "NRO",
    "XLRE",
    "GHY",
    "ISD",
    "ASGI",
    "THQ",
    "AWF",
    "XLV"
]


SP500_Dividend_Aristocrats_stocks = ["ABBV", "ABT", "ADM", "ADP", "AFL", "ALB", "AMCR", "AOS", "APD", "ATO", "BDX", "BEN", "BF.B", "BRO", "CAH", "CAT", "CB", "CHD", "CHRW", "CINF", "CL", "CLX", "CTAS", "CVX", "DOV", "ECL", "ED", "EMR", "ERIE", "ES", "ESS", "EXPD", "FAST", "FDS", "FRT", "GD", "GPC", "GWW", "HRL", "IBM", "ITW", "JNJ", "KMB", "KO", "KVUE", "LIN", "LOW", "MCD", "MDT", "MKC", "NDSN", "NEE", "NOBL", "NUE", "O", "PEP", "PG", "PNR", "PPG", "ROP", "SHW", "SJM", "SPGI", "SWK", "SYY", "TGT", "TROW", "WMT", "WST", "XOM"]









hot_stocks = ["ALAB", "APP", "PLTR", "CRDO", "MU", "SNDK", "WDC", "NVDA", "AVGO", "LRCX", "AMAT", "IREN", "PG", "KO"]












competitors_IREN_stocks = ["IREN", "MARA", "RIOT", "CLSK", "HUT"]


half_year_stars_stocks = ["IREN", "BE", "SNDK", "BMNR", "QBTS", "OKLO", "QS", "NBIS", "RGTI", "CLS", "WDC", "CRDO", "LITE", "SYM", "W", "SATS", "STX", "MU", "HOOD", "ASTS", "INSM", "CIEN", "RNA", "WBD", "LUMN", "ALAB", "NXT", "AMD", "RKLB", "TER", "GH", "JOBY", "FN", "SOFI", "STRL", "FIX", "SMR", "IONS", "UI", "LRCX", "AVAV", "KTOS", "FSLR", "RMBS", "CRWV", "KRMN", "U", "PSTG", "APP", "MDB", "VRT", "MEDP", "GLW", "COHR", "CCJ", "IONQ", "NIO", "INTC", "GOOGL", "NET", "NVT", "GOOG", "RDDT", "FUTU", "BWXT", "ROIV", "ELAN", "CAT", "INCY", "KLAC", "ROKU", "AVGO", "BBIO", "CHRW", "TLN", "ASX", "SHOP", "FLEX", "APH", "NVMI", "PLTR", "B", "TSM", "ORCL", "CELH", "TSLA", "NVDA", "TEL", "GFI", "MRVL", "ALB", "IVZ", "COIN", "ANET", "CW", "WCC", "KEP", "ILMN", "CX", "IBKR", "MDGL", "DELL", "LVS", "RVMD", "TME", "ALNY", "MPWR", "SCCO", "EME", "BBD", "SNOW", "AMAT", "KGC", "EL", "XYZ", "AU", "LOGI", "GM", "NEM", "CRS", "MGA", "ASML", "WYNN", "SPXC", "FERG", "IDXX", "FTAI", "CMI", "DD", "MLI", "TPR", "RCI", "DDOG", "EVR", "NRG", "UTHR", "SHG", "VLO", "FTI", "GE", "RBLX", "AYI", "C", "WF", "BBVA", "GS", "EBR", "HPE", "JBL", "SAN", "BLD", "BIDU", "APTV", "GMAB", "CVE", "RKT", "CYBR", "PWR", "TEM", "HAL", "TEVA", "TMO", "AFRM", "ENSG", "ZS", "MS", "ONC", "ATI", "SYF", "AAPL", "GEV", "EMBJ", "PAAS", "IQV", "MTZ", "THC", "CCL", "DB", "A", "RTX", "AEM", "LTM", "VST", "MFG", "CEG", "NOK", "CFG", "WWD", "BABA", "JCI", "DASH", "SF", "HOLX", "AMX", "ARGX", "TLK", "VALE", "BCS", "ITT", "HII", "TIMB", "WST", "ULTA", "VIK", "SNX", "HUBB", "MTD", "AMZN", "AXP", "MTSI", "RL", "LHX", "SGI", "HWM", "HCA", "BIIB", "NTRS", "J", "BNS", "DCI", "NU", "EXAS", "FER", "NTES", "TOL", "CM", "PSKY", "CG", "ACM", "IMO", "SNN", "URI", "BKR", "BK", "SMCI", "PUK", "BAP", "DAL", "EA", "ON", "ARM", "KB", "PSX", "MPC", "EXPE", "UAL", "CAH", "BAC", "PEGA", "JBHT", "ROK", "VIV", "E", "TD", "STT", "EXPD", "NSC", "ETN", "F", "FOXA", "GD", "TWLO", "SBS", "BP", "ING", "WDS", "NMR", "MCHP", "NBIX", "CRWD", "XPO", "RYAAY", "WSM", "NUE", "LDOS", "JLL", "RBC", "HSBC", "CRH", "BG", "DAY", "LECO", "APG", "GSK", "BMO", "MT", "HAS", "VTRS", "JPM", "NTRA", "PDD", "QCOM", "CSX", "XYL", "NTAP", "KEYS", "PH", "UBS", "DLTR", "CACI", "GILD", "VEEV", "BNT", "AER", "QXO", "IX", "RPRX", "FOX", "NDSN", "BN", "WELL", "MUFG", "ALLY", "ASND", "TECK", "ITUB", "ULS", "YPF", "UHS", "NEE", "BSBR", "SRE", "TS", "LYG", "TRMB", "SU", "RIVN", "BTI", "RY", "NWG", "WMS", "IBM", "CSCO", "STN", "OHI", "EMR", "CBRE", "VOD", "EC", "CVNA", "LLY", "BSAC", "PODD", "WFC", "FDX", "HEI", "COF", "MMM", "DHI", "BBY", "JNJ", "ALLE", "SWK", "ADI", "PTC", "RTO", "STLD", "PHM", "ICLR", "PLD", "CVS", "PNR", "ADM", "PHG", "AGNC", "NOC", "RF", "AS", "ES", "PHYS", "BCH", "MSFT", "MCK", "HEI-A", "AME", "BLK", "ERIC", "GRAB", "DKS", "COR", "EBAY", "AMGN", "AZN", "FITB", "FHN", "PKX", "ABBV", "EWBC", "TPG", "PFGC", "SMFG", "SONY", "RIO", "KEY", "REGN", "L", "BHP", "SHEL", "TXT", "HTHT", "ETR", "MLM", "AGI", "CR", "CTRA", "TFC", "XEL", "WPM", "CPNG", "XPEV", "PCAR", "VTR", "TCOM", "NLY", "ROST", "PANW", "USB", "ENTG", "COKE", "CASY", "NXPI", "TTWO", "RCL", "CVX", "LEN", "NVS", "SPG", "PNC", "ESLT", "MNST", "TROW", "LPLA", "AEG", "PBR-A", "SSNC", "TJX", "SCHW", "BNTX", "AEP", "H", "RJF", "BEN", "BURL", "HST", "PCOR", "UBER", "AFG", "CNQ", "PBR", "AIZ", "SE", "JEF", "PKG", "MAR", "GL", "EW", "TKO", "FNV", "EVRG", "CINF", "DHR", "GEHC", "SWKS", "TOST", "EXEL", "DOCS", "HLT", "BIP", "HLI", "GRMN", "IHG", "SLB", "DIS", "DTM", "FCX", "SNA", "NDAQ", "MDT", "YMM", "XOM", "WAB", "STLA", "FWONA", "FWONK", "DG", "BXP", "PEP", "D", "LNT", "STE", "PFE", "WPC", "RNR", "WAT", "Z", "SAIL", "GPC", "CBOE", "VMC", "SCI", "USFD", "TT", "ORI", "DVN", "EMA", "MRK", "META", "TTE", "BA", "TDY", "SN", "WES", "FE", "MKL", "ZM", "AXON", "CDNS", "ADSK", "FAST", "MFC", "ATO", "BX", "FANG", "NKE", "NI", "TSCO", "WTW", "ZG", "BDX", "MTB", "TM", "CF", "BCE", "BLDR", "MAS", "LOW", "ROL", "BSY", "DOV", "MPLX", "HUM", "PFG", "LUV", "STM", "EQNR", "AFL", "VLTO", "MCO", "UMC", "NGG", "SYY", "EQT", "SUI", "DOC", "ZBRA", "MSCI", "APO", "SOLV", "ORLY", "GWRE", "IRM", "PRU", "HD", "HPQ", "WRB", "TRV", "SUZ", "WMT", "MET", "TRGP", "ISRG", "KKR", "SEIC", "INTU", "HBAN", "WEC", "PR", "SLF", "ITW", "LH", "CTVA", "OXY", "EIX", "ECL", "CNM", "MELI", "PAG", "AEE", "AVY", "UNP", "HMC", "CCK", "FTS", "NVR", "EPD", "RVTY", "YUMC", "GFS", "DGX", "DUK", "K", "HSY", "IR", "EXE", "LYV", "GPN", "LMT", "FCNCA", "HIG", "GGG", "DE", "RMD", "SBUX", "ET", "PPL", "PEG", "LAMR", "ZTO", "COP", "YUM", "ENB", "DLR", "EXC", "CNP", "ABNB", "TXN", "SO", "RDY", "WMG", "PYPL", "DT", "KIM", "O", "UPS", "QSR", "HDB", "TRP", "RPM", "SPGI", "EHC", "CHT", "CMS", "ARCC", "FTV", "MANH", "PAA", "TDG", "AZO", "MA", "WMB", "BMY", "CB", "HON", "EOG", "TU", "V", "CCEP", "GLPI", "ABEV", "TXRH", "WTRG", "OMC", "BAM", "DTE", "CRBG", "BALL", "REG", "TGT", "CNI", "MCD", "BUD", "VRTX", "PBA", "CTSH", "UL", "SHW", "AMP", "RBA", "PAGP", "RBRK", "ALL", "NTR", "UNM", "CP", "VICI", "NFLX", "MSI", "TSN", "AON", "RGLD", "JD", "KO", "BRK-A", "PCG", "NTNX", "ARES", "BKNG", "ACGL", "BSX", "SPOT", "BRK-B", "ELS", "PNW", "EQIX", "CHD", "PINS", "EG", "CME", "RGA", "VNOM", "KMI", "MO", "SJM", "OTIS", "SNY", "CSGP", "IEX", "AIG", "TRU", "FMX", "SYK", "JKHY", "RS", "LIN", "ZBH", "PSA", "ABT", "CLH", "COST", "GEN", "INFY", "WDAY", "BR", "CNA", "IOT", "GWW", "FFIV", "PG", "ESS", "NWSA", "DRI", "CRM", "WIT", "JBS", "ODFL", "VZ", "EXR", "TAK", "PPG", "SW", "IBN", "SNAP", "APD", "NOW", "AR", "NWS", "FLUT", "BMRN", "EQH", "AMCR", "CHKP", "AWK", "AKAM", "CQP", "SAP", "T", "MBLY", "KR", "ADBE", "WY", "IFF", "OWL", "ED", "HLN", "CTAS", "LII", "CHWY", "DOCU", "MKC", "LNG", "VG", "FMS", "CNH", "CCI", "FIS", "FNF", "CSL", "GFL", "CPT", "KHC", "AVB", "RACE", "GIS", "WM", "SNPS", "ADP", "VRSN", "COO", "EQR", "CL", "PM", "GIB", "TTD", "DPZ", "UNH", "WCN", "MDLZ", "CDW", "ICE", "AMH", "TYL", "OKE", "IP", "RSG", "TEF", "AMT", "TMUS", "SBAC", "INVH", "RELX", "CMCSA", "CARR", "DEO", "KSPI", "ACN", "BEKE", "FTNT", "DKNG", "UDR", "PAYC", "CPAY", "DOW", "CLX", "JHX", "ERIE", "LYB", "BF-B", "KDP", "TEAM", "MMC", "RYAN", "MAA", "EFX", "FICO", "BF-A", "ROP", "CI", "PAYX", "ALC", "ZTS", "TRI", "LI", "ELV", "WSO", "BJ", "KMB", "SMMT", "ONON", "HRL", "OKTA", "AJG", "PGR", "GDDY", "TW", "HUBS", "NVO", "MMYT", "DXCM", "BAH", "KVUE", "BRO", "VRSK", "STZ", "CPRT", "TPL", "DECK", "MSTR", "CMG", "LULU", "CNC", "CHTR", "IT", "DUOL", "FI", "CRCL", "HONIV", "FIG", "KLAR", "AMRZ", "FRMI", "STRC", "Q"]


earnings_11_6_stocks = []

earnings_ww_11_17_25_stocks = ["NVDA", "PANW", "BIDU", "BULL", "KC", "INTU", "VEEV", "WIX", "WMT", "TGT", "HD", "LOW", "TJX"]


options_11_24_stocks = ["NXT", "SOXX", "SEI", "CARR", "DHI", "AR", "CAPR", "VLO", "XBI", "ANF", "INFY", "HPQ", "PCT", "CELH", "SVIX", "DKS", "BURL", "FMC", "SYM", "LQDA", "NVO", "BIIB", "CPRT", "ZM", "BBY", "OSCR", "RUM", "AS", "SGHC", "MRK", "CNC", "ADI", "WDAY", "CRMD", "FIGR", "SIRI", "CPRI", "HSAI", "ALT", "NEE", "EPD", "BFLY", "COMP", "FRMI", "GDS", "FIVN", "PRME", "UPWK", "EA", "EH", "QQQM", "QQQ"]

demo_stocks = ["NVDA", "INTC", "GOOG", "AAPL", "OKTA", "COST", "CRDO", "ALAB", "LASE"]





comet_stocks = ["QQQ", "CRDO", "ALAB", "AMD", "INTC"]






AI_semi_stocks = ["AAPL", "AMD", "AMAT", "AMKR", "AMZN", "ARM", "ASMIY", "ASML", "ASX", "ATEYY", "AVGO", "BESIY", "CAMT", "CDNS", "DD", "ENTG", "GOOG", "INTC", "KLAC", "LRCX", "META", "MRVL", "MSFT", "NVDA", "ONTO", "QCOM", "SHECY", "SIEGY", "SNPS", "SSNLF", "SUOPY", "TER", "TOELY", "TSM"]


AA_spy_stocks = ["VOO", "IVV", "SPY", "CSPX", "TWLO", "SPXC", "SPXD", "SPXE", "SPXL", "SPXS", "SPXT", "SPXU", "SPXV", "SPXX"]






ticker_1_stocks = ["FBTC", "VOO", "QQQM", "OKTA", "TQQQ"]




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



temp_stocks = [
    "FRHC",
    "FLNG",
    "META",
    "MU",
    "BABA",
    "GOOG",
    "GOOGL",
    "AMZN",
    "MSFT",
    "ORA",
    "HIVE",
    "SPY",
    "BHP"
]



