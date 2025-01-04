# Stock Ticker Lists: use _stocks or _tickers to tell main program to process them

# List of tickers to process
Jues401k_stocks = ['ALAB','PSTR','QQQ', 'IWM', 'GLD','AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B','AVGO',
           'COST','MCD','BABA','AMD','NIO','AFRM','CQQQ','SPYX','SPYV','SPYU','CRM','ADI','TXN','AAOI','EWS','NKE',
           'AMZA','YINN','JD','BIDU','TNA','TECS','TECL','INTC','TSM','LRCX','MRVL','SPMO','WDC']
 
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