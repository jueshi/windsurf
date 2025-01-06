# -*- coding: utf-8 -*-
import sys
import io

# Set the default encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import json

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

# Optional: print tickers_comment_dict for debugging
if __name__ == '__main__':
    import json
    print(json.dumps(tickers_comment_dict, indent=2, ensure_ascii=False))
