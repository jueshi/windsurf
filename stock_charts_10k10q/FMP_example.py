import requests

API_KEY = "sprOVI3aZi1pc0hZRHVGgkwxp00ylMNR"
BASE_URL_STABLE = "https://financialmodelingprep.com/stable"
# /stable/income-statement?symbol=AAPL&apikey=...
# /stable/balance-sheet-statement?...
# /stable/profile?symbol=AAPL&apikey=...

def get_income_statement(symbol: str):
    url = f"{BASE_URL_STABLE}/income-statement?symbol={symbol}&apikey={API_KEY}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

def get_balance_sheet(symbol: str):
    url = f"{BASE_URL_STABLE}/balance-sheet-statement?symbol={symbol}&apikey={API_KEY}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

def get_profile(symbol: str):
    url = f"{BASE_URL_STABLE}/profile?symbol={symbol}&apikey={API_KEY}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

def get_latest_institutional_filings(page: int = 0, limit: int = 100):
    url = f"{BASE_URL_STABLE}/institutional-ownership/latest"
    params = {
        "page": page,
        "limit": limit,
        "apikey": API_KEY,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

if __name__ == "__main__":
    data = get_income_statement("AAPL")
    print(len(data))
    print(data[:5])
    # filings = get_latest_institutional_filings(page=0, limit=50)
    # print("records:", len(filings))
    # for f in filings[:5]:
    #     print(f)


# import requests

# API_KEY = "sprOVI3aZi1pc0hZRHVGgkwxp00ylMNR"
# BASE_URL_V3 = "https://financialmodelingprep.com/api/v3"

# def get_institutional_holders_v3(symbol: str):
#     url = f"{BASE_URL_V3}/institutional-holder/{symbol}"
#     params = {"apikey": API_KEY}
#     resp = requests.get(url, params=params, timeout=10)
#     resp.raise_for_status()
#     return resp.json()

# if __name__ == "__main__":
#     data = get_institutional_holders_v3("AAPL")
#     print(len(data))
#     print(data[:5])
