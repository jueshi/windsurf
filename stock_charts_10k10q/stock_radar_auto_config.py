import json
import pandas as pd
import yfinance as yf

CONFIG_FILE = "config.json"
CSV_FILE = "stock_list.csv"

def convert_a_stock_code_for_yfinance(code: str) -> str:
    """
    将纯数字A股代码转换为yfinance标准格式（带交易所后缀）。
    沪市: .SS  深市: .SZ
    非A股代码直接返回原字符串（多用于美股代码）。
    """
    code = code.strip()
    if len(code) != 6 or not code.isdigit():
        return code  # 非纯6位数字，认为是美股或其他市场代码，直接返回
    if code.startswith(('6', '9')):
        return f"{code}.SS"  # 上海证券交易所
    elif code.startswith(('0', '3', '2')):
        return f"{code}.SZ"  # 深圳证券交易所
    else:
        return code  # 其他情况直接返回

def get_stock_info(symbol: str):
    """
    用 yfinance 查询股票的行业(Sector)和市场(Market)信息。
    返回 (sector, market)，查询失败返回 ("Unknown", "Unknown")
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        sector = info.get('sector', 'Unknown')
        market = info.get('market', 'Unknown')
        return sector, market
    except Exception as e:
        print(f"⚠️ 查询股票 {symbol} 信息失败: {e}")
        return "Unknown", "Unknown"

def update_stock_list_from_csv():
    # 1. 读取CSV文件，必须包含"代码"列
    df = pd.read_csv(CSV_FILE, dtype=str)
    if '代码' not in df.columns:
        print("⚠️ CSV文件缺少'代码'列，请检查格式！")
        return False

    # 2. 遍历股票代码，转换格式并查询行业和市场信息
    stock_list = []
    for raw_code in df['代码']:
        yf_code = convert_a_stock_code_for_yfinance(raw_code)
        sector, market = get_stock_info(yf_code)
        print(f"{raw_code} -> {yf_code}，行业: {sector}，市场: {market}")
        stock_list.append({
            "代码": raw_code,
            "行业": sector,
            "市场": market
        })

    # 3. 读取现有 config.json
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 4. 更新 STOCK_LIST 字段
    config['STOCK_LIST'] = stock_list

    # 5. 写回配置文件
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"✅ 成功更新配置文件 {CONFIG_FILE} 中的股票清单，共 {len(stock_list)} 支股票。")
    return True

if __name__ == "__main__":
    update_stock_list_from_csv()
