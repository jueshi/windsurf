import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1. 拉取 QQQ 2025 日线数据
ticker = "QQQ"
start = "2025-01-01"
end   = "2025-12-31"

qqq = yf.download(ticker, start=start, end=end)

# 2. 定义 2025 年月度期权到期日（OPEX）
opex_dates = pd.to_datetime([
    "2025-01-17",
    "2025-02-21",
    "2025-03-21",
    "2025-04-17",  # 提前到周四（耶稣受难日假期）
    "2025-05-16",
    "2025-06-20",
    "2025-07-18",
    "2025-08-15",
    "2025-09-19",
    "2025-10-17",
    "2025-11-21",
    "2025-12-19",
])

# 只保留真实存在于交易数据索引中的日期（防止遇到非交易日）
opex_trading_days = [d for d in opex_dates if d in qqq.index]

# 3. 画 QQQ 收盘价，并标注 OPEX
plt.figure(figsize=(14, 7))
plt.plot(qqq.index, qqq["Close"], label="QQQ Close", color="black")

for d in opex_trading_days:
    # 画垂直虚线
    plt.axvline(d, color="red", linestyle="--", alpha=0.6)
    # 在该日上方标注日期（MM-DD）
    y = qqq.loc[d, "Close"]
    plt.text(
        d, y * 1.01, d.strftime("%m-%d"),
        rotation=90, va="bottom", ha="center",
        fontsize=8, color="red"
    )

plt.title("QQQ 2025 with Monthly Options Expiration Dates (OPEX)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
