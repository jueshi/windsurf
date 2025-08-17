
import os
from dotenv import load_dotenv
from sec_edgar_downloader import Downloader

# Check environment variables
load_dotenv()
sec_email = os.getenv("SEC_EDGAR_EMAIL")
print(f"SEC_EDGAR_EMAIL: {'Set' if sec_email else 'Not set'}")

# 初始化下载器，需要提供公司名和邮箱（SEC要求用于User-Agent）
dl = Downloader("Stone & Associates Inc", sec_email)

# 下载微软（MSFT）的所有10-K文件，保存在默认目录
dl.get("10-K", "MSFT")

# 下载苹果（AAPL）最近1份10-K文件
res=dl.get("10-K", "GOOG", limit=1)
print(res)

# 下载特定时间段内的10-K文件（例如2018年后，2020年前）
dl.get("10-K", "AAPL", after="2018-01-01", before="2020-01-01")
