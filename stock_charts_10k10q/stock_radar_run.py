import os
import base64
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# ================= 配置区域 =================
stock_list = [
    {"代码": "AAPL", "行业": "Technology"},
    {"代码": "NVDA", "行业": "Technology"},
    {"代码": "MSFT", "行业": "Technology"}
]
industry_etf_map = {"Technology": "XLK", "Default": "^GSPC"}

# ================= 字体工具 =================
def find_chinese_ttf_font():
    """在Windows系统中查找常见的中文TTF字体，返回(字体族名, 字体路径)。"""
    candidates = [
        ("SimHei", r"C:\\Windows\\Fonts\\simhei.ttf"),       # 黑体
        ("MSYH",  r"C:\\Windows\\Fonts\\msyh.ttf"),        # 微软雅黑
        ("MSYHL", r"C:\\Windows\\Fonts\\msyhl.ttf"),       # 微软雅黑Light
        ("MSJH",  r"C:\\Windows\\Fonts\\msjh.ttf"),        # 微软正黑体（部分系统）
    ]
    for family, path in candidates:
        if os.path.exists(path):
            return family, path
    # 兜底：返回英文核心字体，后续使用时仍可能报错（提醒用户安装字体）
    return "Arial", None

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SMTP_USER = "jueshi@gmail.com"
SMTP_PASSWORD = "xond wlco mygx abyd"
MAIL_TO = ["jueshi@gmail.com"]

# ================= 工具函数 =================
trend_color_map = {"↑": "#00AA00", "↓": "#CC0000", "→": "#888888"}
conclusion_color_map = {
    "双强 - 加仓": ("#00AA00", "#FFFFFF"),
    "基本面走强/价格回调 - 关注买点": ("#0066CC", "#FFFFFF"),
    "价格走强/基本面下滑 - 谨慎追高": ("#FF9900", "#000000"),
    "双弱 - 规避": ("#CC0000", "#FFFFFF"),
    "观望": ("#888888", "#FFFFFF"),
}

def get_score_change(stock_code, days=14):
    hist_file = f"{stock_code}_history.csv"
    if not os.path.exists(hist_file):
        return 0
    df = pd.read_csv(hist_file)
    df = df.sort_values("日期").tail(days)
    return df["平均分"].iloc[-1] - df["平均分"].iloc[0]

def get_price_change_pct(stock_code, days=30):
    df = yf.download(stock_code, period=f"{days}d", interval="1d")
    if df.empty:
        return 0
    close = df["Close"]
    return (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100

def classify_signal_action(score_change, price_change):
    delta = 0.5; pct = 0.5
    if score_change > delta and price_change > pct:
        return "双强 - 加仓"
    elif score_change > delta and price_change < -pct:
        return "基本面走强/价格回调 - 关注买点"
    elif score_change < -delta and price_change > pct:
        return "价格走强/基本面下滑 - 谨慎追高"
    elif score_change < -delta and price_change < -pct:
        return "双弱 - 规避"
    else:
        return "观望"

# ================= 图表生成 =================
def generate_single_score_chart(stock_code, trend_arrow=None):
    hist_file = f"{stock_code}_history.csv"
    if not os.path.exists(hist_file):
        return ""
    df = pd.read_csv(hist_file)
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期")
    df["MA30"] = df["平均分"].rolling(window=30, min_periods=1).mean()
    df["STD30"] = df["平均分"].rolling(window=30, min_periods=1).std().fillna(0)
    arrow_color = trend_color_map.get(trend_arrow, "#000000") if trend_arrow else "#000000"
    plt.figure(figsize=(6,4))
    plt.plot(df["日期"], df["平均分"], marker="o", label="平均分")
    plt.plot(df["日期"], df["MA30"], linestyle="--", label="30日均线")
    plt.fill_between(df["日期"], df["MA30"]-df["STD30"], df["MA30"]+df["STD30"], alpha=0.15)
    plt.title(f"{stock_code} 评分趋势 {trend_arrow or ''}", fontsize=10, color=arrow_color)
    plt.ylabel("分数"); plt.legend(); plt.tight_layout()
    buf = BytesIO(); plt.savefig(buf, format="png"); plt.close()
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

def generate_price_vs_sector_chart(stock_code, industry, trend_arrow=None, days=90):
    etf = industry_etf_map.get(industry, industry_etf_map["Default"])
    stock_data = yf.download(stock_code, period=f"{days}d", interval="1d")["Close"]
    etf_data = yf.download(etf, period=f"{days}d", interval="1d")["Close"]
    if len(stock_data) < 2 or len(etf_data) < 2: return ""
    stock_norm = stock_data / stock_data.iloc[0] * 100
    etf_norm = etf_data / etf_data.iloc[0] * 100
    arrow_color = trend_color_map.get(trend_arrow, "#000000")
    plt.figure(figsize=(7,4))
    plt.plot(stock_norm.index, stock_norm.values, label=f"{stock_code} 价格", color="blue")
    plt.plot(etf_norm.index, etf_norm.values, label=f"{etf} 行业指数", color="orange")
    plt.title(f"{stock_code} vs {etf} {trend_arrow or ''}", fontsize=11, color=arrow_color)
    plt.ylabel("相对基准（首日=100）"); plt.legend(); plt.tight_layout()
    buf = BytesIO(); plt.savefig(buf, format="png"); plt.close()
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

def generate_sparkline_base64(stock_code, days=14):
    hist_file = f"{stock_code}_history.csv"
    if not os.path.exists(hist_file): return ""
    df = pd.read_csv(hist_file)
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期").tail(days)
    plt.figure(figsize=(1.2,0.4))
    plt.plot(df["平均分"], color="#007ACC", linewidth=1)
    plt.axis('off'); plt.tight_layout(pad=0)
    buf = BytesIO(); plt.savefig(buf, format="png", dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(); buf.seek(0)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.read()).decode()}" width="40" height="12"/>'

def generate_price_sparkline_base64(stock_code, days=30):
    try:
        data = yf.download(stock_code, period=f"{days}d", interval="1d")["Close"]
        if data.empty: return ""
        plt.figure(figsize=(1.2,0.4))
        plt.plot(data.values, color="#FF6600", linewidth=1)
        plt.axis('off'); plt.tight_layout(pad=0)
        buf = BytesIO(); plt.savefig(buf, format="png", dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(); buf.seek(0)
        return f'<img src="data:image/png;base64,{base64.b64encode(buf.read()).decode()}" width="40" height="12"/>'
    except: return ""

# ================= HTML生成 =================
def generate_overview_bar_with_links(df):
    color_map = {"高": "#00AA00", "中": "#FFCC00", "低": "#CC0000"}
    html = '<div style="font-size:14px; margin-bottom:10px;">'
    for level in ["高","中","低"]:
        subset = df[df["信号等级"]==level]
        count = len(subset); avg_score = round(subset["平均分"].mean(),2) if count>0 else "-"
        html += f'<a href="#{level}" style="text-decoration:none;"><span style="background-color:{color_map[level]};color:#fff;padding:4px 8px;border-radius:4px;margin-right:8px;">{level} 信号: {count}只 | 均分:{avg_score}</span></a>'
    html += "</div>"; return html

def generate_html_high_signal_table(df):
    color_map = {"高": "#00AA00"}
    trend_color_map = trend_color_map_global = {"↑": "#00AA00", "↓": "#CC0000", "→": "#888888"}
    high_df = df[df["信号等级"]=="高"].sort_values("排名")
    if high_df.empty: return "<p>今日无高信号股票。</p>"
    html = '<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;font-size:13px;text-align:center;"><tr><th>信号</th><th>股票代码</th><th>平均分</th><th>趋势</th><th>评分趋势</th><th>价格趋势</th><th>AI点评</th><th>结论</th></tr>'
    for _, row in high_df.iterrows():
        dot = f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background-color:{color_map["高"]};"></span>'
        trend_arrow = f'<span style="color:{trend_color_map.get(row["趋势"], "#000")};font-weight:bold;">{row["趋势"]}</span>'
        score_spark = generate_sparkline_base64(row["股票代码"])
        price_spark = generate_price_sparkline_base64(row["股票代码"])
        score_chg = get_score_change(row["股票代码"])
        price_chg = get_price_change_pct(row["股票代码"])
        conclusion = classify_signal_action(score_chg, price_chg)
        bg,text = conclusion_color_map.get(conclusion,("#FFF","#000"))
        conclusion_cell = f'<span style="background-color:{bg};color:{text};padding:2px 4px;border-radius:3px;">{conclusion}</span>'
        html += f"<tr><td>{dot}</td><td>{row['股票代码']}</td><td>{row['平均分']}</td><td>{trend_arrow}</td><td>{score_spark}</td><td>{price_spark}</td><td style='text-align:left;'>{row['点评']}</td><td>{conclusion_cell}</td></tr>"
    html += "</table>"; return html

# ================= PDF生成 =================
def add_summary_table_with_colors(pdf, df_summary):
    signal_bg_map = {
        "高": (0,170,0, 255,255,255),
        "中": (255,204,0, 0,0,0),
        "低": (204,0,0, 255,255,255)
    }
    trend_rgb_map = {"↑": (0,170,0), "↓": (204,0,0), "→": (136,136,136)}
    pdf.set_font("CN_FONT", 'B', 10)
    pdf.cell(25,8,"股票代码",border=1)
    pdf.cell(35,8,"所属行业",border=1)
    pdf.cell(20,8,"平均分",border=1)
    pdf.cell(15,8,"趋势",border=1)
    pdf.cell(20,8,"信号",border=1)
    pdf.cell(30,8,"趋势图",border=1)
    pdf.cell(55,8,"AI点评",border=1,ln=True)
    pdf.set_font("CN_FONT", '', 9)
    high_df = df_summary[df_summary["信号等级"]=="高"].sort_values("排名")
    for _, row in high_df.iterrows():
        pdf.cell(25,8,row["股票代码"],border=1)
        pdf.cell(35,8,row["所属行业"],border=1)
        pdf.cell(20,8,str(row["平均分"]),border=1)
        tr_color = trend_rgb_map.get(row["趋势"], (0,0,0))
        pdf.set_text_color(*tr_color)
        pdf.cell(15,8,row["趋势"],border=1)
        pdf.set_text_color(0,0,0)
        bg_r,bg_g,bg_b,fg_r,fg_g,fg_b = signal_bg_map.get(row["信号等级"])
        pdf.set_fill_color(bg_r,bg_g,bg_b)
        pdf.set_text_color(fg_r,fg_g,fg_b)
        pdf.cell(20,8,row["信号等级"],border=1,fill=True)
        pdf.set_text_color(0,0,0)
        spark_data_url = generate_sparkline_base64(row["股票代码"])
        if spark_data_url and "," in spark_data_url:
            spark_img = spark_data_url.split(",", 1)[1]
            x, y = pdf.get_x(), pdf.get_y()
            pdf.cell(30,8,"",border=1)
            # 此处需要保存图片再插入，略
        else:
            pdf.cell(30,8,"-",border=1)
        pdf.multi_cell(55,8,row["点评"],border=1)

# ================= 邮件发送 =================
def send_email(subject, body_html, file_paths=[]):
    msg = MIMEMultipart()
    msg["From"] = SMTP_USER; msg["To"] = ", ".join(MAIL_TO); msg["Subject"] = subject
    msg.attach(MIMEText(body_html, "html", "utf-8"))
    for path in file_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                part = MIMEApplication(f.read())
                part.add_header("Content-Disposition", "attachment", filename=os.path.basename(path))
                msg.attach(part)
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, MAIL_TO, msg.as_string())
    print("✅ 邮件已发送")
    
def load_real_scores():
    # 这里假设你有一个 score_history 文件夹，每只股票一个 CSV
    all_latest = []
    for stock in stock_list:
        hist_file = f"score_history/{stock['代码']}_history.csv"
        if not os.path.exists(hist_file):
            continue
        df = pd.read_csv(hist_file)
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.sort_values("日期")
        latest = df.iloc[-1]
        
        # 趋势判断
        score_change = latest["平均分"] - df.iloc[-8]["平均分"]  # 过去7天变化
        if score_change > 0.5:
            trend = "↑"
        elif score_change < -0.5:
            trend = "↓"
        else:
            trend = "→"

        # 信号等级（你可以根据行业自适应阈值计算）
        if latest["平均分"] >= 85:
            signal = "高"
        elif latest["平均分"] >= 70:
            signal = "中"
        else:
            signal = "低"

        all_latest.append({
            "股票代码": stock["代码"],
            "所属行业": stock["行业"],
            "平均分": round(latest["平均分"], 2),
            "趋势": trend,
            "信号等级": signal,
            "点评": latest.get("点评", ""),
        })
    df_summary = pd.DataFrame(all_latest)
    df_summary["排名"] = df_summary["平均分"].rank(ascending=False).astype(int)
    return df_summary

# ================= 主流程 =================
def run_stock_radar():
    # df_summary = pd.DataFrame([
    #     {"股票代码": s["代码"], "所属行业": s["行业"], "平均分": 88+i*3,
    #      "趋势": "↑", "信号等级": "高", "排名": i+1, "点评": "示例点评"}
    #     for i, s in enumerate(stock_list)
    # ])
    df_summary = load_real_scores()

    overview_bar = generate_overview_bar_with_links(df_summary)
    high_signal_table = generate_html_high_signal_table(df_summary)
    collapsed_html, expanded_html = "", ""  # 可加折叠/展开逻辑
    body_html = f"{overview_bar}<h3>🔥 高信号股票概览</h3>{high_signal_table}{collapsed_html}{expanded_html}"
    pdf_file = f"stock_report_{datetime.today().strftime('%Y%m%d')}.pdf"
    pdf = FPDF()
    pdf.add_page()
    # 注册中文字体
    font_family, font_path = find_chinese_ttf_font()
    if font_path:
        pdf.add_font("CN_FONT", '', font_path, uni=True)
        pdf.add_font("CN_FONT", 'B', font_path, uni=True)
    else:
        # 未找到中文字体，仍尝试英文；但可能导致Unicode报错
        pdf.set_font("Arial", '', 12)
    # 使用中文字体输出
    pdf.set_font("CN_FONT", 'B', 20)
    pdf.cell(0,10,f"股票优先级日报 - {datetime.today().strftime('%Y-%m-%d')}",ln=True,align="C")
    pdf.ln(10)
    pdf.set_font("CN_FONT", '', 12)
    pdf.multi_cell(0,8,"本报告包含每日多因子打分、行业自适应信号评级、趋势分析与个性化点评。")
    pdf.ln(10); pdf.set_font("CN_FONT", 'B', 14)
    pdf.cell(0,10,"高信号股票摘要",ln=True)
    add_summary_table_with_colors(pdf, df_summary)
    pdf.output(pdf_file)
    excel_file = pdf_file.replace(".pdf",".xlsx")
    df_summary.to_excel(excel_file,index=False)
    send_email(f"股票优先级报告 - {datetime.today().strftime('%Y-%m-%d')}", body_html, [pdf_file, excel_file])

if __name__ == "__main__":
    run_stock_radar()
