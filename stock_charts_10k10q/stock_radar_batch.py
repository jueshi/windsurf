import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager

# ===== Matplotlib 字体配置（中文显示） =====
def _configure_chinese_font():
    # 常见中文字体（Windows优先），按优先级排列
    candidates = [
        'Microsoft YaHei',  # C:\\Windows\\Fonts\\msyh.ttf
        'SimHei',           # C:\\Windows\\Fonts\\simhei.ttf
        'Noto Sans CJK SC',
        'MS Gothic',
        'DejaVu Sans',
    ]
    rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块
    available = []
    for fam in candidates:
        try:
            # Validate availability; skip if not found
            font_manager.findfont(font_manager.FontProperties(family=fam), fallback_to_default=False)
            available.append(fam)
        except Exception:
            continue
    if not available:
        available = ['DejaVu Sans']
    rcParams['font.sans-serif'] = available

_configure_chinese_font()
import numpy as np
import pandas as pd
import yfinance as yf
import re
import os
from datetime import datetime, timedelta
import google.generativeai as genai
import os

# ===== 用户配置 =====
stock_list = ["META", "NVDA", "MSFT"]  # 你要跟踪的股票代码列表
api_key = os.getenv("GEMINI_API_KEY")
# Prefer env override; default to a current flash model. We'll also try fallbacks automatically.
model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

_MODEL_FALLBACKS = [
    # Order matters; most capable first
    model_name,
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash-8b",
]

def analyze_stock(stock_code):
    """调用AI分析单只股票，并返回结果字典"""
    analysis_prompt = f"""
    请按以下框架分析 {stock_code}：
    第一部分：沃伦·巴菲特价值投资分析（每项0-10分并解释）
    1. 能力圈
    2. 护城河
    3. 内在价值与安全边际
    4. 财务健康
    5. 管理层素质
    6. 买入时机
    7. 长期持有逻辑
    8. 风险控制
    计算综合评分（0-100）and list the top 3 competitors with highest score

    第二部分：CANSLIM 成长型投资分析（每项0-10分并解释）
    1. C 当前季度盈利增长
    2. A 年度盈利增长
    3. N 新催化剂
    4. S 供需关系
    5. L 行业领先度
    6. I 机构支持
    7. M 市场趋势
    计算综合评分（0-100）and list the top 3 competitors with highest score

    第三部分：请单独列出纯数字格式：
    Buffett分数列表（长度8，用逗号隔开）
    Buffett综合评分（数字）
    CANSLIM分数列表（长度7，用逗号隔开）
    CANSLIM综合评分（数字）
    """
    genai.configure(api_key=api_key)
    last_err = None
    text = ""
    # Try candidates until one succeeds
    for _name in _MODEL_FALLBACKS:
        try:
            if not _name:
                continue
            model = genai.GenerativeModel(_name)
            resp = model.generate_content(analysis_prompt)
            text = (resp.text or "").strip()
            if text:
                break
        except Exception as e:
            last_err = e
            # Try next model if 404/not supported
            if any(tok in str(e) for tok in ["404", "not found", "not supported", "Unsupported", "NOT_FOUND"]):
                continue
            # Other errors: re-raise to surface auth/rate-limit issues
            raise
    else:
        raise Exception(f"Failed to generate content with Gemini models. Last error: {last_err}")
    # Robust extraction with validation
    m_blist = re.search(r'Buffett分数列表.*?(\d+(?:,\d+){7})', text)
    m_btotal = re.search(r'Buffett综合评分.*?(\d+)', text)
    m_clist = re.search(r'CANSLIM分数列表.*?(\d+(?:,\d+){6})', text)
    m_ctotal = re.search(r'CANSLIM综合评分.*?(\d+)', text)

    if not (m_blist and m_btotal and m_clist and m_ctotal):
        snippet = text[:400].replace('\n', ' ')
        raise ValueError(f"无法从模型输出解析评分，请重试或更换提示。输出片段: {snippet}")

    buffett_scores = list(map(int, m_blist.group(1).split(',')))
    buffett_total = int(m_btotal.group(1))
    canslim_scores = list(map(int, m_clist.group(1).split(',')))
    canslim_total = int(m_ctotal.group(1))

    # 投资者画像匹配
    if abs(buffett_total - canslim_total) <= 5 and buffett_total >= 70 and canslim_total >= 70:
        investor_type = "平衡型"
    elif buffett_total > canslim_total:
        investor_type = "价值型"
    else:
        investor_type = "成长型"

    return {
        "text": text,
        "buffett_scores": buffett_scores,
        "buffett_total": buffett_total,
        "canslim_scores": canslim_scores,
        "canslim_total": canslim_total,
        "investor_type": investor_type
    }

def save_history(stock_code, record):
    """保存分析记录到CSV"""
    file_name = f"{stock_code}_analysis_history.csv"
    record["日期"] = datetime.today().strftime("%Y-%m-%d")
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        df = pd.DataFrame(columns=["日期", "Buffett综分", "CANSLIM综分", "投资者类型"])
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(file_name, index=False)
    return df

def plot_analysis(stock_code, buffett_scores, buffett_total, canslim_scores, canslim_total, hist_df):
    """绘制雷达图和时间序列趋势"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 雷达图
    labels = [
        "能力圈", "护城河", "内在价值与安全边际", "财务健康",
        "管理层素质", "买入时机", "长期持有逻辑", "风险控制"
    ]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    b_plot = buffett_scores + [buffett_scores[0]]
    c_plot = (canslim_scores + [0]) + [canslim_scores[0]]
    angles_plot = angles + [angles[0]]

    axes[0] = plt.subplot(1, 2, 1, polar=True)
    axes[0].plot(angles_plot, b_plot, color='blue', linewidth=2, label='Buffett Style')
    axes[0].fill(angles_plot, b_plot, color='blue', alpha=0.25)
    axes[0].plot(angles_plot, c_plot, color='red', linewidth=2, label='CANSLIM Style')
    axes[0].fill(angles_plot, c_plot, color='red', alpha=0.25)
    axes[0].set_theta_offset(np.pi / 2)
    axes[0].set_theta_direction(-1)
    axes[0].set_thetagrids(np.degrees(angles), labels, fontsize=10)
    axes[0].set_ylim(0, 10)
    axes[0].set_title(f"{stock_code} 雷达图\nBuffett:{buffett_total} | CANSLIM:{canslim_total}")
    axes[0].legend(loc='upper right')

    # 历史价格 + 评分趋势
    end_date = datetime.today()
    start_date = end_date - timedelta(days=180)
    price_data = yf.download(stock_code, start=start_date, end=end_date)["Close"]

    # 解决中文显示问题（避免使用不存在的字体）
    try:
        from matplotlib import font_manager as _fm
        _cands = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'MS Gothic', 'DejaVu Sans']
        _avail = []
        for _f in _cands:
            try:
                _fm.findfont(_fm.FontProperties(family=_f), fallback_to_default=False)
                _avail.append(_f)
            except Exception:
                continue
        if not _avail:
            _avail = ['DejaVu Sans']
        plt.rcParams['font.sans-serif'] = _avail
    except Exception:
        pass
    plt.rcParams['axes.unicode_minus'] = False

    ax_price = axes[1]
    ax_price.plot(price_data.index, price_data.values, label="价格", color="black")
    ax_price.set_ylabel("价格")
    ax_score = ax_price.twinx()
    ax_score.plot(pd.to_datetime(hist_df["日期"]), hist_df["Buffett综分"], label="Buffett综分", color="blue", linestyle="--", marker="o")
    ax_score.plot(pd.to_datetime(hist_df["日期"]), hist_df["CANSLIM综分"], label="CANSLIM综分", color="red", linestyle="--", marker="o")
    ax_price.set_title(f"{stock_code} 价格 vs 历史评分")
    ax_price.legend(loc="upper left")
    ax_score.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

# ===== 主程序 =====

if __name__ == "__main__":
    for stock in stock_list:
        print(f"\n=== 分析 {stock} ===")
        result = analyze_stock(stock)
        print(result)
        hist_df = save_history(stock, {
            "Buffett综分": result["buffett_total"],
        "CANSLIM综分": result["canslim_total"],
        "投资者类型": result["investor_type"]
    })
    plot_analysis(stock, result["buffett_scores"], result["buffett_total"], result["canslim_scores"], result["canslim_total"], hist_df)
