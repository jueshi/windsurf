import os
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

try:
    from google import genai
except ImportError:
    genai = None
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import rcParams
from matplotlib import font_manager

# Configure Chinese font (Windows-first)
_DEF_FONT_CANDIDATES = [
    'Microsoft YaHei',
    'SimHei',
    'MS Gothic',
    'Noto Sans CJK SC',
]
rcParams['axes.unicode_minus'] = False
_available = []
for _fam in _DEF_FONT_CANDIDATES + ['DejaVu Sans']:
    try:
        font_manager.findfont(font_manager.FontProperties(family=_fam), fallback_to_default=False)
        _available.append(_fam)
    except Exception:
        continue
if not _available:
    _available = ['DejaVu Sans']
rcParams['font.sans-serif'] = _available

MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-flash')

# Local import: this file sits next to stock_radar_batch.py
from stock_radar_batch import analyze_stock

def analyze_stock_scores(ticker: str) -> Dict[str, object]:
    """Reuse analyze_stock() from stock_radar_batch.py and normalize keys.

    Returns a dict containing:
      - buffett_scores: List[int] len=8
      - buffett_total: int
      - canslim_scores: List[int] len=7
      - canslim_total: int
      - investor_type: str
      - raw_text: str (alias of 'text' from analyze_stock)
    """
    res = analyze_stock(ticker)
    # Ensure expected keys exist
    out = {
        'buffett_scores': res.get('buffett_scores', []),
        'buffett_total': res.get('buffett_total', 0),
        'canslim_scores': res.get('canslim_scores', []),
        'canslim_total': res.get('canslim_total', 0),
        'investor_type': res.get('investor_type', ''),
        'raw_text': res.get('text', ''),
    }
    return out


def build_analysis_figure(
    ticker: str,
    buffett_scores: List[int],
    buffett_total: int,
    canslim_scores: List[int],
    canslim_total: int,
    history_df: Optional[pd.DataFrame] = None,
    price_days: int = 180,
):
    """
    Borrowed from plot_analysis() in stock_radar_batch.py, modified to return a Matplotlib Figure
    and not call plt.show(). If history_df is None, a minimal DF is built from current result.

    history_df: DataFrame with columns ["日期", "Buffett综分", "CANSLIM综分"].
    Returns: matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Radar chart
    labels = [
        "能力圈", "护城河", "内在价值与安全边际", "财务健康",
        "管理层素质", "买入时机", "长期持有逻辑", "风险控制"
    ]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles_plot = angles + [angles[0]]

    b_plot = buffett_scores + [buffett_scores[0]]
    # Expand CANSLIM to 8 points for overlay (append 0)
    c_plot = (canslim_scores + [0]) + [canslim_scores[0]]

    ax0 = axes[0] = plt.subplot(1, 2, 1, polar=True)
    ax0.plot(angles_plot, b_plot, color='blue', linewidth=2, label='Buffett Style')
    ax0.fill(angles_plot, b_plot, color='blue', alpha=0.25)
    ax0.plot(angles_plot, c_plot, color='red', linewidth=2, label='CANSLIM Style')
    ax0.fill(angles_plot, c_plot, color='red', alpha=0.25)
    ax0.set_theta_offset(np.pi / 2)
    ax0.set_theta_direction(-1)
    ax0.set_thetagrids(np.degrees(angles), labels, fontsize=10)
    ax0.set_ylim(0, 10)
    ax0.set_title(f"{ticker} 雷达图\nBuffett:{buffett_total} | CANSLIM:{canslim_total}")
    ax0.legend(loc='upper right')

    # Price + score trend
    end_date = datetime.today()
    start_date = end_date - timedelta(days=price_days)
    try:
        price_data = yf.download(ticker, start=start_date, end=end_date)["Close"]
    except Exception:
        price_data = pd.Series(dtype=float)

    if history_df is None or history_df.empty:
        history_df = pd.DataFrame([
            {
                "日期": datetime.today().strftime("%Y-%m-%d"),
                "Buffett综分": buffett_total,
                "CANSLIM综分": canslim_total,
            }
        ])

    ax_price = axes[1]
    if not price_data.empty:
        ax_price.plot(price_data.index, price_data.values, label="价格", color="black")
    ax_price.set_ylabel("价格")
    ax_score = ax_price.twinx()
    ax_score.plot(pd.to_datetime(history_df["日期"]), history_df["Buffett综分"], label="Buffett综分", color="blue", linestyle="--", marker="o")
    ax_score.plot(pd.to_datetime(history_df["日期"]), history_df["CANSLIM综分"], label="CANSLIM综分", color="red", linestyle="--", marker="o")
    ax_price.set_title(f"{ticker} 价格 vs 历史评分")
    ax_price.legend(loc="upper left")
    ax_score.legend(loc="upper right")

    plt.tight_layout()
    return fig


def save_study_markdown(ticker: str, result: Dict[str, object], figure=None, base_dir: Optional[str] = None) -> Tuple[str, Optional[str]]:
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'buffett_canslim')
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    img_path = None
    if figure is not None:
        img_path = os.path.join(base_dir, f"{ticker}_{ts}.png")
        try:
            figure.savefig(img_path, dpi=150, bbox_inches='tight')
        except Exception:
            img_path = None
    md_path = os.path.join(base_dir, f"{ticker}.md")
    with open(md_path, 'a', encoding='utf-8') as f:
        f.write(f"\n\n## {ticker} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n- Buffett total: {result.get('buffett_total', 0)}\n")
        f.write(f"\n- CANSLIM total: {result.get('canslim_total', 0)}\n")
        f.write(f"\n- Investor type: {result.get('investor_type', '')}\n")
        f.write(f"\n- Buffett scores: {result.get('buffett_scores', [])}\n")
        f.write(f"\n- CANSLIM scores: {result.get('canslim_scores', [])}\n")
        if img_path:
            f.write(f"\n![{ticker} analysis]({os.path.basename(img_path)})\n")
        raw = result.get('raw_text', '')
        if raw:
            f.write(f"\n\n{raw}\n")
    return md_path, img_path
