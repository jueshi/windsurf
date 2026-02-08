"""
Buffett/CANSLIM Stock Analysis Module

Provides AI-powered investment analysis using two frameworks:
1. Warren Buffett's Value Investing (8 dimensions)
2. CANSLIM Growth Investing (7 dimensions)

Ported from desktop app's stock_radar_batch.py
"""
import os
import re
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Gemini model configuration with fallbacks
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
_MODEL_FALLBACKS = [
    MODEL_NAME,
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]

# Buffett analysis dimensions (8 total)
BUFFETT_LABELS = [
    "Circle of Competence",
    "Economic Moat",
    "Intrinsic Value & Margin of Safety",
    "Financial Health",
    "Management Quality",
    "Buy Timing",
    "Long-term Holding Logic",
    "Risk Control"
]

# CANSLIM analysis dimensions (7 total)
CANSLIM_LABELS = [
    "C - Current Quarterly Earnings",
    "A - Annual Earnings Growth",
    "N - New Products/Catalysts",
    "S - Supply & Demand",
    "L - Leader or Laggard",
    "I - Institutional Sponsorship",
    "M - Market Direction"
]


def analyze_stock(ticker: str) -> Dict[str, Any]:
    """
    Analyze a stock using Buffett and CANSLIM frameworks via Gemini AI.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        
    Returns:
        Dict containing:
        - text: Full AI analysis text
        - buffett_scores: List of 8 scores (0-10)
        - buffett_total: Total Buffett score (0-100)
        - canslim_scores: List of 7 scores (0-10)
        - canslim_total: Total CANSLIM score (0-100)
        - investor_type: "Value", "Growth", or "Balanced"
        - error: Error message if analysis failed
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY not configured"}
    
    # Analysis prompt (bilingual for better parsing)
    analysis_prompt = f"""
    Analyze {ticker} stock using the following frameworks:
    
    Part 1: Warren Buffett Value Investing Analysis (score each 0-10 with explanation)
    1. Circle of Competence - Is the business easy to understand?
    2. Economic Moat - Does it have sustainable competitive advantages?
    3. Intrinsic Value & Margin of Safety - Is it trading below intrinsic value?
    4. Financial Health - Strong balance sheet, low debt, good cash flow?
    5. Management Quality - Honest, capable, shareholder-friendly?
    6. Buy Timing - Is now a good time to buy?
    7. Long-term Holding Logic - Can you hold for 10+ years?
    8. Risk Control - What are the key risks?
    Calculate total score (0-100) and list top 3 competitors.
    
    Part 2: CANSLIM Growth Investing Analysis (score each 0-10 with explanation)
    1. C - Current quarterly earnings growth (>25% ideal)
    2. A - Annual earnings growth (>25% for 5 years)
    3. N - New products, management, or price highs
    4. S - Supply and demand (shares outstanding, volume)
    5. L - Leader or laggard in its industry
    6. I - Institutional sponsorship (quality funds buying)
    7. M - Market direction (bull or bear market)
    Calculate total score (0-100) and list top 3 competitors.
    
    Part 3: Output these EXACT lines for parsing:
    Buffett分数列表: X,X,X,X,X,X,X,X (8 numbers, comma-separated)
    Buffett综合评分: XX (single number 0-100)
    CANSLIM分数列表: X,X,X,X,X,X,X (7 numbers, comma-separated)
    CANSLIM综合评分: XX (single number 0-100)
    """
    
    try:
        genai.configure(api_key=api_key)
        text = ""
        last_err = None
        
        # Try model fallbacks
        for model_name in _MODEL_FALLBACKS:
            if not model_name:
                continue
            try:
                model = genai.GenerativeModel(model_name)
                resp = model.generate_content(analysis_prompt)
                text = (resp.text or "").strip()
                if text:
                    logging.info(f"Buffett/CANSLIM analysis successful with {model_name}")
                    break
            except Exception as e:
                last_err = e
                error_str = str(e)
                # Try next model if 404/not supported
                if any(tok in error_str for tok in ["404", "not found", "not supported", "Unsupported", "NOT_FOUND"]):
                    continue
                # Other errors: re-raise
                raise
        else:
            return {"error": f"All Gemini models failed. Last error: {last_err}"}
        
        # Parse scores from response
        m_blist = re.search(r'Buffett分数列表.*?(\d+(?:,\s*\d+){7})', text)
        m_btotal = re.search(r'Buffett综合评分.*?(\d+)', text)
        m_clist = re.search(r'CANSLIM分数列表.*?(\d+(?:,\s*\d+){6})', text)
        m_ctotal = re.search(r'CANSLIM综合评分.*?(\d+)', text)
        
        if not (m_blist and m_btotal and m_clist and m_ctotal):
            # Try alternative parsing patterns
            m_blist = re.search(r'(\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+)', text)
            m_clist = re.search(r'(\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+)(?!,)', text)
            
            if not (m_blist and m_btotal and m_clist and m_ctotal):
                snippet = text[:500].replace('\n', ' ')
                return {
                    "error": f"Could not parse scores from AI response",
                    "text": text,
                    "snippet": snippet
                }
        
        buffett_scores = [int(x.strip()) for x in m_blist.group(1).split(',')]
        buffett_total = int(m_btotal.group(1))
        canslim_scores = [int(x.strip()) for x in m_clist.group(1).split(',')]
        canslim_total = int(m_ctotal.group(1))
        
        # Determine investor type
        if abs(buffett_total - canslim_total) <= 5 and buffett_total >= 70 and canslim_total >= 70:
            investor_type = "Balanced"
        elif buffett_total > canslim_total:
            investor_type = "Value"
        else:
            investor_type = "Growth"
        
        return {
            "ticker": ticker,
            "text": text,
            "buffett_scores": buffett_scores,
            "buffett_total": buffett_total,
            "buffett_labels": BUFFETT_LABELS,
            "canslim_scores": canslim_scores,
            "canslim_total": canslim_total,
            "canslim_labels": CANSLIM_LABELS,
            "investor_type": investor_type,
            "analyzed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Buffett/CANSLIM analysis failed for {ticker}: {e}")
        return {"error": str(e)}


def format_analysis_html(result: Dict[str, Any]) -> str:
    """
    Format analysis result as HTML for display.
    
    Args:
        result: Analysis result from analyze_stock()
        
    Returns:
        HTML string for rendering
    """
    if "error" in result:
        return f'<div class="alert alert-danger">{result["error"]}</div>'
    
    ticker = result.get("ticker", "Unknown")
    buffett_total = result.get("buffett_total", 0)
    canslim_total = result.get("canslim_total", 0)
    investor_type = result.get("investor_type", "Unknown")
    buffett_scores = result.get("buffett_scores", [])
    canslim_scores = result.get("canslim_scores", [])
    
    # Determine badge colors
    buffett_color = "success" if buffett_total >= 70 else "warning" if buffett_total >= 50 else "danger"
    canslim_color = "success" if canslim_total >= 70 else "warning" if canslim_total >= 50 else "danger"
    type_color = "primary" if investor_type == "Balanced" else "info" if investor_type == "Value" else "success"
    
    # Build score bars
    buffett_bars = ""
    for i, (label, score) in enumerate(zip(BUFFETT_LABELS, buffett_scores)):
        pct = score * 10
        bar_color = "bg-success" if score >= 7 else "bg-warning" if score >= 5 else "bg-danger"
        buffett_bars += f'''
        <div class="mb-2">
            <small class="text-muted">{label}</small>
            <div class="progress" style="height: 20px;">
                <div class="progress-bar {bar_color}" style="width: {pct}%">{score}/10</div>
            </div>
        </div>
        '''
    
    canslim_bars = ""
    for i, (label, score) in enumerate(zip(CANSLIM_LABELS, canslim_scores)):
        pct = score * 10
        bar_color = "bg-success" if score >= 7 else "bg-warning" if score >= 5 else "bg-danger"
        canslim_bars += f'''
        <div class="mb-2">
            <small class="text-muted">{label}</small>
            <div class="progress" style="height: 20px;">
                <div class="progress-bar {bar_color}" style="width: {pct}%">{score}/10</div>
            </div>
        </div>
        '''
    
    html = f'''
    <div class="card mb-3">
        <div class="card-header d-flex justify-content-between align-items-center">
            <h5 class="mb-0">{ticker} Investment Analysis</h5>
            <span class="badge bg-{type_color}">{investor_type} Investor Profile</span>
        </div>
        <div class="card-body">
            <div class="row">
                <div class="col-md-6">
                    <h6>
                        <span class="badge bg-{buffett_color}">{buffett_total}/100</span>
                        Buffett Value Analysis
                    </h6>
                    {buffett_bars}
                </div>
                <div class="col-md-6">
                    <h6>
                        <span class="badge bg-{canslim_color}">{canslim_total}/100</span>
                        CANSLIM Growth Analysis
                    </h6>
                    {canslim_bars}
                </div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <div class="card-header">
            <h6 class="mb-0">Detailed Analysis</h6>
        </div>
        <div class="card-body" style="max-height: 400px; overflow-y: auto;">
            <pre style="white-space: pre-wrap; font-size: 0.85em;">{result.get("text", "")}</pre>
        </div>
    </div>
    '''
    
    return html


def get_radar_chart_data(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get data formatted for Plotly.js radar chart.
    
    Args:
        result: Analysis result from analyze_stock()
        
    Returns:
        Dict with Plotly radar chart data
    """
    if "error" in result:
        return {"error": result["error"]}
    
    buffett_scores = result.get("buffett_scores", [0] * 8)
    canslim_scores = result.get("canslim_scores", [0] * 7)
    
    # Pad CANSLIM to 8 dimensions for overlay
    canslim_padded = canslim_scores + [0]
    
    return {
        "buffett": {
            "r": buffett_scores + [buffett_scores[0]],  # Close the polygon
            "theta": BUFFETT_LABELS + [BUFFETT_LABELS[0]],
            "name": f"Buffett ({result.get('buffett_total', 0)}/100)",
            "fill": "toself",
            "fillcolor": "rgba(0, 0, 255, 0.25)",
            "line": {"color": "blue"}
        },
        "canslim": {
            "r": canslim_padded + [canslim_padded[0]],
            "theta": BUFFETT_LABELS + [BUFFETT_LABELS[0]],  # Use same labels for overlay
            "name": f"CANSLIM ({result.get('canslim_total', 0)}/100)",
            "fill": "toself",
            "fillcolor": "rgba(255, 0, 0, 0.25)",
            "line": {"color": "red"}
        },
        "ticker": result.get("ticker", ""),
        "investor_type": result.get("investor_type", "")
    }
