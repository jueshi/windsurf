from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from datetime import datetime
from ..data_manager import data_manager
from ..schemas import filter_fundamental_data
from .. import gemini_analyzer
from .. import buffett_canslim
from ..strategy_cache import strategy_cache
import logging
import markdown
import asyncio
import os

router = APIRouter(
    prefix="/analysis",
    tags=["analysis"]
)

templates = Jinja2Templates(directory="webapp/templates")


def _safe_markdown(text: str) -> str:
    if not isinstance(text, str):
        return "<div class='alert alert-warning mb-0'>No analysis text generated.</div>"
    t = text.strip()
    if not t:
        return "<div class='alert alert-warning mb-0'>No analysis text generated.</div>"
    try:
        return markdown.markdown(t, extensions=['tables', 'fenced_code'])
    except Exception as e:
        logging.error(f"Markdown render error: {e}")
        return f"<div class='alert alert-danger mb-0'>Error rendering markdown: {str(e)}</div>"

def _save_md(subdir: str, name: str, text: str) -> str:
    try:
        base = os.path.join("webapp", "output", subdir)
        os.makedirs(base, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe = "".join([c for c in name if c.isalnum() or c in ("-", "_")]) or "analysis"
        path = os.path.join(base, f"{safe}_{ts}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text if isinstance(text, str) else str(text))
        return path
    except Exception as e:
        logging.error(f"Save markdown failed: {e}")
        return ""

class TickerStrategyRequest(BaseModel):
    ticker: str
    scenario: str = "neutral"
    timeframe: str = "swing"
    benchmark: str | None = None
    price_context: dict | None = None


# NOTE: /fundamental/compare must come BEFORE /fundamental/{ticker} 
# because FastAPI matches routes in order
@router.get("/fundamental/compare", response_class=HTMLResponse)
async def compare_fundamental_data(request: Request, tickers: str):
    """
    Get fundamental data comparison for multiple tickers.
    tickers: Comma-separated list of ticker symbols.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    
    if not ticker_list:
        return HTMLResponse("<p>No tickers provided</p>")
    
    if len(ticker_list) == 1:
        # Redirect to single ticker view
        raw_data = data_manager.get_fundamental_data(ticker_list[0])
        data = filter_fundamental_data(raw_data)
        return templates.TemplateResponse("components/fundamental.html", {
            "request": request,
            "data": data,
            "ticker": ticker_list[0],
            "raw_data": raw_data
        })
    
    # Fetch data for all tickers with small delay to avoid rate limiting
    all_data = {}
    all_raw_data = {}
    for i, ticker in enumerate(ticker_list):
        try:
            # Add small delay between requests to avoid rate limiting (skip first)
            if i > 0:
                await asyncio.sleep(0.2)
            
            raw = data_manager.get_fundamental_data(ticker)
            if raw:
                all_raw_data[ticker] = raw
                all_data[ticker] = filter_fundamental_data(raw)
                logging.info(f"Fetched fundamental data for {ticker}: {len(raw)} fields")
            else:
                all_raw_data[ticker] = {}
                all_data[ticker] = {}
                logging.warning(f"No fundamental data returned for {ticker}")
        except Exception as e:
            logging.error(f"Error fetching fundamental data for {ticker}: {e}")
            all_raw_data[ticker] = {}
            all_data[ticker] = {}
    
    return templates.TemplateResponse("components/fundamental_compare.html", {
        "request": request,
        "tickers": ticker_list,
        "all_data": all_data,
        "all_raw_data": all_raw_data
    })


@router.get("/fundamental/{ticker}", response_class=HTMLResponse)
async def get_fundamental_data(request: Request, ticker: str):
    """Get filtered fundamental data for a ticker."""
    raw_data = data_manager.get_fundamental_data(ticker)
    # Filter to key metrics only
    data = filter_fundamental_data(raw_data)
    # Keep raw data for AI analysis
    return templates.TemplateResponse("components/fundamental.html", {
        "request": request, 
        "data": data, 
        "ticker": ticker,
        "raw_data": raw_data  # Pass raw data for AI analysis button
    })

@router.post("/business/{ticker}", response_class=HTMLResponse)
async def run_business_analysis(request: Request, ticker: str):
    """Run AI business analysis using Gemini.
    Returns HTML with analysis and extracted competitor/related tickers.
    """
    data = data_manager.get_fundamental_data(ticker)
    if not data:
        return HTMLResponse(content="<div class='alert alert-danger'>Error fetching fundamental data</div>")

    try:
        analysis_text = gemini_analyzer.analyze_ticker(ticker, data)
    except Exception as e:
        logging.error(f"Business AI analysis error for {ticker}: {e}")
        return HTMLResponse(content=f"<div class='alert alert-danger mb-0'>Error running analysis: {str(e)}</div>")

    if not isinstance(analysis_text, str) or not analysis_text.strip():
        logging.warning(f"Business AI analysis returned no content for {ticker}")
        return HTMLResponse(content="<div class='alert alert-warning mb-0'>No analysis text generated.</div>")

    saved_path = _save_md("business", ticker, analysis_text)
    html_content = _safe_markdown(analysis_text)
    
    # Extract competitor/related tickers from the analysis
    extracted_tickers = gemini_analyzer.extract_stock_tickers(analysis_text, exclude_ticker=ticker)
    
    # Build the response with extracted tickers section
    if extracted_tickers:
        tickers_csv = ','.join(extracted_tickers)
        tickers_badges = ' '.join([
            f'<span class="badge bg-primary me-1 mb-1" style="cursor:pointer" onclick="loadTicker(\'{t}\')">{t}</span>'
            for t in extracted_tickers
        ])
        tickers_section = f'''
        <div class="card mt-3 border-info">
            <div class="card-header bg-info bg-opacity-10 d-flex justify-content-between align-items-center">
                <span><strong>🔍 Extracted Tickers</strong> <small class="text-muted">({len(extracted_tickers)} competitors/related)</small></span>
                <button class="btn btn-sm btn-outline-primary" onclick="saveExtractedTickers('{tickers_csv}')">
                    💾 Save to List
                </button>
            </div>
            <div class="card-body py-2">
                {tickers_badges}
            </div>
        </div>
        '''
    else:
        tickers_section = ''
    
    saved_note = f"<div class='text-muted small mt-2'>Saved to: {saved_path}</div>" if saved_path else ""
    return HTMLResponse(content=f"{html_content}{tickers_section}{saved_note}")


@router.post("/fundamental/ai/{ticker}", response_class=HTMLResponse)
async def run_fundamental_ai_analysis(request: Request, ticker: str):
    """Run AI analysis on fundamental data using Gemini.
    Accepts optional JSON body with 'metrics' dict to analyze only filtered metrics.
    """
    # Try to get filtered metrics from request body
    filtered_metrics = None
    try:
        body = await request.json()
        filtered_metrics = body.get('metrics', None)
    except:
        pass  # No JSON body, use all data
    
    if filtered_metrics:
        # Use the filtered metrics passed from frontend
        data = filtered_metrics
        filter_note = f"Analyzing {len(data)} filtered metrics"
    else:
        # Fall back to all fundamental data
        data = data_manager.get_fundamental_data(ticker)
        filter_note = "Analyzing all metrics"
    
    if not data:
        return HTMLResponse(content="<div class='alert alert-warning mb-0'>No fundamental data available for analysis.</div>")
    
    try:
        analysis_text = gemini_analyzer.analyze_fundamentals(ticker, data)
        saved_path = _save_md("fundamental", ticker, analysis_text)
        html_content = _safe_markdown(analysis_text)
        saved_note = f"<div class='text-muted small mt-2'>Saved to: {saved_path}</div>" if saved_path else ""
        return HTMLResponse(content=f"<div class='markdown-content'><small class='text-muted'>{filter_note}</small>{html_content}{saved_note}</div>")
    except Exception as e:
        logging.error(f"Fundamental AI analysis error for {ticker}: {e}")
        return HTMLResponse(content=f"<div class='alert alert-danger mb-0'>Error running analysis: {str(e)}</div>")


@router.post("/strategy/ticker", response_class=JSONResponse)
async def run_ticker_strategy(payload: TickerStrategyRequest):
    ticker = payload.ticker.strip().upper()
    if not ticker:
        return JSONResponse({"error": "Ticker is required"}, status_code=400)

    cache_key = f"ticker:{ticker}:{payload.scenario}:{payload.timeframe}:{payload.benchmark or 'auto'}"
    cached = strategy_cache.get(cache_key)
    if cached:
        return JSONResponse({**cached, "cached": True})

    fundamentals = data_manager.get_fundamental_data(ticker)
    strategy_markdown = gemini_analyzer.recommend_strategy_for_ticker(
        ticker,
        scenario=payload.scenario,
        fundamentals=fundamentals,
        price_context=payload.price_context,
        timeframe=payload.timeframe,
        benchmark=payload.benchmark,
    )

    if strategy_markdown.startswith("Error:"):
        return JSONResponse({"error": strategy_markdown}, status_code=500)

    md_path = _save_md("strategy", ticker, strategy_markdown)
    html = _safe_markdown(strategy_markdown)
    generated_at = datetime.utcnow().isoformat() + "Z"
    payload_record = {"html": html, "generated_at": generated_at, "md_path": md_path}
    strategy_cache.set(cache_key, payload_record)
    return JSONResponse({**payload_record, "cached": False})


@router.post("/chat", response_class=HTMLResponse)
async def chat_with_ai(request: Request):
    """General AI chat endpoint for asking any questions."""
    try:
        body = await request.json()
        message = body.get('message', '').strip()
        ticker = body.get('ticker', None)  # Optional ticker context
    except:
        return HTMLResponse(content="<p class='text-danger'>Invalid request</p>")
    
    if not message:
        return HTMLResponse(content="<p class='text-muted'>Please enter a message</p>")
    
    try:
        # Get ticker data if available for context
        ticker_data = None
        if ticker:
            ticker_data = data_manager.get_fundamental_data(ticker)
        
        analysis_text = gemini_analyzer.chat_response(message, ticker, ticker_data)
        html_content = _safe_markdown(analysis_text)
        return HTMLResponse(content=html_content)
    except Exception as e:
        logging.error(f"Chat error: {e}")
        return HTMLResponse(content=f"<p class='text-danger'>Error: {str(e)}</p>")


@router.post("/fundamental/compare/ai", response_class=HTMLResponse)
async def run_comparative_ai_analysis(request: Request):
    """Run AI comparative analysis on multiple tickers' fundamental data."""
    try:
        body = await request.json()
        tickers = body.get('tickers', [])
        metrics = body.get('metrics', {})
    except:
        return HTMLResponse(content="<div class='alert alert-danger mb-0'>Invalid request data.</div>")
    
    if not tickers or len(tickers) < 2:
        return HTMLResponse(content="<div class='alert alert-warning mb-0'>Need at least 2 tickers for comparison.</div>")
    
    if not metrics:
        return HTMLResponse(content="<div class='alert alert-warning mb-0'>No metrics data provided for analysis.</div>")
    
    try:
        analysis_text = gemini_analyzer.analyze_comparison(tickers, metrics)
        name = "-".join([t.strip().upper() for t in tickers])[:40]
        saved_path = _save_md("compare", name, analysis_text)
        html_content = _safe_markdown(analysis_text)
        saved_note = f"<div class='text-muted small mt-2'>Saved to: {saved_path}</div>" if saved_path else ""
        return HTMLResponse(content=f"<div class='markdown-content'><small class='text-muted'>Comparing {len(tickers)} stocks</small>{html_content}{saved_note}</div>")
    except Exception as e:
        logging.error(f"Comparative AI analysis error: {e}")
        return HTMLResponse(content=f"<div class='alert alert-danger mb-0'>Error running analysis: {str(e)}</div>")


@router.post("/buffett/{ticker}", response_class=HTMLResponse)
async def run_buffett_canslim_analysis(request: Request, ticker: str):
    """
    Run Buffett/CANSLIM investment analysis using Gemini AI.
    Returns HTML with score breakdowns and detailed analysis.
    """
    result = buffett_canslim.analyze_stock(ticker)
    html_content = buffett_canslim.format_analysis_html(result)
    return HTMLResponse(content=html_content)


@router.get("/buffett/{ticker}/data", response_class=JSONResponse)
async def get_buffett_canslim_data(ticker: str):
    """
    Get Buffett/CANSLIM analysis data as JSON for radar chart.
    """
    result = buffett_canslim.analyze_stock(ticker)
    if "error" in result:
        return JSONResponse(content={"error": result["error"]}, status_code=400)
    
    chart_data = buffett_canslim.get_radar_chart_data(result)
    return JSONResponse(content=chart_data)
