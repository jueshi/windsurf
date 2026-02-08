from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from .. import news_fetcher
from .. import gemini_analyzer
from .. import finviz_news
import markdown
import re
from urllib.parse import quote_plus

router = APIRouter(
    prefix="/news",
    tags=["news"]
)

templates = Jinja2Templates(directory="webapp/templates")

class ClipboardText(BaseModel):
    """Model for clipboard text input."""
    text: str

@router.get("/feed/{ticker}", response_class=HTMLResponse)
async def get_news_feed(request: Request, ticker: str):
    ticker = ticker.upper()
    news = news_fetcher.fetch_news(ticker)
    return templates.TemplateResponse("components/news.html", {"request": request, "news": news, "ticker": ticker})

@router.post("/analyze/{ticker}", response_class=HTMLResponse)
async def analyze_news(request: Request, ticker: str):
    """Analyze news for a specific ticker using AI."""
    ticker = ticker.upper()
    news = news_fetcher.fetch_news(ticker)

    if news and "error" not in news[0]:
        analysis = gemini_analyzer.analyze_news(news)
        import markdown
        analysis_html = markdown.markdown(analysis)
        return HTMLResponse(content=analysis_html)
    else:
        return HTMLResponse(content="<p>Could not fetch news to analyze.</p>")


async def _get_news_with_summary(news_type: str, title: str) -> str:
    """
    Helper function to fetch news and generate AI summary.
    """
    news_items, tickers = finviz_news.fetch_finviz_news(news_type=news_type)
    
    if not news_items:
        return f'<div class="alert alert-warning">No {title.lower()} available. Try again later.</div>'
    
    # Generate AI summary of headlines
    headlines = [item['title'] for item in news_items[:20]]
    headlines_text = "\n".join([f"- {h}" for h in headlines])
    
    try:
        summary = gemini_analyzer.summarize_market_news(headlines_text, news_type)
        summary_html = markdown.markdown(summary, extensions=['tables', 'fenced_code'])
    except Exception as e:
        summary_html = f'<p class="text-muted">Could not generate summary: {str(e)}</p>'
    
    # Format the complete HTML with summary and news list
    html_content = finviz_news.format_news_html_with_summary(
        news_items, tickers, title=title, summary_html=summary_html
    )
    return html_content


@router.get("/market", response_class=HTMLResponse)
async def get_market_news(request: Request):
    """
    Fetch market-wide news from Finviz v=3 feed with AI summary.
    """
    html_content = await _get_news_with_summary("market", "Market News")
    return HTMLResponse(content=html_content)


@router.get("/stocks", response_class=HTMLResponse)
async def get_stock_news(request: Request):
    """
    Fetch stock-specific news from Finviz v=1 feed with AI summary.
    """
    html_content = await _get_news_with_summary("stocks", "Stock News")
    return HTMLResponse(content=html_content)


@router.get("/etf", response_class=HTMLResponse)
async def get_etf_news(request: Request):
    """
    Fetch ETF-related news with AI summary.
    """
    html_content = await _get_news_with_summary("etf", "ETF News")
    return HTMLResponse(content=html_content)


@router.get("/crypto", response_class=HTMLResponse)
async def get_crypto_news(request: Request):
    """
    Fetch cryptocurrency news with AI summary.
    """
    html_content = await _get_news_with_summary("crypto", "Crypto News")
    return HTMLResponse(content=html_content)


@router.post("/summarize-clipboard", response_class=HTMLResponse)
async def summarize_clipboard(data: ClipboardText):
    """
    Summarize text from clipboard using AI.
    """
    text = data.text.strip()
    if not text:
        return HTMLResponse(content='<div class="alert alert-warning">No text provided to summarize.</div>')
    
    # Limit text length to avoid token limits
    if len(text) > 10000:
        text = text[:10000] + "..."
    
    try:
        summary = gemini_analyzer.summarize_text(text)
        summary_html = markdown.markdown(summary, extensions=['tables', 'fenced_code'])
        
        return HTMLResponse(content=f'''
            <div class="card">
                <div class="card-header">
                    <strong>📋 Clipboard Summary</strong>
                    <small class="text-muted ms-2">({len(data.text)} characters)</small>
                </div>
                <div class="card-body">
                    {summary_html}
                </div>
            </div>
        ''')
    except Exception as e:
        return HTMLResponse(content=f'<div class="alert alert-danger">Error summarizing text: {str(e)}</div>')

@router.get("/{news_id}/{slug}", response_class=HTMLResponse)
async def get_news_item(request: Request, news_id: int, slug: str):
    parts = [p for p in slug.split("-") if p]
    ticker = ""
    if parts:
        last = parts[-1].upper()
        if re.fullmatch(r"[A-Z]{1,5}", last):
            ticker = last
    if ticker:
        items = news_fetcher.fetch_news(ticker) or []
        titles = [it.get("title") for it in items if it.get("title")]
        headlines = "\n".join([f"- {t}" for t in titles[:20]])
        try:
            summary = gemini_analyzer.summarize_market_news(headlines, "stocks")
            summary_html = markdown.markdown(summary, extensions=["tables", "fenced_code"])
        except Exception:
            summary_html = ""
        links = "".join([
            f'<a href="{it.get("url","#")}" target="_blank" rel="noopener" class="list-group-item list-group-item-action py-2">{it.get("title","")}</a>'
            for it in items[:20]
        ]) or '<div class="alert alert-warning">No related articles found.</div>'
        html = f'''
        <div class="card mb-3 border-primary">
            <div class="card-header py-2"><strong>Related articles for {ticker}</strong></div>
            <div class="card-body">{summary_html}</div>
        </div>
        <div class="list-group">{links}</div>
        '''
        return HTMLResponse(content=html)
    query = slug.replace("-", " ")
    search_q = quote_plus(query)
    fallback = f"https://www.google.com/search?q={search_q}"
    html = f'''
    <div class="alert alert-info mb-2">No local article found for "{slug}". Showing search link.</div>
    <a href="{fallback}" target="_blank" rel="noopener" class="btn btn-outline-primary">Open search</a>
    '''
    return HTMLResponse(content=html)
