from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from .. import news_fetcher
from .. import gemini_analyzer

router = APIRouter(
    prefix="/news",
    tags=["news"]
)

templates = Jinja2Templates(directory="webapp/templates")

@router.get("/feed/{ticker}", response_class=HTMLResponse)
async def get_news_feed(request: Request, ticker: str):
    ticker = ticker.upper()
    news = news_fetcher.fetch_news(ticker)
    return templates.TemplateResponse("components/news.html", {"request": request, "news": news, "ticker": ticker})

@router.post("/analyze/{ticker}", response_class=HTMLResponse)
async def analyze_news(request: Request, ticker: str):
    ticker = ticker.upper()
    news = news_fetcher.fetch_news(ticker)

    if news and "error" not in news[0]:
        analysis = gemini_analyzer.analyze_news(news)
        import markdown
        analysis_html = markdown.markdown(analysis)
        return HTMLResponse(content=analysis_html)
    else:
        return HTMLResponse(content="<p>Could not fetch news to analyze.</p>")
