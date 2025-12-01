from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from ..data_manager import data_manager
from .. import gemini_analyzer
import logging
import markdown

router = APIRouter(
    prefix="/analysis",
    tags=["analysis"]
)

templates = Jinja2Templates(directory="webapp/templates")

@router.get("/fundamental/{ticker}", response_class=HTMLResponse)
async def get_fundamental_data(request: Request, ticker: str):
    data = data_manager.get_fundamental_data(ticker)
    return templates.TemplateResponse("components/fundamental.html", {"request": request, "data": data, "ticker": ticker})

@router.post("/business/{ticker}", response_class=HTMLResponse)
async def run_business_analysis(request: Request, ticker: str):
    data = data_manager.get_fundamental_data(ticker)
    if not data:
        return "Error fetching fundamental data"

    analysis_text = gemini_analyzer.analyze_ticker(ticker, data)
    html_content = markdown.markdown(analysis_text)

    return HTMLResponse(content=html_content)
