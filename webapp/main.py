from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from .database import engine, Base
from .routers import tickers, charts, analysis, news, sec, portfolios
from . import gemini_analyzer

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Stock Toolbox Web")

app.mount("/static", StaticFiles(directory="webapp/static"), name="static")

templates = Jinja2Templates(directory="webapp/templates")

app.include_router(tickers.router)
app.include_router(charts.router)
app.include_router(analysis.router)
app.include_router(news.router)
app.include_router(sec.router)
app.include_router(portfolios.router)

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/settings/language")
async def get_language():
    """Get current AI response language setting."""
    return JSONResponse({"language": gemini_analyzer.RESPONSE_LANGUAGE})

@app.post("/settings/language/{lang}")
async def set_language(lang: str):
    """Set AI response language. lang: 'en' for English, 'zh' for Chinese."""
    if lang not in ['en', 'zh']:
        return JSONResponse({"error": "Invalid language. Use 'en' or 'zh'"}, status_code=400)
    gemini_analyzer.RESPONSE_LANGUAGE = lang
    return JSONResponse({"language": lang, "message": f"Language set to {lang}"})
