from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .database import engine, Base
from .routers import tickers, charts, analysis, news, sec

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

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
