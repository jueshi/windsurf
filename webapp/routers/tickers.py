from fastapi import APIRouter, Request, Depends, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from ..database import get_db
from .. import models

router = APIRouter(
    prefix="/tickers",
    tags=["tickers"]
)

templates = Jinja2Templates(directory="webapp/templates")

@router.get("/list", response_class=HTMLResponse)
async def get_ticker_lists(request: Request, db: Session = Depends(get_db)):
    lists = db.query(models.TickerList).all()
    return templates.TemplateResponse("components/ticker_lists.html", {"request": request, "lists": lists})

@router.get("/list/{list_id}", response_class=HTMLResponse)
async def get_tickers_in_list(request: Request, list_id: int, db: Session = Depends(get_db)):
    ticker_list = db.query(models.TickerList).filter(models.TickerList.id == list_id).first()
    tickers = ticker_list.tickers if ticker_list else []
    return templates.TemplateResponse("components/ticker_list_items.html", {"request": request, "tickers": tickers, "list_id": list_id})

@router.post("/list", response_class=HTMLResponse)
async def create_ticker_list(request: Request, name: str = Form(...), db: Session = Depends(get_db)):
    new_list = models.TickerList(name=name)
    db.add(new_list)
    db.commit()
    db.refresh(new_list)
    lists = db.query(models.TickerList).all()
    return templates.TemplateResponse("components/ticker_lists.html", {"request": request, "lists": lists})

@router.post("/add/{list_id}", response_class=HTMLResponse)
async def add_ticker(request: Request, list_id: int, symbol: str = Form(...), db: Session = Depends(get_db)):
    symbol = symbol.upper().strip()
    if not symbol:
        return "" # Handle empty error better

    # Check if exists in list
    existing = db.query(models.Ticker).filter(models.Ticker.list_id == list_id, models.Ticker.symbol == symbol).first()
    if not existing:
        new_ticker = models.Ticker(list_id=list_id, symbol=symbol)
        db.add(new_ticker)
        db.commit()

    ticker_list = db.query(models.TickerList).filter(models.TickerList.id == list_id).first()
    return templates.TemplateResponse("components/ticker_list_items.html", {"request": request, "tickers": ticker_list.tickers, "list_id": list_id})

@router.delete("/{ticker_id}", response_class=HTMLResponse)
async def delete_ticker(request: Request, ticker_id: int, db: Session = Depends(get_db)):
    ticker = db.query(models.Ticker).filter(models.Ticker.id == ticker_id).first()
    if ticker:
        list_id = ticker.list_id
        db.delete(ticker)
        db.commit()

        ticker_list = db.query(models.TickerList).filter(models.TickerList.id == list_id).first()
        return templates.TemplateResponse("components/ticker_list_items.html", {"request": request, "tickers": ticker_list.tickers, "list_id": list_id})
    return ""

@router.get("/search", response_class=HTMLResponse)
async def search_tickers(request: Request, q: str = "", db: Session = Depends(get_db)):
    if not q:
        return ""

    q = q.upper()
    tickers = db.query(models.Ticker).filter(models.Ticker.symbol.contains(q)).limit(10).all()

    html = ""
    for ticker in tickers:
        html += f"""
        <button class="list-group-item list-group-item-action d-flex justify-content-between align-items-center"
                onclick="loadTicker('{ticker.symbol}')">
            {ticker.symbol} <small class="text-muted">{ticker.ticker_list.name}</small>
        </button>
        """

    if not tickers:
        html = '<div class="p-2 text-muted">No tickers found.</div>'

    return html
