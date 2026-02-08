import re
from datetime import datetime
from fastapi import APIRouter, Request, Depends, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from ..database import get_db
from .. import models

router = APIRouter(
    prefix="/tickers",
    tags=["tickers"]
)

templates = Jinja2Templates(directory="webapp/templates")

class TempTickersRequest(BaseModel):
    """Model for saving temp tickers."""
    tickers: str

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

def validate_ticker_symbol(symbol: str) -> tuple[bool, str]:
    """
    Validate a ticker symbol.
    Returns (is_valid, cleaned_symbol_or_error_message).
    """
    symbol = symbol.upper().strip()
    if not symbol:
        return False, "Ticker symbol cannot be empty"
    # Match 1-5 letters, optionally followed by .A, .B, etc. for share classes
    if not re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', symbol):
        return False, "Invalid ticker format. Use 1-5 letters (e.g., AAPL, BRK.A)"
    return True, symbol


@router.post("/add/{list_id}", response_class=HTMLResponse)
async def add_ticker(request: Request, list_id: int, symbol: str = Form(...), db: Session = Depends(get_db)):
    """Add a ticker to a list with validation."""
    is_valid, result = validate_ticker_symbol(symbol)
    
    if not is_valid:
        # Return error message in the list area
        ticker_list = db.query(models.TickerList).filter(models.TickerList.id == list_id).first()
        return templates.TemplateResponse("components/ticker_list_items.html", {
            "request": request, 
            "tickers": ticker_list.tickers if ticker_list else [], 
            "list_id": list_id,
            "error": result
        })
    
    symbol = result  # Use validated/cleaned symbol

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
    """Delete a single ticker from a list."""
    ticker = db.query(models.Ticker).filter(models.Ticker.id == ticker_id).first()
    if ticker:
        list_id = ticker.list_id
        db.delete(ticker)
        db.commit()

        ticker_list = db.query(models.TickerList).filter(models.TickerList.id == list_id).first()
        return templates.TemplateResponse("components/ticker_list_items.html", {"request": request, "tickers": ticker_list.tickers, "list_id": list_id})
    return ""


@router.delete("/list/{list_id}", response_class=HTMLResponse)
async def delete_ticker_list(request: Request, list_id: int, db: Session = Depends(get_db)):
    """Delete an entire ticker list and all its tickers."""
    ticker_list = db.query(models.TickerList).filter(models.TickerList.id == list_id).first()
    if ticker_list:
        db.delete(ticker_list)
        db.commit()
    
    # Return updated list of all ticker lists
    lists = db.query(models.TickerList).all()
    return templates.TemplateResponse("components/ticker_lists.html", {"request": request, "lists": lists})


@router.put("/list/{list_id}/rename")
async def rename_ticker_list(request: Request, list_id: int, db: Session = Depends(get_db)):
    """Rename a ticker list."""
    try:
        body = await request.json()
        new_name = body.get('name', '').strip()
    except:
        return {"error": "Invalid request"}
    
    if not new_name:
        return {"error": "Name cannot be empty"}
    
    ticker_list = db.query(models.TickerList).filter(models.TickerList.id == list_id).first()
    if ticker_list:
        ticker_list.name = new_name
        db.commit()
        return {"success": True, "name": new_name}
    
    return {"error": "List not found"}

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


@router.post("/save-temp", response_class=JSONResponse)
async def save_temp_tickers(data: TempTickersRequest, db: Session = Depends(get_db)):
    """
    Save extracted tickers to a temporary ticker list.
    Creates or updates a list named "News Tickers (temp)" with the provided tickers.
    """
    tickers_csv = data.tickers.strip()
    if not tickers_csv:
        return JSONResponse(content={"error": "No tickers provided"}, status_code=400)
    
    # Parse and validate tickers
    ticker_symbols = [t.strip().upper() for t in tickers_csv.split(",") if t.strip()]
    valid_tickers = []
    for symbol in ticker_symbols:
        is_valid, result = validate_ticker_symbol(symbol)
        if is_valid:
            valid_tickers.append(result)
    
    if not valid_tickers:
        return JSONResponse(content={"error": "No valid tickers found"}, status_code=400)
    
    # Create list name with timestamp
    timestamp = datetime.now().strftime("%m/%d %H:%M")
    list_name = f"News Tickers ({timestamp})"
    
    # Create new list
    new_list = models.TickerList(name=list_name)
    db.add(new_list)
    db.commit()
    db.refresh(new_list)
    
    # Add tickers to the list
    for symbol in valid_tickers:
        new_ticker = models.Ticker(list_id=new_list.id, symbol=symbol)
        db.add(new_ticker)
    
    db.commit()
    
    return JSONResponse(content={
        "success": True,
        "list_id": new_list.id,
        "list_name": list_name,
        "count": len(valid_tickers),
        "tickers": valid_tickers
    })
