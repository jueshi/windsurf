from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from ..sec_api import sec_api
import pandas as pd

router = APIRouter(
    prefix="/sec",
    tags=["sec"]
)

templates = Jinja2Templates(directory="webapp/templates")

@router.get("/filings/{ticker}", response_class=HTMLResponse)
async def get_filings(request: Request, ticker: str):
    ticker = ticker.upper()
    cik = sec_api.get_company_cik(ticker)

    filings = []
    if cik:
        # Get latest 10-K and 10-Q
        f10k = sec_api.get_latest_filing_info(cik, "10-K")
        if f10k: filings.append(f10k)

        f10q = sec_api.get_latest_filing_info(cik, "10-Q")
        if f10q: filings.append(f10q)

    return templates.TemplateResponse("components/sec.html", {"request": request, "filings": filings, "ticker": ticker, "cik": cik})

@router.post("/extract/{ticker}/{accession_number}", response_class=HTMLResponse)
async def extract_tables(request: Request, ticker: str, accession_number: str, cik: str):
    content = sec_api.get_filing_content(cik, accession_number)

    tables_html = ""
    if content:
        dfs = sec_api.extract_tables(content)
        for i, df in enumerate(dfs):
            # Limit rows for display
            if len(df) > 50: df = df.head(50)

            tables_html += f"<h6>Table {i+1}</h6>"
            tables_html += df.to_html(classes="table table-sm table-bordered table-striped", index=False, na_rep="")
            tables_html += "<hr>"

            if i >= 10: # Limit to first 10 tables for web view
                tables_html += "<p>... more tables truncated ...</p>"
                break
    else:
        tables_html = "<p>Could not load filing content.</p>"

    return HTMLResponse(content=tables_html)
