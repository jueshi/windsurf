from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from ..sec_api import sec_api
from .. import gemini_analyzer
import pandas as pd
import markdown

router = APIRouter(
    prefix="/sec",
    tags=["sec"]
)

templates = Jinja2Templates(directory="webapp/templates")

# Cache for filing content to avoid re-fetching
_filing_content_cache = {}

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

@router.post("/sections/{ticker}/{accession_number}", response_class=HTMLResponse)
async def get_sections(request: Request, ticker: str, accession_number: str, cik: str, form_type: str = "10-K"):
    """
    Get the list of key sections found in a filing for navigation.
    """
    cache_key = f"{cik}_{accession_number}"
    
    # Check cache first
    if cache_key in _filing_content_cache:
        content = _filing_content_cache[cache_key]
    else:
        content = sec_api.get_filing_content(cik, accession_number)
        if content:
            _filing_content_cache[cache_key] = content
    
    sections = []
    if content:
        sections = sec_api.extract_sections(content, form_type)
    
    return templates.TemplateResponse("components/sec_sections.html", {
        "request": request,
        "sections": sections,
        "ticker": ticker,
        "accession_number": accession_number,
        "cik": cik,
        "form_type": form_type
    })

@router.post("/section-content/{ticker}/{accession_number}/{section_id}", response_class=HTMLResponse)
async def get_section_content(request: Request, ticker: str, accession_number: str, section_id: str, cik: str, form_type: str = "10-K"):
    """
    Get the content of a specific section from a filing.
    """
    cache_key = f"{cik}_{accession_number}"
    
    # Check cache first
    if cache_key in _filing_content_cache:
        content = _filing_content_cache[cache_key]
    else:
        content = sec_api.get_filing_content(cik, accession_number)
        if content:
            _filing_content_cache[cache_key] = content
    
    section_html = "<p>Could not load filing content.</p>"
    if content:
        section_html = sec_api.get_section_content(content, section_id, form_type)
    
    return HTMLResponse(content=f'<div class="section-content">{section_html}</div>')

@router.post("/analyze-section/{ticker}/{accession_number}/{section_id}", response_class=HTMLResponse)
async def analyze_section(request: Request, ticker: str, accession_number: str, section_id: str, cik: str, form_type: str = "10-K"):
    """
    Analyze a specific section from a filing using AI.
    """
    import logging
    logging.info(f"analyze_section called: ticker={ticker}, accession={accession_number}, section={section_id}, cik={cik}, form={form_type}")
    
    if not cik or cik == 'null' or cik == 'None':
        return HTMLResponse(content=f'<div class="alert alert-danger">Invalid CIK: {cik}. Please select a ticker first.</div>')
    
    if not accession_number or accession_number == 'null':
        return HTMLResponse(content=f'<div class="alert alert-danger">Invalid accession number. Please try again.</div>')
    
    cache_key = f"{cik}_{accession_number}"
    
    # Check cache first
    if cache_key in _filing_content_cache:
        content = _filing_content_cache[cache_key]
        logging.info(f"Using cached content for {cache_key}")
    else:
        logging.info(f"Fetching content for CIK={cik}, accession={accession_number}")
        content = sec_api.get_filing_content(cik, accession_number)
        if content:
            _filing_content_cache[cache_key] = content
            logging.info(f"Cached content for {cache_key}, length={len(content)}")
        else:
            logging.error(f"Failed to fetch content for CIK={cik}, accession={accession_number}")
    
    if not content:
        return HTMLResponse(content=f'<div class="alert alert-danger">Could not load filing content for CIK {cik}, accession {accession_number}. Check server logs for details.</div>')
    
    # Get section content
    section_text = sec_api.get_section_content(content, section_id, form_type)
    
    # Get section label and description from the section_id
    section_map_10k = {
        "item1": ("Item 1", "Business"),
        "item1a": ("Item 1A", "Risk Factors"),
        "item1b": ("Item 1B", "Unresolved Staff Comments"),
        "item1c": ("Item 1C", "Cybersecurity"),
        "item2": ("Item 2", "Properties"),
        "item3": ("Item 3", "Legal Proceedings"),
        "item4": ("Item 4", "Mine Safety Disclosures"),
        "item5": ("Item 5", "Market for Registrant's Common Equity"),
        "item6": ("Item 6", "Reserved"),
        "item7": ("Item 7", "Management's Discussion and Analysis"),
        "item7a": ("Item 7A", "Quantitative and Qualitative Disclosures"),
        "item8": ("Item 8", "Financial Statements"),
        "item9": ("Item 9", "Changes in and Disagreements With Accountants"),
        "item9a": ("Item 9A", "Controls and Procedures"),
        "item9b": ("Item 9B", "Other Information"),
        "item9c": ("Item 9C", "Disclosure Regarding Foreign Jurisdictions"),
        "item10": ("Item 10", "Directors and Corporate Governance"),
        "item11": ("Item 11", "Executive Compensation"),
        "item12": ("Item 12", "Security Ownership"),
        "item13": ("Item 13", "Certain Relationships"),
        "item14": ("Item 14", "Principal Accountant Fees"),
        "item15": ("Item 15", "Exhibits"),
        "item16": ("Item 16", "Form 10-K Summary"),
    }
    
    section_map_10q = {
        "part1item1": ("Part I, Item 1", "Financial Statements"),
        "part1item2": ("Part I, Item 2", "Management's Discussion and Analysis"),
        "part1item3": ("Part I, Item 3", "Quantitative and Qualitative Disclosures"),
        "part1item4": ("Part I, Item 4", "Controls and Procedures"),
        "part2item1": ("Part II, Item 1", "Legal Proceedings"),
        "part2item1a": ("Part II, Item 1A", "Risk Factors"),
        "part2item2": ("Part II, Item 2", "Unregistered Sales"),
        "part2item3": ("Part II, Item 3", "Defaults"),
        "part2item4": ("Part II, Item 4", "Mine Safety"),
        "part2item5": ("Part II, Item 5", "Other Information"),
        "part2item6": ("Part II, Item 6", "Exhibits"),
    }
    
    section_map = section_map_10k if form_type == "10-K" else section_map_10q
    section_info = section_map.get(section_id, ("Unknown", "Unknown Section"))
    section_label, section_description = section_info
    
    # Call AI analysis
    analysis = gemini_analyzer.analyze_sec_section(
        ticker=ticker.upper(),
        section_label=section_label,
        section_description=section_description,
        section_content=section_text,
        form_type=form_type
    )
    
    # Convert markdown to HTML
    analysis_html = markdown.markdown(analysis, extensions=['tables', 'fenced_code'])
    
    return HTMLResponse(content=f'''
        <div class="section-analysis">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h5 class="mb-0"><i class="bi bi-robot"></i> AI Analysis: {section_label} - {section_description}</h5>
                <span class="badge bg-info">{ticker.upper()} {form_type}</span>
            </div>
            <div class="analysis-content">
                {analysis_html}
            </div>
        </div>
    ''')

@router.post("/extract/{ticker}/{accession_number}", response_class=HTMLResponse)
async def extract_tables(request: Request, ticker: str, accession_number: str, cik: str):
    cache_key = f"{cik}_{accession_number}"
    
    # Check cache first
    if cache_key in _filing_content_cache:
        content = _filing_content_cache[cache_key]
    else:
        content = sec_api.get_filing_content(cik, accession_number)
        if content:
            _filing_content_cache[cache_key] = content

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
