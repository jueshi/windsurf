"""
Finviz News Fetcher Module

Fetches market news from Finviz v=3 feed without requiring a ticker.
Also extracts mentioned tickers from the news page.

Ported from desktop app functionality.
"""
import re
import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Finviz news URLs for different news types
FINVIZ_NEWS_URLS = {
    "market": "https://finviz.com/news.ashx?v=3",      # General market news
    "stocks": "https://finviz.com/news.ashx?v=1",      # Stock-specific news  
    "etf": "https://finviz.com/news.ashx?v=4",         # ETF news
    "crypto": "https://finviz.com/news.ashx?v=5",      # Crypto news
}
FINVIZ_NEWS_URL = FINVIZ_NEWS_URLS["market"]  # Default

# Common headers to avoid 403
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

# Regex pattern for extracting stock tickers (1-5 uppercase letters)
TICKER_PATTERN = re.compile(r'\b([A-Z]{1,5})\b')

# Common words to exclude from ticker extraction
EXCLUDED_WORDS = {
    'A', 'I', 'AM', 'PM', 'CEO', 'CFO', 'COO', 'CTO', 'IPO', 'ETF', 'GDP', 'CPI',
    'FED', 'SEC', 'NYSE', 'NASDAQ', 'DOW', 'US', 'UK', 'EU', 'AI', 'IT', 'TV',
    'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS',
    'ONE', 'OUR', 'OUT', 'DAY', 'HAD', 'HAS', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW',
    'NOW', 'OLD', 'SEE', 'WAY', 'WHO', 'BOY', 'DID', 'GET', 'HIM', 'LET', 'PUT',
    'SAY', 'SHE', 'TOO', 'USE', 'Q1', 'Q2', 'Q3', 'Q4', 'YOY', 'QOQ', 'MOM',
    'EST', 'PST', 'CST', 'MST', 'UTC', 'GMT', 'USD', 'EUR', 'GBP', 'JPY', 'CNY',
    'BUY', 'SELL', 'HOLD', 'LONG', 'SHORT', 'CALL', 'PUT', 'ATH', 'ATL',
    'EPS', 'PE', 'PB', 'PS', 'ROE', 'ROA', 'ROI', 'EBITDA', 'FCF', 'DCF',
    'TOP', 'UP', 'DOWN', 'HIGH', 'LOW', 'OPEN', 'CLOSE', 'BID', 'ASK',
    'NEWS', 'BLOG', 'POST', 'READ', 'MORE', 'LIVE', 'JUST', 'SAYS', 'SAID',
    'WILL', 'COULD', 'WOULD', 'SHOULD', 'MIGHT', 'MUST', 'BEEN', 'HAVE',
    'WITH', 'THIS', 'THAT', 'FROM', 'THEY', 'WERE', 'BEEN', 'HAVE', 'WHAT',
    'WHEN', 'WHERE', 'WHICH', 'WHILE', 'ABOUT', 'AFTER', 'BEFORE', 'BETWEEN',
    'INTO', 'THROUGH', 'DURING', 'UNDER', 'AGAIN', 'FURTHER', 'THEN', 'ONCE',
    'HERE', 'THERE', 'BOTH', 'EACH', 'FEW', 'MORE', 'MOST', 'OTHER', 'SOME',
    'SUCH', 'THAN', 'VERY', 'JUST', 'ALSO', 'ONLY', 'EVEN', 'BACK', 'WELL',
}


def fetch_finviz_news(news_type: str = "market") -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Fetch news from Finviz based on news type.
    
    Args:
        news_type: Type of news to fetch - "market", "stocks", "etf", or "crypto"
    
    Returns:
        Tuple containing:
        - List of news articles (title, url, source, time)
        - List of extracted ticker symbols mentioned in headlines
    """
    url = FINVIZ_NEWS_URLS.get(news_type, FINVIZ_NEWS_URL)
    logging.info(f"Fetching {news_type} news from {url}")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        news_items = []
        mentioned_tickers = set()
        
        # Find news table - Finviz uses table-based layout
        news_tables = soup.find_all('table', class_='styled-table-new')
        
        if not news_tables:
            # Try alternative selectors
            news_tables = soup.find_all('table', class_='t-home-table')
        
        for table in news_tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    # Extract time and headline
                    time_cell = cells[0]
                    headline_cell = cells[1] if len(cells) > 1 else cells[0]
                    
                    # Get link
                    link = headline_cell.find('a')
                    if link:
                        title = link.get_text(strip=True)
                        url = link.get('href', '')
                        
                        # Get source if available
                        source_span = headline_cell.find('span', class_='nn-tab-link')
                        source = source_span.get_text(strip=True) if source_span else ''
                        
                        # Get time
                        time_text = time_cell.get_text(strip=True) if time_cell else ''
                        
                        if title:
                            news_items.append({
                                'title': title,
                                'url': url,
                                'source': source,
                                'time': time_text
                            })
                            
                            # Extract tickers from headline
                            potential_tickers = TICKER_PATTERN.findall(title)
                            for ticker in potential_tickers:
                                if ticker not in EXCLUDED_WORDS and len(ticker) >= 2:
                                    mentioned_tickers.add(ticker)
        
        # If no news found with structured parsing, try simpler approach
        if not news_items:
            # Find all links in news section
            all_links = soup.find_all('a', class_='nn-tab-link')
            for link in all_links[:30]:  # Limit to first 30
                title = link.get_text(strip=True)
                url = link.get('href', '')
                if title and url:
                    news_items.append({
                        'title': title,
                        'url': url,
                        'source': '',
                        'time': ''
                    })
                    # Extract tickers
                    potential_tickers = TICKER_PATTERN.findall(title)
                    for ticker in potential_tickers:
                        if ticker not in EXCLUDED_WORDS and len(ticker) >= 2:
                            mentioned_tickers.add(ticker)
        
        logging.info(f"Fetched {len(news_items)} news items, found {len(mentioned_tickers)} tickers")
        return news_items, sorted(list(mentioned_tickers))
        
    except requests.RequestException as e:
        logging.error(f"Error fetching Finviz news: {e}")
        return [], []
    except Exception as e:
        logging.error(f"Error parsing Finviz news: {e}")
        return [], []


def format_news_html(news_items: List[Dict[str, Any]], tickers: List[str], title: str = "News") -> str:
    """
    Format news items and extracted tickers as HTML.
    
    Args:
        news_items: List of news article dicts
        tickers: List of extracted ticker symbols
        title: Title for the news section
        
    Returns:
        HTML string for rendering
    """
    if not news_items:
        return f'<div class="alert alert-warning">No {title.lower()} available. Try again later.</div>'
    
    # Title header
    title_html = f'<h5 class="mb-3">📰 {title}</h5>'
    
    # Ticker badges section
    ticker_html = ""
    if tickers:
        ticker_badges = " ".join([
            f'<span class="badge bg-primary me-1 mb-1 ticker-badge" style="cursor: pointer;" onclick="loadTicker(\'{t}\')">{t}</span>'
            for t in tickers[:20]  # Limit to 20 tickers
        ])
        # Create comma-separated list for saving
        tickers_csv = ",".join(tickers[:20])
        ticker_html = f'''
        <div class="card mb-3">
            <div class="card-header py-2 d-flex justify-content-between align-items-center">
                <div>
                    <strong>📊 Mentioned Tickers</strong>
                    <small class="text-muted ms-2">(click to load)</small>
                </div>
                <button class="btn btn-sm btn-outline-success" onclick="saveExtractedTickers('{tickers_csv}')" title="Save to temp list">
                    💾 Save to List
                </button>
            </div>
            <div class="card-body py-2">
                {ticker_badges}
            </div>
        </div>
        '''
    
    # News list
    news_html = '<div class="list-group">'
    for item in news_items[:25]:  # Limit to 25 items
        source_badge = f'<span class="badge bg-secondary me-2">{item["source"]}</span>' if item.get("source") else ""
        time_badge = f'<small class="text-muted">{item["time"]}</small>' if item.get("time") else ""
        
        news_html += f'''
        <a href="{item['url']}" target="_blank" rel="noopener" class="list-group-item list-group-item-action py-2">
            <div class="d-flex justify-content-between align-items-start">
                <div>
                    {source_badge}
                    {item['title']}
                </div>
                {time_badge}
            </div>
        </a>
        '''
    news_html += '</div>'
    
    return title_html + ticker_html + news_html


def format_news_html_with_summary(
    news_items: List[Dict[str, Any]], 
    tickers: List[str], 
    title: str = "News",
    summary_html: str = ""
) -> str:
    """
    Format news items with AI summary, extracted tickers, and news list.
    
    Args:
        news_items: List of news article dicts
        tickers: List of extracted ticker symbols
        title: Title for the news section
        summary_html: Pre-rendered HTML of the AI summary
        
    Returns:
        HTML string for rendering
    """
    if not news_items:
        return f'<div class="alert alert-warning">No {title.lower()} available. Try again later.</div>'
    
    # Title header
    title_html = f'<h5 class="mb-3">📰 {title}</h5>'
    
    # AI Summary card
    summary_card = ""
    if summary_html:
        summary_card = f'''
        <div class="card mb-3 border-primary">
            <div class="card-header bg-primary text-white py-2">
                <strong>🤖 AI Analysis</strong>
            </div>
            <div class="card-body py-2">
                {summary_html}
            </div>
        </div>
        '''
    
    # Ticker badges section
    ticker_html = ""
    if tickers:
        ticker_badges = " ".join([
            f'<span class="badge bg-primary me-1 mb-1 ticker-badge" style="cursor: pointer;" onclick="loadTicker(\'{t}\')">{t}</span>'
            for t in tickers[:20]
        ])
        tickers_csv = ",".join(tickers[:20])
        ticker_html = f'''
        <div class="card mb-3">
            <div class="card-header py-2 d-flex justify-content-between align-items-center">
                <div>
                    <strong>📊 Mentioned Tickers</strong>
                    <small class="text-muted ms-2">(click to load)</small>
                </div>
                <button class="btn btn-sm btn-outline-success" onclick="saveExtractedTickers('{tickers_csv}')" title="Save to temp list">
                    💾 Save to List
                </button>
            </div>
            <div class="card-body py-2">
                {ticker_badges}
            </div>
        </div>
        '''
    
    # News list in a collapsible card
    news_html = f'''
    <div class="card">
        <div class="card-header py-2 d-flex justify-content-between align-items-center" 
             data-bs-toggle="collapse" data-bs-target="#news-list-collapse" 
             style="cursor: pointer;">
            <strong>📋 Headlines ({len(news_items[:25])} articles)</strong>
            <small class="text-muted">Click to expand/collapse</small>
        </div>
        <div class="collapse show" id="news-list-collapse">
            <div class="list-group list-group-flush" style="max-height: 400px; overflow-y: auto;">
    '''
    
    for item in news_items[:25]:
        source_badge = f'<span class="badge bg-secondary me-2">{item["source"]}</span>' if item.get("source") else ""
        time_badge = f'<small class="text-muted">{item["time"]}</small>' if item.get("time") else ""
        
        news_html += f'''
            <a href="{item['url']}" target="_blank" rel="noopener" class="list-group-item list-group-item-action py-2">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        {source_badge}
                        {item['title']}
                    </div>
                    {time_badge}
                </div>
            </a>
        '''
    
    news_html += '''
            </div>
        </div>
    </div>
    '''
    
    return title_html + summary_card + ticker_html + news_html


if __name__ == "__main__":
    # Test the fetcher
    logging.basicConfig(level=logging.INFO)
    news, tickers = fetch_finviz_news()
    print(f"Found {len(news)} news items")
    print(f"Extracted tickers: {tickers}")
    for item in news[:5]:
        print(f"  - {item['title'][:60]}...")
