import os
import re
import json
import time
import base64
import requests
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from functools import wraps
from datetime import datetime, timedelta
from threading import Lock
import random

# Global language setting: "en" for English, "zh" for Chinese
# Can be set via environment variable GEMINI_RESPONSE_LANGUAGE
RESPONSE_LANGUAGE = os.getenv("GEMINI_RESPONSE_LANGUAGE", "en")

SCENARIO_ALIASES = {
    "up": "bullish",
    "bull": "bullish",
    "bullish": "bullish",
    "rally": "bullish",
    "down": "bearish",
    "bear": "bearish",
    "bearish": "bearish",
    "drop": "bearish",
    "selloff": "bearish",
    "flat": "neutral",
    "sideways": "neutral",
    "range": "neutral",
    "neutral": "neutral"
}

SCENARIO_DISPLAY = {
    "bullish": "⚡ Bullish (market expected to rise)",
    "bearish": "🛡️ Bearish (market expected to fall)",
    "neutral": "🎯 Neutral/Rangebound"
}


def _normalize_market_scenario(value: str | None) -> str:
    """Normalize scenario input into bullish/bearish/neutral."""
    if not value:
        return "neutral"
    value = value.strip().lower()
    return SCENARIO_ALIASES.get(value, "neutral")

def get_language_instruction():
    """Get the language instruction for prompts based on global setting."""
    if RESPONSE_LANGUAGE == "zh":
        return "Please respond in Chinese (中文)."
    return "Please respond in English."


def extract_stock_tickers(text: str, exclude_ticker: str = None) -> list:
    """
    Extract stock ticker symbols from text.
    Looks for common patterns like $AAPL, (AAPL), AAPL:, or standalone uppercase tickers.
    
    Args:
        text: The text to extract tickers from
        exclude_ticker: Optional ticker to exclude (e.g., the main ticker being analyzed)
        
    Returns:
        List of unique ticker symbols found
    """
    # Common patterns for stock tickers
    patterns = [
        r'\$([A-Z]{1,5})\b',           # $AAPL format
        r'\(([A-Z]{1,5})\)',            # (AAPL) format
        r'\b([A-Z]{1,5}):\s',           # AAPL: format
        r'ticker[:\s]+([A-Z]{1,5})\b',  # ticker: AAPL or ticker AAPL
        r'\b([A-Z]{2,5})\b(?=\s+(?:stock|shares|company|Inc|Corp|Ltd))',  # AAPL stock, AAPL shares
    ]
    
    tickers = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            ticker = match.upper()
            # Filter out common words that look like tickers
            common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 
                          'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'HAS',
                          'CEO', 'CFO', 'COO', 'IPO', 'ETF', 'GDP', 'USA', 'USD',
                          'AI', 'IT', 'OR', 'AN', 'AS', 'AT', 'BE', 'BY', 'DO',
                          'IF', 'IN', 'IS', 'NO', 'OF', 'ON', 'SO', 'TO', 'UP',
                          'VS', 'WE', 'PE', 'EPS', 'ROE', 'ROA', 'YOY', 'QOQ'}
            if ticker not in common_words and len(ticker) >= 2:
                tickers.add(ticker)
    
    # Also look for well-known company tickers mentioned by name
    company_tickers = {
        'APPLE': 'AAPL', 'MICROSOFT': 'MSFT', 'GOOGLE': 'GOOGL', 'ALPHABET': 'GOOGL',
        'AMAZON': 'AMZN', 'META': 'META', 'FACEBOOK': 'META', 'TESLA': 'TSLA',
        'NVIDIA': 'NVDA', 'NETFLIX': 'NFLX', 'INTEL': 'INTC', 'AMD': 'AMD',
        'SALESFORCE': 'CRM', 'ORACLE': 'ORCL', 'IBM': 'IBM', 'CISCO': 'CSCO',
        'ADOBE': 'ADBE', 'PAYPAL': 'PYPL', 'SHOPIFY': 'SHOP', 'SPOTIFY': 'SPOT',
        'UBER': 'UBER', 'LYFT': 'LYFT', 'AIRBNB': 'ABNB', 'COINBASE': 'COIN',
        'WALMART': 'WMT', 'TARGET': 'TGT', 'COSTCO': 'COST', 'HOME DEPOT': 'HD',
        'NIKE': 'NKE', 'STARBUCKS': 'SBUX', 'MCDONALDS': 'MCD', 'DISNEY': 'DIS',
        'BOEING': 'BA', 'LOCKHEED': 'LMT', 'RAYTHEON': 'RTX', 'GENERAL ELECTRIC': 'GE',
        'FORD': 'F', 'GM': 'GM', 'GENERAL MOTORS': 'GM', 'TOYOTA': 'TM',
        'JPMORGAN': 'JPM', 'GOLDMAN': 'GS', 'MORGAN STANLEY': 'MS', 'BANK OF AMERICA': 'BAC',
        'VISA': 'V', 'MASTERCARD': 'MA', 'AMERICAN EXPRESS': 'AXP',
        'JOHNSON': 'JNJ', 'PFIZER': 'PFE', 'MERCK': 'MRK', 'ABBVIE': 'ABBV',
        'EXXON': 'XOM', 'CHEVRON': 'CVX', 'CONOCOPHILLIPS': 'COP',
        'VERIZON': 'VZ', 'AT&T': 'T', 'T-MOBILE': 'TMUS',
        'BERKSHIRE': 'BRK.B', 'BLACKROCK': 'BLK', 'SCHWAB': 'SCHW'
    }
    
    text_upper = text.upper()
    for company, ticker in company_tickers.items():
        if company in text_upper:
            tickers.add(ticker)
    
    # Remove the excluded ticker (the one being analyzed)
    if exclude_ticker:
        tickers.discard(exclude_ticker.upper())
    
    return sorted(list(tickers))

def get_news_labels():
    """Get localized labels for news analysis based on language setting."""
    if RESPONSE_LANGUAGE == "zh":
        return {
            "title": "## 新闻分析",
            "bullish": "### 利好消息",
            "bearish": "### 利空消息",
            "no_bullish": "近期无明显利好消息。",
            "no_bearish": "近期无明显利空消息。",
        }
    return {
        "title": "## News Analysis",
        "bullish": "### Bullish News",
        "bearish": "### Bearish News",
        "no_bullish": "No significant bullish news recently.",
        "no_bearish": "No significant bearish news recently.",
    }

# Helper to list accessible Gemini models that support generateContent
def _list_supported_gemini_models() -> list:
    """Return the Gemini model names accessible to the current API key."""
    try:
        models = genai.list_models()
    except Exception as err:
        logging.warning("Unable to list Gemini models: %s", err)
        return []

    supported = []
    for model in models:
        methods = getattr(model, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            supported.append(model.name)
    return supported


# Helper to format Gemini API errors with actionable guidance
def _format_gemini_error(e: Exception) -> str:
    """
    Return a user-friendly error message for common Gemini API failures.
    """
    msg = str(e)
    # Common signals for API not enabled
    service_disabled_tokens = [
        "SERVICE_DISABLED",
        "Generative Language API has not been used",
        "API has not been used in project",
        "it is disabled"
    ]
    if any(tok in msg for tok in service_disabled_tokens):
        return f"Gemini API is disabled for your Google Cloud project. Original error: {msg}"

    unsupported_tokens = [
        "404",
        "not found",
        "NOT_FOUND",
        "not supported",
        "Unsupported",
    ]
    if any(tok in msg for tok in unsupported_tokens):
        return f"Requested Gemini model is unavailable. Original error: {msg}"

    # Default: return the raw error message
    return f"An error occurred while communicating with the Gemini API: {msg}"

class GeminiRateLimiter:
    _instance = None
    _lock = Lock()
    _last_call_time = 0
    MIN_INTERVAL = 2.0
    MAX_RETRIES = 3
    BASE_DELAY = 5.0

    # Patterns that indicate rate limiting or quota exceeded
    RATE_LIMIT_PATTERNS = [
        "RATE_LIMIT_EXCEEDED",
        "429",
        "quota",
        "exceeded your current quota",
        "rate limit",
        "too many requests",
        "Resource has been exhausted",
    ]

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GeminiRateLimiter, cls).__new__(cls)
        return cls._instance

    def _is_rate_limit_error(self, error_msg: str) -> bool:
        """Check if an error message indicates a rate limit issue"""
        error_lower = error_msg.lower()
        return any(pattern.lower() in error_lower for pattern in self.RATE_LIMIT_PATTERNS)

    def wait_for_rate_limit(self):
        with self._lock:
            current_time = time.time()
            time_since_last_call = current_time - self._last_call_time
            wait_time = max(0, self.MIN_INTERVAL - time_since_last_call)

            if wait_time > 0:
                time.sleep(wait_time)

            self._last_call_time = time.time()

    def make_api_call_with_retry(self, api_call_func, *args, **kwargs):
        """
        Makes an API call with retry logic for rate limit errors.
        """
        last_exception = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                # Apply rate limiting
                self.wait_for_rate_limit()

                # Make the API call
                return api_call_func(*args, **kwargs)

            except Exception as e:
                last_exception = e
                error_msg = str(e)

                if self._is_rate_limit_error(error_msg) and attempt < self.MAX_RETRIES:
                    delay = self.BASE_DELAY * (2 ** attempt) * (0.5 + random.random())
                    logging.warning(f"Rate limit exceeded. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    continue
                else:
                    raise

        raise last_exception or Exception("Unknown error in API call with retry")

# ----- Gemini model selection helpers -----
def _get_gemini_model_candidates() -> list:
    env_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
    candidates = [env_name, "gemini-2.0-flash", "gemini-1.5-flash"]
    seen = set()
    ordered = []
    for n in candidates:
        if n and n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered

def _init_gemini_model_with_fallback() -> genai.GenerativeModel:
    last_err = None
    for name in _get_gemini_model_candidates():
        try:
            return genai.GenerativeModel(name)
        except Exception as e:
            last_err = e
            msg = str(e)
            if any(tok in msg for tok in ["404", "not found", "NOT_FOUND", "not supported", "Unsupported"]):
                continue
            raise
    raise last_err or Exception("No supported Gemini model found")

def analyze_ticker(ticker, company_info):
    """
    Analyzes a stock ticker using Google Gemini API.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    rate_limiter = GeminiRateLimiter()
    genai.configure(api_key=api_key)

    try:
        model = _init_gemini_model_with_fallback()
    except Exception as e:
        return _format_gemini_error(e)
    lang_instruction = get_language_instruction()
    snapshot = _format_fundamental_snapshot(company_info)

    prompt = f"""
    Provide a structured business analysis for ticker {ticker}.

    ## Company Snapshot
    {snapshot}

    ## Deliverables
    1. Business Model
    2. Competitive Landscape (key competitors, differentiators)
    3. Financial Health (valuation, margins, growth, leverage)
    4. Growth Drivers
    5. Key Risks

    Keep it concise and actionable. Use markdown headers and bullet lists.

    {lang_instruction}
    """

    try:
        response = rate_limiter.make_api_call_with_retry(
            model.generate_content, prompt
        )
        return response.text
    except Exception as e:
        return _format_gemini_error(e)

def analyze_trading_journey(portfolio_name: str, context: dict) -> str:
    """Summarize trading journey using trades + equity analytics."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    rate_limiter = GeminiRateLimiter()
    genai.configure(api_key=api_key)

    try:
        model = _init_gemini_model_with_fallback()
    except Exception as e:
        return _format_gemini_error(e)

    lang_instruction = get_language_instruction()

    trades = context.get("trades", [])[:200]
    strategies = context.get("strategies", [])[:50]
    equity_points = context.get("equity_points", [])[:200]

    trades_text = json.dumps(trades, default=str)
    strategies_text = json.dumps(strategies, default=str)
    equity_text = json.dumps(equity_points, default=str)

    prompt = f"""
    You are reviewing the trading journey for portfolio "{portfolio_name}".

    Recent trades (chronological, capped at 200):
    {trades_text}

    Per-strategy stats + sample equity points:
    {strategies_text}

    Overall equity curve samples:
    {equity_text}

    Provide:
    1. Narrative of key inflection points and discipline patterns.
    2. Strategy attribution insights (what's working, what's not, mention live vs simulated).
    3. Risk management callouts (drawdowns, position sizing issues, psychological notes if present).
    4. Actionable next steps (experiments, habit checks, scenario drills).

    Keep it concise (under 350 words) with markdown sections and bullet lists.
    {lang_instruction}
    """

    try:
        response = rate_limiter.make_api_call_with_retry(
            model.generate_content, prompt
        )
        return response.text
    except Exception as e:
        return _format_gemini_error(e)

    


def analyze_fundamentals(ticker, fundamental_data):
    """
    Analyzes fundamental data for a stock using Google Gemini API.
    Provides insights on valuation, profitability, growth, and financial health.

    Args:
        ticker (str): The stock ticker symbol.
        fundamental_data (dict): Dictionary of fundamental metrics.

    Returns:
        str: AI-generated analysis of the fundamental data.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    rate_limiter = GeminiRateLimiter()
    genai.configure(api_key=api_key)
    try:
        model = _init_gemini_model_with_fallback()
    except Exception as e:
        return _format_gemini_error(e)

    lang_instruction = get_language_instruction()
    
    # Format metrics for prompt - handle both raw data and pre-formatted filtered data
    metrics_text = "\n".join([f"- **{k}:** {v}" for k, v in fundamental_data.items() if v and v != 'N/A' and v != '-'])
    
    prompt = f"""
    Analyze the following fundamental metrics for {ticker} and provide investment insights.

    **Key Metrics:**
    {metrics_text}

    Please provide a concise analysis covering:

    1. **Valuation Assessment:** Is the stock overvalued, fairly valued, or undervalued based on P/E, P/B, P/S, and PEG ratios? Compare to typical industry ranges.

    2. **Profitability Analysis:** Evaluate profit margins, ROE, and ROA. Are these metrics strong or weak for this sector?

    3. **Growth Outlook:** Based on revenue and earnings growth rates, what's the growth trajectory?

    4. **Financial Health:** Assess debt levels, liquidity ratios (current/quick ratio), and cash flow. Any red flags?

    5. **Dividend Analysis:** If applicable, evaluate dividend sustainability based on yield and payout ratio.

    6. **Key Takeaways:** Summarize 2-3 main investment considerations (both positive and negative).

    Keep the analysis focused and actionable. Use bullet points where appropriate.
    
    {lang_instruction}
    """

    try:
        response = rate_limiter.make_api_call_with_retry(
            model.generate_content, prompt
        )
        return response.text
    except Exception as e:
        return _format_gemini_error(e)


def chat_response(message, ticker=None, ticker_data=None):
    """
    General chat function for answering any user questions.
    Optionally includes ticker context if available.

    Args:
        message (str): User's question or message.
        ticker (str): Optional ticker symbol for context.
        ticker_data (dict): Optional fundamental data for the ticker.

    Returns:
        str: AI-generated response.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    rate_limiter = GeminiRateLimiter()
    genai.configure(api_key=api_key)
    try:
        model = _init_gemini_model_with_fallback()
    except Exception as e:
        return _format_gemini_error(e)

    lang_instruction = get_language_instruction()
    
    # Build context if ticker data is available
    context = ""
    if ticker and ticker_data:
        context = f"""
        Context: The user is currently viewing {ticker}.
        Key data for {ticker}:
        - Company: {ticker_data.get('longName', 'N/A')}
        - Sector: {ticker_data.get('sector', 'N/A')}
        - Industry: {ticker_data.get('industry', 'N/A')}
        - Market Cap: {ticker_data.get('marketCap', 'N/A')}
        - P/E Ratio: {ticker_data.get('trailingPE', 'N/A')}
        - Price: {ticker_data.get('currentPrice', 'N/A')}
        
        Use this context if relevant to the user's question.
        """
    elif ticker:
        context = f"Context: The user is currently viewing {ticker}. Use this context if relevant."
    
    prompt = f"""
    You are a helpful financial assistant in a stock analysis application.
    Answer the user's question clearly and concisely.
    
    {context}
    
    User's question: {message}
    
    Guidelines:
    - Be helpful and informative
    - If the question is about stocks/investing, provide actionable insights
    - Use bullet points for lists
    - Keep responses focused and not too long
    - If you don't know something, say so
    
    {lang_instruction}
    """

    try:
        response = rate_limiter.make_api_call_with_retry(
            model.generate_content, prompt
        )
        return response.text
    except Exception as e:
        return _format_gemini_error(e)


def analyze_comparison(tickers, metrics_data):
    """
    Analyzes and compares fundamental data for multiple stocks using Google Gemini API.
    Focuses on comparative insights and relative valuation.

    Args:
        tickers (list): List of stock ticker symbols.
        metrics_data (dict): Dictionary with ticker as key and metrics dict as value.

    Returns:
        str: AI-generated comparative analysis.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    rate_limiter = GeminiRateLimiter()
    genai.configure(api_key=api_key)
    try:
        model = _init_gemini_model_with_fallback()
    except Exception as e:
        return _format_gemini_error(e)

    lang_instruction = get_language_instruction()
    
    # Format metrics for each ticker
    comparison_text = ""
    for ticker in tickers:
        ticker_metrics = metrics_data.get(ticker, {})
        if ticker_metrics:
            metrics_list = "\n".join([f"  - {k}: {v}" for k, v in ticker_metrics.items()])
            comparison_text += f"\n**{ticker}:**\n{metrics_list}\n"
    
    # Build ticker header for tables
    ticker_headers = " | ".join(tickers)
    
    prompt = f"""
    Perform a comparative analysis of the following stocks based on their fundamental metrics.

    {comparison_text}

    **IMPORTANT: Format your response using markdown tables for easy comparison.**

    ## Comparison Summary Table
    Create a summary table with key metrics. Example format:
    | Metric | {ticker_headers} | Winner |
    |--------|{"|".join(["---" for _ in tickers])}|--------|
    | P/E Ratio | ... | ... | TICKER |

    Include these metrics in the table (if available): P/E, Forward P/E, P/B, P/S, Profit Margin, ROE, ROA, Debt/Equity, Current Ratio, Revenue Growth, Beta

    ## Analysis by Category

    ### 1. Valuation
    Brief analysis of which stock is cheapest/most expensive and why.

    ### 2. Profitability  
    Brief analysis of margins and returns.

    ### 3. Growth
    Brief analysis of growth rates and sustainability.

    ### 4. Financial Health
    Brief analysis of debt and liquidity.

    ### 5. Risk
    Brief analysis of volatility and risk factors.

    ## Final Verdict Table
    | Category | Best Pick | Reason |
    |----------|-----------|--------|
    | Best Value | TICKER | ... |
    | Best Growth | TICKER | ... |
    | Safest | TICKER | ... |
    | **Overall Winner** | **TICKER** | ... |

    Keep text analysis brief (2-3 sentences per section). Focus on the tables for easy scanning.
    
    {lang_instruction}
    """

    try:
        response = rate_limiter.make_api_call_with_retry(
            model.generate_content, prompt
        )
        return response.text
    except Exception as e:
        return _format_gemini_error(e)


def analyze_news(news_articles):
    """
    Analyzes a list of news articles using Google Gemini API.

    Args:
        news_articles (list): A list of news articles from Tavily.

    Returns:
        str: A structured summary of the news.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    try:
        model = _init_gemini_model_with_fallback()
    except Exception as e:
        return _format_gemini_error(e)

    good_news = []
    bad_news = []

    rate_limiter = GeminiRateLimiter()

    lang_instruction = get_language_instruction()
    
    for article in news_articles:
        prompt = f"""
        Summarize the following news article and classify it as "bullish", "bearish", or "neutral".
        Include specific data points and percentage changes if mentioned.
        Return your response in JSON format with "summary" and "sentiment" fields.
        {lang_instruction}

        Title: {article.get('title', 'N/A')}
        Content: {article.get('content', 'N/A')}
        """
        try:
            # Use make_api_call_with_retry to handle 429 quota errors automatically
            response = rate_limiter.make_api_call_with_retry(
                model.generate_content, prompt
            )
            # Clean the response to make it valid JSON
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
            result = json.loads(cleaned_response)

            summary = result.get('summary', 'Unable to generate summary.')
            sentiment = result.get('sentiment', 'neutral').lower()

            if "bullish" in sentiment or "positive" in sentiment or "利好" in sentiment:
                good_news.append(f"- {summary} ([source]({article.get('url', '#')}))")
            elif "bearish" in sentiment or "negative" in sentiment or "利空" in sentiment:
                bad_news.append(f"- {summary} ([source]({article.get('url', '#')}))")
        except Exception as e:
            logging.error(f"Error processing article: {e}")
            continue

    # Format the final output using localized labels
    labels = get_news_labels()
    output = f"{labels['title']}\n\n"
    output += f"{labels['bullish']}\n"
    if good_news:
        output += "\n".join(good_news)
    else:
        output += f"{labels['no_bullish']}\n"

    output += f"\n\n{labels['bearish']}\n"
    if bad_news:
        output += "\n".join(bad_news)
    else:
        output += f"{labels['no_bearish']}\n"

    return output


def analyze_sec_section(ticker: str, section_label: str, section_description: str, section_content: str, form_type: str = "10-K"):
    """
    Analyzes a specific section from an SEC filing using Google Gemini API.
    
    Args:
        ticker: Stock ticker symbol
        section_label: The section label (e.g., "Item 1A")
        section_description: The section description (e.g., "Risk Factors")
        section_content: The extracted text content of the section
        form_type: The filing type (10-K or 10-Q)
    
    Returns:
        str: AI-generated analysis of the section
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    rate_limiter = GeminiRateLimiter()
    genai.configure(api_key=api_key)
    
    try:
        model = _init_gemini_model_with_fallback()
    except Exception as e:
        return _format_gemini_error(e)

    lang_instruction = get_language_instruction()
    
    # Truncate content if too long for API
    max_content_length = 30000
    if len(section_content) > max_content_length:
        section_content = section_content[:max_content_length] + "\n\n[Content truncated...]"
    
    # Customize prompt based on section type
    section_prompts = {
        "Risk Factors": """
            Analyze these risk factors and provide:
            1. **Top 5 Key Risks**: List the most significant risks mentioned
            2. **Risk Categories**: Group risks by type (operational, financial, regulatory, competitive, etc.)
            3. **New/Emerging Risks**: Highlight any risks that seem new or particularly relevant to current market conditions
            4. **Investor Implications**: What should investors be most concerned about?
        """,
        "Business": """
            Analyze this business description and provide:
            1. **Business Model Summary**: How does the company make money?
            2. **Key Products/Services**: Main revenue drivers
            3. **Competitive Position**: Market position and advantages
            4. **Geographic Presence**: Where does the company operate?
            5. **Key Takeaways**: 3-4 bullet points for investors
        """,
        "Management's Discussion": """
            Analyze this MD&A section and provide:
            1. **Performance Summary**: How did the company perform?
            2. **Key Drivers**: What drove revenue/earnings changes?
            3. **Management's Outlook**: What is management saying about the future?
            4. **Red Flags**: Any concerning language or trends?
            5. **Key Metrics**: Important numbers mentioned
        """,
        "Financial Statements": """
            Analyze this financial section and provide:
            1. **Key Financial Highlights**: Important numbers and trends
            2. **Balance Sheet Health**: Assets, liabilities, equity observations
            3. **Cash Flow Insights**: Operating, investing, financing activities
            4. **Notable Changes**: Year-over-year or quarter-over-quarter changes
            5. **Investor Takeaways**: What matters most for investors?
        """,
        "Controls and Procedures": """
            Analyze this controls section and provide:
            1. **Control Effectiveness**: Are controls effective?
            2. **Material Weaknesses**: Any reported weaknesses?
            3. **Changes in Controls**: Any significant changes?
            4. **Risk Assessment**: Any governance concerns?
        """,
    }
    
    # Find matching prompt or use default
    specific_prompt = ""
    for key, prompt in section_prompts.items():
        if key.lower() in section_description.lower():
            specific_prompt = prompt
            break
    
    if not specific_prompt:
        specific_prompt = """
            Analyze this section and provide:
            1. **Key Points**: Main takeaways from this section
            2. **Important Details**: Specific facts, numbers, or disclosures
            3. **Investor Relevance**: Why this matters for investors
            4. **Summary**: 2-3 sentence summary
        """
    
    prompt = f"""
    You are analyzing the {section_label} ({section_description}) section from {ticker}'s {form_type} SEC filing.
    
    **Section Content:**
    {section_content}
    
    **Analysis Instructions:**
    {specific_prompt}
    
    Format your response with clear markdown headers and bullet points.
    Be concise but thorough. Focus on actionable insights for investors.
    
    {lang_instruction}
    """

    try:
        response = rate_limiter.make_api_call_with_retry(
            model.generate_content, prompt
        )
        return response.text
    except Exception as e:
        return _format_gemini_error(e)


def _format_fundamental_snapshot(fundamentals: dict | None) -> str:
    """Reduce the noisy fundamentals blob to a concise prompt block."""
    if not fundamentals:
        return "Not available"

    key_map = {
        "longName": "Company",
        "sector": "Sector",
        "industry": "Industry",
        "marketCap": "Market Cap",
        "currentPrice": "Price",
        "fiftyTwoWeekHigh": "52W High",
        "fiftyTwoWeekLow": "52W Low",
        "beta": "Beta",
        "trailingPE": "PE (TTM)",
        "forwardPE": "Forward PE",
        "profitMargins": "Profit Margin",
        "operatingMargins": "Operating Margin",
        "revenueGrowth": "Revenue Growth",
        "earningsGrowth": "Earnings Growth",
    }

    lines = []
    for key, label in key_map.items():
        value = fundamentals.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            if key in {"profitMargins", "operatingMargins", "revenueGrowth", "earningsGrowth"}:
                value = f"{value * 100:.2f}%"
            elif key == "marketCap" and value:
                if abs(value) >= 1e12:
                    value = f"${value / 1e12:.2f}T"
                elif abs(value) >= 1e9:
                    value = f"${value / 1e9:.2f}B"
                elif abs(value) >= 1e6:
                    value = f"${value / 1e6:.2f}M"
                else:
                    value = f"${value:,.0f}"
            else:
                value = f"{value:.2f}"
        lines.append(f"- {label}: {value}")

    return "\n".join(lines) if lines else "Not available"


def _format_price_context(price_context: dict | None) -> str:
    if not price_context:
        return "Not provided"

    parts = []
    if price_context.get("current_price"):
        parts.append(f"Current Price: {price_context['current_price']}")
    if price_context.get("support_levels"):
        supports = ", ".join(str(x) for x in price_context["support_levels"])
        parts.append(f"Support: {supports}")
    if price_context.get("resistance_levels"):
        resist = ", ".join(str(x) for x in price_context["resistance_levels"])
        parts.append(f"Resistance: {resist}")
    if price_context.get("volatility"):
        parts.append(f"Volatility: {price_context['volatility']}")
    if price_context.get("notes"):
        parts.append(f"Notes: {price_context['notes']}")

    return " | ".join(parts) if parts else "Not provided"


def recommend_strategy_for_ticker(
    ticker: str,
    scenario: str = "neutral",
    fundamentals: dict | None = None,
    price_context: dict | None = None,
    timeframe: str = "swing",
    benchmark: str | None = None,
) -> str:
    """Return an AI-generated trading & options plan for a specific ticker."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    scenario = _normalize_market_scenario(scenario)
    scenario_label = SCENARIO_DISPLAY.get(scenario, "🎯 Neutral/Rangebound")

    rate_limiter = GeminiRateLimiter()
    genai.configure(api_key=api_key)
    try:
        model = _init_gemini_model_with_fallback()
    except Exception as e:
        return _format_gemini_error(e)

    lang_instruction = get_language_instruction()

    fundamentals_block = _format_fundamental_snapshot(fundamentals)
    price_block = _format_price_context(price_context)
    benchmark_text = benchmark or "SPY"

    prompt = f"""
    You are an institutional-grade trading strategist. Build a complete trade plan for {ticker}.

    ## Inputs
    - Primary Scenario: {scenario_label}
    - Timeframe Focus: {timeframe}
    - Benchmark for context: {benchmark_text}
    - Key Fundamentals:\n{fundamentals_block}
    - Price Context: {price_block}

    ## Requirements
    1. Provide separate sections for **Bullish**, **Bearish**, and **Neutral/Rangebound** paths. Emphasize the primary scenario first.
    2. For each path, include:
       - **Equity Plan** (entries, adds, exits, stop discipline)
       - **Options Overlay** (specific structures with strikes/expirations guidance)
       - **Risk Controls** (position sizing, hedges, what invalidates the trade)
       - **Trigger Checklist** (technicals or catalysts to monitor)
    3. Reference current fundamentals or price context when justifying tactics.
    4. End with a concise **Playbook Summary Table** comparing the three paths.

    {lang_instruction}
    """

    try:
        response = rate_limiter.make_api_call_with_retry(
            model.generate_content, prompt
        )
        return response.text
    except Exception as e:
        return _format_gemini_error(e)


def _format_portfolio_positions(positions: list[dict]) -> tuple[str, float]:
    if not positions:
        return ("No valid positions provided", 0.0)

    lines = []
    total_value = 0.0
    for pos in positions:
        ticker = pos.get("ticker", "?")
        qty = float(pos.get("quantity", 0) or 0)
        price = float(pos.get("current_price") or pos.get("avg_cost") or 0)
        sector = pos.get("sector") or "Unknown"
        weight = float(pos.get("weight", 0) or 0)
        value = qty * price
        total_value += max(value, 0)
        lines.append(
            f"- {ticker}: qty {qty:.2f}, px {price:.2f}, sector {sector}, weight {weight:.2f}%, notes: {pos.get('notes', 'n/a')}"
        )

    return ("\n".join(lines), total_value)


def recommend_strategy_for_portfolio(
    portfolio_name: str,
    positions: list[dict],
    scenario: str = "neutral",
    timeframe: str = "swing",
    synthetic_symbol: str | None = None,
    notes: str | None = None,
) -> str:
    """Return a portfolio-level allocation, hedge, and options plan."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    scenario = _normalize_market_scenario(scenario)
    scenario_label = SCENARIO_DISPLAY.get(scenario, "🎯 Neutral/Rangebound")

    rate_limiter = GeminiRateLimiter()
    genai.configure(api_key=api_key)
    try:
        model = _init_gemini_model_with_fallback()
    except Exception as e:
        return _format_gemini_error(e)

    lang_instruction = get_language_instruction()

    positions_block, total_value = _format_portfolio_positions(positions)
    exposure_notes = notes or "None provided"
    synthetic_symbol = synthetic_symbol or "PORT-AUTO"

    prompt = f"""
    You are managing the portfolio "{portfolio_name}" (synthetic ticker {synthetic_symbol}).

    ## Portfolio Snapshot
    - Total Market Value: ${total_value:,.2f}
    - Holdings Detail:\n{positions_block}
    - Scenario Focus: {scenario_label}
    - Timeframe: {timeframe}
    - Additional Notes: {exposure_notes}

    ## Deliverables
    1. Provide **allocation adjustments** for bullish, bearish, and neutral regimes.
    2. Recommend **index or single-name hedges**, plus at least one options package (e.g., collars, spreads, ratio hedges) sized for this book.
    3. Call out **sector/position concentrations** and how to rebalance or pair-trade them.
    4. Explain how to utilize the synthetic ticker for benchmarking and alerting.
    5. Summarize in a markdown table listing scenario, target beta, gross/net exposure, and key action items.

    {lang_instruction}
    """

    try:
        response = rate_limiter.make_api_call_with_retry(
            model.generate_content, prompt
        )
        return response.text
    except Exception as e:
        return _format_gemini_error(e)


def extract_positions_from_image(
    image_bytes: bytes,
    portfolio_hint: str | None = None,
    mime_type: str = "image/png",
    text_hint: str | None = None,
    preprocessing_notes: dict | None = None,
) -> dict:
    """Use Gemini vision models to parse a brokerage screenshot into structured holdings.

    Args:
        image_bytes: Raw (optionally preprocessed) image payload sent to Gemini Vision.
        portfolio_hint: Friendly name used to personalize responses.
        mime_type: MIME string for the provided image bytes.
        text_hint: Optional OCR transcription used to reinforce difficult screenshots.
        preprocessing_notes: Metadata describing preprocessing pipeline steps.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY not configured"}

    rate_limiter = GeminiRateLimiter()
    genai.configure(api_key=api_key)
    try:
        model = _init_gemini_model_with_fallback()
    except Exception as e:
        return {"error": _format_gemini_error(e)}

    lang_instruction = get_language_instruction()
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    hint_block = ""
    if text_hint:
        truncated_hint = text_hint.strip()
        if len(truncated_hint) > 2000:
            truncated_hint = truncated_hint[:2000] + "…"
        hint_block = f"""
        ## OCR Text Hint
        {truncated_hint}
        """

    preprocess_block = ""
    if preprocessing_notes:
        steps = ", ".join(preprocessing_notes.get("steps", [])) or "none"
        preprocess_block = f"""
        ## Image Preprocessing Metadata
        Steps: {steps}
        """

    prompt_text = f"""
    You are an AI portfolio assistant. Extract the visible holdings from this screenshot of a brokerage account.

    Return JSON with this schema:
    {{
        "positions": [
            {{"ticker": "AAPL", "quantity": 10, "avg_cost": 150.25, "current_price": 189.30, "notes": ""}}
        ],
        "summary": "Short explanation of what you found"
    }}

    - Use ticker symbols as they appear (convert to uppercase, strip class suffixes if obvious).
    - If cost basis is not visible, set avg_cost to null.
    - If only market value is shown, infer price = value / shares.
    - Ignore cash balances.
    - If the image quality is low, still attempt to guess tickers/quantities and note low confidence.

    {lang_instruction}
    """

    message = {
        "role": "user",
        "parts": [
            {"text": prompt_text},
            {
                "inline_data": {
                    "mime_type": mime_type or "image/png",
                    "data": encoded,
                }
            }
        ],
    }

    try:
        response = rate_limiter.make_api_call_with_retry(
            model.generate_content,
            [message],
        )
        text = response.text.strip()
        cleaned = text.replace("```json", "").replace("```", "")
        payload = json.loads(cleaned)
        positions = payload.get("positions", [])
        if portfolio_hint:
            payload["portfolio"] = portfolio_hint
        base_payload = payload | {"positions": positions}
        if text_hint:
            base_payload["ocr_hint_used"] = True
        if preprocessing_notes:
            base_payload["preprocessing"] = preprocessing_notes
        return base_payload
    except Exception as e:
        logging.error("Screenshot extraction failed: %s", e)
        return {"error": str(e)}


def summarize_text(text: str) -> str:
    """
    Summarize arbitrary text using AI.
    Useful for summarizing clipboard content, articles, etc.
    
    Args:
        text: The text to summarize
        
    Returns:
        AI-generated summary as markdown string
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."
    
    rate_limiter = GeminiRateLimiter()
    genai.configure(api_key=api_key)
    
    try:
        model = _init_gemini_model_with_fallback()
    except Exception as e:
        return _format_gemini_error(e)
    
    lang_instruction = get_language_instruction()
    
    prompt = f"""
    Please summarize the following text. Provide:
    
    1. **Key Points**: Main takeaways (bullet points)
    2. **Summary**: A concise 2-3 paragraph summary
    3. **Relevance**: If this appears to be financial/market related, note any investment implications
    
    Text to summarize:
    ---
    {text[:8000]}
    ---
    
    {lang_instruction}
    """
    
    try:
        response = rate_limiter.make_api_call_with_retry(
            model.generate_content, prompt
        )
        return response.text
    except Exception as e:
        return _format_gemini_error(e)


def summarize_market_news(headlines: str, news_type: str = "market") -> str:
    """
    Summarize market news headlines using AI.
    Provides key themes, market sentiment, and actionable insights.
    
    Args:
        headlines: Newline-separated list of news headlines
        news_type: Type of news (market, stocks, etf, crypto)
        
    Returns:
        AI-generated summary as markdown string
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."
    
    rate_limiter = GeminiRateLimiter()
    genai.configure(api_key=api_key)
    
    try:
        model = _init_gemini_model_with_fallback()
    except Exception as e:
        return _format_gemini_error(e)
    
    lang_instruction = get_language_instruction()
    
    # Customize prompt based on news type
    type_context = {
        "market": "general market and economic news",
        "stocks": "individual stock and company news",
        "etf": "ETF and fund-related news",
        "crypto": "cryptocurrency and blockchain news"
    }
    context = type_context.get(news_type, "financial news")
    
    prompt = f"""
    You are a financial news analyst. Analyze these {context} headlines and provide a concise summary.
    
    **Headlines:**
    {headlines}
    
    **Provide:**
    1. **🎯 Key Themes** (2-3 bullet points): What are the main topics driving the news?
    2. **📊 Market Sentiment**: Is the overall tone bullish, bearish, or neutral? Why?
    3. **⚡ Notable Movers**: Any specific stocks, sectors, or assets mentioned prominently?
    4. **💡 Actionable Insight**: One key takeaway for investors (1-2 sentences)
    
    Keep the response concise and focused. Use bullet points for clarity.
    
    {lang_instruction}
    """
    
    try:
        response = rate_limiter.make_api_call_with_retry(
            model.generate_content, prompt
        )
        return response.text
    except Exception as e:
        return _format_gemini_error(e)
