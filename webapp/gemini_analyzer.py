import os
import re
import json
import time
import requests
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from functools import wraps
from datetime import datetime, timedelta
from threading import Lock
import random

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

    prompt = f"""
    对以下公司进行详细的商业分析，公司股票代码为 '{ticker}'。
    这是该公司的一些基本数据：
    - **公司名称:** {company_info.get('longName', 'N/A')}
    - **行业板块:** {company_info.get('sector', 'N/A')}
    - **具体行业:** {company_info.get('industry', 'N/A')}
    - **市值:** {company_info.get('marketCap', 'N/A')}
    - **市盈率（过去12个月）:** {company_info.get('trailingPE', 'N/A')}
    - **远期市盈率:** {company_info.get('forwardPE', 'N/A')}
    - **股息率:** {company_info.get('dividendYield', 'N/A')}
    - **贝塔系数:** {company_info.get('beta', 'N/A')}
    - **52周最高价:** {company_info.get('fiftyTwoWeekHigh', 'N/A')}
    - **52周最低价:** {company_info.get('fiftyTwoWeekLow', 'N/A')}
    - **业务摘要:** {company_info.get('longBusinessSummary', 'N/A')}

    请提供一份结构良好、详细的中文商业分析，涵盖以下方面：
    1.  **商业模式:** 描述公司的主要商业模式及其收入来源。
    2.  **竞争格局:** 主要竞争对手是谁？这家公司的竞争优势是什么？
    3.  **财务状况:** 根据所提供的指标，简要评估公司的财务状况。
    4.  **增长前景:** 这家公司潜在的增长动力是什么？
    5.  **潜在风险:** 与这家公司相关的主要风险是什么？

    请用Chinese提供结构良好且详细的分析。Followed by an English version of the response in a separate paragraph as well.
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

    for article in news_articles:
        prompt = f"""
        请用中文总结以下新闻文章，并将其分类为“利好”、“利空”或“中性”。
        并提供具体数据和百分比变化。Followed by an English version of the response in a separate paragraph as well.
        请以JSON格式返回，包含“summary”和“sentiment”两个字段。

        新闻标题: {article.get('title', 'N/A')}
        新闻内容: {article.get('content', 'N/A')}
        """
        try:
            # Use make_api_call_with_retry to handle 429 quota errors automatically
            response = rate_limiter.make_api_call_with_retry(
                model.generate_content, prompt
            )
            # Clean the response to make it valid JSON
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
            result = json.loads(cleaned_response)

            summary = result.get('summary', '无法生成摘要。')
            sentiment = result.get('sentiment', '中性').lower()

            if "利好" in sentiment:
                good_news.append(f"- {summary} (来源: {article.get('url', 'N/A')})")
            elif "利空" in sentiment:
                bad_news.append(f"- {summary} (来源: {article.get('url', 'N/A')})")
        except Exception as e:
            logging.error(f"Error processing article: {e}")
            continue

    # Format the final output
    output = "## 新闻分析\n\n"
    output += "### 利好消息\n"
    if good_news:
        output += "\n".join(good_news)
    else:
        output += "近期无明显利好消息。\n"

    output += "\n\n### 利空消息\n"
    if bad_news:
        output += "\n".join(bad_news)
    else:
        output += "近期无明显利空消息。\n"

    return output
