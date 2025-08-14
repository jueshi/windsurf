import os
import re
import google.generativeai as genai
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import json

def analyze_ticker(ticker, company_info):
    """
    Analyzes a stock ticker using Google Gemini API.

    Args:
        ticker (str): The stock ticker symbol.
        company_info (dict): A dictionary containing fundamental data about the company.

    Returns:
        str: The business analysis from Gemini API.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        print(f"Could not initialize model: {e}")
        print("Available models:")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
        return "Error: Could not initialize Gemini model."


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

    请用中文提供结构良好且详细的分析。
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while communicating with the Gemini API: {e}"

def _get_filing_url(ticker, filing_type):
    """
    Searches for the latest SEC filing URL for a given ticker.
    """
    query = f'"{ticker}" "{filing_type}" filing site:sec.gov'
    print(f"Searching with query: {query}")
    try:
        for url in search(query, num=5, stop=5, pause=1.0):
            if "ix?doc=" in url or ".htm" in url:
                if re.search(r'\d{10}-\d{2}-\d{6}', url):
                    print(f"Found potential filing URL: {url}")
                    return url
    except Exception as e:
        print(f"An error occurred during web search: {e}")
        return None
    print("Could not find a suitable URL.")
    return None

def _get_text_from_url(url):
    """
    Fetches and extracts plain text from a URL.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text
    except requests.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during text extraction: {e}")
        return None

def analyze_10k_report(ticker):
    """
    Finds the latest 10-K report from the web, analyzes it using Google Gemini API.
    """
    print(f"Starting 10-K analysis for {ticker}...")
    filing_url = _get_filing_url(ticker, "10-K")
    if not filing_url:
        return f"无法为 {ticker} 的10-K报告找到有效的SECファイリングURL。"

    print(f"Found URL: {filing_url}. Fetching content...")
    report_text = _get_text_from_url(filing_url)
    if not report_text:
        return f"无法从URL获取或解析内容: {filing_url}"

    report_text = report_text[:200000]
    print("Content fetched. Analyzing with Gemini...")

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        return f"Error: Could not initialize Gemini model: {e}"

    prompt = f"""
    请对以下10-K报告进行详细的中文分析。

    报告文本:
    ---
    {report_text}
    ---

    请提供一份结构良好、深入的分析报告，涵盖以下方面：
    1.  **整体摘要:** 对整个10-K报告进行高级别摘要。
    2.  **亮点 (Highlights):** 识别并总结报告中的主要积极方面、成就或优势。
    3.  **不足 (Lowlights):** 识别并总结报告中的主要风险、挑战或负面趋势。
    4.  **核心业务分析:** 详细描述公司的核心业务、收入来源和市场定位。
    5.  **财务状况评估:** 基于报告中的财务数据，评估公司的财务健康状况。
    6.  **管理层讨论:** 总结管理层对公司业绩和未来前景的看法。

    请确保分析客观、信息丰富，并以清晰的格式呈现。
    报告最后，请提供在线报告的直接链接: {filing_url}
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"在最终分析过程中发生错误: {e}"

def analyze_10q_report(ticker):
    """
    Finds the latest 10-Q report from the web, analyzes it using Google Gemini API.
    """
    print(f"Starting 10-Q analysis for {ticker}...")
    filing_url = _get_filing_url(ticker, "10-Q")
    if not filing_url:
        return f"无法为 {ticker} 的10-Q报告找到有效的SECファイリングURL。"

    print(f"Found URL: {filing_url}. Fetching content...")
    report_text = _get_text_from_url(filing_url)
    if not report_text:
        return f"无法从URL获取或解析内容: {filing_url}"

    report_text = report_text[:200000]
    print("Content fetched. Analyzing with Gemini...")

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        return f"Error: Could not initialize Gemini model: {e}"

    prompt = f"""
    请对以下10-Q报告进行详细的中文分析。

    报告文本:
    ---
    {report_text}
    ---

    请提供一份结构良好、深入的季度分析报告，涵盖以下方面：
    1.  **整体摘要:** 对整个10-Q报告进行高级别摘要，重点关注本季度的变化。
    2.  **亮点 (Highlights):** 识别并总结报告中的主要积极方面或超出预期的表现。
    3.  **不足 (Lowlights):** 识别并总结报告中的主要风险、挑战或未达预期的表现。
    4.  **财务表现:** 分析本季度的财务报表，总结关键财务指标的变化。
    5.  **管理层讨论:** 总结管理层对本季度业绩和短期前景的看法。

    请确保分析客观、信息丰富，并以清晰的格式呈现。
    报告最后，请提供在线报告的直接链接: {filing_url}
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"在最终分析过程中发生错误: {e}"

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
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        return f"Error: Could not initialize Gemini model: {e}"

    good_news = []
    bad_news = []

    for article in news_articles:
        prompt = f"""
        请用中文总结以下新闻文章，并将其分类为“利好”、“利空”或“中性”。
        请以JSON格式返回，包含“summary”和“sentiment”两个字段。

        新闻标题: {article.get('title', 'N/A')}
        新闻内容: {article.get('content', 'N/A')}
        """
        try:
            response = model.generate_content(prompt)
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
            print(f"Error processing article: {e}")
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

def general_search(ticker, company_info, query):
    """
    Performs a general AI search about a company using Google Gemini API.

    Args:
        ticker (str): The stock ticker symbol.
        company_info (dict): A dictionary containing fundamental data about the company.
        query (str): The user's search query.

    Returns:
        str: The search result from Gemini API.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        print(f"Could not initialize model: {e}")
        return "Error: Could not initialize Gemini model."

    prompt = f"""
    针对股票代码为 '{ticker}' 的公司 '{company_info.get('longName', 'N/A')}'，请回答以下问题。

    用户问题: "{query}"

    请使用中文进行详细回答。
    ---
    公司参考信息:
    - **行业板块:** {company_info.get('sector', 'N/A')}
    - **具体行业:** {company_info.get('industry', 'N/A')}
    - **业务摘要:** {company_info.get('longBusinessSummary', 'N/A')}
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while communicating with the Gemini API: {e}"
