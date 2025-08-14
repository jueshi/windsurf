import os
import google.generativeai as genai
from dotenv import load_dotenv

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

def analyze_10k_report(file_path):
    """
    Analyzes a 10-K report using Google Gemini API.

    Args:
        file_path (str): The path to the 10-K report file.

    Returns:
        str: The comprehensive analysis of the 10-K report.
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

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            report_text = f.read()
    except Exception as e:
        return f"Error reading 10-K file: {e}"

    # Extract relevant sections using regex
    def extract_section(text, start_pattern, end_pattern):
        import re
        start_match = re.search(start_pattern, text, re.IGNORECASE | re.DOTALL)
        if not start_match:
            return None
        start_index = start_match.end()
        end_match = re.search(end_pattern, text[start_index:], re.IGNORECASE | re.DOTALL)
        if not end_match:
            return text[start_index:]
        end_index = end_match.start()
        return text[start_index:start_index + end_index]

    item1_text = extract_section(report_text, r"Item\s+1\.\s+Business", r"Item\s+1A\.")
    item1a_text = extract_section(report_text, r"Item\s+1A\.\s+Risk Factors", r"Item\s+1B\.")
    item7_text = extract_section(report_text, r"Item\s+7\.\s+Management's Discussion and Analysis", r"Item\s+7A\.")

    sections = {
        "业务 (Business)": item1_text,
        "风险因素 (Risk Factors)": item1a_text,
        "管理层的讨论与分析 (MD&A)": item7_text,
    }

    summaries = {}
    for title, text in sections.items():
        if text:
            # Truncate text to avoid being too long
            text = text[:10000]
            prompt = f"请用中文总结以下10-K报告的 '{title}' 部分:\n\n{text}"
            try:
                response = model.generate_content(prompt)
                summaries[title] = response.text
            except Exception as e:
                summaries[title] = f"无法总结该部分: {e}"
        else:
            summaries[title] = "未找到该部分。"

    final_prompt = f"""
    请根据以下10-K报告各部分的摘要，生成一份全面的中文商业分析报告。

    **业务 (Business) 摘要:**
    {summaries.get('业务 (Business)', 'N/A')}

    **风险因素 (Risk Factors) 摘要:**
    {summaries.get('风险因素 (Risk Factors)', 'N/A')}

    **管理层的讨论与分析 (MD&A) 摘要:**
    {summaries.get("管理层的讨论与分析 (MD&A)", 'N/A')}

    请综合以上信息，提供一份深入的、结构化的分析报告，重点突出公司的核心业务、主要风险和管理层对公司未来发展的看法。
    """

    try:
        response = model.generate_content(final_prompt)
        return response.text
    except Exception as e:
        return f"生成最终分析时出错: {e}"

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
