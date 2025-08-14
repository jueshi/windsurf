import os
import re
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


def analyze_10q_report(file_path):
    """
    Analyzes a 10-Q report using Google Gemini API.

    Args:
        file_path (str): The path to the 10-Q report file.

    Returns:
        str: The comprehensive analysis of the 10-Q report.
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
        return f"Error reading 10-Q file: {e}"

    # Extract CIK and accession number to build the URL
    accession_number_match = re.search(r"ACCESSION NUMBER:\s*([\d-]+)", report_text)
    cik_match = re.search(r"CENTRAL INDEX KEY:\s*(\d+)", report_text)

    filing_url = "链接未找到"
    if accession_number_match and cik_match:
        accession_number_no_dashes = accession_number_match.group(1).replace('-', '')
        cik = cik_match.group(1)
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number_no_dashes}/{accession_number_match.group(1)}-index.htm"


    # Extract relevant sections using regex
    def extract_section(text, start_pattern, end_pattern):
        start_match = re.search(start_pattern, text, re.IGNORECASE | re.DOTALL)
        if not start_match:
            return None
        start_index = start_match.end()
        end_match = re.search(end_pattern, text[start_index:], re.IGNORECASE | re.DOTALL)
        if not end_match:
            return text[start_index:]
        end_index = end_match.start()
        return text[start_index:start_index + end_index]

    item1_text = extract_section(report_text, r"Item\s+1\.\s+Financial Statements", r"Item\s+2\.")
    item2_text = extract_section(report_text, r"Item\s+2\.\s+Management's Discussion and Analysis", r"Item\s+3\.")
    item4_text = extract_section(report_text, r"Item\s+4\.\s+Controls and Procedures", r"PART\s+II")

    sections = {
        "财务报表 (Financial Statements)": item1_text,
        "管理层的讨论与分析 (MD&A)": item2_text,
        "控制与程序 (Controls and Procedures)": item4_text,
    }

    summaries = {}
    for title, text in sections.items():
        if text:
            # Truncate text to avoid being too long
            text = text[:10000]
            prompt = f"请用中文总结以下10-Q报告的 '{title}' 部分:\n\n{text}"
            try:
                response = model.generate_content(prompt)
                summaries[title] = response.text
            except Exception as e:
                summaries[title] = f"无法总结该部分: {e}"
        else:
            summaries[title] = "未找到该部分。"

    final_prompt = f"""
    请根据以下10-Q报告各部分的摘要，生成一份全面的中文商业分析报告。

    **财务报表 (Financial Statements) 摘要:**
    {summaries.get('财务报表 (Financial Statements)', 'N/A')}

    **管理层的讨论与分析 (MD&A) 摘要:**
    {summaries.get("管理层的讨论与分析 (MD&A)", 'N/A')}

    **控制与程序 (Controls and Procedures) 摘要:**
    {summaries.get("控制与程序 (Controls and Procedures)", 'N/A')}

    请综合以上信息，提供一份深入的、结构化的分析报告，重点突出公司的最新财务表现、管理层对业绩的看法以及内部控制的有效性。
    报告最后，请提供在线报告的直接链接: {filing_url}
    """

    try:
        response = model.generate_content(final_prompt)
        return response.text
    except Exception as e:
        return f"生成最终分析时出错: {e}"

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
