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
    Conduct a detailed business analysis for the company with the stock ticker '{ticker}'.
    Here is some fundamental data for the company:
    - **Company Name:** {company_info.get('longName', 'N/A')}
    - **Sector:** {company_info.get('sector', 'N/A')}
    - **Industry:** {company_info.get('industry', 'N/A')}
    - **Market Cap:** {company_info.get('marketCap', 'N/A')}
    - **Trailing P/E:** {company_info.get('trailingPE', 'N/A')}
    - **Forward P/E:** {company_info.get('forwardPE', 'N/A')}
    - **Dividend Yield:** {company_info.get('dividendYield', 'N/A')}
    - **Beta:** {company_info.get('beta', 'N/A')}
    - **52 Week High:** {company_info.get('fiftyTwoWeekHigh', 'N/A')}
    - **52 Week Low:** {company_info.get('fiftyTwoWeekLow', 'N/A')}
    - **Business Summary:** {company_info.get('longBusinessSummary', 'N/A')}

    Please provide a comprehensive business analysis covering the following aspects:
    1.  **Business Model:** Describe the company's primary business model and how it generates revenue.
    2.  **Competitive Landscape:** Who are the main competitors, and what is this company's competitive advantage?
    3.  **Financial Health:** Briefly assess the company's financial health based on the provided metrics.
    4.  **Growth Prospects:** What are the potential growth drivers for this company?
    5.  **Potential Risks:** What are the key risks associated with this company?

    Provide a well-structured and detailed analysis.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while communicating with the Gemini API: {e}"
