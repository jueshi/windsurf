import os
import re
import json
import html
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

    # Prepare the prompt
    prompt = f"""
    You are a financial analyst specializing in stock analysis. Analyze the following company:
    
    Ticker: {ticker}
    
    Company Information:
    {json.dumps(company_info, indent=2)}
    
    Provide a comprehensive business analysis including:
    1. Business Overview
    2. Industry Position
    3. Competitive Advantages
    4. Growth Prospects
    5. Risks and Challenges
    
    Format your response in markdown.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating content: {e}")
        return f"Error generating analysis: {str(e)}"

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
        print(f"Error initializing model: {e}")
        return f"Error initializing Gemini model: {str(e)}"

    # Load the 10-K report
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            report_text = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return f"Error reading 10-K file: {str(e)}"

    # Extract CIK and accession number from the file path for linking back to SEC
    filing_url = "https://www.sec.gov/edgar/search-and-access"
    cik_match = re.search(r'\/(\d+)\/10-K\/', file_path)
    accession_number_match = re.search(r'\/10-K\/([^\/]+)\/', file_path)
    
    if cik_match and accession_number_match:
        accession_number_no_dashes = accession_number_match.group(1).replace('-', '')
        cik = cik_match.group(1)
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number_no_dashes}/{accession_number_match.group(1)}-index.htm"
    else:
        try:
            # Fallback to constructing a search link
            ticker = file_path.split(os.path.sep)[1]
            filing_url = f"链接到SEC网站手动搜索: https://www.sec.gov/edgar/search/#/q={ticker}"
        except IndexError:
            filing_url = "链接到SEC网站手动搜索: https://www.sec.gov/edgar/search-and-access"

    # Helper function to clean extracted text
    def clean_extracted_text(text):
        """Clean up extracted text by removing HTML tags and normalizing whitespace."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Decode HTML entities
        text = html.unescape(text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove any non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        return text.strip()

    # Basic section extraction function
    def extract_section(text, start_pattern, end_pattern):
        """Extract a section from the report using regex patterns."""
        start_match = re.search(start_pattern, text, re.IGNORECASE | re.DOTALL)
        if not start_match:
            return None
        start_index = start_match.end()
        end_match = re.search(end_pattern, text[start_index:], re.IGNORECASE | re.DOTALL)
        if not end_match:
            return text[start_index:]
        end_index = end_match.start()
        return text[start_index:start_index + end_index]

    # Enhanced robust section extraction function
    def extract_section_robust(report_text, start_patterns, end_patterns, max_search_chars=100000, section_type=None):
        """
        Extract a section from the report using multiple start and end patterns.
        
        Args:
            report_text (str): The text of the report.
            start_patterns (list): List of regex patterns to match the start of the section.
            end_patterns (list): List of regex patterns to match the end of the section.
            max_search_chars (int): Maximum number of characters to search after the start pattern.
            section_type (str, optional): Type of section being extracted ('business', 'risk', or 'mda').
                                         Used for specialized extraction methods.
            
        Returns:
            str: The extracted section, or None if not found.
        """
        # For MD&A section, try direct position extraction first as it's proven most reliable
        if section_type == 'mda':
            try:
                # Direct position extraction for MD&A section
                print("Attempting direct position extraction for MD&A section...")
                
                # Search for Item 7 and Item 7A markers
                item7_match = re.search(r'<span[^>]*>Item 7\.&#160;&#160;&#160;&#160;Management&#8217;s Discussion', report_text)
                if not item7_match:
                    item7_match = re.search(r'Item 7\.&#160;&#160;&#160;&#160;Management&#8217;s Discussion', report_text)
                
                if not item7_match:
                    item7_match = re.search(r'Item\s+7\.?\s*Management(?:\'|&#8217;)s\s+Discussion', report_text, re.IGNORECASE)
                
                if item7_match:
                    print(f"Found Item 7 marker at position {item7_match.start()}")
                    
                    # Start searching from the Item 7 position
                    start_index = item7_match.start()
                    
                    # Look for Item 7A or Item 8 (end of MD&A section)
                    search_limit = min(len(report_text) - start_index, 300000)  # 300K chars
                    search_text = report_text[start_index:start_index + search_limit]
                    
                    # Try to find the end of the MD&A section
                    end_match = re.search(r'<span[^>]*>Item 7A\.&#160;&#160;&#160;&#160;Quantitative', search_text)
                    if not end_match:
                        end_match = re.search(r'Item 7A\.&#160;&#160;&#160;&#160;Quantitative', search_text)
                    
                    if not end_match:
                        end_match = re.search(r'Item\s+7A\.?\s*Quantitative', search_text, re.IGNORECASE)
                    
                    if not end_match:
                        end_match = re.search(r'Item\s+8\.?\s*Financial', search_text, re.IGNORECASE)
                    
                    if end_match:
                        print(f"Found MD&A end marker at relative position {end_match.start()}")
                        end_index = end_match.start()
                        section_text = search_text[:end_index]
                        cleaned_text = clean_extracted_text(section_text)
                        print(f"Direct position extraction successful: {len(cleaned_text)} characters")
                        return cleaned_text
                    else:
                        print("Could not find MD&A end marker")
                else:
                    print("Could not find Item 7 marker")
            except Exception as e:
                print(f"Error with direct position extraction for MD&A: {e}")
        
        # Standard pattern-based extraction (works well for Business and Risk Factors)
        print(f"Attempting pattern-based extraction for {section_type if section_type else 'unknown'} section...")
        
        # Try each start pattern
        for i, start_pattern in enumerate(start_patterns):
            start_match = re.search(start_pattern, report_text, re.IGNORECASE | re.DOTALL)
            if not start_match:
                continue
            
            print(f"Found start pattern #{i+1} at position {start_match.start()}")
            
            # Get the position right after the start pattern
            start_index = start_match.end()
            
            # Look for the end pattern, but limit the search to a reasonable chunk of text
            search_limit = min(len(report_text) - start_index, max_search_chars)
            search_text = report_text[start_index:start_index + search_limit]
            
            # Try each end pattern
            for j, end_pattern in enumerate(end_patterns):
                end_match = re.search(end_pattern, search_text, re.IGNORECASE | re.DOTALL)
                if not end_match:
                    continue
                
                print(f"Found end pattern #{j+1} at relative position {end_match.start()}")
                
                end_index = end_match.start()
                section_text = search_text[:end_index]
                
                # Clean up the extracted text
                cleaned_text = clean_extracted_text(section_text)
                print(f"Pattern-based extraction successful: {len(cleaned_text)} characters")
                
                return cleaned_text
        
        # If we get here, we couldn't find the section using any method
        print(f"Failed to extract {section_type if section_type else 'unknown'} section")
        return None

    # Define improved extraction patterns based on HTML structure and our testing
    # Business section patterns
    business_start_patterns = [
        # HTML formatted patterns
        r"<span[^>]*>Item\s+1\.?\s*(?:&#160;)*\s*Business</span>",
        r"<span[^>]*>ITEM\s+1\.?\s*(?:&#160;)*\s*BUSINESS</span>",
        # Regular patterns
        r"Item\s+1\.\s*Business",
        r"ITEM\s+1\.\s*BUSINESS",
        r"Item\s+1\s+Business",
        r"ITEM\s+1\s+BUSINESS",
        # Common variations
        r"PART\s+I\s+Item\s+1\.\s+Business",
        r"PART\s+I\s+ITEM\s+1\.\s+BUSINESS",
        # Broader patterns
        r"Business",
        r"BUSINESS",
        r"Company Overview",
        r"COMPANY OVERVIEW"
    ]
    
    business_end_patterns = [
        # HTML formatted patterns
        r"<span[^>]*>Item\s+1A\.?\s*(?:&#160;)*\s*Risk\s+Factors</span>",
        r"<span[^>]*>ITEM\s+1A\.?\s*(?:&#160;)*\s*RISK\s+FACTORS</span>",
        # Regular patterns
        r"Item\s+1A\.\s*Risk\s+Factors",
        r"ITEM\s+1A\.\s*RISK\s+FACTORS",
        r"Item\s+1A\s+Risk\s+Factors",
        r"ITEM\s+1A\s+RISK\s+FACTORS",
        # Broader patterns
        r"Risk Factors",
        r"RISK\s+FACTORS"
    ]
    
    # Risk Factors section patterns
    risk_start_patterns = [
        # HTML formatted patterns
        r"<span[^>]*>Item\s+1A\.?\s*(?:&#160;)*\s*Risk\s+Factors</span>",
        r"<span[^>]*>ITEM\s+1A\.?\s*(?:&#160;)*\s*RISK\s+FACTORS</span>",
        # Regular patterns
        r"Item\s+1A\.\s*Risk\s+Factors",
        r"ITEM\s+1A\.\s*RISK\s+FACTORS",
        r"Item\s+1A\s+Risk\s+Factors",
        r"ITEM\s+1A\s+RISK\s+FACTORS",
        # Broader patterns
        r"Risk Factors",
        r"RISK\s+FACTORS"
    ]
    
    risk_end_patterns = [
        # HTML formatted patterns
        r"<span[^>]*>Item\s+1B\.?\s*(?:&#160;)*\s*</span>",
        r"<span[^>]*>ITEM\s+1B\.?\s*(?:&#160;)*\s*</span>",
        r"<span[^>]*>Item\s+2\.?\s*(?:&#160;)*\s*</span>",
        r"<span[^>]*>ITEM\s+2\.?\s*(?:&#160;)*\s*</span>",
        # Regular patterns
        r"Item\s+1B\.",
        r"ITEM\s+1B\.",
        r"Item\s+2\.",
        r"ITEM\s+2\.",
        # Broader patterns
        r"UNRESOLVED STAFF COMMENTS",
        r"Unresolved Staff Comments"
    ]
    
    # MD&A section patterns - based on our detailed analysis
    mda_start_patterns = [
        # Exact match from analysis
        r"<span style=\"color:#000000;font-family:'Helvetica',sans-serif;font-size:9pt;font-weight:700;line-height:120%\">Item 7\.&#160;&#160;&#160;&#160;Management&#8217;s Discussion and Analysis",
        # More generic patterns based on analysis
        r"<span[^>]*>Item 7\.&#160;&#160;&#160;&#160;Management&#8217;s Discussion and Analysis",
        r"<span[^>]*>Item\s+7\.?\s*(?:&#160;)*\s*Management(?:&#8217;|')?s\s+Discussion\s+and\s+Analysis",
        r"<span[^>]*>ITEM\s+7\.?\s*(?:&#160;)*\s*MANAGEMENT(?:&#8217;|')?S\s+DISCUSSION\s+AND\s+ANALYSIS",
        # Anchor to position from analysis
        r"Item 7\.&#160;&#160;&#160;&#160;Management&#8217;s Discussion and Analysis",
        # Regular patterns
        r"Item\s+7\.\s*Management(?:&#8217;|')?s\s+Discussion\s+and\s+Analysis",
        r"ITEM\s+7\.\s*MANAGEMENT(?:&#8217;|')?S\s+DISCUSSION\s+AND\s+ANALYSIS",
        r"Item\s+7\s+Management(?:&#8217;|')?s\s+Discussion\s+and\s+Analysis",
        r"ITEM\s+7\s+MANAGEMENT(?:&#8217;|')?S\s+DISCUSSION\s+AND\s+ANALYSIS",
        # Common variations
        r"Item\s+7\.\s*Management(?:&#8217;|')?s\s+Discussion",
        r"ITEM\s+7\.\s*MANAGEMENT(?:&#8217;|')?S\s+DISCUSSION",
        # Financial condition variations
        r"Management(?:&#8217;|')?s\s+Discussion\s+and\s+Analysis\s+of\s+Financial\s+Condition",
        r"MANAGEMENT(?:&#8217;|')?S\s+DISCUSSION\s+AND\s+ANALYSIS\s+OF\s+FINANCIAL\s+CONDITION"
    ]
    
    mda_end_patterns = [
        # Exact match from analysis
        r"<span style=\"color:#000000;font-family:'Helvetica',sans-serif;font-size:9pt;font-weight:700;line-height:120%\">Item 7A\.&#160;&#160;&#160;&#160;Quantitative and Qualitative Disclosures About Market Risk</span>",
        # More generic patterns based on analysis
        r"<span[^>]*>Item 7A\.&#160;&#160;&#160;&#160;Quantitative and Qualitative Disclosures About Market Risk</span>",
        r"<span[^>]*>Item\s+7A\.?\s*(?:&#160;)*\s*Quantitative\s+and\s+Qualitative\s+Disclosures\s+About\s+Market\s+Risk</span>",
        r"<span[^>]*>ITEM\s+7A\.?\s*(?:&#160;)*\s*QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES\s+ABOUT\s+MARKET\s+RISK</span>",
        # Regular patterns
        r"Item\s+7A\.\s*Quantitative\s+and\s+Qualitative\s+Disclosures\s+About\s+Market\s+Risk",
        r"ITEM\s+7A\.\s*QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES\s+ABOUT\s+MARKET\s+RISK",
        # Item 8 as fallback
        r"Item\s+8\.\s*Financial\s+Statements",
        r"ITEM\s+8\.\s*FINANCIAL\s+STATEMENTS"
    ]

    # Extract sections using the robust extraction function
    print("Extracting Business section...")
    business_section = extract_section_robust(report_text, business_start_patterns, business_end_patterns, section_type='business')
    print(f"Business section extracted: {len(business_section) if business_section else 0} characters")
    
    print("Extracting Risk Factors section...")
    risk_factors_section = extract_section_robust(report_text, risk_start_patterns, risk_end_patterns, section_type='risk')
    print(f"Risk Factors section extracted: {len(risk_factors_section) if risk_factors_section else 0} characters")
    
    print("Extracting MD&A section...")
    mda_section = extract_section_robust(report_text, mda_start_patterns, mda_end_patterns, section_type='mda')
    print(f"MD&A section extracted: {len(mda_section) if mda_section else 0} characters")

    # Prepare the prompt for Gemini API
    prompt = f"""
    You are a financial analyst specializing in SEC filings analysis. Analyze the following sections from a 10-K report:
    
    BUSINESS SECTION:
    {business_section if business_section else "Not available"}
    
    RISK FACTORS SECTION:
    {risk_factors_section if risk_factors_section else "Not available"}
    
    MANAGEMENT'S DISCUSSION AND ANALYSIS (MD&A) SECTION:
    {mda_section if mda_section else "Not available"}
    
    Provide a comprehensive analysis including:
    1. Business Overview and Key Products/Services
    2. Market Position and Competitive Landscape
    3. Key Risk Factors and Their Potential Impact
    4. Financial Performance Analysis
    5. Future Outlook and Strategic Initiatives
    
    Format your response in markdown. Include a link to the SEC filing: {filing_url}
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating content: {e}")
        return f"Error generating 10-K analysis: {str(e)}"

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
        print(f"Error initializing model: {e}")
        return f"Error initializing Gemini model: {str(e)}"

    # Load the 10-Q report
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            report_text = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return f"Error reading 10-Q file: {str(e)}"

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

    # Extract CIK and accession number from the file path for linking back to SEC
    filing_url = "https://www.sec.gov/edgar/search-and-access"
    cik_match = re.search(r'\/(\d+)\/10-Q\/', file_path)
    accession_number_match = re.search(r'\/10-Q\/([^\/]+)\/', file_path)
    
    if cik_match and accession_number_match:
        accession_number_no_dashes = accession_number_match.group(1).replace('-', '')
        cik = cik_match.group(1)
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number_no_dashes}/{accession_number_match.group(1)}-index.htm"
    else:
        try:
            # Fallback to constructing a search link
            ticker = file_path.split(os.path.sep)[1]
            filing_url = f"链接到SEC网站手动搜索: https://www.sec.gov/edgar/search/#/q={ticker}"
        except IndexError:
            filing_url = "链接到SEC网站手动搜索: https://www.sec.gov/edgar/search-and-access"

    # Prepare the prompt for Gemini API
    prompt = f"""
    You are a financial analyst specializing in SEC filings analysis. Analyze the following sections from a 10-Q report:
    
    FINANCIAL STATEMENTS:
    {item1_text if item1_text else "Not available"}
    
    MANAGEMENT'S DISCUSSION AND ANALYSIS:
    {item2_text if item2_text else "Not available"}
    
    CONTROLS AND PROCEDURES:
    {item4_text if item4_text else "Not available"}
    
    Provide a comprehensive quarterly report analysis including:
    1. Financial Performance Summary
    2. Key Changes from Previous Quarter
    3. Management's Perspective on Results
    4. Notable Risks or Concerns
    5. Future Outlook
    
    Format your response in markdown. Include a link to the SEC filing: {filing_url}
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating content: {e}")
        return f"Error generating 10-Q analysis: {str(e)}"

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
        print(f"Error initializing model: {e}")
        return f"Error initializing Gemini model: {str(e)}"

    # Format the news articles for the prompt
    articles_text = ""
    for i, article in enumerate(news_articles):
        articles_text += f"Article {i+1}:\n"
        articles_text += f"Title: {article.get('title', 'No title')}\n"
        articles_text += f"Content: {article.get('content', 'No content')}\n"
        articles_text += f"URL: {article.get('url', 'No URL')}\n"
        articles_text += f"Published: {article.get('published_date', 'No date')}\n\n"

    # Prepare the prompt for Gemini API
    prompt = f"""
    You are a financial news analyst. Analyze the following news articles:
    
    {articles_text}
    
    Provide a comprehensive news summary including:
    1. Key Developments
    2. Market Sentiment
    3. Potential Impact on Stock Price
    4. Analyst Opinions (if any)
    5. Future Outlook Based on News
    
    Format your response in markdown.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating content: {e}")
        return f"Error generating news analysis: {str(e)}"

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
        print(f"Error initializing model: {e}")
        return f"Error initializing Gemini model: {str(e)}"

    # Prepare the prompt for Gemini API
    prompt = f"""
    You are a financial analyst specializing in stock analysis. Answer the following query about this company:
    
    Ticker: {ticker}
    
    Company Information:
    {json.dumps(company_info, indent=2)}
    
    User Query: {query}
    
    Provide a comprehensive answer to the query based on the company information provided.
    Format your response in markdown.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating content: {e}")
        return f"Error generating search result: {str(e)}"
