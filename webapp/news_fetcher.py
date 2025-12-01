import json
from tavily import TavilyClient
import os
import logging

def fetch_news(ticker):
    """
    Fetches news articles for a given ticker using Tavily API.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        list: A list of news articles, where each article is a dictionary.
    """
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            # As a fallback, try to get it from the .env file for local dev
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("TAVILY_API_KEY")

        if not api_key:
            logging.error("TAVILY_API_KEY not found in environment variables.")
            return [{"error": "TAVILY_API_KEY not found in environment variables."}]

        client = TavilyClient(api_key=api_key)
        query = f"news about {ticker} stock"

        # Use the Tavily search API
        response = client.search(query=query, search_depth="advanced", max_results=7)

        # The response['results'] is a list of dictionaries, each representing a search result
        # We can directly return this list
        return response['results']

    except Exception as e:
        logging.error(f"An error occurred while fetching news: {e}")
        return [{"error": f"An error occurred while fetching news: {e}"}]
