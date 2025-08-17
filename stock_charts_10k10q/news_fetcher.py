import json
from tavily import TavilyClient
import os

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
            return [{"error": "TAVILY_API_KEY not found in environment variables."}]

        client = TavilyClient(api_key=api_key)
        query = f"news about {ticker} stock"

        # Use the Tavily search API
        response = client.search(query=query, search_depth="advanced", max_results=7)

        # The response['results'] is a list of dictionaries, each representing a search result
        # We can directly return this list
        return response['results']

    except Exception as e:
        print(f"An error occurred while fetching news: {e}")
        return [{"error": f"An error occurred while fetching news: {e}"}]

if __name__ == '__main__':
    # Example usage
    # Make sure to have a .env file with TAVILY_API_KEY="your_key"
    ticker_to_search = "NVDA"
    news_results = fetch_news(ticker_to_search)

    if news_results and "error" not in news_results[0]:
        print(f"Found {len(news_results)} news articles for {ticker_to_search}:")
        for i, result in enumerate(news_results, 1):
            print(f"  {i}. {result.get('title', 'No Title')}")
            print(f"     URL: {result.get('url', 'No URL')}")
            # print(f"     Content: {result.get('content', 'No Content')[:150]}...") # Print snippet
            print("-" * 20)
    else:
        print(f"Could not fetch news for {ticker_to_search}.")
        if news_results:
            print(f"Error: {news_results[0]['error']}")
