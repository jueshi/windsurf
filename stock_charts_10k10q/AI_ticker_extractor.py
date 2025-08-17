"""
Changelog:
- 2024-01-04: 
  * Added pyperclip to read clipboard content
  * Implemented AI-assisted stock ticker extraction
  * Added regex to extract stock tickers list from AI response
  * Added functionality to copy extracted tickers to clipboard
"""

from openai import OpenAI
import os
from dotenv import load_dotenv
import pyperclip
import re
import ast

# Load environment variables from .env file
load_dotenv()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

query = "gennerate a pythhon list of stock tickers from the folowing content: "
content = '\n'.join([x.strip() for x in pyperclip.paste().splitlines() if x.strip()])

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", ""),  # Optional
    "X-Title": os.getenv("OPENROUTER_X_TITLE", ""),  # Optional
  },
  model="deepseek/deepseek-chat",
  messages=[
    {
      "role": "user",
      "content": query + content
    }
  ]
)

print(completion.choices[0].message.content)

# Get AI response content
AI_response_content = completion.choices[0].message.content

# Print the full content for debugging
print("Full AI response content:")
print(repr(AI_response_content))

# Regular expression to find a list of stock tickers
code_patterns = [
    # Pattern for tickers with comments
    (r'stock_tickers\s*=\s*(\[(?:\s*"[^"]+",\s*#\s*[^]]+)+\s*\])', 'Tickers with comments pattern'),
    # Pattern for tickers without comments
    (r'stock_tickers\s*=\s*(\[(?:\s*"[^"]+",?)+\s*\])', 'Tickers without comments pattern'),
    # Alternate patterns
    (r'```python\n.*?stock_tickers\s*=\s*(\[(?:[^]]*)\])', 'Markdown code block pattern'),
    (r'\[(?:[^]]*)\]', 'Alternate pattern')
]

# Try each pattern
for code_pattern, pattern_description in code_patterns:
    print(f"\nTrying pattern: {pattern_description}")
    print(f"Regex pattern: {code_pattern}")
    
    code_match = re.search(code_pattern, AI_response_content, re.DOTALL | re.MULTILINE)
    
    if code_match:
        print("Match found!")
        # Get the list part of the match
        stock_tickers_str = code_match.group(1) if code_match.groups() else code_match.group(0)
        
        print("Raw tickers string:", stock_tickers_str)
        
        try:
            # Try to extract tickers with comments first
            ticker_pattern_with_comments = r'"([^"]+)",\s*#\s*(.+)'
            ticker_matches = re.findall(ticker_pattern_with_comments, stock_tickers_str)
            
            # If no comments found, just extract tickers
            if not ticker_matches:
                ticker_pattern = r'"([^"]+)"'
                ticker_matches = [(ticker, ticker) for ticker in re.findall(ticker_pattern, stock_tickers_str)]
            
            # Prepare the tickers and comments
            stock_tickers = [match[0] for match in ticker_matches]
            comments = [match[1] for match in ticker_matches]
            
            print("Extracted stock tickers:")
            for ticker, comment in zip(stock_tickers, comments):
                print(f"{ticker}: {comment}")
            
            # Append to ticker_lists.py
            with open('ticker_lists.py', 'a') as f:
                f.write('\n')
                f.write('AI_ticker_extractor_tickers = [\n')
                for ticker, comment in zip(stock_tickers, comments):
                    f.write(f'    "{ticker}",  # {comment}\n')
                f.write(']\n')
            
            # Copy to clipboard
            pyperclip.copy(stock_tickers_str)
            
            # Process the tickers
            from data_rechiever import StockDataManager
            stock_manager = StockDataManager()
            stock_manager.process_stock_data(tickers=stock_tickers, name='stocks_from_AI_ticker_extractor')
            
            break  # Stop after first successful match
            
        except Exception as e:
            print(f"Error parsing tickers: {e}")

else:
    print("No stock tickers list found in the response.")
