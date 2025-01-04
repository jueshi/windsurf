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

# Extract the Python code block
code_match = re.search(r'stock_tickers\s*=\s*\[(.*?)\]', completion.choices[0].message.content, re.DOTALL)
if code_match:
    stock_tickers_str = code_match.group(0)
    print(stock_tickers_str)
    pyperclip.copy(stock_tickers_str)
else:
    print("No stock tickers list found in the response.")
