from google import genai
import os
import logging
import json
from datetime import datetime
import time
import random
import hashlib

# Configure logging
log_directory = os.getenv("LOG_DIR", "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, f"llm_calls_{datetime.now().strftime('%Y%m%d')}.log")

# Set up logger
logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Simple cache configuration
cache_file = "llm_cache.json"

# By default, we Google Gemini 2.5 pro, as it shows great performance for code understanding
def call_llm(prompt: str, use_cache: bool = True) -> str:
    # Log the prompt
    logger.info(f"PROMPT: {prompt}")
    
    # Check cache if enabled
    if use_cache:
        # Use hash for prompt to avoid issues with very long prompts as dictionary keys
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        
        # Load cache from disk
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, starting with empty cache")
        
        # Return from cache if exists
        if prompt_hash in cache:
            logger.info(f"CACHE HIT: Using cached response for prompt hash {prompt_hash[:8]}...")
            return cache[prompt_hash]
    
    # Call the LLM if not in cache or cache disabled
    # client = genai.Client(
    #     vertexai=True, 
    #     # TODO: change to your own project id and location
    #     project=os.getenv("GEMINI_PROJECT_ID", "your-project-id"),
    #     location=os.getenv("GEMINI_LOCATION", "us-central1")
    # )
    # You can comment the previous line and use the AI Studio key instead:
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY", "AIzaSyABRTy0lZ1dysdKkOv0YHPqnuKvmnQk4Ik"),
    )
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25")
    
    # Add retry logic with exponential backoff for API rate limiting
    max_retries = 5
    base_delay = 2  # starting delay in seconds
    
    for retry_count in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt]
            )
            response_text = response.text
            break  # Success, exit retry loop
        except Exception as e:
            # Check if it's a rate limit error
            if "429" in str(e) or "rate limit" in str(e).lower():
                # Calculate delay with exponential backoff and jitter
                delay = base_delay * (2 ** retry_count) + random.uniform(0, 1)
                logger.warning(f"Rate limit hit, retrying in {delay:.2f} seconds (attempt {retry_count+1}/{max_retries})")
                
                # If this is the last retry, raise the exception
                if retry_count == max_retries - 1:
                    logger.error(f"Max retries reached, giving up: {e}")
                    raise
                    
                time.sleep(delay)
            else:
                # Not a rate limit error, raise immediately
                logger.error(f"API error: {e}")
                raise
    
    # Log the response
    logger.info(f"RESPONSE: {response_text}")
    
    # Update cache if enabled
    if use_cache:
        # Use hash for prompt to avoid issues with very long prompts as dictionary keys
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        
        # Load cache again to avoid overwrites
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache for saving: {e}")
        
        # Add to cache and save
        cache[prompt_hash] = response_text
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    return response_text

# # Use Anthropic Claude 3.7 Sonnet Extended Thinking
# def call_llm(prompt, use_cache: bool = True):
#     from anthropic import Anthropic
#     client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "your-api-key"))
#     response = client.messages.create(
#         model="claude-3-7-sonnet-20250219",
#         max_tokens=21000,
#         thinking={
#             "type": "enabled",
#             "budget_tokens": 20000
#         },
#         messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return response.content[1].text

# # Use OpenAI o1
# def call_llm(prompt, use_cache: bool = True):    
#     from openai import OpenAI
#     client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"))
#     r = client.chat.completions.create(
#         model="o1",
#         messages=[{"role": "user", "content": prompt}],
#         response_format={
#             "type": "text"
#         },
#         reasoning_effort="medium",
#         store=False
#     )
#     return r.choices[0].message.content

if __name__ == "__main__":
    test_prompt = "Hello, how are you?"
    
    # First call - should hit the API
    print("Making call...")
    response1 = call_llm(test_prompt, use_cache=False)
    print(f"Response: {response1}")
    
