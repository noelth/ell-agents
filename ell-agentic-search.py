import os
import json
import requests
import ell
from dotenv import load_dotenv
from typing import List, Dict
import random

# Import provider libraries
import anthropic           # For Anthropic's models
from openai import OpenAI  # Used for both OpenAI and Ollama

# Load API keys and settings from .env file
load_dotenv()

# API keys and base URL (only keys go in .env)
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OLLAMA_API_KEY    = os.getenv("OLLAMA_API_KEY")  # May be empty if running locally
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:3002")

# --- Create Client Instances ---
openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.openai.com/v1") if OPENAI_API_KEY else None
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
# For Ollama, we use the OpenAI interface with our locally hosted Searxng (via Docker) on port 3002.
if OLLAMA_API_KEY:
    ollama_client = OpenAI(api_key=OLLAMA_API_KEY, base_url=OLLAMA_BASE_URL)
else:
    ollama_client = OpenAI(base_url=OLLAMA_BASE_URL)

# Register models (so ell knows which client to use for a given model ID)
ell.config.register_model("deepseek-r1:1.5b", ollama_client)
ell.config.register_model("llama3.2:3b", ollama_client)
# (Also, we register an Anthropic model if needed)
ell.config.register_model("claude-3-5-haiku-latest", anthropic_client)

# Map provider names to their client objects.
CLIENTS = {
    "openai": openai_client,
    "anthropic": anthropic_client,
    "ollama": ollama_client,
}

# Enable verbose logging (optional)
ell.config.verbose = True

# --------------------------------------------------------------------
# 1. Web Search Tool Agent
# --------------------------------------------------------------------
@ell.tool()
def web_search_tool(query: str) -> str:
    """
    A web search tool that takes a search query and uses Searxng (hosted locally) to:
      - Identify 3 URLs that give the best results.
      - Run a basic Google query in the format:
         http://localhost:3002/search?q=%21google%20{query}&language=auto&time_range=&safesearch=0&categories=none
      - For the Google query, access the URLs of the top 5 search results.
      - For each URL (both from the original and the Google query), fetch its content.
      - Return a JSON string with two keys: "google_results" (a list of {url, content}) 
        and "original_results" (a list of {url, content}).
    """
    searx_base = "http://localhost:3002/search"
    # Query for best 3 URLs (default Searxng query)
    params = {
        "q": query,
        "language": "auto",
        "time_range": "",
        "safesearch": "0",
        "categories": "none"
    }
    try:
        resp = requests.get(searx_base, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            original_urls = [result["url"] for result in data.get("results", [])][:3]
        else:
            original_urls = []
    except Exception:
        original_urls = []

    print("Original URLs:", original_urls)

    # Query for Google results using Searxng's !google syntax.
    google_query = f"!google {query}"
    google_url = f"{searx_base}?q={requests.utils.quote(google_query)}&language=auto&time_range=&safesearch=0&categories=none"
    try:
        resp_google = requests.get(google_url, timeout=5)
        if resp_google.status_code == 200:
            data_google = resp_google.json()
            google_urls = [result["url"] for result in data_google.get("results", [])][:5]
        else:
            google_urls = []
    except Exception:
        google_urls = []

    print("Google URLs:", google_urls)

    def fetch_content(url: str) -> str:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                # Remove unnecessary characters (simple cleaning: strip whitespace)
                return r.text.strip()
            else:
                return ""
        except Exception:
            return ""

    original_results = [{"url": url, "content": fetch_content(url)} for url in original_urls]
    google_results = [{"url": url, "content": fetch_content(url)} for url in google_urls]

    output = {"google_results": google_results, "original_results": original_results}
    return json.dumps(output)

# --------------------------------------------------------------------
# 2. Web Search Summarizer Agent
# --------------------------------------------------------------------
@ell.simple(model="llama3.2:3b", client=CLIENTS["ollama"], max_tokens=500)
def web_search_summarizer(query: str, search_output: str) -> str:
    """
    You are a web search summarizer. Given the original search query and the JSON string output 
    from the web_search_tool, produce a concise summary of the key findings.
    """
    return f"Summary for query '{query}': {search_output}"

# --------------------------------------------------------------------
# 3. Brain Agent (Orchestrator)
# --------------------------------------------------------------------
@ell.complex(model="llama3.2:3b", client=CLIENTS["ollama"], temperature=0.5, max_tokens=1500)
def brain_agent(context: str) -> str:
    """
    You are the brain agent. You are provided with a user prompt that starts with a context.
    You must decide whether to call the web search tool if more current information is needed.
    If so, call the web_search_tool and web_search_summarizer agents, update the context with their output,
    and then hand off back with the updated context appended to the original prompt.
    If no additional info is needed, return the final answer based solely on the context.
    """
    # For demonstration, we'll assume that if the prompt contains "web search", we perform the search.
    if "web search" in context.lower():
        # Extract the search query from the user prompt (for simplicity, assume the entire prompt is the query)
        search_output = web_search_tool(context)
        summary = web_search_summarizer(context, search_output)
        updated_context = context + "\n" + summary
        return f"Handoff to requesting agent with updated context:\n{updated_context}"
    else:
        # Otherwise, finalize the result using the existing context.
        return f"Final result based on context:\n{context}"

# --------------------------------------------------------------------
# 4. Build Blog Post (Example of Agent Handoff)
# --------------------------------------------------------------------
def build_blog_post():
    # Initial empty context.
    context = ""
    # Prompt the user for a blog topic.
    topic = input("Enter the blog topic: ").strip()
    # Create a user prompt with context at the top.
    context = f"Context:\n {topic}"
    
    # The brain agent decides what to do.
    result = brain_agent(context)
    print("Brain Agent Output:\n", result)

if __name__ == "__main__":
    ell.init(verbose=True, autocommit=True)
    build_blog_post()