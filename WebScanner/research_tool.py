import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import anthropic
from typing import List, Dict, Optional
import time
import logging
import json
from datetime import datetime
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Load environment variables
# -----------------------------------------------------------------------------
load_dotenv()  # Loads variables from the .env file

# -----------------------------------------------------------------------------
# WebResearchTool Class
# -----------------------------------------------------------------------------
class WebResearchTool:
    def __init__(self, anthropic_api_key: str, searxng_base_url: str, max_urls: int = 4):
        """
        Initialize the research tool with required configuration.

        Args:
            anthropic_api_key: Your Anthropic API key.
            searxng_base_url: Base URL for your locally hosted SearxNG instance.
            max_urls: Maximum number of URLs to process per research query.
        """
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.max_urls = max_urls
        self.searxng_base_url = searxng_base_url
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for debugging and status reporting."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def search_urls(self, query: str) -> Dict:
        """
        Search for relevant URLs using SearxNG.

        The function performs two searches:
          - The default search for up to self.max_urls results.
          - A Google-engine search by adding the "engines" parameter, returning up to self.max_urls results.

        Args:
            query: The search query string.

        Returns:
            A dictionary with keys:
              "original_results": List of result dictionaries from the default engine.
              "google_results": List of result dictionaries from the Google engine.
        """
        searx_url = f"{self.searxng_base_url}/search"
        base_params = {
            "language": "auto",
            "time_range": "",
            "safesearch": "0",
            "categories": "none",
            "format": "json"
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        # ---- Original Search ----
        original_params = base_params.copy()
        original_params["q"] = query
        try:
            self.logger.info(f"Original search: {searx_url} with data: {original_params} and headers: {headers}")
            resp_orig = requests.post(searx_url, data=original_params, headers=headers, timeout=60)
            resp_orig.raise_for_status()
            data_orig = resp_orig.json()
            original_results = []
            for item in data_orig.get("results", [])[:self.max_urls]:
                original_results.append({
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "source": urlparse(item.get("url", "")).netloc
                })
        except Exception as e:
            self.logger.error(f"Error in original search: {str(e)}")
            original_results = []

        # ---- Google Search ----
        google_params = base_params.copy()
        google_params["q"] = query
        google_params["engines"] = "google"
        try:
            self.logger.info(f"Google search: {searx_url} with data: {google_params} and headers: {headers}")
            resp_google = requests.post(searx_url, data=google_params, headers=headers, timeout=10)
            resp_google.raise_for_status()
            data_google = resp_google.json()
            google_results = []
            for item in data_google.get("results", [])[:self.max_urls]:
                google_results.append({
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "source": urlparse(item.get("url", "")).netloc
                })
        except Exception as e:
            self.logger.error(f"Error in google search: {str(e)}")
            google_results = []

        return {"original_results": original_results, "google_results": google_results}

    def filter_urls(self, urls: List[Dict]) -> List[Dict]:
        """
        Filter a list of URLs based on a set of trusted domains.

        Args:
            urls: List of URL dictionaries.

        Returns:
            A filtered list containing only URLs whose domain ends with one of the trusted domains.
        """
        trusted_domains = [
            'nature.com', 'science.org', 'scientificamerican.com',
            'ieee.org', 'acm.org', 'arxiv.org', 'gov', 'edu',
            'github.com', 'stackoverflow.com', 'medium.com'
        ]
        filtered_urls = []
        for url_data in urls:
            domain = urlparse(url_data['url']).netloc
            if any(domain.endswith(trusted) for trusted in trusted_domains):
                filtered_urls.append(url_data)
        return filtered_urls[:self.max_urls]

    def extract_text_from_url(self, url: str) -> Optional[str]:
        """
        Extract the main text content from a given webpage URL.

        Args:
            url: The URL of the webpage.

        Returns:
            A string containing the extracted text, or None if extraction fails.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            self.logger.info(f"Extracting text from URL: {url} with headers: {headers}")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'ads']):
                element.decompose()
            content = []
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'article']):
                text = element.get_text().strip()
                if text:
                    content.append(text)
            return '\n'.join(content)
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {str(e)}")
            return None

    def analyze_webpage(self, url_data: Dict, query: str) -> Dict:
        """
        Analyze a webpage's content in relation to a research query using Anthropic's API.

        Args:
            url_data: Dictionary containing URL, title, snippet, and source.
            query: The research query.

        Returns:
            A dictionary containing the analysis results merged with URL metadata.
        """
        content = self.extract_text_from_url(url_data['url'])
        if not content:
            return {"url": url_data['url'], "error": "Failed to extract content"}
        # Truncate content if it's too long
        max_content_length = 12000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        try:
            prompt = f"""Analyze this webpage about "{query}":

URL: {url_data['url']}
Title: {url_data['title']}
Source: {url_data['source']}

Content: {content}

Provide:
1. Summary of relevant information (3-4 sentences)
2. Key findings related to the query
3. Credibility assessment of the source
4. Publication date or recency (if available)

Format as JSON with keys: summary, key_findings, credibility, date"""
            self.logger.info(f"Analyzing webpage with prompt (first 200 chars): {prompt[:200]}...")
            message = self.client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=1000,
                temperature=0,
                system="You analyze web content and provide structured analysis. Respond only with JSON.",
                messages=[{"role": "user", "content": prompt}]
            )
            analysis = json.loads(message.content[0].text)
            analysis.update(url_data)
            return analysis
        except Exception as e:
            self.logger.error(f"Error analyzing {url_data['url']}: {str(e)}")
            return {"url": url_data['url'], "error": str(e)}

    def research(self, query: str, apply_filter: bool = True) -> Dict:
        """
        Conduct automated research by combining URL search, filtering, analysis, 
        and synthesis of findings.

        Args:
            query: The research question.
            apply_filter: Whether to filter URLs based on trusted domains.

        Returns:
            A dictionary with the research query, analyses, and a synthesis summary.
        """
        self.logger.info(f"Searching for relevant URLs for query: {query}")
        search_results = self.search_urls(query)
        if not search_results:
            return {"error": "No search results found"}
        original_results = search_results.get("original_results", [])
        google_results = search_results.get("google_results", [])
        all_results = original_results + google_results
        if apply_filter:
            final_results = self.filter_urls(all_results)
        else:
            final_results = all_results
        if not final_results:
            return {"error": "No reliable sources found"}
        analyses = []
        for url_data in final_results:
            self.logger.info(f"Analyzing {url_data['url']}")
            analysis = self.analyze_webpage(url_data, query)
            analyses.append(analysis)
            time.sleep(1)  # Rate limiting between analyses

        synthesis_prompt = f"""Research query: {query}

Webpage analyses: {json.dumps(analyses, indent=2)}

Synthesize these findings into:
1. Comprehensive answer to the query
2. Main conclusions
3. Confidence level in findings
4. Areas needing more research
5. Most reliable sources found

Format as JSON with keys: answer, conclusions, confidence, gaps, best_sources"""
        try:
            message = self.client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=1500,
                temperature=0,
                system="You synthesize research findings into clear conclusions. Respond only with JSON.",
                messages=[{"role": "user", "content": synthesis_prompt}]
            )
            raw_response = message.content[0].text
            if not raw_response.strip():
                raise ValueError("Empty synthesis response")
            synthesis = json.loads(raw_response)
            results = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "sources_analyzed": len(analyses),
                "webpage_analyses": analyses,
                "synthesis": synthesis
            }
            return results
        except Exception as e:
            self.logger.error(f"Error synthesizing results: {str(e)}")
            return {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "sources_analyzed": len(analyses),
                "webpage_analyses": analyses,
                "error": str(e)
            }

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    searxng_base_url = os.getenv("SEARXNG_BASE_URL", "http://localhost:3002")
    if not anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY is not set in your .env file.")
        return

    # Initialize researcher with default max_urls of 5
    researcher = WebResearchTool(
        anthropic_api_key=anthropic_api_key,
        searxng_base_url=searxng_base_url,
        max_urls=5
    )
    
    # Get research query from user
    query = input("\nEnter your research question: ")
    # Ask whether to filter URLs, defaulting to 'N'
    filter_choice = input("Filter URLs? (Y/N) [Default N]: ").strip().upper()
    apply_filter = filter_choice == "Y"
    # Get number of URLs to check, defaulting to 4
    num_urls_input = input("Enter number of URLs to check [Default 4]: ").strip()
    max_urls = int(num_urls_input) if num_urls_input.isdigit() else 4
    researcher.max_urls = max_urls  # Update max_urls based on user input

    print("\nResearching... This may take a few minutes.")
    results = researcher.research(query, apply_filter=apply_filter)

    # Create the "search_results" folder in the root directory if it doesn't exist.
    results_folder = "search_results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_folder, f"research_results_{timestamp}.json")
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResearch Results Summary:")
    print(f"Query: {query}")
    print(f"Sources analyzed: {results.get('sources_analyzed', 0)}")
    if "synthesis" in results:
        print("\nFindings:")
        print(json.dumps(results["synthesis"], indent=2))
        print(f"\nFull results saved to: {filename}")
    else:
        print("\nError:", results.get("error", "Unknown error"))

# -----------------------------------------------------------------------------
# Run Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()