import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import anthropic
from typing import List, Dict, Optional
import time
import logging
import json
from googleapiclient.discovery import build
from datetime import datetime

class WebResearchTool:
    def __init__(self, anthropic_api_key: str, google_api_key: str, google_cse_id: str, max_urls: int = 5):
        """
        Initialize the research tool with API keys and configuration.
        
        Args:
            anthropic_api_key: Your Anthropic API key
            google_api_key: Your Google Custom Search API key
            google_cse_id: Your Google Custom Search Engine ID
            max_urls: Maximum number of URLs to process per research query
        """
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.max_urls = max_urls
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for the tool."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def search_urls(self, query: str) -> List[Dict]:
        """
        Search for relevant URLs using Google Custom Search API.
        
        Args:
            query: The search query
            
        Returns:
            List of dictionaries containing URLs and metadata
        """
        try:
            service = build("customsearch", "v1", developerKey=self.google_api_key)
            result = service.cse().list(q=query, cx=self.google_cse_id, num=self.max_urls).execute()
            
            search_results = []
            for item in result.get('items', []):
                search_results.append({
                    'url': item['link'],
                    'title': item['title'],
                    'snippet': item['snippet'],
                    'source': urlparse(item['link']).netloc
                })
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error in search: {str(e)}")
            return []

    def filter_urls(self, urls: List[Dict]) -> List[Dict]:
        """
        Filter URLs based on credibility and relevance.
        
        Args:
            urls: List of URL dictionaries from search
            
        Returns:
            Filtered list of URLs
        """
        # List of trusted domains (expand as needed)
        trusted_domains = [
            'nature.com', 'science.org', 'scientificamerican.com',
            'ieee.org', 'acm.org', 'arxiv.org', 'gov', 'edu',
            'github.com', 'stackoverflow.com', 'medium.com'
        ]
        
        filtered_urls = []
        for url_data in urls:
            domain = urlparse(url_data['url']).netloc
            
            # Check if domain is trusted or ends with trusted suffix
            is_trusted = any(
                domain.endswith(trusted) for trusted in trusted_domains
            )
            
            if is_trusted:
                filtered_urls.append(url_data)
        
        return filtered_urls[:self.max_urls]

    def extract_text_from_url(self, url: str) -> Optional[str]:
        """
        Extract main text content from a webpage.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Extracted text content or None if failed
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'ads']):
                element.decompose()
            
            # Extract text from paragraphs and headers
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
        Analyze a single webpage in the context of the research query.
        
        Args:
            url_data: Dictionary containing URL and metadata
            query: The research query to consider
            
        Returns:
            Dictionary containing analysis results
        """
        content = self.extract_text_from_url(url_data['url'])
        if not content:
            return {"url": url_data['url'], "error": "Failed to extract content"}

        # Truncate content if too long
        max_content_length = 12000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        try:
            # Create analysis prompt
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

            # Get response from Claude
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0,
                system="You analyze web content and provide structured analysis. Respond only with JSON.",
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse JSON response
            analysis = json.loads(message.content[0].text)
            analysis.update(url_data)  # Add URL metadata to analysis
            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing {url_data['url']}: {str(e)}")
            return {"url": url_data['url'], "error": str(e)}

    def research(self, query: str) -> Dict:
        """
        Conduct automated research on a query.
        
        Args:
            query: The research topic or question
            
        Returns:
            Dictionary containing compiled research results
        """
        # Search for relevant URLs
        self.logger.info(f"Searching for relevant URLs for query: {query}")
        search_results = self.search_urls(query)
        
        if not search_results:
            return {"error": "No search results found"}
        
        # Filter URLs
        filtered_urls = self.filter_urls(search_results)
        
        if not filtered_urls:
            return {"error": "No reliable sources found"}
        
        # Analyze each URL
        analyses = []
        for url_data in filtered_urls:
            self.logger.info(f"Analyzing {url_data['url']}")
            analysis = self.analyze_webpage(url_data, query)
            analyses.append(analysis)
            time.sleep(1)  # Rate limiting
        
        # Compile findings
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
            # Get synthesis from Claude
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                temperature=0,
                system="You synthesize research findings into clear conclusions. Respond only with JSON.",
                messages=[{"role": "user", "content": synthesis_prompt}]
            )
            
            synthesis = json.loads(message.content[0].text)
            
            # Compile final results
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

def main():
    """
    Main function to run the research tool interactively.
    """
    # Load configuration
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found. Please create it with your API keys.")
        return
    
    # Initialize researcher
    researcher = WebResearchTool(
        anthropic_api_key=config['anthropic_api_key'],
        google_api_key=config['google_api_key'],
        google_cse_id=config['google_cse_id']
    )
    
    # Get query from user
    query = input("\nEnter your research question: ")
    
    print("\nResearching... This may take a few minutes.")
    
    # Conduct research
    results = researcher.research(query)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_results_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nResearch Results Summary:")
    print(f"Query: {query}")
    print(f"Sources analyzed: {results.get('sources_analyzed', 0)}")
    
    if "synthesis" in results:
        print("\nFindings:")
        print(json.dumps(results["synthesis"], indent=2))
        print(f"\nFull results saved to: {filename}")
    else:
        print("\nError:", results.get("error", "Unknown error"))

if __name__ == "__main__":
    main()