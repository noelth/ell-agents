import os
import ell
from dotenv import load_dotenv

# Import provider libraries
import anthropic           # For Anthropic's models
from openai import OpenAI  # Used for both OpenAI and Ollama

# Import the research tool from the WebScanner folder.
# Adjust the import path as needed. Here we assume WebScanner is a folder
# at the same level as this script.
from WebScanner.research_tool import WebResearchTool

# Load environment variables from .env file
load_dotenv()

# API keys and base URL (only keys go in .env)
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OLLAMA_API_KEY    = os.getenv("OLLAMA_API_KEY")  # May be empty if not used
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL")

# --- Create Client Instances ---
openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.openai.com/v1") if OPENAI_API_KEY else None
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
ollama_client = OpenAI(api_key=OLLAMA_API_KEY, base_url=OLLAMA_BASE_URL)

# Register models with ell
ell.config.register_model("deepseek-r1:1.5b", ollama_client)
ell.config.register_model("claude-3-5-haiku-latest", anthropic_client)
ell.config.register_model("llama3.2:3b", ollama_client)

# Map provider names to their client objects.
CLIENTS = {
    "openai": openai_client,
    "anthropic": anthropic_client,
    "ollama": ollama_client,
}

# Enable verbose logging (optional)
ell.config.verbose = True

# --- Define the First Agent (Topic Analysis) ---
@ell.simple(model="llama3.2:3b", client=CLIENTS["ollama"], max_tokens=1000)
def analyze_topic(topic: str) -> str:
    """
    You are a blog strategist. Use a system prompt and a user prompt to provide a concise analysis.
    """
    system_prompt = "You are a blog strategist. Analyze the topic and return key insights."
    user_prompt = f"Please analyze the topic: '{topic}'."
    return [ell.system(system_prompt), ell.user(user_prompt)]

# --- Define the Storyteller Agent (Write Blog) ---
@ell.complex(model="llama3.2:3b", client=CLIENTS["ollama"], temperature=0.5, max_tokens=1500)
def write_blog(topic: str, context: str) -> str:
    """
    You are an experienced storyteller and blog writer. Using the provided topic and context,
    generate an engaging and comprehensive blog post.
    Construct your response using multiple message formats.
    """
    system_prompt = (
        "You are an engaging blog writer. Your writing is vivid, descriptive, and informative. "
        "Craft a blog post that is both creative and well-structured."
    )
    user_prompt = (
        f"Write a blog post about the topic '{topic}'. Use the following context to inform your response:\n{context}\n"
        "Ensure your response is coherent, detailed, and flows naturally."
    )
    return [ell.system(system_prompt), ell.user(user_prompt)]

# --- Workflow Orchestration (Terminal-Based) ---
def build_blog_post():
    # Prompt the user for a blog topic.
    topic = input("Enter the blog topic: ").strip()
    
    print("\nAnalyzing topic...")
    analysis_output = analyze_topic(topic)
    if isinstance(analysis_output, list):
        analysis_text = " ".join(str(msg) for msg in analysis_output)
    else:
        analysis_text = str(analysis_output)
    print("Topic Analysis:", analysis_text)
    
    # Ask if a web search is needed.
    search_choice = input("Perform web search for current information? (Y/N) [Default N]: ").strip().upper()
    research_context = ""
    if search_choice == "Y":
        # Prompt for filtering and number of URLs
        filter_choice = input("Filter URLs? (Y/N) [Default N]: ").strip().upper()
        apply_filter = filter_choice == "Y"
        num_urls_input = input("Enter number of URLs to check [Default 4]: ").strip()
        max_urls = int(num_urls_input) if num_urls_input.isdigit() else 4
        
        # Initialize the research tool from the WebScanner folder.
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        searxng_base_url = os.getenv("SEARXNG_BASE_URL")
        researcher = WebResearchTool(
            anthropic_api_key=anthropic_api_key,
            searxng_base_url=searxng_base_url,
            max_urls=max_urls
        )
        print("\nResearching... This may take a few minutes.")
        research_results = researcher.research(topic, apply_filter=apply_filter)
        
        # Check if synthesis is available, and extract the "answer" field.
        if "synthesis" in research_results and "answer" in research_results["synthesis"]:
            research_context = research_results["synthesis"]["answer"]
            print("\nResearch Findings (Answer):")
            print(research_context)
        else:
            print("\nNo synthesis available from research; proceeding without additional context.")
    
    # Combine initial analysis with research context
    combined_context = analysis_text
    if research_context:
        combined_context += "\nAdditional Current Information:\n" + research_context
    
    print("\nWriting blog post...")
    blog_post = write_blog(topic, combined_context)
    print("\nBlog Post:\n", blog_post.content)

if __name__ == "__main__":
    ell.init(verbose=True, autocommit=True)
    build_blog_post()   