from typing import List, Dict, Tuple
from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
import chromadb
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# Load variables from .env into environment (primarily for Anthropic API key)
load_dotenv()

def fetch_article_content(url: str) -> str:
    """
    Fetches and parses the content of an article from a URL using Playwright.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")
        article_content = ""
        for paragraph in soup.find_all("p"):
            article_content += paragraph.get_text() + "\n"
        
        return article_content.strip()
    except Exception as e:
        print(f"Error fetching article: {str(e)}")
        return None

def answer_based_on_article(query: str, url: str) -> str:
    """
    Answers a user's question based on the content of a provided article URL.
    """
    # Fetch the article content
    article_content = fetch_article_content(url)
    if not article_content:
        return "Failed to fetch the article content. Please check the URL and try again."
    
    # Initialize Chroma client
    chroma_client = chromadb.PersistentClient(path="db/chroma-articles")
    chroma_collection = chroma_client.get_or_create_collection("articles")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Set the embedding model
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize the LLM (Anthropic in this example - you'll need an API key for this too)
    llm = Anthropic(model="claude-3-haiku-20240307")
    
    # Create a document from the article content
    document = Document(text=article_content)
    
    # Create Settings for this session
    Settings.embed_model = embed_model
    Settings.llm = llm  # Set the LLM for this session
    
    # Create a VectorStoreIndex
    index = VectorStoreIndex.from_documents(
        [document],
        vector_store=vector_store,
    )
    
    # Query the index with explicit LLM
    query_engine = index.as_query_engine(llm=llm)  # Explicitly pass the LLM
    response = query_engine.query(query)
    
    return str(response)

# Create the tool
url_based_qa_tool = FunctionTool.from_defaults(
    fn=answer_based_on_article,
    name="url_based_qa",
    description="""
    Use this tool when the user provides a URL and asks a question based on the article.
    Example: "Based on [https://example.com/article], which areas were affected by the typhoon?"
    """
)

# Example usage
if __name__ == "__main__":
    query = "Give me a TLDR of the following"
    url = "https://www.rappler.com/sports/gilas-pilipinas/tim-cone-sticking-with-same-lineup-despite-recent-losses"
    answer = answer_based_on_article(query, url)
    print(answer)