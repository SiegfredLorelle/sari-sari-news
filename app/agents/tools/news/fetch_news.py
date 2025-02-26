from typing import List
import requests
import feedparser
from llama_index.core.tools import FunctionTool
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def fetch_rss_news(source: str) -> List:
    """
    Fetches recent news articles from a specific source using RSS feeds.

    Usage:
    - Use only when user explicitly mentions a specific source
    - Example: "News from GMA about politics"
    
    Args:
        source (str): Must be one of: "GMA", "Philippine Daily Inquirer", 
                     "Manila Bulletin", "ABS-CBN", "Rappler", "Philstar", 
                     "Manila Times", "BusinessWorld", "The Daily Tribune"

    Returns:
        List of recent news articles from the specified source.
    """
    rss_urls = {
        "GMA": "https://data.gmanetwork.com/gno/rss/news/feed.xml",
        "Daily Inquirer": "https://www.inquirer.net/fullfeed",
        "Manila Bulletin": "https://mb.com.ph/rss/news",
        "ABS-CBN": "https://news.abs-cbn.com/feed/",
        "Rappler": "https://www.rappler.com/feed/",
        "Philstar": "https://www.philstar.com/rss/headlines",
        "Manila Times": "https://www.manilatimes.net/news/feed/",
        "BusinessWorld": "https://www.bworldonline.com/feed",
        "The Daily Tribune": "https://www.tribune.net.ph/feed",
    }

    if source not in rss_urls:
        return []

    url = rss_urls[source]
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        # Parse RSS feed
        feed = feedparser.parse(response.content)
        return feed.entries
    else:
        return []


model = SentenceTransformer('all-MiniLM-L6-v2')
def fetch_news_with_embeddings(source: str, query: str = None, similarity_threshold: float = 0.3) -> List:
    """
    Fetches news articles from a source and filters them using semantic similarity.
    
    Args:
        source (str): News source (e.g., "GMA").
        query (str, optional): Query to match against articles.
        similarity_threshold (float): Minimum similarity score (0-1). Default: 0.3.
    
    Returns:
        List of articles sorted by relevance to the query.
    """
    # Fetch articles from RSS
    articles = fetch_rss_news(source)
    if not query or not articles:
        return articles
    
    # Generate embeddings for query and articles
    query_embedding = model.encode([query], convert_to_tensor=True)
    article_texts = [f"{article.title} {article.summary}" for article in articles]
    article_embeddings = model.encode(article_texts, convert_to_tensor=True)
    
    # Calculate similarity scores
    similarities = cosine_similarity(query_embedding.cpu(), article_embeddings.cpu())
    similarities = similarities.flatten()
    
    # Filter and sort articles by similarity
    filtered_articles = []
    for idx, article in enumerate(articles):
        if similarities[idx] >= similarity_threshold:
            article.similarity = similarities[idx]  # Attach similarity score
            filtered_articles.append(article)
    
    # Sort by descending similarity
    filtered_articles.sort(key=lambda x: x.similarity, reverse=True)
    return filtered_articles

def fetch_all_news(query: str, similarity_threshold: float = 0.25) -> List:
    """
    Fetches news from ALL valid sources and filters by semantic similarity.
    
    Args:
        query (str, optional): General query to match against all articles.
        similarity_threshold (float): Lower threshold for broader queries.
    
    Returns:
        List of articles from all sources, sorted by relevance.
    """
    all_articles = []
    sources = [
        "GMA", "Daily Inquirer", "Manila Bulletin", 
        "ABS-CBN", "Rappler", "Philstar", 
        "Manila Times", "BusinessWorld", "The Daily Tribune"
    ]
    
    for source in sources:
        if query:
            articles = fetch_news_with_embeddings(source, query, similarity_threshold)
        else:
            articles = fetch_rss_news(source)
        all_articles.extend(articles)
    
    # Deduplicate articles
    seen = set()
    unique_articles = []
    for article in all_articles:
        identifier = f"{article.title}-{article.link}"
        if identifier not in seen:
            seen.add(identifier)
            unique_articles.append(article)
    
    return unique_articles

get_latest_specific_news_tool = FunctionTool.from_defaults(
    fn=fetch_rss_news,
    name="GetLatestSpecificNews",
    description="Useful for getting news on explicitly mentioned specific source"
    )
search_latest_news_tool = FunctionTool.from_defaults(
    fn=fetch_news_with_embeddings,
    name="SearchLatestNews",
    description="Useful for searching for relatively new specific news"
    )
get_latest_general_news_tool = FunctionTool.from_defaults(
    fn=fetch_all_news,
    name="GetLatestGeneralNews",
    description="Useful for broad requests without specific sources"
    )
# print(fetch_rss_news("The Daily Tribune", "gma")) # Uncomment to test the tool manually
# print(fetch_news_with_embeddings("The Daily Tribune", "Japan")) # Uncomment to test the tool manually