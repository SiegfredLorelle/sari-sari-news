from typing import List
import requests
import feedparser
from llama_index.core.tools import FunctionTool

def fetch_news(source: str, keyword: str = None) -> List:
    """
    Fetches recent news articles from a specific source using RSS feeds.
    
    Args:
        source (str): Must be one of: "GMA", "Philippine Daily Inquirer", 
                     "Manila Bulletin", "ABS-CBN", "Rappler", "Philstar", 
                     "Manila Times", "BusinessWorld", "The Daily Tribune"
        keyword (str, optional): Filter articles by keyword. Defaults to None.

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
        print(url)
        feed = feedparser.parse(response.content)
        entries = feed.entries
        if keyword:
            keyword = keyword.lower()
            entries = [
                entry for entry in entries 
                if keyword in entry.title.lower() or 
                keyword in entry.summary.lower()
            ]
        return entries
    else:
        return []


news_tool = FunctionTool.from_defaults(fn=fetch_news)

print(fetch_news("The Daily Tribune", "gma")) # Uncomment to test the tool manually