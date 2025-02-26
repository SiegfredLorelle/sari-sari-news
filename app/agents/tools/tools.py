from app.agents.tools.news.fetch_news import get_latest_specific_news_tool, search_latest_news_tool, get_latest_general_news_tool
from app.agents.tools.news.url_based import url_based_qa_tool
tools = [
    get_latest_specific_news_tool,
    search_latest_news_tool,
    get_latest_general_news_tool,
    url_based_qa_tool,
]