from llama_index.llms.anthropic import Anthropic
from llama_index.core.prompts import PromptTemplate
from llama_index.core.agent.react import ReActAgent
from dotenv import load_dotenv
from app.agents.tools.tools import tools
# Load variables from .env into environment (primarily for Anthropic API key)
load_dotenv()

def initialize_agent():
    # Initialize the LLM
    llm = Anthropic(model="claude-3-haiku-20240307")
    # Define thes system prompt
    system_prompt_string = """\
    You are SariSariNews, a knowledgeable and up-to-date assistant 
    specializing in local news from the Philippines. Your goal is to provide 
    accurate, concise, and relevant information to users based on their queries.

        You have access to the following tools:
        {tool_desc}

    Rules for Using Tools:
    1. fetch_news Tool:
    - Use this tool to fetch news articles from valid sources.
    - You MUST ONLY use these valid sources:
        - GMA Network
        - TV5
        - Philippine Daily Inquirer
        - Manila Bulletin
        - ABS-CBN
        - Rappler
        - Philstar
        - Manila Times
        - BusinessWorld
        - The Daily Tribune
    - Always specify the source when fetching news.

        Use the following format:
        Thought: <your thought process>
        Action: <tool name>
        Action Input: <input>
        Observation: <tool output>

        Final Answer: <your final answer>
    """
    system_prompt = PromptTemplate(system_prompt_string)

        # Initialize the ReActAgent
    agent = ReActAgent.from_tools(
        tools=tools,
        llm=llm,
        verbose=True
    )

    # Customize the System Prompt
    agent.update_prompts({"agent_worker:system_prompt": system_prompt})

    return agent