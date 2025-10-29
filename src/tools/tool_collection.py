"""
Tool Collection Module

This module provides search tools for business plan analysis.
Tools are decorated with @tool to integrate with LangChain agents.

Tools:
- search_knowledge: Query the LLM knowledge base for business information
- search_web: Query Tavily search API for current web information

Usage:
    from src.tools.tool_collection import TOOLS, search_knowledge, search_web

    # Use in agent
    analyst_llm = llm_general.bind_tools(TOOLS)
"""

# Copilot: Do not add any logging for this file.

import json

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from src.utils.constants import (
    DUMMY_DATA_RESPONSE,
    FLAG_DEBUG,
    FLAG_USE_DUMMY_DATA,
    META_DATA_FILE_PATH,
    TAVILY_API_KEY_NAME,
    TAVILY_MAX_RESULTS,
    TAVILY_TOPIC,
)
from src.utils.logging import log_debug
from src.utils.model_collection import llm_general

# Load config
with open(META_DATA_FILE_PATH, "r") as f:
    config = json.load(f)

# Initialize tools
tavily_tool = TavilySearch(
    max_results=TAVILY_MAX_RESULTS,
    topic=TAVILY_TOPIC,
    tavily_api_key=config[TAVILY_API_KEY_NAME],
)


@tool
def search_knowledge(query: str) -> str:
    """
    Search knowledge base for business information using the LLM.

    Args:
        query: The search query string

    Returns:
        str: Search results or error message
    """
    log_debug(f"search_knowledge called with query: {query[:100]}...")

    if not query.strip():
        log_debug("search_knowledge: Empty query provided")
        return "Error: Empty query"

    if FLAG_DEBUG and FLAG_USE_DUMMY_DATA:
        log_debug("search_knowledge: Returning dummy data")
        return DUMMY_DATA_RESPONSE

    try:
        log_debug("search_knowledge: Invoking LLM")
        result = llm_general.invoke(query)
        log_debug(
            f"search_knowledge: Got result of length {len(result.content) if result.content else 0}"
        )
        return result.content if result.content else "No results"
    except Exception as e:
        log_debug(f"search_knowledge: Error occurred - {str(e)}")
        return f"Error: {e}"


@tool
def search_web(query: str) -> str:
    """
    Search web for current information using Tavily API.

    Args:
        query: The search query string

    Returns:
        str: Search results or error message
    """
    log_debug(f"search_web called with query: {query[:100]}...")

    if not query.strip():
        log_debug("search_web: Empty query provided")
        return "Error: Empty query"

    if FLAG_DEBUG:
        log_debug("search_web: Returning dummy data (DEBUG mode)")
        return DUMMY_DATA_RESPONSE

    try:
        log_debug("search_web: Invoking Tavily search")
        result = tavily_tool.invoke({"query": query})
        log_debug(
            f"search_web: Got result of length {len(str(result)) if result else 0}"
        )
        return str(result) if result else "No results"
    except Exception as e:
        log_debug(f"search_web: Error occurred - {str(e)}")
        return f"Error: {e}"


if FLAG_DEBUG:
    TOOLS = [search_knowledge]
else:
    TOOLS = [search_knowledge, search_web]
