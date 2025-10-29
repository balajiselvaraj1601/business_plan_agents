"""
Test suite for src/tools/tool_collection.py

This module contains comprehensive unit tests for all functions defined in tool_collection.py.
Each function has exactly two test functions: one for expected behavior and one for edge cases.
Tests use mocking to isolate dependencies and ensure efficient, reliable testing.
"""

from unittest.mock import Mock, patch

from src.tools.tool_collection import search_knowledge, search_web


@patch("src.tools.tool_collection.llm_general")
@patch("src.tools.tool_collection.log_debug")
def test_search_knowledge_success(mock_log_debug, mock_llm):
    """Test successful knowledge search."""
    mock_response = Mock()
    mock_response.content = "Knowledge search result"
    mock_llm.invoke.return_value = mock_response

    # Test the tool by calling invoke method
    result = search_knowledge.invoke({"query": "test query"})

    assert result == "Knowledge search result"
    mock_llm.invoke.assert_called_once_with("test query")


@patch("src.tools.tool_collection.llm_general")
@patch("src.tools.tool_collection.log_debug")
def test_search_knowledge_empty_query(mock_log_debug, mock_llm):
    """Test knowledge search with empty query."""
    result = search_knowledge.invoke({"query": ""})

    assert result == "Error: Empty query"
    mock_llm.invoke.assert_not_called()


def test_search_web_success():
    """Test successful web search."""
    with (
        patch("src.tools.tool_collection.FLAG_DEBUG", False),
        patch("src.tools.tool_collection.tavily_tool") as mock_tavily,
    ):
        mock_tavily.invoke.return_value = "Web search result"

        result = search_web.invoke({"query": "test query"})

        assert result == "Web search result"
        mock_tavily.invoke.assert_called_once_with({"query": "test query"})


@patch("src.tools.tool_collection.tavily_tool")
@patch("src.tools.tool_collection.log_debug")
def test_search_web_empty_query(mock_log_debug, mock_tavily):
    """Test web search with empty query."""
    result = search_web.invoke({"query": ""})

    assert result == "Error: Empty query"
    mock_tavily.invoke.assert_not_called()
