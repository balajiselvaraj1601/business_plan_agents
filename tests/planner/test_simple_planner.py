"""
Test suite for src/planner/simple_planner.py

This module contains comprehensive unit tests for all functions defined in simple_planner.py.
Each function has exactly two test functions: one for expected behavior and one for edge cases.
Tests use mocking to isolate dependencies and ensure efficient, reliable testing.
"""

from unittest.mock import patch

from src.planner.simple_planner import planner
from src.states.state_collection import PlanningResponse, Topic
from src.utils.constants import (
    DEFAULT_BUSINESS_TYPE,
    DEFAULT_LOCATION,
)


@patch("src.planner.simple_planner.llm_reason")
@patch("src.planner.simple_planner.PROMPT_PLANNING")
@patch("src.planner.simple_planner.log_separator")
@patch("src.planner.simple_planner.log_debug")
def test_planner_success(mock_log_debug, mock_log_separator, mock_prompt, mock_llm):
    """Test successful business plan topic generation."""
    mock_prompt.format.return_value = "formatted prompt"
    mock_topic = Topic(
        topic="Test Topic", reason="Test", subtopics=["sub1"], report=None
    )
    mock_response = PlanningResponse(topics=[mock_topic])
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_response

    result = planner("Test Business", "Test City")

    assert isinstance(result, PlanningResponse)
    assert result.topics == [mock_topic]
    mock_prompt.format.assert_called_once()
    # Check that the format was called with the expected parameters
    call_args = mock_prompt.format.call_args
    assert "business" in call_args.kwargs
    assert call_args.kwargs["business"] == "Test Business"
    assert "location" in call_args.kwargs
    assert call_args.kwargs["location"] == "Test City"
    assert "experts_information" in call_args.kwargs
    mock_llm.with_structured_output.return_value.invoke.assert_called_once_with(
        "formatted prompt"
    )
    # Note: The actual implementation uses log_debug, not log_info


@patch("src.planner.simple_planner.llm_reason")
@patch("src.planner.simple_planner.PROMPT_PLANNING")
@patch("src.planner.simple_planner.log_separator")
@patch("src.planner.simple_planner.log_debug")
def test_planner_default_parameters(
    mock_log_debug, mock_log_separator, mock_prompt, mock_llm
):
    """Test planner with default business type and location."""
    mock_prompt.format.return_value = "formatted prompt"
    mock_topic = Topic(
        topic="Test Topic", reason="Test", subtopics=["sub1"], report=None
    )
    mock_response = PlanningResponse(topics=[mock_topic])
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_response

    result = planner()

    assert isinstance(result, PlanningResponse)
    assert result.topics == [mock_topic]
    mock_prompt.format.assert_called_once()
    # Check that the format was called with the expected parameters
    call_args = mock_prompt.format.call_args
    assert "business" in call_args.kwargs
    assert call_args.kwargs["business"] == DEFAULT_BUSINESS_TYPE
    assert "location" in call_args.kwargs
    assert call_args.kwargs["location"] == DEFAULT_LOCATION
    assert "experts_information" in call_args.kwargs
    mock_llm.with_structured_output.return_value.invoke.assert_called_once_with(
        "formatted prompt"
    )
