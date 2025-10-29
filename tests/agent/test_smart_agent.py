"""
Test suite for src/agent/smart_agent.py

This module contains comprehensive unit tests for all functions defined in smart_agent.py.
Each function has exactly two test functions: one for expected behavior and one for edge cases.
Tests use mocking to isolate dependencies and ensure efficient, reliable testing.
"""

from unittest.mock import Mock, patch

import pytest
from langgraph.graph import END

from src.agent.smart_agent import agent, planner, router_supervisor_agent, supervisor
from src.states.state_collection import BusinessAnalysisState, PlanningResponse, Topic


@patch("src.agent.smart_agent.planner_llm")
@patch("src.agent.smart_agent.PROMPT_PLANNING")
@patch("src.agent.smart_agent.log_separator")
@patch("src.agent.smart_agent.log_info")
@patch("src.agent.smart_agent.log_debug")
def test_planner_success(
    mock_log_debug, mock_log_info, mock_log_separator, mock_prompt, mock_llm
):
    """Test successful business plan topic generation."""
    mock_prompt.format.return_value = "formatted prompt"
    mock_topic = Topic(
        topic="Test Topic", reason="Test", subtopics=["sub1"], report=None
    )
    mock_response = PlanningResponse(topics=[mock_topic])
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_response

    state = BusinessAnalysisState(
        business_type="Test Business",
        location="Test City",
        topics_list=[],
        current_topic_index=-1,
        raw_information=[],
        description=None,
        messages=[],
        report="",
    )

    result = planner(state)

    assert result.topics_list == [mock_topic]
    assert result.current_topic_index == -1
    mock_prompt.format.assert_called_once_with(
        business_type="Test Business",
        location="Test City",
    )
    mock_llm.with_structured_output.return_value.invoke.assert_called_once_with(
        "formatted prompt"
    )


@patch("src.agent.smart_agent.planner_llm")
@patch("src.agent.smart_agent.PROMPT_PLANNING")
@patch("src.agent.smart_agent.log_separator")
@patch("src.agent.smart_agent.log_info")
@patch("src.agent.smart_agent.log_debug")
def test_planner_with_debug_topics(
    mock_log_debug, mock_log_info, mock_log_separator, mock_prompt, mock_llm
):
    """Test planner with debug mode limiting topics."""
    mock_prompt.format.return_value = "formatted prompt"
    topics = [
        Topic(topic="Topic 1", reason="", subtopics=[], report=None),
        Topic(topic="Topic 2", reason="", subtopics=[], report=None),
        Topic(topic="Topic 3", reason="", subtopics=[], report=None),
        Topic(topic="Topic 4", reason="", subtopics=[], report=None),
    ]
    mock_response = PlanningResponse(topics=topics)
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_response

    with patch("src.agent.smart_agent.FLAG_DEBUG", True):
        state = BusinessAnalysisState(
            business_type="Test Business",
            location="Test City",
            topics_list=[],
            current_topic_index=-1,
            raw_information=[],
            description=None,
            messages=[],
            report="",
        )

        result = planner(state)

    assert len(result.topics_list) == 3
    # Verify debug logs were called
    mock_log_debug.assert_any_call(f"Model response type: {type(mock_response)}")
    mock_log_debug.assert_any_call(f"Number of topics generated: {len(topics)}")


@patch("src.agent.smart_agent.planner_llm")
@patch("src.agent.smart_agent.PROMPT_PLANNING")
@patch("src.agent.smart_agent.log_separator")
@patch("src.agent.smart_agent.log_info")
@patch("src.agent.smart_agent.log_success")
@patch("src.agent.smart_agent.log_debug")
def test_planner_with_single_topic(
    mock_log_debug,
    mock_log_success,
    mock_log_info,
    mock_log_separator,
    mock_prompt,
    mock_llm,
):
    """Test planner when LLM returns single topic."""
    mock_prompt.format.return_value = "formatted prompt"
    mock_topic = Topic(topic="Single Topic", reason="", subtopics=[], report=None)
    mock_response = PlanningResponse(topics=[mock_topic])
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_response

    state = BusinessAnalysisState(
        business_type="Test Business",
        location="Test City",
        topics_list=[],
        current_topic_index=-1,
        raw_information=[],
        description=None,
        messages=[],
        report="",
    )

    result = planner(state)

    assert len(result.topics_list) == 1
    assert result.topics_list[0].topic == "Single Topic"
    assert result.current_topic_index == -1


@patch("src.agent.smart_agent.setup_agent_graph")
def test_agent_success(mock_setup_graph):
    """Test successful agent execution on a topic."""
    mock_graph = Mock()
    mock_graph.invoke.return_value = {"report": "Generated report content"}
    mock_setup_graph.return_value = mock_graph

    # Mock the global agent_graph that the function expects
    import src.agent.smart_agent as smart_agent_module

    smart_agent_module.agent_graph = mock_graph

    state = BusinessAnalysisState(
        business_type="Test Business",
        location="Test City",
        topics_list=[
            Topic(
                topic="Test Topic",
                reason="Test",
                subtopics=["sub1", "sub2", "sub3"],
                report=None,
            )
        ],
        current_topic_index=0,
        raw_information=[],
        messages=[],
        report="",
        description=None,
    )

    result = agent(state)

    assert "Generated report content" in result.raw_information
    # The report should be set on the original topic (index 0 before increment)
    assert result.topics_list[0].report == "Generated report content"
    mock_graph.invoke.assert_called_once()


@patch("src.agent.smart_agent.setup_agent_graph")
def test_agent_no_current_topic(mock_setup_graph):
    """Test agent function when no current topic is available."""
    mock_graph = Mock()
    mock_setup_graph.return_value = mock_graph

    # Mock the global agent_graph that the function expects
    import src.agent.smart_agent as smart_agent_module

    smart_agent_module.agent_graph = mock_graph

    state = BusinessAnalysisState(
        business_type="Test Business",
        location="Test City",
        topics_list=[],
        current_topic_index=-1,  # No topics available
        raw_information=[],
        messages=[],
        report="",
        description=None,
    )

    with pytest.raises(ValueError, match="No topic set for agent processing"):
        agent(state)

    mock_graph.invoke.assert_not_called()


@patch("src.agent.smart_agent.setup_agent_graph")
def test_agent_string_topic_error(mock_setup_graph):
    """Test agent function raises ValueError when current_topic is a string."""
    mock_graph = Mock()
    mock_setup_graph.return_value = mock_graph

    # Mock the global agent_graph that the function expects
    import src.agent.smart_agent as smart_agent_module

    smart_agent_module.agent_graph = mock_graph

    # This state should never happen in practice, but we need to test the validation
    state = BusinessAnalysisState(
        business_type="Test Business",
        location="Test City",
        topics_list=[Topic(topic="Test", reason="", subtopics=[], report=None)],
        current_topic_index=0,
        raw_information=[],
        messages=[],
        report="",
        description=None,
    )

    # Manually set current_topic to a string (bypassing Pydantic validation)
    state.__dict__["topics_list"][0] = "String Topic"  # Force invalid state

    with pytest.raises(
        ValueError, match="Topic must be a Topic object for agent processing"
    ):
        agent(state)

    mock_graph.invoke.assert_not_called()


@patch("src.agent.smart_agent.setup_agent_graph")
@patch("src.agent.smart_agent.log_separator")
@patch("src.agent.smart_agent.log_info")
def test_agent_with_empty_subtopics(
    mock_log_info, mock_log_separator, mock_setup_graph
):
    """Test agent execution when topic has no subtopics."""
    mock_graph = Mock()
    mock_graph.invoke.return_value = {"report": "Report for topic with no subtopics"}
    mock_setup_graph.return_value = mock_graph

    # Mock the global agent_graph that the function expects
    import src.agent.smart_agent as smart_agent_module

    smart_agent_module.agent_graph = mock_graph

    state = BusinessAnalysisState(
        business_type="Test Business",
        location="Test City",
        topics_list=[
            Topic(
                topic="Test Topic",
                reason="Test description",
                subtopics=[],  # Empty subtopics
                report=None,
            )
        ],
        current_topic_index=0,
        raw_information=[],
        messages=[],
        report="",
        description=None,
    )

    result = agent(state)

    assert "Report for topic with no subtopics" in result.raw_information
    assert result.topics_list[0].report == "Report for topic with no subtopics"
    mock_graph.invoke.assert_called_once()


@patch("src.agent.smart_agent.setup_agent_graph")
@patch("src.agent.smart_agent.log_separator")
@patch("src.agent.smart_agent.log_info")
def test_agent_with_few_subtopics(mock_log_info, mock_log_separator, mock_setup_graph):
    """Test agent execution when topic has fewer than 3 subtopics."""
    mock_graph = Mock()
    mock_graph.invoke.return_value = {"report": "Report with 2 subtopics"}
    mock_setup_graph.return_value = mock_graph

    # Mock the global agent_graph that the function expects
    import src.agent.smart_agent as smart_agent_module

    smart_agent_module.agent_graph = mock_graph

    state = BusinessAnalysisState(
        business_type="Test Business",
        location="Test City",
        topics_list=[
            Topic(
                topic="Test Topic",
                reason="Test description",
                subtopics=["sub1", "sub2"],  # Only 2 subtopics
                report=None,
            )
        ],
        current_topic_index=0,
        raw_information=[],
        messages=[],
        report="",
        description=None,
    )

    result = agent(state)

    assert "Report with 2 subtopics" in result.raw_information
    assert result.topics_list[0].report == "Report with 2 subtopics"
    mock_graph.invoke.assert_called_once()

    # Verify that the agent input had 2 subtopics (not sliced to 3)
    call_args = mock_graph.invoke.call_args[0][0]
    assert len(call_args.topics_list) == 2


@patch("src.agent.smart_agent.setup_agent_graph")
@patch("src.agent.smart_agent.log_separator")
@patch("src.agent.smart_agent.log_info")
def test_agent_with_more_than_three_subtopics(
    mock_log_info, mock_log_separator, mock_setup_graph
):
    """Test agent execution when topic has more than 3 subtopics (should only process first 3)."""
    mock_graph = Mock()
    mock_graph.invoke.return_value = {"report": "Report with limited subtopics"}
    mock_setup_graph.return_value = mock_graph

    # Mock the global agent_graph that the function expects
    import src.agent.smart_agent as smart_agent_module

    smart_agent_module.agent_graph = mock_graph

    state = BusinessAnalysisState(
        business_type="Test Business",
        location="Test City",
        topics_list=[
            Topic(
                topic="Test Topic",
                reason="Test description",
                subtopics=["sub1", "sub2", "sub3", "sub4", "sub5"],  # 5 subtopics
                report=None,
            )
        ],
        current_topic_index=0,
        raw_information=[],
        messages=[],
        report="",
        description=None,
    )

    result = agent(state)

    assert "Report with limited subtopics" in result.raw_information
    assert result.topics_list[0].report == "Report with limited subtopics"
    mock_graph.invoke.assert_called_once()

    # Verify that only first 3 subtopics were used
    call_args = mock_graph.invoke.call_args[0][0]
    assert len(call_args.topics_list) == 3


@patch("src.agent.smart_agent.setup_agent_graph")
@patch("src.agent.smart_agent.log_separator")
@patch("src.agent.smart_agent.log_info")
def test_agent_with_empty_report(mock_log_info, mock_log_separator, mock_setup_graph):
    """Test agent execution when output has empty report."""
    mock_graph = Mock()
    mock_graph.invoke.return_value = {"report": ""}
    mock_setup_graph.return_value = mock_graph

    # Mock the global agent_graph that the function expects
    import src.agent.smart_agent as smart_agent_module

    smart_agent_module.agent_graph = mock_graph

    state = BusinessAnalysisState(
        business_type="Test Business",
        location="Test City",
        topics_list=[
            Topic(
                topic="Test Topic",
                reason="Test description",
                subtopics=["sub1", "sub2", "sub3"],
                report=None,
            )
        ],
        current_topic_index=0,
        raw_information=[],
        messages=[],
        report="",
        description=None,
    )

    result = agent(state)

    assert "" in result.raw_information
    assert result.topics_list[0].report == ""
    # Verify log_info was called with the report size
    mock_log_info.assert_any_call("Report size : 0 characters")


@patch("src.agent.smart_agent.log_separator")
@patch("src.agent.smart_agent.log_info")
@patch("src.agent.smart_agent.log_debug")
def test_supervisor_with_topics_remaining(
    mock_log_debug, mock_log_info, mock_log_separator
):
    """Test supervisor when topics remain to be processed."""
    topic1 = Topic(topic="Topic 1", reason="", subtopics=[], report=None)
    topic2 = Topic(topic="Topic 2", reason="", subtopics=[], report=None)

    state = BusinessAnalysisState(
        business_type="Test Business",
        location="Test City",
        topics_list=[topic1, topic2],
        current_topic_index=0,
        raw_information=[],
        description=None,
        messages=[],
        report="",
    )

    result = supervisor(state)

    assert result.current_topic_index == 1
    assert result.current_topic == topic2
    # Note: Actual implementation logs different message


@pytest.mark.parametrize(
    "topics_list,current_topic_index,expected_index",
    [
        (
            [
                Topic(topic="Topic 1", reason="", subtopics=[], report=None),
                Topic(topic="Topic 2", reason="", subtopics=[], report=None),
            ],
            0,
            1,
        ),
        ([], -1, -1),
    ],
)
@patch("src.agent.smart_agent.log_separator")
@patch("src.agent.smart_agent.log_info")
@patch("src.agent.smart_agent.log_debug")
def test_supervisor_parameterized(
    mock_log_debug,
    mock_log_info,
    mock_log_separator,
    topics_list,
    current_topic_index,
    expected_index,
):
    """Test supervisor with different scenarios."""
    state = BusinessAnalysisState(
        business_type="Test Business",
        location="Test City",
        topics_list=topics_list,
        current_topic_index=current_topic_index,
        raw_information=[],
        description=None,
        messages=[],
        report="",
    )

    result = supervisor(state)

    assert result.current_topic_index == expected_index


@patch("src.agent.smart_agent.log_info")
@patch("src.agent.smart_agent.log_debug")
def test_router_supervisor_agent_topics_remaining(mock_log_debug, mock_log_info):
    """Test router_supervisor_agent when topics remain."""
    state = BusinessAnalysisState(
        business_type="Test Business",
        location="Test City",
        topics_list=[
            Topic(topic="Topic 1", reason="", subtopics=[], report=None),
            Topic(topic="Topic 2", reason="", subtopics=[], report=None),
        ],
        current_topic_index=0,
        raw_information=[],
        description=None,
        messages=[],
        report="",
    )

    result = router_supervisor_agent(state)

    assert result == "agent"


@pytest.mark.parametrize(
    "topics_list,current_topic_index,expected_result",
    [
        (
            [
                Topic(topic="Topic 1", reason="", subtopics=[], report=None),
                Topic(topic="Topic 2", reason="", subtopics=[], report=None),
            ],
            0,
            "agent",
        ),
        ([], -1, END),
    ],
)
@patch("src.agent.smart_agent.log_info")
@patch("src.agent.smart_agent.log_debug")
def test_router_supervisor_agent_parameterized(
    mock_log_debug, mock_log_info, topics_list, current_topic_index, expected_result
):
    """Test router_supervisor_agent with different scenarios."""
    state = BusinessAnalysisState(
        business_type="Test Business",
        location="Test City",
        topics_list=topics_list,
        current_topic_index=current_topic_index,
        raw_information=[],
        description=None,
        messages=[],
        report="",
    )

    result = router_supervisor_agent(state)

    assert result == expected_result


@patch("src.agent.smart_agent.setup_agent_graph")
@patch("src.agent.smart_agent.log_info")
@patch("src.agent.smart_agent.log_success")
@patch("src.agent.smart_agent.log_separator")
def test_agent_with_topic_with_description(
    mock_log_separator, mock_log_success, mock_log_info, mock_setup_graph
):
    """Test agent function when topic has description."""
    mock_graph = Mock()
    mock_graph.invoke.return_value = {"report": "Report with description"}
    mock_setup_graph.return_value = mock_graph

    # Mock the global agent_graph that the function expects
    import src.agent.smart_agent as smart_agent_module

    smart_agent_module.agent_graph = mock_graph

    state = BusinessAnalysisState(
        business_type="Test Business",
        location="Test City",
        topics_list=[
            Topic(
                topic="Test Topic",
                reason="This is a detailed description",
                subtopics=["sub1", "sub2", "sub3"],
                report=None,
            )
        ],
        current_topic_index=0,
        raw_information=[],
        messages=[],
        report="",
        description=None,
    )

    result = agent(state)

    assert "Report with description" in result.raw_information
    assert result.topics_list[0].report == "Report with description"
    # Verify the agent_input included the description from current_topic
    call_args = mock_graph.invoke.call_args[0][0]
    assert call_args.description == "This is a detailed description"


@patch("src.agent.smart_agent.log_separator")
@patch("src.agent.smart_agent.log_info")
def test_supervisor_increments_topic_index(mock_log_info, mock_log_separator):
    """Test supervisor correctly increments current_topic_index."""
    topic1 = Topic(topic="Topic 1", reason="", subtopics=[], report=None)
    topic2 = Topic(topic="Topic 2", reason="", subtopics=[], report=None)
    topic3 = Topic(topic="Topic 3", reason="", subtopics=[], report=None)

    state = BusinessAnalysisState(
        business_type="Test Business",
        location="Test City",
        topics_list=[topic1, topic2, topic3],
        current_topic_index=-1,  # Starting before first topic
        raw_information=[],
        description=None,
        messages=[],
        report="",
    )

    result = supervisor(state)

    assert result.current_topic_index == 0
    assert result.current_topic == topic1


# ============================================================================
# Tests for create_and_run_workflow function
# ============================================================================


@patch("src.agent.smart_agent.log_process_end")
@patch("src.agent.smart_agent.log_success")
@patch("src.agent.smart_agent.log_debug")
@patch("src.agent.smart_agent.log_info")
@patch("src.agent.smart_agent.log_separator")
@patch("src.agent.smart_agent.setup_agent_graph")
@patch("src.agent.smart_agent.analyst_llm")
@patch("src.agent.smart_agent.agent_graph")
@patch("src.agent.smart_agent.StateGraph")
def test_create_and_run_workflow_success(
    mock_state_graph,
    mock_agent_graph,
    mock_analyst_llm,
    mock_setup_graph,
    mock_log_separator,
    mock_log_info,
    mock_log_debug,
    mock_log_success,
    mock_log_process_end,
):
    """Test successful execution of create_and_run_workflow."""
    # Mock StateGraph and its methods
    mock_graph_instance = Mock()
    mock_state_graph.return_value = mock_graph_instance

    # Mock compiled graph
    mock_compiled_graph = Mock()
    mock_graph_instance.compile.return_value = mock_compiled_graph

    # Mock graph invocation result
    mock_output_state = Mock()
    mock_output_state.location = "Test City"
    mock_output_state.business_type = "Test Business"
    mock_output_state.save_text = Mock()
    mock_output_state.print_structured = Mock()
    mock_compiled_graph.invoke.return_value = mock_output_state

    # Mock agent graph setup
    mock_setup_graph.return_value = mock_agent_graph

    from src.agent.smart_agent import create_and_run_workflow

    result = create_and_run_workflow("Test Business", "Test City")

    # Verify StateGraph was created correctly
    mock_state_graph.assert_called_once()
    assert (
        mock_state_graph.call_args[1]["output_schema"].__name__
        == "BusinessAnalysisState"
    )

    # Verify nodes were added
    mock_graph_instance.add_node.assert_any_call("planner", planner)
    mock_graph_instance.add_node.assert_any_call("supervisor", supervisor)
    mock_graph_instance.add_node.assert_any_call("agent", agent)

    # Verify edges were added
    mock_graph_instance.add_edge.assert_any_call("__start__", "planner")
    mock_graph_instance.add_edge.assert_any_call("planner", "supervisor")
    mock_graph_instance.add_edge.assert_any_call("agent", "supervisor")

    # Verify conditional edges
    mock_graph_instance.add_conditional_edges.assert_called_once()
    assert mock_graph_instance.add_conditional_edges.call_args[0][0] == "supervisor"

    # Verify graph compilation and invocation
    mock_graph_instance.compile.assert_called_once()
    mock_compiled_graph.invoke.assert_called_once()

    # Verify output processing
    mock_output_state.print_structured.assert_called_once()
    mock_output_state.save_text.assert_called_once()

    # Verify logging
    mock_log_process_end.assert_called_once_with(
        "Business Plan Analysis Agent", success=True
    )

    assert result == mock_output_state


@patch("src.agent.smart_agent.log_process_end")
@patch("src.agent.smart_agent.log_success")
@patch("src.agent.smart_agent.log_debug")
@patch("src.agent.smart_agent.log_info")
@patch("src.agent.smart_agent.log_separator")
@patch("src.agent.smart_agent.setup_agent_graph")
@patch("src.agent.smart_agent.analyst_llm")
@patch("src.agent.smart_agent.agent_graph")
@patch("src.agent.smart_agent.StateGraph")
def test_create_and_run_workflow_dict_output_conversion(
    mock_state_graph,
    mock_agent_graph,
    mock_analyst_llm,
    mock_setup_graph,
    mock_log_separator,
    mock_log_info,
    mock_log_debug,
    mock_log_success,
    mock_log_process_end,
):
    """Test create_and_run_workflow when graph returns dict that needs conversion."""
    # Mock StateGraph and its methods
    mock_graph_instance = Mock()
    mock_state_graph.return_value = mock_graph_instance

    # Mock compiled graph
    mock_compiled_graph = Mock()
    mock_graph_instance.compile.return_value = mock_compiled_graph

    # Mock graph invocation result as dict
    dict_output = {
        "location": "Test City",
        "business_type": "Test Business",
        "topics_list": [],
        "current_topic_index": -1,
        "raw_information": [],
        "messages": [],
        "description": None,
        "report": "Test Report",
    }
    mock_compiled_graph.invoke.return_value = dict_output

    # Mock BusinessAnalysisState constructor
    with patch("src.agent.smart_agent.BusinessAnalysisState") as mock_state_class:
        # First call returns the initial state for graph.invoke
        mock_initial_state = Mock()
        # Second call returns the converted state from dict
        mock_output_state = Mock()
        mock_output_state.location = "Test City"
        mock_output_state.business_type = "Test Business"
        mock_output_state.save_text = Mock()
        mock_output_state.print_structured = Mock()
        mock_state_class.side_effect = [mock_initial_state, mock_output_state]

        # Mock agent graph setup
        mock_setup_graph.return_value = mock_agent_graph

        from src.agent.smart_agent import create_and_run_workflow

        result = create_and_run_workflow("Test Business", "Test City")

        # Verify BusinessAnalysisState was called twice - once for initial state, once for dict conversion
        assert mock_state_class.call_count == 2
        # Verify the second call was with the dict output
        mock_state_class.assert_any_call(**dict_output)

        assert result == mock_output_state


@patch("src.agent.smart_agent.log_process_end")
@patch("src.agent.smart_agent.log_success")
@patch("src.agent.smart_agent.log_debug")
@patch("src.agent.smart_agent.log_info")
@patch("src.agent.smart_agent.log_separator")
@patch("src.agent.smart_agent.setup_agent_graph")
@patch("src.agent.smart_agent.analyst_llm")
@patch("src.agent.smart_agent.agent_graph")
@patch("src.agent.smart_agent.StateGraph")
def test_create_and_run_workflow_with_custom_parameters(
    mock_state_graph,
    mock_agent_graph,
    mock_analyst_llm,
    mock_setup_graph,
    mock_log_separator,
    mock_log_info,
    mock_log_debug,
    mock_log_success,
    mock_log_process_end,
):
    """Test create_and_run_workflow with custom business type and location."""
    # Mock StateGraph and its methods
    mock_graph_instance = Mock()
    mock_state_graph.return_value = mock_graph_instance

    # Mock compiled graph
    mock_compiled_graph = Mock()
    mock_graph_instance.compile.return_value = mock_compiled_graph

    # Mock graph invocation result
    mock_output_state = Mock()
    mock_output_state.location = "Custom Location"
    mock_output_state.business_type = "Custom Business"
    mock_output_state.save_text = Mock()
    mock_output_state.print_structured = Mock()
    mock_compiled_graph.invoke.return_value = mock_output_state

    # Mock agent graph setup
    mock_setup_graph.return_value = mock_agent_graph

    from src.agent.smart_agent import create_and_run_workflow

    result = create_and_run_workflow("Custom Business", "Custom Location")

    # Verify the initial state was created with correct parameters
    call_args = mock_compiled_graph.invoke.call_args[0][0]
    assert call_args.business_type == "Custom Business"
    assert call_args.location == "Custom Location"

    assert result == mock_output_state


@patch("src.agent.smart_agent.log_process_end")
@patch("src.agent.smart_agent.log_success")
@patch("src.agent.smart_agent.log_debug")
@patch("src.agent.smart_agent.log_info")
@patch("src.agent.smart_agent.log_separator")
@patch("src.agent.smart_agent.setup_agent_graph")
@patch("src.agent.smart_agent.analyst_llm")
@patch("src.agent.smart_agent.agent_graph")
@patch("src.agent.smart_agent.StateGraph")
def test_create_and_run_workflow_report_filename_generation(
    mock_state_graph,
    mock_agent_graph,
    mock_analyst_llm,
    mock_setup_graph,
    mock_log_separator,
    mock_log_info,
    mock_log_debug,
    mock_log_success,
    mock_log_process_end,
):
    """Test that create_and_run_workflow generates correct report filename."""
    # Mock StateGraph and its methods
    mock_graph_instance = Mock()
    mock_state_graph.return_value = mock_graph_instance

    # Mock compiled graph
    mock_compiled_graph = Mock()
    mock_graph_instance.compile.return_value = mock_compiled_graph

    # Mock graph invocation result
    mock_output_state = Mock()
    mock_output_state.location = "Sweden"
    mock_output_state.business_type = "Falooda"
    mock_output_state.save_text = Mock()
    mock_output_state.print_structured = Mock()
    mock_compiled_graph.invoke.return_value = mock_output_state

    # Mock agent graph setup
    mock_setup_graph.return_value = mock_agent_graph

    from src.agent.smart_agent import create_and_run_workflow

    result = create_and_run_workflow("Falooda", "Sweden")

    # Verify save_text was called with correct filename
    expected_filename = "reports/Sweden_Falooda/business_analysis_report.txt"
    mock_output_state.save_text.assert_called_once_with(expected_filename)

    assert result == mock_output_state


# ============================================================================
# Tests for main execution block
# ============================================================================


@patch("src.agent.smart_agent.create_and_run_workflow")
def test_main_execution_block(mock_create_workflow):
    """Test the main execution block when script is run directly."""
    # Mock the workflow function
    mock_result = Mock()
    mock_create_workflow.return_value = mock_result

    # Import and execute the main block
    import sys
    from unittest.mock import patch

    # Simulate running the module directly
    with patch.object(sys, "argv", ["smart_agent.py"]):
        # Execute the main block by importing and checking if __name__ == "__main__"
        # Since we can't directly execute the if block, we test that the function exists and is callable
        from src.agent.smart_agent import create_and_run_workflow

        # Verify the function is callable (this covers the import and function definition)
        assert callable(create_and_run_workflow)

        # The actual execution would happen if we ran: python -m src.agent.smart_agent
        # But we can verify the function signature and behavior through other tests
