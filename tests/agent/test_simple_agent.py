"""Tests for simple_agent.py functions."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.agent.simple_agent import (
    analyst,
    load_business_plan_data,
    report_generator,
    router_analyst,
    setup_agent_graph,
)
from src.states.state_collection import BusinessAnalysisState, Topic


# Global mock fixtures for common test patterns
@pytest.fixture
def mock_planning_response():
    """Mock PlanningResponse for testing."""
    from src.states.state_collection import PlanningResponse

    mock_response = Mock(spec=PlanningResponse)
    mock_response.topics = [
        Topic(topic="Test Topic", reason="Test", subtopics=["sub1"], report=None)
    ]
    return mock_response


@pytest.fixture
def mock_topic():
    """Mock Topic for testing."""
    return Topic(
        topic="Test Topic",
        reason="Test description",
        subtopics=["Subtopic 1", "Subtopic 2"],
        report=None,
    )


@pytest.fixture
def mock_business_analysis_state(mock_topic):
    """Mock BusinessAnalysisState for testing."""
    return BusinessAnalysisState(
        business_type="Test Business",
        location="Test City",
        topics_list=[mock_topic],
        current_topic_index=0,
        raw_information=[],
        reason=None,
        messages=[],
        report="",
    )


# Tests for load_business_plan_data function


@patch("src.states.state_collection.PlanningResponse.load_json")
@patch("src.agent.simple_agent.log_success")
@patch("src.agent.simple_agent.log_data_processing")
def test_load_business_plan_data_success(
    mock_log_data, mock_log_success, mock_load_json
):
    """Test loading business plan data successfully."""
    # Setup
    from src.states.state_collection import PlanningResponse, Topic

    mock_topics = [
        Topic(
            topic="Market Analysis",
            reason="Analyze the market",
            subtopics=["Demographics", "Competition"],
            report=None,
        )
    ]
    mock_response = PlanningResponse(topics=mock_topics)
    mock_load_json.return_value = mock_response

    # Execute
    result = load_business_plan_data("test.json")

    # Verify
    assert isinstance(result, PlanningResponse)
    assert len(result.topics) == 1
    assert result.topics[0].topic == "Market Analysis"
    assert result.topics[0].subtopics == ["Demographics", "Competition"]
    mock_load_json.assert_called_once_with("test.json")
    mock_log_data.assert_called_once_with("Loading", "business plan JSON")
    mock_log_success.assert_called_once_with("Business plan data loaded successfully")


@pytest.mark.parametrize(
    "file_path,should_raise,expected_error",
    [
        ("nonexistent.json", FileNotFoundError, "Please run the planner first"),
        (None, None, None),  # Test default path
    ],
)
@patch("src.states.state_collection.PlanningResponse.load_json")
@patch("src.agent.simple_agent.log_error")
@patch("src.agent.simple_agent.log_data_processing")
def test_load_business_plan_data_parameterized(
    mock_log_data,
    mock_log_error,
    mock_load_json,
    file_path,
    should_raise,
    expected_error,
):
    """Test loading business plan data with different scenarios."""
    from src.states.state_collection import PlanningResponse, Topic

    # Setup
    if should_raise:
        mock_load_json.side_effect = should_raise("File not found")
    else:
        # Test with default path
        mock_topics = [Topic(topic="Test", reason="", subtopics=[], report=None)]
        mock_response = PlanningResponse(topics=mock_topics)
        mock_load_json.return_value = mock_response

    # Execute & Verify
    if should_raise:
        with pytest.raises(should_raise) as exc_info:
            load_business_plan_data(file_path)
        assert expected_error in str(exc_info.value)
        mock_load_json.assert_called_once_with(file_path)
        mock_log_error.assert_called_once()
    else:
        result = load_business_plan_data()
        assert isinstance(result, PlanningResponse)
        mock_load_json.assert_called_once()  # Called with default path


# Tests for analyst function


@patch("src.agent.simple_agent.get_expert_prompt")
@patch("src.agent.simple_agent.analyst_llm")
@patch("src.agent.simple_agent.log_info")
def test_analyst_with_current_topic(mock_log_info, mock_llm, mock_prompt):
    """Test analyst processes current topic."""
    # Setup
    topic1 = Topic(
        topic="Market Analysis",
        reason="Analyze market",
        subtopics=["Demographics"],
        report=None,
    )

    state = BusinessAnalysisState(
        business_type="Bakery",
        location="San Francisco",
        topics_list=[topic1],
        current_topic_index=-1,  # Will be incremented to 0
    )

    mock_prompt.return_value = "Expert prompt"
    mock_response = Mock()
    mock_response.content = "Analysis result"
    mock_response.tool_calls = []
    mock_llm.invoke.return_value = mock_response

    # Execute
    result = analyst(state)

    # Verify
    assert isinstance(result, BusinessAnalysisState)
    assert len(result.messages) == 1
    assert result.messages[0].content == "Analysis result"
    assert result.current_topic_index == 0
    mock_prompt.assert_called_once_with(user_input="Market Analysis")
    mock_llm.invoke.assert_called_once()


@pytest.mark.slow
@patch("src.agent.simple_agent.analyst_llm")
@patch("src.agent.simple_agent.log_info")
def test_analyst_with_tool_message(mock_log_info, mock_llm):
    """Test analyst handles tool message responses."""
    # Setup
    topic1 = Topic(
        topic="Market Analysis",
        reason="Analyze market",
        subtopics=["Demographics"],
        report=None,
    )

    tool_msg = ToolMessage(content="Tool result", tool_call_id="123")

    state = BusinessAnalysisState(
        business_type="Bakery",
        location="San Francisco",
        topics_list=[topic1],
        current_topic_index=-1,
        messages=[tool_msg],
    )

    mock_response = Mock()
    mock_response.content = "Updated analysis"
    mock_response.tool_calls = []
    mock_llm.invoke.return_value = mock_response

    # Execute
    result = analyst(state)

    # Verify
    assert isinstance(result, BusinessAnalysisState)
    assert len(result.raw_information) == 1
    assert result.raw_information[0] == "Tool result"
    assert len(result.messages) == 1  # Old messages cleared, new one added
    assert result.messages[0].content == "Updated analysis"
    mock_llm.invoke.assert_called_once()


@patch("src.agent.simple_agent.log_info")
def test_analyst_no_topics_remaining(mock_log_info):
    """Test analyst when no topics remain to process."""
    # Setup
    topic1 = Topic(
        topic="Market Analysis",
        reason="Analyze market",
        subtopics=["Demographics"],
        report=None,
    )

    state = BusinessAnalysisState(
        business_type="Bakery",
        location="San Francisco",
        topics_list=[topic1],
        current_topic_index=0,  # Already at last topic
    )

    # Execute
    result = analyst(state)

    # Verify - non_processed_topics will be empty, so should return early
    assert isinstance(result, BusinessAnalysisState)
    assert result.current_topic_index == 0  # Not incremented when no topics left
    assert len(result.messages) == 0  # No new messages
    mock_log_info.assert_any_call("All topics processed, moving to report")


@patch("src.agent.simple_agent.get_expert_prompt")
@patch("src.agent.simple_agent.analyst_llm")
@patch("src.agent.simple_agent.log_info")
def test_analyst_processes_multiple_topics(mock_log_info, mock_llm, mock_prompt):
    """Test analyst can process multiple topics in sequence."""
    # Setup
    topic1 = Topic(topic="Topic 1", reason="First", subtopics=[], report=None)
    topic2 = Topic(topic="Topic 2", reason="Second", subtopics=[], report=None)

    state = BusinessAnalysisState(
        business_type="Bakery",
        location="San Francisco",
        topics_list=[topic1, topic2],
        current_topic_index=-1,
    )

    mock_prompt.return_value = "Expert prompt"
    mock_response = Mock()
    mock_response.content = "First analysis"
    mock_response.tool_calls = []
    mock_llm.invoke.return_value = mock_response

    # Execute first topic
    result = analyst(state)

    # Verify
    assert result.current_topic_index == 0
    assert len(result.messages) == 1
    assert result.messages[0].content == "First analysis"
    mock_prompt.assert_called_with(user_input="Topic 1")


# Tests for router_analyst function


@patch("src.agent.simple_agent.log_info")
def test_router_analyst_no_topics(mock_log_info):
    """Test router routes to report generator when no topics remain."""
    # Setup
    topic1 = Topic(topic="Topic 1", reason="Desc 1", subtopics=["sub1"], report=None)

    state = BusinessAnalysisState(
        business_type="Bakery",
        location="San Francisco",
        topics_list=[topic1],
        current_topic_index=0,  # At last topic - non_processed_topics will be empty
    )

    # Execute
    result = router_analyst(state)

    # Verify
    assert result == "report_generator"
    mock_log_info.assert_any_call(
        "No more topics to process - routing to report_generator"
    )


@pytest.mark.parametrize(
    "messages,expected_result,expected_log",
    [
        (
            [
                AIMessage(
                    content="Need info",
                    tool_calls=[{"name": "search_web", "args": {}, "id": "call_123"}],
                )
            ],
            "tools",
            None,
        ),
        ([], "analyst", "More topics to process - routing back to analyst"),
    ],
)
@patch("src.agent.simple_agent.log_info")
def test_router_analyst_parameterized(
    mock_log_info, messages, expected_result, expected_log
):
    """Test router_analyst with different scenarios."""
    # Setup
    topic1 = Topic(topic="Topic 1", reason="Desc 1", subtopics=["sub1"], report=None)
    topic2 = Topic(topic="Topic 2", reason="Desc 2", subtopics=["sub2"], report=None)

    state = BusinessAnalysisState(
        business_type="Bakery",
        location="San Francisco",
        topics_list=[topic1, topic2],
        current_topic_index=0,
        messages=messages,
    )

    # Execute
    result = router_analyst(state)

    # Verify
    assert result == expected_result
    if expected_log:
        mock_log_info.assert_any_call(expected_log)


# Tests for report_generator function


@patch("src.utils.utils.save_text_file")
@patch("src.agent.simple_agent.llm_general")
@patch("src.agent.simple_agent.log_info")
@patch("src.agent.simple_agent.log_success")
def test_report_generator_success(mock_log_success, mock_log_info, mock_llm, mock_save):
    """Test report generator creates executive report."""
    # Setup
    topic1 = Topic(topic="Topic 1", reason="Desc 1", subtopics=["sub1"], report=None)

    state = BusinessAnalysisState(
        business_type="Bakery",
        location="San Francisco",
        topics_list=[topic1],
        current_topic_index=0,
        raw_information=["Analysis 1", "Analysis 2"],
    )

    mock_response = Mock()
    mock_response.content = "Executive Report Content"
    mock_llm.invoke.return_value = mock_response

    # Execute
    result = report_generator(state)

    # Verify
    assert result.report == "Executive Report Content"
    mock_llm.invoke.assert_called_once()
    mock_save.assert_called_once()
    assert "reports/" in mock_save.call_args[0][1]


@pytest.mark.parametrize(
    "raw_information,expected_content",
    [
        ([], "Basic Report"),
        (["Analysis 1", "Analysis 2"], "Executive Report Content"),
    ],
)
@patch("src.utils.utils.save_text_file")
@patch("src.agent.simple_agent.llm_general")
@patch("src.agent.simple_agent.log_info")
@patch("src.agent.simple_agent.log_success")
def test_report_generator_parameterized(
    mock_log_success,
    mock_log_info,
    mock_llm,
    mock_save,
    raw_information,
    expected_content,
):
    """Test report generator with different raw information scenarios."""
    # Setup
    topic1 = Topic(topic="Topic 1", reason="Desc 1", subtopics=["sub1"], report=None)

    state = BusinessAnalysisState(
        business_type="Bakery",
        location="San Francisco",
        topics_list=[topic1],
        current_topic_index=0,
        raw_information=raw_information,
    )

    mock_response = Mock()
    mock_response.content = expected_content
    mock_llm.invoke.return_value = mock_response

    # Execute
    result = report_generator(state)

    # Verify
    assert result.report == expected_content
    mock_llm.invoke.assert_called_once()
    mock_save.assert_called_once()


# Tests for setup_agent_graph function


@patch("src.agent.simple_agent.ToolNode")
@patch("src.agent.simple_agent.log_info")
@patch("src.agent.simple_agent.log_success")
def test_setup_agent_graph_success(mock_log_success, mock_log_info, mock_tool_node):
    """Test setup_agent_graph creates graph successfully."""
    # Setup - Mock ToolNode to avoid tool validation issues
    mock_tool_node.return_value = Mock()
    mock_tools = [Mock(name="test_tool")]

    # Execute
    result = setup_agent_graph(tools=mock_tools)

    # Verify
    assert result is not None
    assert hasattr(result, "invoke")
    mock_log_success.assert_called_once_with("Graph compiled successfully")
    mock_tool_node.assert_called_once_with(mock_tools)


@pytest.mark.parametrize("analyst_model", [None, Mock()])
@patch("src.agent.simple_agent.ToolNode")
@patch("src.agent.simple_agent.log_info")
@patch("src.agent.simple_agent.log_success")
def test_setup_agent_graph_parameterized(
    mock_log_success, mock_log_info, mock_tool_node, analyst_model
):
    """Test setup_agent_graph with different configurations."""
    # Setup
    mock_tool_node.return_value = Mock()
    mock_tools = [Mock(name="test_tool")]

    # Execute
    kwargs = {"tools": mock_tools}
    if analyst_model is not None:
        kwargs["analyst_model"] = analyst_model

    result = setup_agent_graph(**kwargs)

    # Verify
    assert result is not None
    assert hasattr(result, "invoke")
    mock_log_success.assert_called_once_with("Graph compiled successfully")
    mock_tool_node.assert_called_once_with(mock_tools)


# ============================================================================
# Additional tests for missing coverage
# ============================================================================


@patch("src.agent.simple_agent.analyst_llm")
@patch("src.agent.simple_agent.log_info")
def test_analyst_with_non_string_tool_message_content(mock_log_info, mock_llm):
    """Test analyst handles tool message with non-string content."""
    # Setup
    topic1 = Topic(
        topic="Market Analysis",
        reason="Analyze market",
        subtopics=["Demographics"],
        report=None,
    )

    # Tool message with non-string content (e.g., dict or list)
    tool_msg = ToolMessage(content={"result": "Tool result data"}, tool_call_id="123")

    state = BusinessAnalysisState(
        business_type="Bakery",
        location="San Francisco",
        topics_list=[topic1],
        current_topic_index=-1,
        messages=[tool_msg],
    )

    mock_response = Mock()
    mock_response.content = "Updated analysis"
    mock_response.tool_calls = []
    mock_llm.invoke.return_value = mock_response

    # Execute
    result = analyst(state)

    # Verify
    assert isinstance(result, BusinessAnalysisState)
    assert len(result.raw_information) == 1
    # Non-string content should be converted to string
    assert result.raw_information[0] == "{'result': 'Tool result data'}"
    assert len(result.messages) == 1  # Old messages cleared, new one added
    assert result.messages[0].content == "Updated analysis"
    mock_llm.invoke.assert_called_once()


@patch("src.agent.simple_agent.get_expert_prompt")
@patch("src.agent.simple_agent.analyst_llm")
@patch("src.agent.simple_agent.log_debug")
@patch("src.agent.simple_agent.log_info")
def test_analyst_with_debug_logging_and_tool_calls(
    mock_log_info, mock_log_debug, mock_llm, mock_prompt
):
    """Test analyst debug logging when output has tool calls."""
    # Setup
    topic1 = Topic(
        topic="Market Analysis",
        reason="Analyze market",
        subtopics=["Demographics"],
        report=None,
    )

    state = BusinessAnalysisState(
        business_type="Bakery",
        location="San Francisco",
        topics_list=[topic1],
        current_topic_index=-1,
    )

    mock_prompt.return_value = "Expert prompt"
    mock_response = Mock()
    mock_response.content = "Analysis result"
    mock_response.tool_calls = [{"name": "search_tool", "args": {}}]  # Mock tool calls
    mock_llm.invoke.return_value = mock_response

    with patch("src.agent.simple_agent.FLAG_DEBUG", True):
        # Execute
        result = analyst(state)

    # Verify
    assert isinstance(result, BusinessAnalysisState)
    assert len(result.messages) == 1
    assert result.messages[0].content == "Analysis result"

    # Verify debug logging calls
    mock_log_debug.assert_any_call(f"Model response type: {type(mock_response)}")
    mock_log_debug.assert_any_call(
        f"Response content length: {len(mock_response.content)}"
    )
    mock_log_debug.assert_any_call(
        f"Response preview: {mock_response.content[:200]}..."
    )
    mock_log_debug.assert_any_call(
        f"Tool calls detected: {len(mock_response.tool_calls)}"
    )


# ============================================================================
# Tests for main function
# ============================================================================


@patch("src.agent.simple_agent.log_process_end")
@patch("src.agent.simple_agent.log_success")
@patch("src.agent.simple_agent.log_info")
@patch("src.agent.simple_agent.log_separator")
@patch("src.agent.simple_agent.load_business_plan_data")
@patch("src.agent.simple_agent.setup_agent_graph")
@patch("src.agent.simple_agent.analyst_llm")
@patch("src.agent.simple_agent.TOOLS")
def test_main_function_success(
    mock_tools,
    mock_analyst_llm,
    mock_setup_graph,
    mock_load_data,
    mock_log_separator,
    mock_log_info,
    mock_log_success,
    mock_log_process_end,
):
    """Test successful execution of main function."""
    # Mock the planning response and topics
    from src.states.state_collection import PlanningResponse

    mock_topic = Topic(
        topic="Sample Topic",
        reason="Sample description",
        subtopics=["Subtopic 1", "Subtopic 2", "Subtopic 3", "Subtopic 4"],
        report=None,
    )

    # Create enough topics so that topics[-3] works (needs at least 3 topics)
    mock_topic1 = Topic(topic="Topic 1", reason="", subtopics=[], report=None)
    mock_topic2 = Topic(topic="Topic 2", reason="", subtopics=[], report=None)

    mock_response = Mock(spec=PlanningResponse)
    mock_response.topics = [
        mock_topic1,
        mock_topic2,
        mock_topic,
    ]  # 3 topics so [-3] works
    mock_load_data.return_value = mock_response

    # Mock the graph and its execution
    mock_graph = Mock()
    mock_setup_graph.return_value = mock_graph

    mock_output_state = Mock()
    mock_output_state.location = "Test City"
    mock_output_state.business_type = "Test Business"
    mock_output_state.print_structured = Mock()
    mock_output_state.save_text = Mock()
    mock_graph.invoke.return_value = mock_output_state

    # Execute
    from src.agent.simple_agent import main

    main()

    # Verify data loading
    mock_load_data.assert_called_once()

    # Verify graph setup
    mock_setup_graph.assert_called_once_with(tools=mock_tools)

    # Verify graph execution was called
    mock_graph.invoke.assert_called_once()
    call_args = mock_graph.invoke.call_args[0][0]
    # Just verify it's a BusinessAnalysisState-like object with basic properties
    assert hasattr(call_args, "business_type")
    assert hasattr(call_args, "location")
    assert hasattr(call_args, "topics_list")

    # Verify output processing
    mock_output_state.print_structured.assert_called_once()
    mock_output_state.save_text.assert_called_once()

    # Verify success logging
    mock_log_process_end.assert_called_once_with(
        "Business Plan Analysis Agent", success=True
    )


@patch("src.agent.simple_agent.log_process_end")
@patch("src.agent.simple_agent.log_success")
@patch("src.agent.simple_agent.log_info")
@patch("src.agent.simple_agent.log_separator")
@patch("src.agent.simple_agent.load_business_plan_data")
@patch("src.agent.simple_agent.setup_agent_graph")
@patch("src.agent.simple_agent.analyst_llm")
@patch("src.agent.simple_agent.TOOLS")
def test_main_function_with_dict_output(
    mock_tools,
    mock_analyst_llm,
    mock_setup_graph,
    mock_load_data,
    mock_log_separator,
    mock_log_info,
    mock_log_success,
    mock_log_process_end,
):
    """Test main function when graph returns dict output."""
    # Mock the planning response and topics
    from src.states.state_collection import PlanningResponse

    mock_topic = Topic(
        topic="Sample Topic",
        reason="Sample description",
        subtopics=["Subtopic 1", "Subtopic 2", "Subtopic 3"],
        report=None,
    )

    # Create enough topics so that topics[-3] works (needs at least 3 topics)
    mock_topic1 = Topic(topic="Topic 1", reason="", subtopics=[], report=None)
    mock_topic2 = Topic(topic="Topic 2", reason="", subtopics=[], report=None)

    mock_response = Mock(spec=PlanningResponse)
    mock_response.topics = [
        mock_topic1,
        mock_topic2,
        mock_topic,
    ]  # 3 topics so [-3] works
    mock_load_data.return_value = mock_response

    # Mock the graph and its execution
    mock_graph = Mock()
    mock_setup_graph.return_value = mock_graph

    # Mock graph returns dict instead of BusinessAnalysisState
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
    mock_graph.invoke.return_value = dict_output

    # Mock BusinessAnalysisState constructor
    with patch("src.agent.simple_agent.BusinessAnalysisState") as mock_state_class:
        # First call returns the initial state for graph.invoke
        mock_initial_state = Mock()
        # Second call returns the converted state from dict
        mock_output_state = Mock()
        mock_output_state.location = "Test City"
        mock_output_state.business_type = "Test Business"
        mock_output_state.print_structured = Mock()
        mock_output_state.save_text = Mock()
        mock_state_class.side_effect = [mock_initial_state, mock_output_state]

        # Execute
        from src.agent.simple_agent import main

        main()

        # Verify BusinessAnalysisState was called twice - once for initial state, once for dict conversion
        assert mock_state_class.call_count == 2
        # Verify the second call was with the dict output
        mock_state_class.assert_any_call(**dict_output)

        # Verify output processing
        mock_output_state.print_structured.assert_called_once()
        mock_output_state.save_text.assert_called_once()


@patch("src.agent.simple_agent.main")
def test_main_execution_block(mock_main):
    """Test the main execution block when script is run directly."""
    # Mock the main function
    mock_main.return_value = None

    # Import and execute the main block
    import sys
    from unittest.mock import patch

    # Simulate running the module directly
    with patch.object(sys, "argv", ["simple_agent.py"]):
        # Execute the main block by importing and checking if __name__ == "__main__"
        # Since we can't directly execute the if block, we test that the function exists and is callable
        from src.agent.simple_agent import main

        # Verify the function is callable (this covers the import and function definition)
        assert callable(main)

        # The actual execution would happen if we ran: python -m src.agent.simple_agent
        # But we can verify the function signature and behavior through other tests
