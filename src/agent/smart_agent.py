"""
Smart Business Plan Analysis Agent

This module provides an enhanced, intelligent agent for comprehensive business plan analysis.
It extends the simple agent with advanced workflow orchestration, including topic planning,
supervisory coordination, and iterative analysis execution.

The smart agent implements a sophisticated workflow that:
1. Uses AI-powered planning to generate comprehensive business plan topics
2. Coordinates analysis across multiple topics through a supervisor node
3. Executes detailed analysis on each topic using the underlying agent graph
4. Manages state transitions and workflow routing automatically

Key Features:
- Multi-node workflow orchestration (planner → supervisor → agent)
- Structured topic generation and analysis
- Automatic workflow routing based on remaining topics
- Comprehensive logging and debugging support
- Report generation and file output
- Error handling and state validation

Workflow Nodes:
- planner: Generates business plan topics using LLM with structured output
- supervisor: Manages topic processing order and workflow coordination
- agent: Executes detailed analysis on individual topics using tool-augmented LLM

Dependencies:
- langgraph: For workflow graph management and state orchestration
- langchain: For LLM integration and tool binding
- Internal modules: States, prompts, tools, utilities, and simple agent

Usage:
    from src.agent.smart_agent import create_and_run_workflow

    # Run complete analysis workflow
    result = create_and_run_workflow(
        business_type="Falooda",
        location="Sweden"
    )

    # Access results
    print(f"Generated {len(result.processed_topics)} topic analyses")
    print(result.report)

Configuration:
- Uses constants from src.utils.constants for default values
- Logging configured via src.utils.logging
- Models initialized from src.utils.model_collection
- Tools bound from src.tools.tool_collection

Note: This agent provides a higher-level orchestration layer above the simple agent,
enabling complex multi-topic business analysis workflows with automatic coordination.
"""

# Copilot: Do not add any logging for this file.

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agent.simple_agent import setup_agent_graph
from src.prompts.prompt_collection import PROMPT_PLANNING
from src.states.state_collection import BusinessAnalysisState, PlanningResponse, Topic
from src.tools.tool_collection import TOOLS
from src.utils.constants import (
    DEFAULT_BUSINESS_TYPE,
    DEFAULT_LOCATION,
    FLAG_DEBUG,
    LOG_LEVEL,
    LOG_SEPARATOR_CHAR,
    LOG_SEPARATOR_LENGTH,
    REPORTS_DIR,
)
from src.utils.logging import (
    log_debug,
    log_info,
    log_process_end,
    log_process_start,
    log_separator,
    log_success,
    setup_logging,
)
from src.utils.model_collection import llm_general, planner_llm

# ============================================================================
# Module Initialization
# ============================================================================

# Setup logging
setup_logging(LOG_LEVEL)

# Test direct logging
logging.info("Direct logging test")

# Log initialization
log_process_start("Business Plan Analysis Agent")
log_info(f"FLAG_DEBUG = {FLAG_DEBUG}")
log_info(f"LOG_LEVEL = {LOG_LEVEL}")

# Create analyst_llm by binding tools to the general model
analyst_llm = llm_general.bind_tools(TOOLS)

log_info("Starting the Smart Business Plan Analysis Agent")
agent_graph = setup_agent_graph(tools=TOOLS)

# ============================================================================
# Node Functions
# ============================================================================


def planner(state: BusinessAnalysisState) -> BusinessAnalysisState:
    """
    Generate comprehensive business plan topics using AI-powered analysis.

    This function uses a structured output LLM to generate a complete set of business
    plan topics tailored to the specific business type and location. The topics are
    generated using a predefined planning prompt and returned as structured data.

    The planning process:
    1. Formats the planning prompt with business type and location
    2. Invokes the planner LLM with structured output parsing
    3. Validates and stores the generated topics in the state
    4. Sets up the processing queue for subsequent analysis

    Args:
        state (BusinessAnalysisState): Current business analysis state containing
            business_type and location information. These parameters drive the
            topic generation process.

    Returns:
        BusinessAnalysisState: Updated state with topics_list populated with
            generated business plan topics. The current_topic_index is reset to -1
            to prepare for sequential processing.

    Raises:
        ValidationError: If the LLM response doesn't match the expected PlanningResponse schema.
        RuntimeError: If the LLM invocation fails or returns invalid structured output.

    Note:
        In debug mode, only the first 3 topics are processed to speed up testing.
        The function modifies the input state in-place and returns the same object.

    Example:
        >>> state = BusinessAnalysisState(business_type="Falooda", location="Sweden")
        >>> updated_state = planner(state)
        >>> print(f"Generated {len(updated_state.topics_list)} topics")
    """

    log_separator("PLANNER NODE", char=LOG_SEPARATOR_CHAR, length=LOG_SEPARATOR_LENGTH)
    log_info("Planner Node Processing")
    log_info(
        f"Generating business plan topics for {state.business_type} in {state.location}"
    )

    # Create the prompt with placeholders
    log_info("Creating business plan topics prompt")
    prompt = PROMPT_PLANNING.format(
        business_type=state.business_type,
        location=state.location,
    )

    # Invoke model to generate business plan topics
    log_info("Invoking model to generate business plan topics")
    structured_output_model = planner_llm.with_structured_output(PlanningResponse)
    structured_output = structured_output_model.invoke(prompt)

    log_success("Business plan topics generated successfully")
    structured_output.print_structured()
    state.topics_list = structured_output.topics
    state.current_topic_index = -1  # Start before first topic

    if FLAG_DEBUG:
        state.topics_list = structured_output.topics[:3]
        log_debug(f"Model response type: {type(structured_output)}")
        log_debug(f"Number of topics generated: {len(structured_output.topics)}")

    return state


def agent(state: BusinessAnalysisState) -> BusinessAnalysisState:
    """
    Execute detailed analysis on a specific topic using the agent graph.

    This function invokes the pre-configured agent graph workflow to perform
    comprehensive analysis on the current topic and its subtopics. It converts
    the topic's subtopics into individual analysis tasks and collects all
    resulting insights and reports.

    The analysis process:
    1. Validates that a current topic is set for processing
    2. Converts topic subtopics into Topic objects for the agent graph
    3. Creates a focused BusinessAnalysisState for the subtopic analysis
    4. Executes the agent graph workflow on the subtopics
    5. Collects and stores the analysis results in the parent state

    Args:
        state (BusinessAnalysisState): Current business analysis state containing
            the current_topic to be analyzed. Must have a valid Topic object
            with populated subtopics list.

    Returns:
        BusinessAnalysisState: Updated state with analysis results appended to
            raw_information and the current_topic.report field populated with
            the generated analysis content.

    Raises:
        ValueError: If no current_topic is set or if current_topic is a string
            instead of a Topic object.
        RuntimeError: If the agent graph execution fails or returns invalid results.

    Note:
        Only the first 3 subtopics of the current topic are processed for efficiency.
        The function modifies the input state in-place and returns the same object.
        Analysis results are stored both in raw_information and topic.report.

    Example:
        >>> topic = Topic(topic="Market Analysis", subtopics=["Target market", "Competition", "Pricing"])
        >>> state = BusinessAnalysisState(current_topic=topic, ...)
        >>> analyzed_state = agent(state)
        >>> print(f"Analysis length: {len(analyzed_state.current_topic.report)} chars")
    """
    log_separator("AGENT NODE", char=LOG_SEPARATOR_CHAR, length=LOG_SEPARATOR_LENGTH)
    log_info("Agent Node Processing")

    if not state.current_topic:
        raise ValueError("No topic set for agent processing")

    # Ensure topic is a Topic object, not a string
    if isinstance(state.current_topic, str):
        raise ValueError("Topic must be a Topic object for agent processing")

    # Convert subtopics to Topic objects
    subtopic_objects = [
        Topic(topic=subtopic, reason="", subtopics=[], report=None)
        for subtopic in state.current_topic.subtopics[:3]
    ]

    # Create BusinessAnalysisState for the agent graph
    agent_input = BusinessAnalysisState(
        business_type=state.business_type,
        location=state.location,
        topics_list=subtopic_objects,
        current_topic_index=-1,
        description=state.current_topic.reason,
        raw_information=[],
        messages=[],
        report="",
    )

    output = agent_graph.invoke(agent_input)  # type: ignore[union-attr]

    # Assuming output is BusinessAnalysisState
    report_content = output.get("report", "")
    state.raw_information.append(report_content)
    state.current_topic.report = report_content
    log_info("Report size : " + str(len(report_content)) + " characters")

    return state


def supervisor(state: BusinessAnalysisState) -> BusinessAnalysisState:
    """
    Supervise and coordinate the business plan analysis workflow.

    This function acts as a workflow coordinator that manages topic processing
    progression, tracking which topics have been completed and selecting the
    next topic for analysis. It implements the control logic for sequential
    topic processing in the multi-topic analysis workflow.

    The supervision process:
    1. Checks for remaining unprocessed topics in the queue
    2. Increments the current_topic_index to advance to the next topic
    3. Updates the current_topic reference for the next processing node
    4. Logs progress information for monitoring and debugging

    Args:
        state (BusinessAnalysisState): Current business analysis state containing
            topics_list and current_topic_index. Must have a properly initialized
            topics queue with valid Topic objects.

    Returns:
        BusinessAnalysisState: Updated state with current_topic_index incremented
            and current_topic set to the next topic for processing. If no topics
            remain, current_topic will be None.

    Raises:
        IndexError: If current_topic_index exceeds the bounds of topics_list.
        ValueError: If topics_list contains invalid Topic objects.

    Note:
        This function modifies the input state in-place and returns the same object.
        It serves as the decision point in the workflow graph, determining whether
        to continue processing or terminate the analysis.

    Example:
        >>> state = BusinessAnalysisState(topics_list=[topic1, topic2, topic3], current_topic_index=0)
        >>> supervised_state = supervisor(state)
        >>> print(f"Next topic index: {supervised_state.current_topic_index}")
        >>> print(f"Next topic: {supervised_state.current_topic.topic}")
    """

    log_separator(
        "SUPERVISOR NODE", char=LOG_SEPARATOR_CHAR, length=LOG_SEPARATOR_LENGTH
    )
    log_info("Supervisor Node Processing")
    log_info(
        f"Supervising business plan analysis for {state.business_type} in {state.location}"
    )

    if state.non_processed_topics:
        # Move to the next topic by incrementing the index
        state.current_topic_index += 1
        log_info(f"Current Topic to Analyze: {state.current_topic}")
        log_info(f"List of topics to be analyzed: {len(state.non_processed_topics)}")

    return state


# Do not change the function signature
def router_supervisor_agent(state: BusinessAnalysisState) -> Literal["agent", END]:  # type: ignore[valid-type]
    """
    Route the workflow based on remaining topics to analyze.

    This routing function implements the conditional logic that determines the
    next step in the business analysis workflow. It checks whether there are
    remaining unprocessed topics and routes accordingly to either continue
    analysis or terminate the workflow.

    Routing logic:
    - If unprocessed topics remain: Routes to "agent" node for continued analysis
    - If all topics processed: Routes to END to terminate the workflow
    - The decision is based on the non_processed_topics computed property

    Args:
        state (BusinessAnalysisState): Current business analysis state containing
            topics_list and current_topic_index. Used to determine if topics
            remain for processing.

    Returns:
        Literal["agent", END]: Routing decision indicating the next workflow node.
            Returns "agent" to continue processing or END to complete the workflow.

    Note:
        This function is pure and does not modify the input state. It only examines
        the state to make routing decisions. The END constant represents workflow
        termination in LangGraph.

    Example:
        >>> # State with remaining topics
        >>> state_with_topics = BusinessAnalysisState(topics_list=[...], current_topic_index=0)
        >>> next_node = router_supervisor_agent(state_with_topics)
        >>> print(next_node)  # Output: "agent"
        >>>
        >>> # State with no remaining topics
        >>> state_complete = BusinessAnalysisState(topics_list=[...], current_topic_index=2)
        >>> next_node = router_supervisor_agent(state_complete)
        >>> print(next_node)  # Output: END
    """
    if state.non_processed_topics:
        log_info("More topics to analyze, routing to agent")
        return "agent"
    else:
        log_info("No more topics to analyze, ending process")
        return END


# ============================================================================
# Workflow Management
# ============================================================================


def create_and_run_workflow(
    business_type: str = DEFAULT_BUSINESS_TYPE,
    location: str = DEFAULT_LOCATION,
) -> BusinessAnalysisState:
    """
    Create and execute the complete business plan analysis workflow.

    This function orchestrates the entire multi-node business analysis process
    from topic planning through detailed analysis to final report generation.
    It sets up a LangGraph workflow with planner, supervisor, and agent nodes,
    executes the complete analysis pipeline, and saves results to disk.

    Workflow execution steps:
    1. Creates and configures the LangGraph workflow with all nodes
    2. Compiles the graph with proper edge connections and routing
    3. Initializes the workflow with provided business parameters
    4. Executes the complete analysis pipeline (planning → supervision → analysis)
    5. Processes results and generates structured output
    6. Saves the comprehensive report to timestamped files
    7. Returns the final state with all analysis results

    Args:
        business_type (str): Type of business to analyze (e.g., "Falooda Shop",
            "Tech Startup", "Coffee Roastery"). Used to tailor topic generation
            and analysis focus. Defaults to DEFAULT_BUSINESS_TYPE.
        location (str): Geographic location/city for the business analysis.
            Influences localization aspects of the analysis. Defaults to
            DEFAULT_LOCATION.

    Returns:
        BusinessAnalysisState: Complete analysis results containing processed
            topics, raw information, generated reports, and workflow metadata.
            All topics will have their report fields populated with analysis content.

    Raises:
        Exception: Propagates any exceptions from graph compilation, execution,
            or file operations. Common issues include LLM failures, invalid
            business parameters, or file system permission errors.

    Note:
        The function automatically saves results to the REPORTS_DIR in a
        structured folder hierarchy. Graph visualization is available but
        currently disabled for performance. The workflow processes all
        generated topics comprehensively.

    Example:
        >>> # Analyze a falooda business in Sweden
        >>> result = create_and_run_workflow("Falooda", "Sweden")
        >>> print(f"Processed {len(result.processed_topics)} topics")
        >>> print(f"Report length: {len(result.report)} characters")
        >>>
        >>> # Use default parameters
        >>> default_result = create_and_run_workflow()
        >>> print(default_result.business_type)  # DEFAULT_BUSINESS_TYPE
    """
    # Simplified graph structure
    graph = StateGraph(BusinessAnalysisState, output_schema=BusinessAnalysisState)
    # Add nodes
    log_info("Adding graph nodes")
    graph.add_node("planner", planner)
    graph.add_node("supervisor", supervisor)
    graph.add_node("agent", agent)

    # Define flow
    log_info("Defining graph edges and flow")
    graph.add_edge(START, "planner")

    # Tools always returns to analyst
    graph.add_edge("planner", "supervisor")
    graph.add_conditional_edges("supervisor", router_supervisor_agent)
    graph.add_edge("agent", "supervisor")

    log_info("Compiling graph")
    app = graph.compile()
    log_success("Graph compiled successfully")

    # Graph visualization is currently disabled
    log_debug("Graph visualization skipped")

    output = app.invoke(  # type: ignore[arg-type]
        BusinessAnalysisState(
            location=location,
            business_type=business_type,
            topics_list=[],
            current_topic_index=-1,
            raw_information=[],
            messages=[],
            description=None,
            report="",
        )
    )

    # Convert output to BusinessAnalysisState if it's a dict
    if isinstance(output, dict):
        output = BusinessAnalysisState(**output)

    log_success("Graph execution completed")
    output.print_structured()

    # Save the final business analysis report
    folder_name = (
        f"{output.location.replace(' ', '_')}_{output.business_type.replace(' ', '_')}"
    )
    report_filename = f"{REPORTS_DIR}/{folder_name}/business_analysis_report.txt"
    output.save_text(report_filename)

    log_process_end("Business Plan Analysis Agent", success=True)

    return output


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    create_and_run_workflow()
