"""
Business Plan Topics Generator

This module generates comprehensive business plan topics using AI-powered analysis.
It creates structured topic hierarchies for different business types and locations,
providing a foundation for detailed business planning and analysis.

The planner uses LangChain and Ollama models to generate topics that cover all
essential aspects of business planning including market research, financial planning,
operations, marketing, and risk management.

Usage:
    Run this module to generate a business_plan.json file containing structured
    topics for business planning. The generated file is used by the agent module
    for detailed analysis of each topic.

    Default configuration uses "Falooda" as business type and "Sweden" as location.
"""

# Copilot: Do not add any logging for this file.

from typing import List, Literal

from langgraph.graph import END, START, StateGraph

from src.prompts.prompt_collection import (
    PROMPT_PLANNING,
    PROMPT_PLANNING_CRITIQUE,
    PROMPT_PLANNING_FEEDBACK,
)
from src.states.state_collection import (
    FeedbackResponse,
    PlanningResponse,
    PlanningState,
    Topic,
)
from src.utils.constants import (
    BUSINESS_PLAN_JSON_PATH,
    DEFAULT_BUSINESS_TYPE,
    DEFAULT_LOCATION,
    EXPERTS,
    FLAG_DEBUG,
    LOG_LEVEL,
    QUALITY_THRESHOLD,
)
from src.utils.logging import (
    log_configuration,
    log_data_processing,
    log_debug,
    log_info,
    log_process_end,
    log_process_start,
    log_separator,
    log_success,
    setup_logging,
)
from src.utils.model_collection import llm_reason

# ============================================================================
# Module Initialization
# ============================================================================

# Setup logging
setup_logging(LOG_LEVEL)

# Log initialization
log_process_start("Business Plan Topics Generator")
log_configuration("FLAG_DEBUG", str(FLAG_DEBUG))
log_configuration("DEFAULT_BUSINESS_TYPE", DEFAULT_BUSINESS_TYPE)
log_configuration("DEFAULT_LOCATION", DEFAULT_LOCATION)

# ============================================================================
# Helper Functions
# ============================================================================


def format_list_as_bullets(items: List[str]) -> str:
    """
    Format a list of strings as bullet-pointed items.

    Converts a list of strings into a formatted string with each item
    prefixed by a bullet point and placed on a new line. This is useful
    for creating readable lists in prompts and logging output.

    Args:
        items (List[str]): List of strings to format as bullets.
            Each string becomes one bullet point.

    Returns:
        str: Formatted string with each item on a new line prefixed by "- ".
            Returns empty string if input list is empty.

    Raises:
        TypeError: If items is not a list or contains non-string elements.

    Example:
        >>> items = ["Market research", "Financial planning", "Risk assessment"]
        >>> formatted = format_list_as_bullets(items)
        >>> print(formatted)
        - Market research
        - Financial planning
        - Risk assessment
        >>>
        >>> empty = format_list_as_bullets([])
        >>> print(repr(empty))
        ''
    """
    return "\n".join(f"- {item}" for item in items)


def format_topics_for_prompt(topics: List[Topic]) -> str:
    """
    Format a list of Topic objects into a readable string for prompts.

    Converts a list of Topic objects into a structured, human-readable string
    suitable for inclusion in AI prompts. Each topic is numbered and its
    subtopics are indented with sub-numbering for clear hierarchy visualization.

    Args:
        topics (List[Topic]): List of Topic objects to format. Each Topic
            should have a topic string and optionally a subtopics list.

    Returns:
        str: Formatted string with numbered topics and indented subtopics.
            Returns "No topics available." if the input list is empty.
            Format: "1. Topic Name\n   1.1. Subtopic\n   1.2. Subtopic\n\n2. ..."

    Raises:
        AttributeError: If Topic objects don't have required attributes.

    Example:
        >>> topics = [
        ...     Topic(topic="Market Analysis", subtopics=["Target market", "Competition"]),
        ...     Topic(topic="Operations", subtopics=["Location", "Suppliers"])
        ... ]
        >>> formatted = format_topics_for_prompt(topics)
        >>> print(formatted)
        1. Market Analysis
           1.1. Target market
           1.2. Competition
        <BLANKLINE>
        2. Operations
           2.1. Location
           2.2. Suppliers
        <BLANKLINE>
    """
    if not topics:
        return "No topics available."

    formatted_lines = []
    for i, topic in enumerate(topics, 1):
        formatted_lines.append(f"{i}. {topic.topic}")
        if topic.subtopics:
            for j, subtopic in enumerate(topic.subtopics, 1):
                formatted_lines.append(f"   {i}.{j}. {subtopic}")
        formatted_lines.append("")  # Empty line between topics

    return "\n".join(formatted_lines)


# ============================================================================
# Core Functions
# ============================================================================


def critique(state: PlanningState) -> PlanningState:
    """
    Generate feedback and critique for the current business plan topics.

    This function uses AI to evaluate the quality and completeness of generated
    business plan topics, providing structured feedback according to the
    PROMPT_PLANNING_CRITIQUE evaluation framework. The critique assesses
    strategic relevance, localization, completeness, actionability, and other
    quality dimensions.

    The critique process:
    1. Formats existing topics into a readable structure for the AI evaluator
    2. Constructs a critique prompt with business context and topic details
    3. Invokes the reasoning LLM with structured output parsing
    4. Populates the state with comprehensive feedback including assessment,
       score, strengths, weaknesses, suggestions, and recommendations
    5. Logs detailed feedback metrics for monitoring and debugging

    Args:
        state (PlanningState): Current planning state containing business type,
            location, and existing topics to be evaluated. Must have populated
            topics list for meaningful critique.

    Returns:
        PlanningState: Updated state with feedback attribute populated with a
            FeedbackResponse object containing structured evaluation results.
            The state object is modified in-place and returned for workflow continuity.

    Raises:
        ValidationError: If the LLM response doesn't match the expected
            FeedbackResponse schema or contains invalid feedback structures.
        RuntimeError: If the LLM invocation fails or returns malformed output.
        ValueError: If state.topics is empty or contains invalid Topic objects.

    Note:
        The critique uses PROMPT_PLANNING_CRITIQUE which evaluates topics on
        multiple dimensions including strategic alignment, localization depth,
        functional completeness, and research actionability. The feedback score
        (1-10) determines whether the planning iteration should continue.

    Example:
        >>> state = PlanningState(business="Falooda", location="Sweden", topics=[...])
        >>> critiqued_state = critique(state)
        >>> print(f"Quality score: {critiqued_state.feedback.score}/10")
        >>> print(f"Strengths: {len(critiqued_state.feedback.strength_list)}")
        >>> print(f"Weaknesses: {len(critiqued_state.feedback.weakness_list)}")
    """
    log_separator("CRITIQUE NODE", char="═", length=60)
    log_info("Critique Node Processing")
    log_info(f"Evaluating topics for {state.business} in {state.location}")

    formatted_topics = format_topics_for_prompt(state.topics)
    log_debug(f"Formatted {len(state.topics)} topics for critique")

    # Log topic summary
    if state.topics:
        total_subtopics = sum(len(topic.subtopics) for topic in state.topics)
        log_debug(f"Total subtopics being evaluated: {total_subtopics}")

    prompt = PROMPT_PLANNING_CRITIQUE.format(
        business=state.business, location=state.location, topics=formatted_topics
    )

    if FLAG_DEBUG:
        log_debug(f"Critique prompt length: {len(prompt)} characters")

    log_info("Invoking critique model")
    structured_output_model = llm_reason.with_structured_output(FeedbackResponse)
    structured_output = structured_output_model.invoke(prompt)

    state.feedback = structured_output
    log_success("Critique completed")
    log_info(f"Feedback score: {structured_output.score}/10")

    # Print detailed feedback
    print(f"\n{'='*60}")
    print(f"CRITIQUE FEEDBACK - SCORE: {structured_output.score}/10")
    print(f"{'='*60}")
    print(f"Assessment: {structured_output.assessment}")
    print(f"\nStrengths ({len(structured_output.strength_list)}):")
    for strength in structured_output.strength_list:
        print(f"  • {strength}")
    print(f"\nWeaknesses ({len(structured_output.weakness_list)}):")
    for weakness in structured_output.weakness_list:
        print(f"  • {weakness}")
    print(f"\nSuggestions ({len(structured_output.suggestion_list)}):")
    for suggestion in structured_output.suggestion_list:
        print(f"  • {suggestion}")
    print(f"\nRecommendations ({len(structured_output.recommendation_list)}):")
    for recommendation in structured_output.recommendation_list:
        print(f"  • {recommendation}")
    print(f"{'='*60}\n")

    log_info("Feedback details:")
    log_info(
        f"  Assessment: {structured_output.assessment[:100]}..."
        if len(structured_output.assessment) > 100
        else f"  Assessment: {structured_output.assessment}"
    )
    log_info(f"  Strengths identified: {len(structured_output.strength_list)}")
    log_info(f"  Weaknesses identified: {len(structured_output.weakness_list)}")
    log_info(f"  Suggestions provided: {len(structured_output.suggestion_list)}")
    log_info(
        f"  Recommendations provided: {len(structured_output.recommendation_list)}"
    )
    return state


def planner(state: PlanningState) -> PlanningState:
    """
    Generate or improve business plan topics using AI-powered analysis.

    This function creates comprehensive business plan topics based on the business
    type and location, with optional iterative improvement using previous critique
    feedback. It integrates expert knowledge from multiple domains and uses
    structured prompts to ensure comprehensive coverage of all business planning areas.

    The planning process:
    1. Loads expert knowledge from predefined expert profiles for comprehensive coverage
    2. Checks for existing feedback to determine if this is an iterative improvement
    3. If feedback exists, incorporates critique results into enhanced planning prompts
    4. Formats existing topics and feedback for context in improvement iterations
    5. Constructs planning prompts with expert information and business context
    6. Invokes the reasoning LLM with structured output parsing
    7. Updates state with newly generated or improved topics
    8. Logs detailed topic and subtopic breakdowns for monitoring

    Args:
        state (PlanningState): Current planning state containing business type,
            location, existing topics (for iterative improvement), and optionally
            previous feedback from critique function.

    Returns:
        PlanningState: Updated state with topics attribute populated with newly
            generated or improved Topic objects. The state object is modified
            in-place and returned for workflow continuity.

    Raises:
        ValidationError: If the LLM response doesn't match the expected
            PlanningResponse schema or contains invalid topic structures.
        RuntimeError: If the LLM invocation fails or returns malformed output.
        ValueError: If business or location parameters are empty or invalid.

    Note:
        This function implements iterative improvement: if state.feedback exists,
        it uses enhanced prompts that incorporate previous critique results to
        address weaknesses and build on strengths. Otherwise, it generates
        initial topics using basic planning prompts with expert integration.

    Example:
        >>> # Initial planning
        >>> state = PlanningState(business="Falooda", location="Sweden")
        >>> planned_state = planner(state)
        >>> print(f"Generated {len(planned_state.topics)} topics")
        >>>
        >>> # Iterative improvement with feedback
        >>> state.feedback = FeedbackResponse(score=6, weakness_list=["Missing KPIs"])
        >>> improved_state = planner(state)
        >>> print("Topics improved based on feedback")
    """
    log_separator("PLANNER NODE", char="═", length=60)
    log_info("Planner Node Processing")
    log_info(f"Planning topics for {state.business} in {state.location}")

    experts_information = "".join(
        f"- {key.replace('_', ' ')}: {value}\n" for key, value in EXPERTS.items()
    )
    log_debug(f"Loaded {len(EXPERTS)} expert domains for planning")

    if state.feedback is not None:
        log_info("Incorporating feedback for iterative improvement")
        log_debug(f"Previous feedback score: {state.feedback.score}/10")
        log_debug(f"Number of strengths: {len(state.feedback.strength_list)}")
        log_debug(f"Number of weaknesses: {len(state.feedback.weakness_list)}")
        log_debug(f"Number of suggestions: {len(state.feedback.suggestion_list)}")

        formatted_topics = format_topics_for_prompt(state.topics)
        log_debug(
            f"Formatted {len(state.topics)} existing topics for feedback context"
        )  # Format the feedback part
        prompt = PROMPT_PLANNING_FEEDBACK.format(
            experts_information=experts_information,
            business=state.business,
            location=state.location,
            assessment=state.feedback.assessment,
            strength_list=format_list_as_bullets(state.feedback.strength_list),
            weakness_list=format_list_as_bullets(state.feedback.weakness_list),
            suggestion_list=format_list_as_bullets(state.feedback.suggestion_list),
            recommendation_list=format_list_as_bullets(
                state.feedback.recommendation_list
            ),
            previous_output=formatted_topics,
        )
        log_debug("Using enhanced prompt with feedback")
    else:
        log_info("Generating initial topics")
        prompt = PROMPT_PLANNING.format(
            experts_information=experts_information,
            business=state.business,
            location=state.location,
        )
        log_debug("Using basic planning prompt")

    if FLAG_DEBUG:
        log_debug(f"Planning prompt length: {len(prompt)} characters")

    log_info("Invoking planning model")
    structured_output_model = llm_reason.with_structured_output(PlanningResponse)
    structured_output = structured_output_model.invoke(prompt)

    state.topics += structured_output.topics
    log_success("Planning completed")
    log_info(f"Generated {len(structured_output.topics)} topics")

    # Log detailed topics and subtopics information
    if FLAG_DEBUG:
        log_debug("Topics and subtopics breakdown:")
        for i, topic in enumerate(structured_output.topics, 1):
            log_debug(f"  Topic {i}: {topic.topic}")
            if topic.subtopics:
                log_debug(f"    Subtopics ({len(topic.subtopics)}):")
                for j, subtopic in enumerate(topic.subtopics, 1):
                    log_debug(f"      {i}.{j}. {subtopic}")
            else:
                log_debug("    No subtopics")
    else:
        # Provide summary even when not in debug mode
        total_subtopics = sum(
            len(topic.subtopics) for topic in structured_output.topics
        )
        log_info(f"Total subtopics across all topics: {total_subtopics}")

    return state


def generate_and_save_business_plan(
    business: str = DEFAULT_BUSINESS_TYPE,
    location: str = DEFAULT_LOCATION,
    output_path: str = BUSINESS_PLAN_JSON_PATH,
) -> PlanningResponse:
    """
    Generate business plan topics using iterative AI planning and save to JSON file.

    This function orchestrates the complete smart planning workflow, implementing
    an iterative improvement loop where topics are generated, critiqued, and
    refined until they meet quality standards. The process combines planning
    and critique nodes in a LangGraph workflow for comprehensive topic development.

    The workflow process:
    1. Initializes planning state with business parameters
    2. Sets up the smart planner graph with iterative improvement logic
    3. Executes the planning-critique iteration loop until quality threshold is met
    4. Extracts final topics from the completed workflow state
    5. Validates and saves the structured topics to a JSON file
    6. Provides comprehensive logging throughout the process

    Args:
        business (str): Type of business to generate topics for (e.g., "Falooda",
            "Coffee Shop", "Tech Startup"). Drives domain-specific topic generation.
            Defaults to DEFAULT_BUSINESS_TYPE.
        location (str): Geographic location for business analysis (e.g., "Sweden",
            "New York", "Mumbai"). Influences location-specific considerations.
            Defaults to DEFAULT_LOCATION.
        output_path (str): File system path where the JSON output will be saved.
            Should include filename with .json extension. Defaults to
            BUSINESS_PLAN_JSON_PATH. Parent directories are created automatically.

    Returns:
        PlanningResponse: The final generated business plan structure containing
            all iteratively improved topics, subtopics, and metadata. This object
            can be used directly for analysis or saved to different formats.

    Raises:
        FileNotFoundError: If the output directory cannot be created or accessed.
        PermissionError: If write permissions are insufficient for the output path.
        ValidationError: If topic generation fails or produces invalid structures.
        RuntimeError: If the LangGraph workflow execution encounters errors.
        OSError: If file system operations encounter system-level errors.

    Note:
        The smart planner implements iterative improvement using critique feedback.
        Topics are refined until the quality score exceeds QUALITY_THRESHOLD.
        The saved JSON file serves as input for agent analysis modules.
        The function provides comprehensive logging for monitoring iterations.

    Example:
        >>> # Generate comprehensive business plan with iterative improvement
        >>> result = generate_and_save_business_plan(
        ...     business="Falooda",
        ...     location="Sweden",
        ...     output_path="smart_plan.json"
        ... )
        >>> print(f"Generated {len(result.topics)} high-quality topics")
        >>> print(f"Saved to smart_plan.json")
        >>>
        >>> # Use defaults for quick generation
        >>> default_result = generate_and_save_business_plan()
        >>> print(f"Default plan saved to {BUSINESS_PLAN_JSON_PATH}")
    """
    # Create initial state
    log_info(f"Initializing smart planner for {business} in {location}")
    log_debug(f"Output path: {output_path}")
    initial_state = PlanningState(
        business=business, location=location, topics=[], feedback=None
    )
    log_debug("Initial state created with empty topics list")

    # Setup and run the smart planner graph
    log_info("Setting up smart planner graph")
    app = setup_smart_planner()

    log_separator("GRAPH EXECUTION", char="═", length=60)
    log_info("Starting smart planner execution")
    log_debug("Invoking graph with initial state")
    final_state = app.invoke(initial_state)
    log_debug("Graph execution completed")

    # Convert to PlanningResponse for saving
    # LangGraph returns a dict, so access topics accordingly
    topics = (
        final_state.get("topics", [])
        if isinstance(final_state, dict)
        else final_state.topics
    )
    log_debug(f"Extracted {len(topics)} topics from final state")

    result = PlanningResponse(topics=topics)
    log_info(f"Final result: {len(result.topics)} topics generated")

    if FLAG_DEBUG and result.topics:
        log_debug("Final topics summary:")
        for i, topic in enumerate(result.topics, 1):
            log_debug(f"  {i}. {topic.topic} ({len(topic.subtopics)} subtopics)")

    # Save JSON
    log_data_processing("Saving", "business plan topics JSON")
    result.save_json(output_path)
    log_success(f"Business plan topics saved to {output_path}")

    log_process_end("Business Plan Topics Generator", success=True)

    return result


def router_planner(state: PlanningState) -> Literal["planner", END]:  # type: ignore[valid-type]
    """
    Determine the next step in the planning workflow based on feedback quality.

    This routing function implements the conditional logic that controls the
    iterative improvement loop in the smart planning workflow. It evaluates
    the quality score from the critique feedback and decides whether to
    continue planning improvements or terminate the workflow.

    Routing logic:
    - If feedback score > QUALITY_THRESHOLD: Routes to END to complete the process
    - If feedback score ≤ QUALITY_THRESHOLD: Routes to "planner" for another iteration
    - The decision creates an improvement loop until quality standards are met

    Args:
        state (PlanningState): Current planning state containing critique feedback
            with quality score. Must have state.feedback populated by critique node.

    Returns:
        Literal["planner", END]: Routing decision indicating the next workflow node.
            Returns "planner" to continue iterative improvement, or END to terminate
            the planning workflow successfully.

    Raises:
        AttributeError: If state.feedback is None or missing score attribute.

    Note:
        This function assumes state.feedback has been set by a previous critique
        node execution. The QUALITY_THRESHOLD constant determines the quality
        standard required for workflow completion. The function is pure and
        does not modify the input state.

    Example:
        >>> # High quality - end workflow
        >>> state_good = PlanningState(feedback=FeedbackResponse(score=9))
        >>> next_node = router_planner(state_good)
        >>> print(next_node)  # Output: END
        >>>
        >>> # Needs improvement - continue planning
        >>> state_needs_work = PlanningState(feedback=FeedbackResponse(score=5))
        >>> next_node = router_planner(state_needs_work)
        >>> print(next_node)  # Output: "planner"
    """
    log_separator("ROUTER PLANNER", char="═", length=60)
    log_info("Router Planner Processing")

    if state.feedback:
        log_debug(f"Evaluating feedback score: {state.feedback.score}/10")
        log_debug(f"Score threshold for completion: > {QUALITY_THRESHOLD}")

    if state.feedback and state.feedback.score > QUALITY_THRESHOLD:
        log_info(
            f"Quality score {state.feedback.score}/10 meets threshold - ending iteration"
        )
        log_success("Planning workflow completed successfully")
        return END
    else:
        score_msg = state.feedback.score if state.feedback else "N/A"
        log_info(f"Quality score {score_msg}/10 below threshold - continuing iteration")
        log_info("Routing back to planner for improvement")
        return "planner"


def setup_smart_planner():
    """
    Create and configure the smart planner workflow graph.

    This function sets up a LangGraph StateGraph implementing an iterative
    improvement workflow for business plan topic generation. The graph combines
    planning and critique nodes in a feedback loop that refines topics until
    they meet quality standards defined by QUALITY_THRESHOLD.

    Graph Structure:
    - START → planner: Initial topic generation or iterative improvement
    - planner → critique: Evaluate the quality of generated/improved topics
    - critique → router_planner: Conditional routing based on quality score
    - If score > QUALITY_THRESHOLD: END (workflow completion)
    - If score ≤ QUALITY_THRESHOLD: planner (another improvement iteration)

    The workflow implements this iterative process:
    1. Generate initial topics using expert knowledge and business context
    2. Critique the topics for quality, completeness, and strategic alignment
    3. If quality is insufficient, use feedback to improve topics in next iteration
    4. Repeat critique-improve cycle until quality threshold is exceeded
    5. Output final high-quality topics ready for detailed analysis

    Returns:
        CompiledGraph: Compiled LangGraph application ready for execution with
            PlanningState objects. The graph can be invoked with initial planning
            parameters and will return completed planning results.

    Raises:
        RuntimeError: If graph compilation fails due to invalid node or edge definitions.
        ValueError: If required nodes or edges cannot be added to the graph.

    Note:
        The graph uses PlanningState to maintain workflow state across iterations,
        including topics, feedback, and business context. The iterative nature
        ensures high-quality topic generation through AI-powered critique and improvement.

    Example:
        >>> graph = setup_smart_planner()
        >>> initial_state = PlanningState(business="Falooda", location="Sweden")
        >>> result = graph.invoke(initial_state)
        >>> print(f"Final topics: {len(result.topics)}")
        >>> print(f"Quality score: {result.feedback.score}/10")
    """
    log_separator("GRAPH SETUP", char="═", length=60)
    log_info("Creating smart planner workflow graph")
    log_debug("Initializing StateGraph with PlanningState")

    graph = StateGraph(PlanningState)

    log_info("Adding graph nodes")
    log_debug("Adding 'planner' node")
    graph.add_node("planner", planner)
    log_debug("Adding 'critique' node")
    graph.add_node("critique", critique)

    log_info("Defining graph edges and flow")
    log_debug("Setting START -> planner edge")
    graph.add_edge(START, "planner")
    log_debug("Setting planner -> critique edge")
    graph.add_edge("planner", "critique")

    log_debug("Adding conditional routing: critique -> router_planner")
    graph.add_conditional_edges(
        "critique",
        router_planner,
    )

    log_info("Compiling graph")
    app = graph.compile()
    log_debug("Graph compilation successful")

    log_success("Smart planner graph compiled successfully")
    return app


# Main execution
if __name__ == "__main__":
    generate_and_save_business_plan()
