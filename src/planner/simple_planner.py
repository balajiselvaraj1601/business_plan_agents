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

from src.prompts.prompt_collection import PROMPT_PLANNING
from src.states.state_collection import PlanningResponse
from src.utils.constants import (
    BUSINESS_PLAN_JSON_PATH,
    DEFAULT_BUSINESS_TYPE,
    DEFAULT_LOCATION,
    EXPERTS,
    FLAG_DEBUG,
    LOG_LEVEL,
    LOG_SEPARATOR_CHAR,
    LOG_SEPARATOR_LENGTH,
)
from src.utils.logging import (
    log_configuration,
    log_data_processing,
    log_debug,
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
# Core Functions
# ============================================================================


def planner(
    business: str = DEFAULT_BUSINESS_TYPE, location: str = DEFAULT_LOCATION
) -> PlanningResponse:
    """
    Generate comprehensive business plan topics using AI-powered analysis.

    This function uses a reasoning-focused LLM to create a structured set of business
    plan topics tailored to the specific business type and location. The topics are
    generated using expert knowledge integration and structured output parsing to
    ensure comprehensive coverage of all business planning aspects.

    The planning process:
    1. Integrates expert knowledge from predefined expert profiles
    2. Formats the planning prompt with business context and expert information
    3. Invokes the reasoning LLM with structured output requirements
    4. Validates the generated topics against the PlanningResponse schema
    5. Returns structured topics ready for detailed analysis

    Args:
        business (str): Type of business to generate topics for (e.g., "Falooda",
            "Coffee Shop", "Tech Startup"). This drives the domain-specific
            topic generation. Defaults to DEFAULT_BUSINESS_TYPE.
        location (str): Geographic location/city for business analysis (e.g.,
            "Sweden", "New York", "Mumbai"). Influences location-specific
            considerations in topic generation. Defaults to DEFAULT_LOCATION.

    Returns:
        PlanningResponse: Structured response containing a list of Topic objects,
            each with comprehensive subtopics covering all aspects of business
            planning including market research, operations, finance, marketing,
            and risk management.

    Raises:
        ValidationError: If the LLM response doesn't match the expected
            PlanningResponse schema or contains invalid topic structures.
        RuntimeError: If the LLM invocation fails or returns malformed output.
        ValueError: If business or location parameters are empty or invalid.

    Note:
        The function integrates expert knowledge from the EXPERTS constant to
        ensure comprehensive topic coverage. In debug mode, additional logging
        provides insight into prompt construction and model responses.

    Example:
        >>> # Generate topics for a falooda business in Sweden
        >>> topics = planner("Falooda", "Sweden")
        >>> print(f"Generated {len(topics.topics)} main topics")
        >>> for topic in topics.topics[:3]:
        ...     print(f"- {topic.topic}: {len(topic.subtopics)} subtopics")
        >>>
        >>> # Use default parameters
        >>> default_topics = planner()
        >>> print(f"Business: {default_topics.business_type}")
    """

    log_separator("PLANNER NODE", char=LOG_SEPARATOR_CHAR, length=LOG_SEPARATOR_LENGTH)
    log_debug("Planner Node Processing")
    log_debug(f"Generating business plan topics for {business} in {location}")

    # Use the general model from model_collection
    llm = llm_reason

    # Create the prompt with placeholders
    log_debug("Creating business plan topics prompt")
    experts_information = (
        "".join(
            f"- {key.replace('_', ' ')}: {value}\n" for key, value in EXPERTS.items()
        ),
    )
    prompt = PROMPT_PLANNING.format(
        experts_information=experts_information,
        business=business,
        location=location,
    )

    if FLAG_DEBUG:
        log_debug(f"Planner prompt length: {len(prompt)} characters")
        log_debug(f"Prompt preview: {prompt[:200]}...")

    # Invoke model to generate business plan topics
    log_debug("Invoking model to generate business plan topics")
    structured_output_model = llm.with_structured_output(PlanningResponse)
    structured_output = structured_output_model.invoke(prompt)

    if FLAG_DEBUG:
        log_debug(f"Model response type: {type(structured_output)}")
        log_debug(f"Number of topics generated: {len(structured_output.topics)}")

    log_success("Business plan topics generated successfully")
    structured_output.print_structured()

    return structured_output


def generate_and_save_business_plan(
    business: str = DEFAULT_BUSINESS_TYPE,
    location: str = DEFAULT_LOCATION,
    output_path: str = BUSINESS_PLAN_JSON_PATH,
) -> PlanningResponse:
    """
    Generate business plan topics and save to JSON file.

    This function orchestrates the complete business plan topic generation workflow,
    from AI-powered topic creation to persistent storage. It combines topic generation
    with file I/O operations to create a reusable business plan structure that can
    be consumed by analysis agents.

    The process includes:
    1. Generates comprehensive business plan topics using the planner function
    2. Validates the generated topic structure and content
    3. Saves the structured topics to a JSON file with proper formatting
    4. Provides logging and feedback throughout the process
    5. Returns the generated topics for immediate use or further processing

    Args:
        business (str): Type of business to generate topics for. Passed directly
            to the planner function. Defaults to DEFAULT_BUSINESS_TYPE.
        location (str): Geographic location for business analysis. Influences
            location-specific topic considerations. Defaults to DEFAULT_LOCATION.
        output_path (str): File system path where the JSON output will be saved.
            Should include filename with .json extension. Defaults to
            BUSINESS_PLAN_JSON_PATH. Parent directories are created automatically.

    Returns:
        PlanningResponse: The complete generated business plan structure containing
            all topics, subtopics, and metadata. This object can be used directly
            for analysis or saved to different formats.

    Raises:
        FileNotFoundError: If the output directory cannot be created or accessed.
        PermissionError: If write permissions are insufficient for the output path.
        ValidationError: If topic generation fails or produces invalid structures.
        OSError: If file system operations encounter system-level errors.

    Note:
        The saved JSON file serves as input for the agent analysis modules.
        The function provides comprehensive logging for monitoring and debugging.
        Existing files at the output path will be overwritten.

    Example:
        >>> # Generate and save custom business plan
        >>> result = generate_and_save_business_plan(
        ...     business="Coffee Shop",
        ...     location="Portland",
        ...     output_path="custom_plan.json"
        ... )
        >>> print(f"Saved {len(result.topics)} topics to custom_plan.json")
        >>>
        >>> # Use defaults
        >>> default_result = generate_and_save_business_plan()
        >>> print(f"Default plan saved to {BUSINESS_PLAN_JSON_PATH}")
    """
    # Generate business plan topics
    structured_output = planner(business, location)

    # Save JSON
    log_data_processing("Saving", "business plan topics JSON")
    structured_output.save_json(output_path)
    log_success(f"Business plan topics saved to {output_path}")

    log_process_end("Business Plan Topics Generator", success=True)

    return structured_output


# Main execution
if __name__ == "__main__":
    generate_and_save_business_plan()
