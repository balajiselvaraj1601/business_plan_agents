"""
Utility Functions for Business Plan Agents

This module provides essential utility functions for the Business Plan Agents project,
including model loading, file operations, logging configuration, and expert routing.
These utilities form the foundation layer that supports the higher-level agent and
planner functionality.

Core Functionality:
- Model Management: Load and configure AI models with automatic debug/production switching
- File Operations: Safe file saving with directory creation and error handling
- Logging Setup: Centralized logging configuration with noise reduction
- Expert Routing: Intelligent routing of queries to appropriate business experts

Key Functions:
- setup_logging(): Configure application-wide logging with noise suppression
- load_model(): Initialize chat models with automatic lite/full model selection
- save_text_file(): Safe file writing with directory creation
- choose_expert(): AI-powered expert selection for query routing
- get_expert_prompt(): Generate expert-specific prompts based on query analysis

Architecture Notes:
- Functions are designed to be stateless and reusable
- Error handling includes logging and appropriate exception propagation
- Model loading automatically switches between lite/debug and full/production models
- Expert routing uses structured AI outputs for reliable decision making

Usage Patterns:
    # Setup and model loading
    from src.utils.utils import setup_logging, load_model

    setup_logging("INFO")
    general_model = load_model("general", temperature=0.0)

    # File operations
    from src.utils.utils import save_text_file

    save_text_file("Business plan content", "output/plan.txt")

    # Expert routing
    from src.utils.utils import get_expert_prompt

    prompt = get_expert_prompt("How to improve our marketing strategy?")
    # Returns formatted prompt for marketing expert

Dependencies:
- langchain: For model initialization and prompt templating
- pathlib: For robust file path handling
- Internal modules: constants, logging, model_collection, prompt_collection, states
"""

# Copilot: Do not add any logging for this file.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

from src.prompts.prompt_collection import EXPERT_PROMPTS, EXPERT_ROUTER_PROMPT
from src.states.state_collection import ExpertDecision
from src.utils.constants import (
    FLAG_DEBUG,
    OLLAMA_GENERAL,
    OLLAMA_GENERAL_LITE,
    OLLAMA_REASON,
    OLLAMA_REASON_LITE,
    ExpertType,
)
from src.utils.logging import (
    log_error,
    log_model_loading,
    log_success,
)

# ============================================================================
# Logging Configuration
# ============================================================================


def setup_logging(level: str = "INFO") -> None:
    """
    Configure application-wide logging with noise reduction for third-party libraries.

    This function sets up the Python logging system with a standardized format and
    suppresses noisy loggers from external dependencies. It provides consistent
    timestamp formatting and structured log output across the entire application.

    The logging configuration includes:
    - Timestamp formatting: YYYY-MM-DD HH:MM:SS
    - Structured format: timestamp - logger_name - level - message
    - Force reconfiguration to override any existing handlers
    - Automatic noise reduction for httpcore, httpx, and urllib3 libraries

    Args:
        level: Logging level string. Must be one of "DEBUG", "INFO", "WARNING",
               "ERROR", or "CRITICAL". Case-insensitive. Defaults to "INFO".

    Raises:
        AttributeError: If an invalid logging level is provided (not a valid
                       logging level name).

    Example:
        >>> # Setup for development with detailed logging
        >>> setup_logging("DEBUG")
        >>>
        >>> # Setup for production with info and above
        >>> setup_logging("INFO")
        >>>
        >>> # Invalid level raises AttributeError
        >>> setup_logging("INVALID")  # Raises AttributeError

    Note:
        This function should be called once at application startup. Calling it
        multiple times will reconfigure logging each time. The configuration is
        global and affects all loggers in the application. Third-party library
        noise is automatically suppressed to keep logs clean and readable.
    """
    # Always configure logging, using force=True to override any existing configuration
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Force reconfiguration
    )

    # Suppress noisy third-party loggers
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# ============================================================================
# Model Loading Functions
# ============================================================================


def load_model(model_type: str, temperature: float = 0.0) -> Any:
    """
    Load and initialize a chat model instance with automatic debug/production model selection.

    This function provides intelligent model loading that automatically switches between
    full production models and lite debug models based on the FLAG_DEBUG setting. It
    handles model initialization, error recovery, and provides comprehensive logging
    for debugging and monitoring.

    The function supports two model types:
    - "general": For general-purpose analysis, reasoning, and text generation
    - "reason": For complex planning, decision-making, and analytical tasks

    Model Selection Logic:
    - Debug mode (FLAG_DEBUG=True): Uses lite models for faster iteration
    - Production mode (FLAG_DEBUG=False): Uses full models for best performance
    - Temperature controls creativity vs determinism (0.0 = deterministic)

    Args:
        model_type: The type of model to load. Must be either "general" or "reason".
                   Case-sensitive. Determines which model family to use.
        temperature: Sampling temperature for model outputs. Range: 0.0 to 1.0.
                    0.0 = deterministic, 1.0 = highly creative. Defaults to 0.0.

    Returns:
        Any: An initialized LangChain chat model instance ready for use. The exact
             type depends on the Ollama model provider and configuration.

    Raises:
        ValueError: If model_type is not "general" or "reason".
        Exception: If model initialization fails (network issues, invalid model name,
                  Ollama service unavailable, etc.). Original exception is logged
                  and re-raised.

    Example:
        >>> # Load general-purpose model for analysis
        >>> model = load_model("general", temperature=0.0)
        >>> response = model.invoke("Analyze this business idea...")
        >>>
        >>> # Load reasoning model for planning
        >>> planner_model = load_model("reason", temperature=0.1)
        >>> plan = planner_model.invoke("Create a business strategy...")
        >>>
        >>> # Invalid model type raises ValueError
        >>> load_model("invalid")  # Raises ValueError

    Note:
        Model loading is logged with success/failure status. In debug mode, lite
        models are automatically selected for faster development iteration. The
        function handles all model provider setup and configuration internally.
    """
    # Determine model name based on type and debug flag
    if model_type == "general":
        model_name = OLLAMA_GENERAL_LITE if FLAG_DEBUG else OLLAMA_GENERAL
    elif model_type == "reason":
        model_name = OLLAMA_REASON_LITE if FLAG_DEBUG else OLLAMA_REASON
    else:
        raise ValueError(
            f"Invalid model_type: {model_type}. Must be 'general' or 'reason'"
        )

    log_model_loading(model_name, is_lite=FLAG_DEBUG)
    try:
        model = init_chat_model(
            model_name, model_provider="ollama", temperature=temperature
        )
        log_success(f"Model {model_name} loaded successfully")
        return model
    except Exception as e:
        log_error(f"Failed to load model {model_name}: {str(e)}")
        raise

    # Determine model name based on type and debug flag
    if model_type == "general":
        model_name = OLLAMA_GENERAL_LITE if FLAG_DEBUG else OLLAMA_GENERAL
    elif model_type == "reason":
        model_name = OLLAMA_REASON_LITE if FLAG_DEBUG else OLLAMA_REASON
    else:
        raise ValueError(
            f"Invalid model_type: {model_type}. Must be 'general' or 'reason'"
        )

    log_model_loading(model_name, is_lite=FLAG_DEBUG)
    try:
        model = init_chat_model(
            model_name, model_provider="ollama", temperature=temperature
        )
        log_success(f"Model {model_name} loaded successfully")
        return model
    except Exception as e:
        log_error(f"Failed to load model {model_name}: {str(e)}")
        raise


# ============================================================================
# File Operations
# ============================================================================


def save_text_file(content: str, filepath: str) -> None:
    """
    Save text content to a file with automatic directory creation and error handling.

    This function provides safe file writing operations using pathlib for robust path
    handling. It automatically creates any missing parent directories and provides
    comprehensive error handling with logging. The function ensures UTF-8 encoding
    for proper text handling across different character sets.

    Key Features:
    - Automatic directory creation (parents=True, exist_ok=True)
    - UTF-8 encoding for international character support
    - Comprehensive error handling with detailed logging
    - Absolute path resolution in success messages

    Args:
        content: The text content to write to the file. Should be a valid string.
                The content is written as-is without any modifications.
        filepath: The file path where to save the content. Can be relative or absolute.
                 Parent directories will be created automatically if they don't exist.

    Raises:
        OSError: If directory creation fails (permission issues, disk full, etc.)
        IOError: If file writing fails (permission issues, disk full, encoding errors, etc.)
        Exception: Any other unexpected error during file operations. Original exception
                  is logged and re-raised.

    Example:
        >>> # Save business plan content
        >>> content = "Executive Summary: This is a great business idea..."
        >>> save_text_file(content, "reports/business_plan.txt")
        >>> # Creates reports/ directory if needed
        >>>
        >>> # Save with absolute path
        >>> save_text_file("Data export", "/tmp/export.json")
        >>>
        >>> # Save to nested directory structure
        >>> save_text_file("Analysis", "output/2024/Q1/analysis.txt")
        >>> # Creates output/2024/Q1/ directories automatically

    Note:
        The function uses pathlib.Path for robust cross-platform path handling.
        Success and failure are both logged with absolute paths for clarity.
        The function overwrites existing files without warning - implement your
        own checks if preservation of existing files is needed.
    """
    path = Path(filepath)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        log_success(f"Saved text file to: {path.resolve()}")
    except Exception as e:
        log_error(f"Failed to save text file to {filepath}: {str(e)}")
        raise


# ============================================================================
# Expert Routing Functions
# ============================================================================


def choose_expert(user_input: str) -> Any:
    """
    Intelligently route a user query to the most appropriate business expert using AI analysis.

    This function employs the expert router LLM to analyze the semantic content of user
    input and determine which specialized expert should handle the request. It uses
    structured AI outputs to ensure reliable and consistent expert selection based on
    the query's domain, context, and requirements.

    The routing decision considers:
    - Business domain (marketing, finance, operations, etc.)
    - Query intent and complexity
    - Required expertise and knowledge areas
    - Context clues and specific terminology

    Available Experts:
    - BUSINESS_ANALYST: Strategy, planning, and business processes
    - COMPETITIVE_INTELLIGENCE_EXPERT: Market research and competitor analysis
    - CUSTOMER_EXPERIENCE_EXPERT: Customer satisfaction and support strategy
    - FINANCIAL_EXPERT: Investments, accounting, and budgeting
    - HR_EXPERT: Hiring, policies, and training
    - LEGAL_EXPERT: Laws, contracts, and compliance
    - MARKETING_EXPERT: Campaigns, branding, and promotional activities
    - OPERATIONS_EXPERT: Logistics, workflows, and operational efficiency
    - PUBLIC_RELATIONS_EXPERT: Media relations, reputation, and communications
    - SALES_EXPERT: Sales strategies, CRM, and revenue generation
    - SUPPLY_CHAIN_EXPERT: Procurement, inventory, and logistics management
    - TECHNOLOGY_EXPERT: Software, AI, hardware, and IT infrastructure

    Args:
        user_input: The user's query or request text to analyze. Should be a clear,
                   natural language description of the business question or task.
                   The more context provided, the better the routing decision.

    Returns:
        ExpertType: An ExpertType enum value representing the recommended expert.
                   This can be used directly with EXPERTS dictionary or converted
                   to string with .value attribute.

    Raises:
        Exception: If the AI routing model fails or returns invalid output. In this
                  case, the function logs the error and returns BUSINESS_ANALYST
                  as a safe default fallback.

    Example:
        >>> # Route marketing-related query
        >>> expert = choose_expert("How can we improve our social media presence?")
        >>> print(expert)  # ExpertType.MARKETING_EXPERT
        >>> print(expert.value)  # "marketing_expert"
        >>>
        >>> # Route financial query
        >>> expert = choose_expert("What's our projected ROI for the next quarter?")
        >>> print(expert)  # ExpertType.FINANCIAL_EXPERT
        >>>
        >>> # Route complex business strategy
        >>> expert = choose_expert("Develop a market entry strategy for Europe")
        >>> print(expert)  # ExpertType.BUSINESS_ANALYST

    Note:
        The function uses lazy importing to avoid circular dependencies with
        model_collection. If the AI routing fails, it gracefully falls back to
        BUSINESS_ANALYST as a general-purpose expert. The routing decision is
        logged for debugging and monitoring purposes.
    """
    # Import here to avoid circular import
    from src.utils.model_collection import expert_router_llm

    router_prompt = PromptTemplate.from_template(EXPERT_ROUTER_PROMPT)
    prompt = router_prompt.format(input=user_input)
    try:
        decision: ExpertDecision = expert_router_llm.invoke(prompt)
        return decision.expert
    except Exception as e:
        log_error(f"Failed to determine expert for input '{user_input}': {str(e)}")
        # Return default expert when LLM fails
        return ExpertType.BUSINESS_ANALYST


def get_expert_prompt(user_input: str) -> str:
    """
    Generate a formatted, expert-specific prompt based on intelligent query analysis.

    This function combines expert routing with prompt templating to create customized
    prompts that are optimized for the selected expert's domain knowledge and expertise.
    It first analyzes the user input to determine the most appropriate expert, then
    formats the corresponding expert-specific prompt template with the user's query.

    The process involves:
    1. AI-powered expert selection using choose_expert()
    2. Retrieval of the appropriate prompt template from EXPERT_PROMPTS
    3. Template formatting with the user's input
    4. Return of the ready-to-use prompt string

    Expert Prompt Templates:
    Each expert has a specialized prompt template designed for their domain:
    - Business Analyst: Strategy and planning focused
    - Marketing Expert: Campaign and branding focused
    - Financial Expert: Investment and accounting focused
    - And so on for all 12 expert domains

    Args:
        user_input: The user's query or request text to analyze and format into
                   an expert-specific prompt. Should be clear and contain sufficient
                   context for proper expert routing and prompt customization.

    Returns:
        str: A fully formatted prompt string ready for use with the appropriate
            expert LLM. The prompt includes expert-specific instructions,
            domain knowledge, and the formatted user query.

    Raises:
        KeyError: If the selected expert doesn't have a corresponding prompt template.
                 This should not occur with properly configured EXPERT_PROMPTS.
        Exception: If expert selection fails (handled gracefully in choose_expert).

    Example:
        >>> # Generate marketing-focused prompt
        >>> prompt = get_expert_prompt("How can we improve brand awareness?")
        >>> print(prompt[:100])  # Shows formatted marketing expert prompt
        >>>
        >>> # Generate financial analysis prompt
        >>> prompt = get_expert_prompt("Analyze our cash flow projections")
        >>> # Returns prompt formatted for financial expert
        >>>
        >>> # Complex business strategy prompt
        >>> prompt = get_expert_prompt("Develop a competitive positioning strategy")
        >>> # Returns business analyst focused prompt

    Note:
        This function provides end-to-end prompt engineering: from raw user input
        to expert-optimized, formatted prompts. The expert selection is cached
        within the function call, so the AI routing decision is made once per
        prompt generation. The returned prompt is immediately usable with any
        LangChain-compatible LLM.
    """
    expert = choose_expert(user_input)
    expert_name = expert.value  # Use .value to get the string key
    template = EXPERT_PROMPTS[expert_name]
    return template.format(input=user_input)
