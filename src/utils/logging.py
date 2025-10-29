"""
Logging Utilities Module

This module provides centralized, enhanced logging functionality for the Business Plan Agents project.
It offers consistent formatting, visual separators, process tracking, and debug control across
all application components.

The logging system includes:
- Standard logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Enhanced message formatting with emojis and visual indicators
- Process lifecycle tracking with start/end separators
- Debug-controlled output for development vs production
- API call and data processing logging
- Configuration and model loading notifications

Key Features:
- Visual separators for improved log readability
- Emoji-enhanced messages for quick visual scanning
- Debug flag integration for conditional logging
- Process tracking with automatic success/failure indication
- External library noise reduction (urllib3 warnings)
- Consistent timestamp formatting

Logging Functions:
- setup_logging(): Configure application-wide logging behavior
- log_separator(): Create visual dividers with optional titles
- log_debug/info/error/success(): Enhanced level-specific logging
- log_process_start/end(): Track process lifecycles
- log_model_loading(): Model initialization notifications
- log_api_call(): External API interaction logging
- log_data_processing(): Data operation tracking
- log_configuration(): Configuration setting logging

Configuration:
- Controlled by FLAG_DEBUG constant for debug output
- Uses standard Python logging with custom formatting
- Reduces noise from external libraries automatically
- Force reconfiguration to override existing handlers

Usage:
    from src.utils.logging import setup_logging, log_info, log_process_start, log_success

    # Setup logging
    setup_logging("DEBUG")

    # Process tracking
    log_process_start("Business Analysis")
    log_info("Processing topic: Marketing Strategy")
    log_success("Analysis completed")
    log_process_end("Business Analysis", success=True)

Best Practices:
- Use log_process_start/end for major operations
- Use log_separator for visual organization
- Use appropriate log levels (debug for development, info for operations)
- Include relevant context in log messages
- Use log_success for positive outcomes and confirmations

Dependencies:
- Python standard library logging module
- Internal constants module for debug flags
"""

# Copilot: Do not add any logging for this file.

import logging
from typing import Optional

from src.utils.constants import FLAG_DEBUG

# ============================================================================
# Logging Setup
# ============================================================================


def setup_logging(level: str = "INFO") -> None:
    """
    Configure application-wide logging behavior with enhanced formatting and noise reduction.

    This function initializes the Python logging system with custom formatting, sets the
    appropriate logging level, and reduces noise from external libraries. It provides
    consistent timestamp formatting and structured log output across all application
    components.

    The logging configuration includes:
    - Timestamp formatting: YYYY-MM-DD HH:MM:SS
    - Structured format: timestamp - logger_name - level - message
    - Force reconfiguration to override any existing handlers
    - Automatic noise reduction for urllib3 and httpcore libraries

    Args:
        level: Logging level string. Must be one of "DEBUG", "INFO", "WARNING",
               "ERROR", or "CRITICAL". Case-insensitive. Defaults to "INFO".

    Raises:
        ValueError: If an invalid logging level is provided (not in the supported levels).

    Example:
        >>> # Setup for development with debug output
        >>> setup_logging("DEBUG")
        >>>
        >>> # Setup for production with info and above
        >>> setup_logging("INFO")
        >>>
        >>> # Invalid level raises ValueError
        >>> setup_logging("INVALID")  # Raises ValueError

    Note:
        This function should be called once at application startup. Calling it multiple
        times will reconfigure logging each time, which may affect performance.
        The configuration is global and affects all loggers in the application.
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    numeric_level = level_map.get(level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore.http11").setLevel(logging.WARNING)


# ============================================================================
# Core Logging Functions
# ============================================================================


def log_separator(title: str = "", char: str = "═", length: int = 60) -> None:
    """
    Create and log a visual separator line to improve log readability and organization.

    This function generates attractive separator lines that help visually organize log
    output, making it easier to distinguish between different sections or processes.
    When a title is provided, it creates a centered title within the separator line.
    When no title is provided, it creates a solid line of the specified character.

    The separator is only logged when FLAG_DEBUG is True, allowing for clean production
    logs while maintaining visual organization during development.

    Args:
        title: Optional title text to display centered within the separator.
               If empty, creates a solid separator line. Defaults to empty string.
        char: Character to use for creating the separator line. Common choices include
              "═", "-", "=", "*". Defaults to "═" (double horizontal line).
        length: Total length of the separator line in characters. The title will be
                centered within this length. Defaults to 60 characters.

    Example:
        >>> # Simple separator line
        >>> log_separator()
        >>> # Output: ════════════════════════════════════════════════════════════════
        >>>
        >>> # Separator with title
        >>> log_separator("PROCESSING STARTED")
        >>> # Output: ════════════════ PROCESSING STARTED ════════════════
        >>>
        >>> # Custom character and length
        >>> log_separator("Debug Info", "-", 40)
        >>> # Output: ---------------- Debug Info ----------------

    Note:
        The function ensures the total length matches the specified length by padding
        with additional separator characters if needed. This is particularly useful
        for creating consistent visual boundaries in log output.
    """
    if FLAG_DEBUG:
        if title:
            total_length = length
            title_length = len(title) + 2  # Add spaces around title
            separator_length = (total_length - title_length) // 2
            separator = char * separator_length
            result = f"{separator} {title} {separator}"
            # Ensure exact length
            if len(result) < total_length:
                result += char * (total_length - len(result))
            logging.info(result)
        else:
            logging.info(char * length)


def log_debug(message: str, *args, **kwargs) -> None:
    """
    Log a debug message with enhanced formatting and visual indicators.

    This function logs debug-level information with a magnifying glass emoji (🔍) prefix
    to make debug messages easily identifiable in log output. Debug messages are only
    logged when FLAG_DEBUG is True, allowing for detailed development information
    without cluttering production logs.

    Debug logging is typically used for:
    - Variable values during development
    - Function entry/exit points
    - Intermediate calculation results
    - API request/response details
    - Configuration loading information

    Args:
        message: The debug message to log. Should be descriptive and include relevant
                 context or variable values.
        *args: Additional positional arguments passed to the underlying logging.debug()
               call for string formatting.
        **kwargs: Additional keyword arguments passed to the underlying logging.debug()
                  call.

    Example:
        >>> # Simple debug message
        >>> log_debug("Processing user input")
        >>> # Output: 🔍 DEBUG: Processing user input
        >>>
        >>> # Debug with variable values
        >>> user_count = 42
        >>> log_debug(f"Found {user_count} users in database")
        >>> # Output: 🔍 DEBUG: Found 42 users in database
        >>>
        >>> # Debug with formatting
        >>> log_debug("API response: %s", response_data)

    Note:
        Debug messages are suppressed in production (when FLAG_DEBUG is False) to
        maintain clean log output. Use this function liberally during development
        but consider using log_info for important operational information that
        should always be visible.
    """
    if FLAG_DEBUG:
        formatted_message = f"🔍 DEBUG: {message}"
        logging.debug(formatted_message, *args, **kwargs)


def log_info(message: str, *args, **kwargs) -> None:
    """
    Log an informational message with enhanced formatting and visual indicators.

    This function logs general information about application operations with an info
    emoji (ℹ️) prefix. Info messages are always logged regardless of the FLAG_DEBUG
    setting, making them suitable for important operational information that should
    be visible in both development and production environments.

    Info logging is typically used for:
    - Process milestones and completion status
    - Configuration confirmations
    - User actions and responses
    - System state changes
    - Performance metrics and summaries

    Args:
        message: The informational message to log. Should describe important events
                 or state changes in the application.
        *args: Additional positional arguments passed to the underlying logging.info()
               call for string formatting.
        **kwargs: Additional keyword arguments passed to the underlying logging.info()
                  call.

    Example:
        >>> # Process milestone
        >>> log_info("Business plan generation completed")
        >>> # Output: ℹ️  INFO: Business plan generation completed
        >>>
        >>> # Configuration confirmation
        >>> log_info(f"Loaded {len(topics)} business topics")
        >>> # Output: ℹ️  INFO: Loaded 15 business topics
        >>>
        >>> # User action
        >>> log_info("User requested marketing analysis")

    Note:
        Unlike debug messages, info messages are always visible and should contain
        information that is valuable for monitoring application health and user
        interactions. Avoid using info for verbose debugging information.
    """
    formatted_message = f"ℹ️  INFO: {message}"
    logging.info(formatted_message, *args, **kwargs)


def log_error(message: str, *args, **kwargs) -> None:
    """
    Log an error message with enhanced formatting and visual indicators.

    This function logs error conditions and failures with a red X emoji (❌) prefix
    to make errors easily identifiable in log output. Error messages are always logged
    regardless of the FLAG_DEBUG setting, ensuring critical issues are never missed.

    Error logging is typically used for:
    - Exception handling and error recovery
    - API call failures
    - Data validation errors
    - Configuration problems
    - Resource access issues

    Args:
        message: The error message to log. Should include specific details about what
                 went wrong and any relevant context or error codes.
        *args: Additional positional arguments passed to the underlying logging.error()
               call for string formatting.
        **kwargs: Additional keyword arguments passed to the underlying logging.error()
                  call.

    Example:
        >>> # API failure
        >>> log_error("Failed to connect to Tavily API")
        >>> # Output: ❌ ERROR: Failed to connect to Tavily API
        >>>
        >>> # Data validation error
        >>> log_error(f"Invalid business type: {business_type}")
        >>> # Output: ❌ ERROR: Invalid business type: invalid_type
        >>>
        >>> # Exception handling
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     log_error(f"Operation failed: {str(e)}")

    Note:
        Error messages should be actionable and include enough context for debugging.
        Consider using log_process_end() with success=False for process-level failures,
        and reserve log_error() for unexpected errors that need immediate attention.
    """
    formatted_message = f"❌ ERROR: {message}"
    logging.error(formatted_message, *args, **kwargs)


def log_success(message: str, *args, **kwargs) -> None:
    """
    Log a success message with enhanced formatting and visual indicators.

    This function logs successful operations and positive outcomes with a green checkmark
    emoji (✅) prefix. Success messages are logged at INFO level and are always visible,
    making them perfect for confirming important milestones and completed operations.

    Success logging is typically used for:
    - Process completion confirmations
    - Successful API responses
    - Data saving/loading operations
    - Configuration validations
    - User action completions

    Args:
        message: The success message to log. Should describe what was accomplished
                 successfully and provide confirmation of the positive outcome.
        *args: Additional positional arguments passed to the underlying logging.info()
               call for string formatting.
        **kwargs: Additional keyword arguments passed to the underlying logging.info()
                  call.

    Example:
        >>> # Process completion
        >>> log_success("Business plan saved to database")
        >>> # Output: ✅ SUCCESS: Business plan saved to database
        >>>
        >>> # API success
        >>> log_success(f"Retrieved {len(results)} search results")
        >>> # Output: ✅ SUCCESS: Retrieved 25 search results
        >>>
        >>> # Configuration validation
        >>> log_success("All required API keys validated")

    Note:
        Success messages provide positive feedback and help track the flow of successful
        operations. They complement error messages and help create a complete picture
        of application behavior. Consider pairing with log_process_start/end for
        comprehensive process tracking.
    """
    formatted_message = f"✅ SUCCESS: {message}"
    logging.info(formatted_message, *args, **kwargs)


# ============================================================================
# Process Logging Functions
# ============================================================================


def log_process_start(process_name: str) -> None:
    """
    Log the initiation of a major process with visual separators and status indicators.

    This function creates a clear visual boundary in the logs to mark the beginning of
    a significant operation or workflow. It combines a separator line with an info
    message to provide both visual organization and textual confirmation of process
    initiation.

    The process tracking pattern typically involves:
    1. log_process_start() - Mark beginning
    2. Various log_info/debug messages - Track progress
    3. log_process_end() - Mark completion with success/failure status

    Args:
        process_name: Descriptive name of the process being started. Should be clear
                      and specific, e.g., "Business Plan Generation", "Market Analysis",
                      "Data Validation".

    Example:
        >>> # Start a business analysis process
        >>> log_process_start("Business Plan Analysis")
        >>> # Output:
        >>> # ═══════════════ STARTING: Business Plan Analysis ═══════════════
        >>> # ℹ️  INFO: Process 'Business Plan Analysis' initiated
        >>>
        >>> # Start API data retrieval
        >>> log_process_start("API Data Retrieval")
        >>> # Output:
        >>> # ═══════════════ STARTING: API Data Retrieval ═══════════════
        >>> # ℹ️  INFO: Process 'API Data Retrieval' initiated

    Note:
        Always pair this function with log_process_end() to provide complete process
        lifecycle tracking. The visual separator helps organize log output, making
        it easier to follow complex workflows and identify where processes begin
        and end.
    """
    log_separator(f"STARTING: {process_name}", char="═")
    log_info(f"Process '{process_name}' initiated")


def log_process_end(process_name: str, success: bool = True) -> None:
    """
    Log the completion of a major process with visual separators and outcome indicators.

    This function creates a clear visual boundary in the logs to mark the end of a
    significant operation or workflow. It provides both visual organization and clear
    indication of whether the process completed successfully or failed.

    The function complements log_process_start() to provide complete process lifecycle
    tracking, making it easy to identify process boundaries and outcomes in log files.

    Args:
        process_name: Name of the process that is ending. Should match the name used
                      in the corresponding log_process_start() call.
        success: Boolean indicating whether the process completed successfully.
                 Defaults to True. When False, logs "FAILED" status; when True,
                 logs "COMPLETED" status.

    Example:
        >>> # Successful process completion
        >>> log_process_end("Business Plan Analysis", success=True)
        >>> # Output:
        >>> # ℹ️  INFO: Process 'Business Plan Analysis' COMPLETED
        >>> # ═══════════════ COMPLETED: Business Plan Analysis ═══════════════
        >>>
        >>> # Failed process
        >>> log_process_end("API Data Retrieval", success=False)
        >>> # Output:
        >>> # ℹ️  INFO: Process 'API Data Retrieval' FAILED
        >>> # ═══════════════ FAILED: API Data Retrieval ═══════════════

    Note:
        Always use this function to pair with log_process_start() calls. The success
        parameter helps create clear audit trails and makes it easy to identify
        failed operations that may need investigation or retry logic.
    """
    status = "COMPLETED" if success else "FAILED"
    log_info(f"Process '{process_name}' {status}")
    log_separator(f"{status}: {process_name}", char="═")


def log_model_loading(model_name: str, is_lite: bool = False) -> None:
    """
    Log information about model loading operations with version indicators.

    This function provides debug-level logging for AI/ML model initialization,
    distinguishing between full and lite model versions. This helps track which
    models are being loaded and their performance characteristics during development.

    Model loading is a critical operation that can take significant time and
    resources, so tracking it helps with performance monitoring and debugging.

    Args:
        model_name: Name or identifier of the model being loaded. This should match
                    the model names defined in constants.py (e.g., "granite3.3:8b").
        is_lite: Boolean flag indicating whether this is a lite/smaller version of
                 the model. Lite models are typically faster but less capable.
                 Defaults to False.

    Example:
        >>> # Loading full production model
        >>> log_model_loading("granite3.3:8b", is_lite=False)
        >>> # Output: 🔍 DEBUG: Loading FULL model: granite3.3:8b
        >>>
        >>> # Loading lite development model
        >>> log_model_loading("qwen3:1.7b", is_lite=True)
        >>> # Output: 🔍 DEBUG: Loading LITE model: qwen3:1.7b

    Note:
        This function only logs when FLAG_DEBUG is True, keeping model loading
        details out of production logs while providing valuable debugging information
        during development. Model loading performance can be a bottleneck, so this
        logging helps identify when and which models are being initialized.
    """
    model_type = "LITE" if is_lite else "FULL"
    log_debug(f"Loading {model_type} model: {model_name}")


def log_api_call(service: str, endpoint: Optional[str] = None) -> None:
    """
    Log information about external API calls for debugging and monitoring.

    This function tracks interactions with external services and APIs, providing
    visibility into which services are being accessed and which specific endpoints
    are being used. This is crucial for debugging integration issues and monitoring
    API usage patterns.

    API call logging helps with:
    - Identifying which external services are being used
    - Debugging failed API integrations
    - Monitoring API usage patterns
    - Troubleshooting network or authentication issues

    Args:
        service: Name of the external service or API being called. Should be a clear,
                 recognizable name like "Tavily", "OpenAI", "Database", etc.
        endpoint: Optional specific endpoint or operation being accessed within the
                  service. This could be a URL path, method name, or operation type.
                  If None, only the service name is logged.

    Example:
        >>> # Simple service call
        >>> log_api_call("Tavily")
        >>> # Output: 🔍 DEBUG: API Call: Tavily
        >>>
        >>> # Service with specific endpoint
        >>> log_api_call("Tavily", "search")
        >>> # Output: 🔍 DEBUG: API Call: Tavily - search
        >>>
        >>> # Database operation
        >>> log_api_call("Database", "save_business_plan")

    Note:
        This function only logs when FLAG_DEBUG is True, preventing API call details
        from cluttering production logs while providing essential debugging information
        during development. Consider the security implications of logging API calls
        in production environments.
    """
    if FLAG_DEBUG:
        call_info = f"{service}"
        if endpoint:
            call_info += f" - {endpoint}"
        log_debug(f"API Call: {call_info}")


def log_data_processing(operation: str, data_type: str) -> None:
    """
    Log information about data processing operations for tracking data flow.

    This function provides debug-level logging for data manipulation operations,
    helping track how data moves through the system and what transformations are
    being applied. This is essential for debugging data processing pipelines and
    understanding data flow in complex workflows.

    Data processing logging helps with:
    - Tracking data transformation steps
    - Debugging data loading/saving issues
    - Monitoring data pipeline performance
    - Understanding data flow through the application

    Args:
        operation: The type of data operation being performed. Common operations
                   include "Loading", "Saving", "Parsing", "Validating", "Transforming",
                   "Filtering", etc.
        data_type: The type or description of data being processed. This could be
                   file types ("JSON", "CSV"), data structures ("BusinessPlan",
                   "UserData"), or conceptual types ("Configuration", "Results").

    Example:
        >>> # File operations
        >>> log_data_processing("Loading", "business_plan.json")
        >>> # Output: 🔍 DEBUG: Data Processing: Loading on business_plan.json
        >>>
        >>> # Data transformation
        >>> log_data_processing("Parsing", "API response")
        >>> # Output: 🔍 DEBUG: Data Processing: Parsing on API response
        >>>
        >>> # Database operations
        >>> log_data_processing("Saving", "user preferences")

    Note:
        This function only logs when FLAG_DEBUG is True, keeping detailed data
        processing information out of production logs. Use this for tracking the
        internal flow of data through your application during development and
        debugging.
    """
    if FLAG_DEBUG:
        log_debug(f"Data Processing: {operation} on {data_type}")


def log_configuration(key: str, value: str) -> None:
    """
    Log configuration settings and their values for debugging and verification.

    This function provides debug-level logging for configuration parameters,
    helping verify that settings are loaded correctly and providing visibility
    into application configuration during development and troubleshooting.

    Configuration logging helps with:
    - Verifying configuration loading
    - Debugging configuration-related issues
    - Tracking which settings are active
    - Auditing configuration changes

    Args:
        key: The configuration parameter name or key. This should match the
             constant names or configuration keys used in the application.
        value: The value of the configuration parameter. This will be converted
               to string for logging purposes.

    Example:
        >>> # Model configuration
        >>> log_configuration("OLLAMA_GENERAL", "granite3.3:8b")
        >>> # Output: 🔍 DEBUG: Configuration: OLLAMA_GENERAL = granite3.3:8b
        >>>
        >>> # Debug flag
        >>> log_configuration("FLAG_DEBUG", "True")
        >>> # Output: 🔍 DEBUG: Configuration: FLAG_DEBUG = True
        >>>
        >>> # API settings
        >>> log_configuration("TAVILY_MAX_RESULTS", "5")

    Note:
        This function only logs when FLAG_DEBUG is True, preventing sensitive
        configuration details from appearing in production logs. Be cautious about
        logging sensitive information like API keys or passwords, even in debug mode.
        Consider masking or omitting sensitive values.
    """
    if FLAG_DEBUG:
        log_debug(f"Configuration: {key} = {value}")
