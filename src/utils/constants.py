"""
Constants Configuration Module

This module defines all configuration constants used across the Business Plan Agents application.
Constants are organized by category for easy maintenance and clear structure, providing a single
source of truth for application configuration.

Categories:
- Model Configurations: LLM model names, types, and settings for different use cases
- File Paths: Standard paths for configuration files, data storage, and reports
- API Keys: Named references to API keys stored in external configuration files
- Business Defaults: Default values for business type, location, and quality thresholds
- Tool Configurations: Settings for external tools and services (Tavily search, etc.)
- Debug and Logging: Debug flags, logging levels, and visual formatting settings
- Expert System Constants: Domain expert definitions and routing configurations

Key Features:
- Environment-aware configuration (debug vs production modes)
- Comprehensive expert system with 12 specialized domains
- Flexible model selection (full-sized vs lite models for development)
- Structured file organization with clear naming conventions

Usage:
    from src.utils.constants import (
        FLAG_DEBUG, DEFAULT_BUSINESS_TYPE, EXPERTS, ExpertType
    )

    # Check debug mode
    if FLAG_DEBUG:
        print(f"Debug mode enabled for {DEFAULT_BUSINESS_TYPE}")

    # Access expert definitions
    business_expert = EXPERTS["business_analyst"]
    print(f"Business expert handles: {business_expert}")

    # Use enum for type safety
    expert_type = ExpertType.BUSINESS_ANALYST

Best Practices:
- All constants use UPPERCASE with underscores for clarity
- Related constants are grouped with descriptive headers
- Constants include inline comments for non-obvious values
- Enum values match EXPERTS dictionary keys for consistency
- Debug flags control lite model usage and verbose logging

Current Defaults:
- DEFAULT_BUSINESS_TYPE: "Falooda" (demonstration business)
- DEFAULT_LOCATION: "Sweden" (demonstration location)
- FLAG_DEBUG: True (development mode with lite models)
- QUALITY_THRESHOLD: 7 (minimum score for planning completion)
"""

# Copilot: Do not add any logging for this file.

from enum import Enum

# Constants for the Business Plan Agents project

# ============================================================================
# Model Configurations
# ============================================================================
# Full-sized models for production use
OLLAMA_GENERAL = "granite3.3:8b"  # General-purpose language model
OLLAMA_REASON = "qwen3:8b"  # Reasoning and planning model

# Lite models for development/debugging (faster but less capable)
OLLAMA_GENERAL_LITE = "granite3.3:2b"
OLLAMA_REASON_LITE = "qwen3:1.7b"

# Model temperature (0 = deterministic, 1 = creative)
MODEL_TEMPERATURE = 0

# Model type identifiers
MODEL_TYPE_GENERAL = "general"
MODEL_TYPE_REASON = "reason"

# ============================================================================
# File Paths
# ============================================================================
META_DATA_FILE_PATH = "./meta_data.json"  # API keys and configuration
BUSINESS_PLAN_JSON_PATH = "./business_plan.json"  # Generated business plan topics
REPORTS_DIR = "reports"  # Directory for generated reports

# ============================================================================
# API Keys
# ============================================================================
TAVILY_API_KEY_NAME = "tavily"  # Key name in meta_data.json

# ============================================================================
# Business Defaults
# ============================================================================
DEFAULT_BUSINESS_TYPE = "Falooda"  # Default business type for examples
DEFAULT_LOCATION = "Sweden"  # Default location for business analysis
QUALITY_THRESHOLD = 9  # Minimum quality score to end planning iteration

# ============================================================================
# Tool Configurations
# ============================================================================
TAVILY_MAX_RESULTS = 5  # Maximum search results from Tavily
TAVILY_TOPIC = "general"  # Tavily search topic category

# ============================================================================
# Debug and Logging
# ============================================================================
FLAG_DEBUG = True  # Enable debug mode (uses lite models, extra logging)
FLAG_USE_DUMMY_DATA = False  # Use dummy data instead of real API calls

LOG_LEVEL = "DEBUG" if FLAG_DEBUG else "INFO"  # Logging level
DUMMY_DATA_RESPONSE = "Dummy Data"  # Response when using dummy data

# ============================================================================
# Logging Visual Settings
# ============================================================================
LOG_SEPARATOR_CHAR = "═"  # Character for log separators
LOG_SEPARATOR_LENGTH = 60  # Length of log separator lines

# ============================================================================
# Tool Messages (deprecated - prefer using log_* functions)
# ============================================================================
TOOL_CALL_SEPARATOR = "*" * 25
KNOWLEDGE_TOOL_MESSAGE = "Tool call : Knowledge Tool"
WEB_TOOL_MESSAGE = "Tool call : Web Tool"

# ============================================================================
# Expert System Constants
# ============================================================================
# Expert definitions for the business plan routing system
EXPERTS = {
    "technology_expert": "software, AI, hardware, IT topics",
    "legal_expert": "laws, contracts, compliance",
    "business_analyst": "strategy, planning, business processes",
    "hr_expert": "hiring, policies, training",
    "competitive_intelligence_expert": "market research, competitor analysis",
    "financial_expert": "investments, accounting, budgeting",
    "operations_expert": "logistics, workflows",
    "sales_expert": "sales strategies, CRM",
    "marketing_expert": "campaigns, branding",
    "customer_experience_expert": "customer satisfaction, support strategy",
    "public_relations_expert": "media, reputation, communications",
    "supply_chain_expert": "procurement, inventory, logistics",
}


# Define ExpertType enum explicitly for mypy compatibility
class ExpertType(Enum):
    """
    Enumeration of available expert types for business plan analysis routing.

    This enum defines all possible expert domains that can be assigned to analyze
    specific business plan topics. Each expert type corresponds to a specialized
    knowledge domain and is used by the routing system to direct queries to the
    most appropriate analytical expertise.

    The enum values match the keys in the EXPERTS dictionary, ensuring consistency
    between the routing system and expert capability definitions.

    Attributes:
        BUSINESS_ANALYST: Strategy, planning, and business process analysis
        COMPETITIVE_INTELLIGENCE_EXPERT: Market research and competitor analysis
        CUSTOMER_EXPERIENCE_EXPERT: Customer satisfaction and support strategy
        FINANCIAL_EXPERT: Investments, accounting, and budgeting
        HR_EXPERT: Hiring, policies, and training
        LEGAL_EXPERT: Laws, contracts, and compliance
        MARKETING_EXPERT: Campaigns, branding, and promotional activities
        OPERATIONS_EXPERT: Logistics, workflows, and operational efficiency
        PUBLIC_RELATIONS_EXPERT: Media relations, reputation, and communications
        SALES_EXPERT: Sales strategies, CRM, and revenue generation
        SUPPLY_CHAIN_EXPERT: Procurement, inventory, and logistics management
        TECHNOLOGY_EXPERT: Software, AI, hardware, and IT infrastructure

    Example:
        >>> # Use enum for type-safe expert selection
        >>> expert = ExpertType.BUSINESS_ANALYST
        >>> print(f"Selected expert: {expert.value}")
        >>> print(f"Expert description: {EXPERTS[expert.value]}")
        >>>
        >>> # Iterate through all experts
        >>> for expert_type in ExpertType:
        ...     print(f"{expert_type.name}: {expert_type.value}")
    """

    BUSINESS_ANALYST = "business_analyst"
    COMPETITIVE_INTELLIGENCE_EXPERT = "competitive_intelligence_expert"
    CUSTOMER_EXPERIENCE_EXPERT = "customer_experience_expert"
    FINANCIAL_EXPERT = "financial_expert"
    HR_EXPERT = "hr_expert"
    LEGAL_EXPERT = "legal_expert"
    MARKETING_EXPERT = "marketing_expert"
    OPERATIONS_EXPERT = "operations_expert"
    PUBLIC_RELATIONS_EXPERT = "public_relations_expert"
    SALES_EXPERT = "sales_expert"
    SUPPLY_CHAIN_EXPERT = "supply_chain_expert"
    TECHNOLOGY_EXPERT = "technology_expert"
