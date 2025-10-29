"""
Model Collection Module

This module provides centralized model initialization and management for the
Business Plan Agents application. It serves as the single source of truth for
all AI model instances, ensuring consistent configuration and avoiding duplicate
model loading across the codebase.

The module initializes and exports pre-configured model instances that are used
throughout the application for different purposes:

Core Models:
- llm_general: General-purpose language model for analysis, reasoning, and text generation
- llm_reason: Specialized reasoning model for complex planning and decision-making tasks

Specialized Configurations:
- planner_llm: Business plan topic generation and strategic planning (uses reasoning model)
- expert_router_llm: Structured output model for routing decisions to appropriate experts

Model Characteristics:
- All models use consistent temperature settings from constants
- Models are initialized once at module import time for efficiency
- Structured output models use Pydantic schemas for type safety
- Models are configured for specific use cases to optimize performance

Important Notes:
- analyst_llm is NOT defined here to avoid circular imports with tool_collection
- Create analyst_llm by binding tools where needed: llm_general.bind_tools(TOOLS)
- Model instances are shared across the application - treat as read-only

Usage Patterns:
    # Import pre-configured models
    from src.utils.model_collection import llm_general, llm_reason, planner_llm

    # Use for general analysis
    response = llm_general.invoke("Analyze this business idea...")

    # Use for complex planning
    plan = planner_llm.invoke("Generate business plan topics...")

    # Create analyst with tools (when needed)
    from src.tools.tool_collection import TOOLS
    analyst_llm = llm_general.bind_tools(TOOLS)

Performance Considerations:
- Models are loaded once at startup to minimize initialization overhead
- Reuse model instances rather than creating new ones
- Consider model size and capabilities for specific use cases

Dependencies:
- src.states.state_collection: For ExpertDecision schema
- src.utils.constants: For model configuration constants
- src.utils.utils: For load_model utility function
"""

# Copilot: Do not add any logging for this file.

from src.states.state_collection import ExpertDecision
from src.utils.constants import MODEL_TEMPERATURE, MODEL_TYPE_GENERAL, MODEL_TYPE_REASON
from src.utils.utils import load_model

# Copilot: No need to add logging in this module.

# ============================================================================
# Model Initialization
# ============================================================================
# Initialize all base models at module load time
llm_general = load_model(MODEL_TYPE_GENERAL, MODEL_TEMPERATURE)
"""
General-purpose language model for analysis, reasoning, and text generation tasks.

This model is the workhorse of the application, handling most conversational AI
tasks including business analysis, content generation, and general reasoning.
It serves as the base model for creating tool-bound variants like analyst_llm.

Capabilities:
- Business analysis and insights
- Content generation and summarization
- Conversational responses
- General reasoning and problem-solving

Configuration:
- Uses MODEL_TYPE_GENERAL from constants (typically a capable general model)
- Temperature set to MODEL_TEMPERATURE for balanced creativity vs consistency
- Can be bound with tools to create specialized variants

Usage:
    from src.utils.model_collection import llm_general

    # Direct usage for analysis
    response = llm_general.invoke("Analyze this market opportunity...")

    # Create tool-bound variant
    analyst_llm = llm_general.bind_tools(TOOLS)
"""

llm_reason = load_model(MODEL_TYPE_REASON, MODEL_TEMPERATURE)
"""
Specialized reasoning model for complex planning and decision-making tasks.

This model excels at structured reasoning, planning, and analytical thinking.
It's used for tasks that require careful consideration, step-by-step reasoning,
and complex decision-making processes.

Capabilities:
- Complex planning and strategy development
- Multi-step reasoning and analysis
- Decision-making under uncertainty
- Structured problem-solving approaches

Configuration:
- Uses MODEL_TYPE_REASON from constants (typically a reasoning-focused model)
- Temperature set to MODEL_TEMPERATURE for analytical consistency
- Optimized for planning and analytical workflows

Usage:
    from src.utils.model_collection import llm_reason

    # Complex planning tasks
    plan = llm_reason.invoke("Develop a 5-year business strategy...")
"""

# ============================================================================
# Specialized Model Configurations
# ============================================================================
# Planner uses the reasoning model
planner_llm = llm_reason
"""
Business plan topic generation and strategic planning model.

This is a specialized configuration of the reasoning model, optimized for generating
comprehensive business plan topics and strategic planning activities. It leverages
the reasoning model's analytical capabilities for structured business planning.

Capabilities:
- Business plan topic generation
- Strategic planning and roadmapping
- Market analysis and opportunity identification
- Competitive positioning and SWOT analysis

Configuration:
- Based on llm_reason for enhanced analytical capabilities
- Same temperature and reasoning characteristics as the base model
- Focused on business planning domain expertise

Usage:
    from src.utils.model_collection import planner_llm

    # Generate business plan topics
    topics = planner_llm.invoke("Generate topics for a restaurant business plan...")

Note:
    This is an alias to llm_reason - they share the same model instance for efficiency.
"""

expert_router_llm = llm_general.with_structured_output(ExpertDecision)
"""
Structured output model for intelligent expert routing decisions.

This model is configured to provide structured outputs using the ExpertDecision
Pydantic schema, enabling type-safe and predictable expert routing decisions.
It analyzes business topics and determines which expert should handle each analysis.

Capabilities:
- Topic analysis and categorization
- Expert domain matching
- Structured decision outputs
- Type-safe routing recommendations

Configuration:
- Based on llm_general with structured output binding
- Uses ExpertDecision schema for consistent output format
- Ensures routing decisions follow predefined expert domains

Usage:
    from src.utils.model_collection import expert_router_llm

    # Route a business topic to appropriate expert
    decision = expert_router_llm.invoke("Analyze our marketing strategy...")
    # Returns: ExpertDecision object with expert_type and reasoning

Output Schema:
    The model returns ExpertDecision objects containing:
    - expert_type: Selected expert domain
    - reasoning: Explanation for the routing decision
"""
