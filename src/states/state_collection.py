"""
State Collection Module

This module defines all Pydantic state models used in the Business Plan Agents application.
States manage data flow, validation, and structure for different agent workflows including
topic planning, business analysis, and expert routing.

The module provides five core state models:

1. Topic: Represents individual business plan topics with descriptions and subtopics
2. PlanningResponse: Structured output for AI-generated business plan topic collections
3. BusinessAnalysisState: Main workflow state for business analysis operations
4. ExpertDecision: Routing decisions for expert-based query handling
5. Feedback: Structured output for planning feedback and critique

Key Features:
- Pydantic validation with custom field validators
- Structured data models with comprehensive type hints
- Computed properties for workflow state management
- JSON serialization/deserialization support
- Integration with LangChain message types
- Comprehensive error handling and validation

State Management:
- BusinessAnalysisState serves as the central state for analysis workflows
- Supports both single-topic and multi-topic analysis patterns
- Tracks processing progress through indexed topic lists
- Manages raw information collection and report synthesis
- Handles message passing between workflow nodes

Validation & Safety:
- Field validators ensure data integrity (non-empty topics, etc.)
- Type-safe property access with proper error handling
- Model validators for complex cross-field validation
- Optional fields with sensible defaults

Dependencies:
- pydantic: For data validation and serialization
- langchain_core: For message type integration
- langgraph: For workflow state management
- Internal modules: Constants and logging utilities

Usage:
    from src.states.state_collection import BusinessAnalysisState, Topic, PlanningResponse

    # Create analysis state
    state = BusinessAnalysisState(
        business_type="Falooda",
        location="Sweden",
        topics_list=[Topic(topic="Marketing Strategy", subtopics=["Digital marketing", "Brand development"])]
    )

    # Access computed properties
    processed = state.processed_topics
    current = state.current_topic
    remaining = state.non_processed_topics

File Operations:
- States support JSON save/load operations
- Text report generation with structured output
- Path-based file management with automatic directory creation
"""

# Copilot: Do not add any logging for this file.

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator, model_validator

from src.utils.constants import ExpertType
from src.utils.logging import log_error, log_info, log_success


class Topic(BaseModel):
    """
    Represents a business plan topic with optional description and subtopics.

    This class captures a structured overview of a topic in a business plan.
    Each topic can have an optional description and a list of subtopics as bullet points.

    ## Attributes:
        topic (str): The main topic title.
        reason (str): Reason why this topic needs to be researched.
        subtopics (List[str]): List of bullet points detailing aspects of the topic.
            Recommended to have at least 10 points for completeness.
        report (Optional[str]): Optional detailed report or analysis for the topic.

    ## Example (JSON)
    ```json
    {
        "topic": "Marketing Strategy",
        "reason": "Outline of key marketing initiatives for the upcoming year.",
        "subtopics": [
            "Define target audience segments",
            "Conduct competitive analysis",
            "Develop brand messaging",
            "Plan content marketing calendar",
            "Leverage social media campaigns",
            "Email marketing campaigns",
            "Influencer collaborations",
            "SEO and website optimization",
            "Paid advertising strategy",
            "Measure KPIs and adjust campaigns"
        ]
    }
    ```

    ## Typical Use Cases:
    - Creating structured business plan documents.
    - Organizing presentations or reports for stakeholders or investors.
    - Outlining departmental strategies (marketing, operations, finance, etc.).
    - Capturing actionable checklists for project planning.

    ## Guidelines:
    - Provide a meaningful topic.
    - Include a reason if context is necessary.
    - Subtopics should be concise and actionable.
    - Aim for at least 10 subtopics for thorough coverage.
    - Subtopics are strings only.
    """

    topic: str = Field(
        ...,
        description="Business plan topic title (e.g., 'Marketing Strategy', 'Financial Projections')",
    )
    reason: str = Field(
        ..., description="Justification for including this topic in the business plan"
    )
    subtopics: List[str] = Field(
        default_factory=list,
        description="Specific subtopics to research and analyze (aim for at least 10 items)",
    )

    report: Optional[str] = Field(
        None, description="Completed analysis report for this topic"
    )

    @field_validator("topic")
    @classmethod
    def topic_must_not_be_empty(cls, v):
        """
        Validate that the topic field is not empty or whitespace-only.

        This validator ensures that all Topic instances have meaningful,
        non-empty topic titles. It strips whitespace and checks for content,
        raising an error if the topic would be effectively empty.

        Args:
            v (str): The topic value to validate, as provided during model creation.

        Returns:
            str: The validated and stripped topic string.

        Raises:
            ValueError: If the topic is None, empty, or contains only whitespace
                characters. Error message: "Topic must not be empty".

        Example:
            >>> # Valid topic
            >>> topic = Topic(topic="Market Analysis", subtopics=[])
            >>> topic.topic
            'Market Analysis'
            >>>
            >>> # Invalid - would raise ValueError
            >>> # Topic(topic="   ", subtopics=[])  # Raises: Topic must not be empty
        """
        if not v or not v.strip():
            raise ValueError("Topic must not be empty")
        return v

    def print_structured(self, indent: int = 0):
        """
        Print the topic and its subtopics in a readable, indented format.

        This method provides a hierarchical visualization of the topic structure,
        displaying the main topic, its reason (if available), and all subtopics
        with proper indentation. Useful for debugging, logging, and user display.

        Args:
            indent (int): Number of spaces to indent the entire output block.
                Useful for nesting within larger structures. Defaults to 0.

        Returns:
            None: This method prints directly to the logging system and doesn't
                return a value.

        Note:
            Output is sent to the logging system using log_info(), not printed
            to stdout. The reason field is only displayed if it exists and is
            not empty. Subtopics are displayed as bullet points under the topic.

        Example:
            >>> topic = Topic(
            ...     topic="Market Analysis",
            ...     reason="Analyze target market and competition",
            ...     subtopics=["Customer demographics", "Competitor analysis"]
            ... )
            >>> topic.print_structured()  # Logs formatted output
            Market Analysis
              Analyze target market and competition
              - Customer demographics
              - Competitor analysis
            >>>
            >>> topic.print_structured(indent=4)  # Indented output
                Market Analysis
                  Analyze target market and competition
                  - Customer demographics
                  - Competitor analysis
        """
        space = " " * indent
        log_info(f"{space}{self.topic}")
        if self.reason:
            log_info(f"{space}  {self.reason}")
        for sub in self.subtopics:
            log_info(f"{space}  - {sub}")


class PlanningResponse(BaseModel):
    """
    Represents a collection of business plan topics.

    This class organizes multiple top-level business plan topics.
    Each topic is a `Topic` object with optional description and subtopics.
    Supports structured printing, JSON saving, and loading.

    ## Attributes:
        topics (List[Topic]): List of top-level business plan topics.

    ## Example (JSON)
    ```json
    {
        "topics": [
            {
                "topic": "Marketing Strategy",
                "reason": "Outline of key marketing initiatives for the upcoming year.",
                "subtopics": [
                    "Define target audience segments",
                    "Conduct competitive analysis",
                    "Develop brand messaging",
                    "Plan content marketing calendar",
                    "Leverage social media campaigns",
                    "Email marketing campaigns",
                    "Influencer collaborations",
                    "SEO and website optimization",
                    "Paid advertising strategy",
                    "Measure KPIs and adjust campaigns"
                ]
            },
            {
                "topic": "Operations Plan",
                "reason": "Overview of operational processes and efficiency measures.",
                "subtopics": [
                    "Supply chain management",
                    "Inventory tracking",
                    "Staffing and HR policies",
                    "Workflow optimization",
                    "Technology and software integration",
                    "Quality control measures",
                    "Vendor management",
                    "Production scheduling",
                    "Cost reduction initiatives",
                    "Reporting and analytics"
                ]
            }
        ]
    }
    ```

    ## Typical Use Cases:
    - Representing full business plan structures.
    - Organizing departmental strategies into a single document.
    - Exporting plans for stakeholder review or presentations.
    - Storing reusable templates for multiple projects.

    ## Guidelines:
    - Each topic should be a well-defined `Topic` object.
    - Subtopics should be clear, concise, and actionable.
    - Maintain consistent format for readability.
    - Use `print_structured()` to visualize the plan.
    - Use `save_json()` and `load_json()` to persist or reload plans.
    """

    topics: List[Topic] = Field(
        default_factory=list,
        description="Complete list of business plan topics generated by the AI planner",
    )

    @model_validator(mode="after")
    def validate_topics(self) -> "PlanningResponse":
        """
        Validate that the topics collection meets business plan requirements.

        This method performs comprehensive validation of the PlanningResponse
        to ensure it contains a valid business plan structure. It checks for
        required topics, validates topic content, and ensures subtopic lists
        are properly formed.

        Validation checks performed:
        1. At least one topic must be present in the collection
        2. Each topic must have a non-empty, non-whitespace topic title
        3. Subtopic lists must either be None or contain at least one item
        4. All topic objects must be properly formed Topic instances

        Returns:
            PlanningResponse: The same instance after successful validation,
                allowing for method chaining.

        Raises:
            ValueError: If any validation check fails. Specific error messages
                indicate which requirement was not met:
                - "Business plan must contain at least one topic"
                - "Topic at index {i} must have a non-empty topic"
                - "Topic '{topic}' has empty subtopics list - use None if no subtopics"

        Note:
            This method is typically called automatically during model creation
            or can be invoked manually to validate existing instances. It does
            not modify the data, only validates it.

        Example:
            >>> # Valid planning response
            >>> topics = [Topic(topic="Market Analysis", subtopics=["Research", "Analysis"])]
            >>> plan = PlanningResponse(topics=topics)
            >>> validated = plan.validate_topics()  # Returns self if valid
            >>>
            >>> # Invalid - empty topics list
            >>> empty_plan = PlanningResponse(topics=[])
            >>> empty_plan.validate_topics()  # Raises: Business plan must contain at least one topic
        """
        if not self.topics:
            raise ValueError("Business plan must contain at least one topic")
        for i, topic in enumerate(self.topics):
            if not topic.topic or not topic.topic.strip():
                raise ValueError(f"Topic at index {i} must have a non-empty topic")
            if topic.subtopics and len(topic.subtopics) == 0:
                raise ValueError(
                    f"Topic '{topic.topic}' has empty subtopics list - use None if no subtopics"
                )
        return self

    def print_structured(self):
        """
        Print all topics and their subtopics in a structured, readable format.

        This method provides a comprehensive visualization of the entire business
        plan structure, displaying each topic with its subtopics in a clean,
        hierarchical format. Topics are separated by blank lines for clarity.

        Returns:
            None: This method prints directly to the logging system and doesn't
                return a value.

        Note:
            Output is sent to the logging system using log_info(), not printed
            to stdout. Each topic is displayed using its own print_structured()
            method, followed by a blank line for visual separation.

        Example:
            >>> topics = [
            ...     Topic(topic="Market Analysis", subtopics=["Research", "Analysis"]),
            ...     Topic(topic="Operations", subtopics=["Planning", "Execution"])
            ... ]
            >>> plan = PlanningResponse(topics=topics)
            >>> plan.print_structured()  # Logs formatted output
            Market Analysis
              - Research
              - Analysis
            <BLANKLINE>
            Operations
              - Planning
              - Execution
            <BLANKLINE>
        """
        for topic in self.topics:
            topic.print_structured()
            log_info("")  # spacing between topics

    def save_json(self, filepath: str):
        """
        Save the entire topics collection to a JSON file with proper formatting.

        This method serializes the PlanningResponse instance to JSON format and
        saves it to the specified file path. It automatically creates parent
        directories if they don't exist and uses proper formatting for readability.

        Args:
            filepath (str): File system path where the JSON file should be saved.
                Can be absolute or relative. Parent directories are created
                automatically if they don't exist.

        Returns:
            None: This method saves to disk and doesn't return a value.

        Raises:
            OSError: If the file cannot be written due to permission issues,
                disk space, or other file system errors.
            ValueError: If the filepath is invalid or the model serialization fails.

        Note:
            The JSON is saved with:
            - UTF-8 encoding for proper character support
            - 4-space indentation for readability
            - ensure_ascii=False to preserve non-ASCII characters
            - Full model serialization including all nested Topic objects

        Example:
            >>> topics = [Topic(topic="Market Analysis", subtopics=["Research"])]
            >>> plan = PlanningResponse(topics=topics)
            >>> plan.save_json("business_plan.json")
            >>> # File saved to business_plan.json with proper formatting
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(), ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        log_success(f"Saved JSON to: {path.resolve()}")

    @classmethod
    def load_json(cls, filepath: str) -> PlanningResponse:
        """
        Load a topics collection from a JSON file.

        This class method reads and parses a JSON file containing business plan
        topics, validates the structure, and creates a properly typed PlanningResponse
        instance. The loaded data is validated against the Pydantic model schema.

        Args:
            filepath (str): File system path to the JSON file to load. Can be
                absolute or relative path.

        Returns:
            PlanningResponse: A new PlanningResponse instance populated with
                the data from the JSON file, including all Topic objects and
                their subtopics.

        Raises:
            FileNotFoundError: If the specified file does not exist at the given path.
            ValueError: If the JSON data is malformed or doesn't match the expected
                schema. This includes invalid JSON syntax or missing required fields.
            OSError: If the file cannot be read due to permission issues or other
                file system errors.

        Note:
            The loaded instance automatically runs validation via the model
            constructor, ensuring data integrity. The JSON file should have
            been created using the save_json() method for guaranteed compatibility.
            Comprehensive error handling provides detailed logging for debugging.

        Example:
            >>> # Load from previously saved file
            >>> plan = PlanningResponse.load_json("business_plan.json")
            >>> print(f"Loaded {len(plan.topics)} topics")
            >>>
            >>> # File not found example
            >>> try:
            ...     plan = PlanningResponse.load_json("nonexistent.json")
            ... except FileNotFoundError as e:
            ...     print(f"Error: {e}")
        """
        path = Path(filepath)
        try:
            if not path.exists():
                raise FileNotFoundError(
                    f"Business plan JSON file not found at {filepath}"
                )
            data = json.loads(path.read_text(encoding="utf-8"))
            log_success(f"Loaded JSON from: {path.resolve()}")
            return cls(**data)
        except FileNotFoundError:
            log_error(f"Business plan JSON file not found at {filepath}")
            raise
        except json.JSONDecodeError as e:
            log_error(f"Invalid JSON in file {filepath}: {str(e)}")
            raise ValueError(f"Invalid JSON format in {filepath}: {str(e)}")
        except Exception as e:
            log_error(f"Failed to load business plan data from {filepath}: {str(e)}")
            raise


class FeedbackResponse(BaseModel):
    """
    Structured output for planning feedback and critique in iterative business plan development.

    This model represents comprehensive AI-generated feedback on business plan topics, enabling
    iterative improvement through systematic evaluation and enhancement. It captures multi-dimensional
    assessment including qualitative evaluation, quantitative scoring, and actionable recommendations
    to refine business plan quality until it meets established thresholds.

    The feedback system supports the smart planning workflow by:
    - Evaluating topic quality against business-specific and location-specific criteria
    - Identifying strengths to preserve and weaknesses to address
    - Providing concrete suggestions for improvement
    - Offering strategic insights for enhanced competitiveness
    - Assigning numerical scores for automated quality gating

    ## Attributes:
        assessment (str): Comprehensive qualitative evaluation covering strategic alignment,
            market relevance, localization depth, functional completeness, research actionability,
            implementation feasibility, and risk coverage. Provides detailed rationale for
            both strengths and areas requiring improvement.
        score (int): Overall quality score (1-10) based on evaluation criteria:
            - 1-3: Poor (major gaps, minimal localization, vague content)
            - 4-6: Adequate (basic coverage, partial localization, some actionable items)
            - 7-8: Good (comprehensive coverage, well-localized, mostly actionable)
            - 9-10: Excellent (strategic depth, fully localized, highly actionable with risk awareness)
        strength_list (List[str]): Specific strengths identified in the current topics,
            such as strategic insights, localization quality, research depth, or innovative approaches.
        weakness_list (List[str]): Critical gaps and weaknesses requiring attention,
            including missing content, vague descriptions, inadequate localization, or lack of specificity.
        suggestion_list (List[str]): Concrete, actionable recommendations to address weaknesses,
            providing specific steps for improvement and enhancement.
        recommendation_list (List[str]): Strategic insights and optional enhancements extending
            beyond immediate weaknesses, offering competitive advantages and future considerations.
    """

    model_config = {"populate_by_name": True}

    assessment: str = Field(
        ...,
        description=(
            "Comprehensive qualitative evaluation of the business plan topics and subtopics."
        ),
    )

    strength_list: List[str] = Field(
        default_factory=list,
        description=(
            "List of identified strengths within the topics and subtopics matching the business context."
        ),
    )

    weakness_list: List[str] = Field(
        default_factory=list,
        description=(
            "List of identified weaknesses or gaps within the topics and subtopics matching the business context."
        ),
    )

    suggestion_list: List[str] = Field(
        default_factory=list,
        description=(
            "List of actionable recommendations to address identified weaknesses and enhance the overall quality, depth, and feasibility of the business plan topics and subtopics."
        ),
    )

    recommendation_list: List[str] = Field(
        default_factory=list,
        description=(
            "List of strategic insights and optional enhancements that extend beyond immediate weaknesses."
        ),
    )
    score: float = Field(
        1.0,
        description=(
            "Overall quality score (1.0–10.0) derived from all the feedback components — assessment, strength_list, weakness_list, suggestion_list, and recommendation_list. Poor (1.0–3.0), Adequate (4.0–6.0), Good (7.0–8.0), Excellent (9.0–10.0)."
            "If sum of length of weakness_list, suggestion_list and recommendation_list is more than 10, score should be lower."
        ),
        ge=1.0,
        le=10.0,
    )


class PlanningState(BaseModel):
    """
    State for the iterative business plan topic generation workflow.

    Tracks the planning process through multiple iterations until quality threshold is met.
    """

    business: str = Field(
        ...,
        description="Type of business being planned (e.g., 'Falooda shop', 'SaaS startup')",
    )
    location: str = Field(
        ...,
        description="Geographic location for the business (e.g., 'Sweden', 'New York')",
    )
    topics: List[Topic] = Field(
        default_factory=list,
        description="Current version of generated business plan topics",
    )
    feedback: Optional[FeedbackResponse] = Field(
        None,
        description="AI critique of current topics (triggers iterative improvement)",
    )


class BusinessAnalysisState(BaseModel):
    """
    Represents the state of a business analysis workflow.

    This unified state class supports both:
    1. Multi-topic analysis workflows (smart_agent)
    2. Single-topic subtopic analysis workflows (simple_agent)

    ## Attributes:
        **location** (str): The location where the business is located.
        **business_type** (str): The type of business being analyzed.
        **topics_list** (List[Topic]): Single list containing all topics in processing order.
        **current_topic_index** (int): Index of the currently active topic in topics_list (-1 if none).
        **description** (Optional[str]): Optional description for the current analysis.
        **raw_information** (List[str]): Raw information collected for the current topic.
        **messages** (List[BaseMessage] | List[str]): Messages exchanged during analysis.
        **report** (str): Detailed report synthesizing insights.

    ## Guidelines:
        - Use topics_list to store all topics in the order they should be processed
        - Use current_topic_index to track which topic is currently being analyzed
        - Use processed_topics property to get topics that have been completed
        - Use non_processed_topics property to get topics that remain to be processed
        - Use current_topic property to get the currently active topic
        - Ensure **raw_information** corresponds to processed items
        - Use **messages** to track communication and progress.
    """

    location: str = Field(
        ..., description="Business location (e.g., 'Sweden', 'California')"
    )
    business_type: str = Field(
        ...,
        description="Type of business being analyzed (e.g., 'Falooda shop', 'Tech startup')",
    )
    topics_list: List[Topic] = Field(
        default_factory=list,
        description="All business plan topics to analyze, in processing order",
    )
    current_topic_index: int = Field(
        default=-1,
        description="Index of topic currently being analyzed (-1 means no active topic)",
    )
    description: Optional[str] = Field(
        None, description="Optional context or notes about this analysis session"
    )
    raw_information: List[str] = Field(
        default_factory=list,
        description="Research data and web search results for the current topic",
    )
    messages: Annotated[list[BaseMessage], add_messages] | List[str] = Field(
        default_factory=list,
        description="Agent conversation history and workflow messages",
    )
    report: str = Field(
        "",
        description="Final synthesized report combining all analyzed topics",
    )

    @property
    def processed_topics(self) -> List[Topic]:
        """
        Get the list of topics that have been processed so far.

        This computed property returns a slice of the topics_list containing
        all topics that have been completed in the analysis workflow. Topics
        are considered processed when they appear before the current_topic_index.

        Returns:
            List[Topic]: Topics that have been fully analyzed and have their
                reports populated. Returns an empty list if no topics have been
                processed yet (current_topic_index < 0).

        Note:
            The property is computed dynamically based on current_topic_index.
            Topics at indices 0 through current_topic_index-1 are considered processed.
            This property is read-only and cannot be set directly.

        Example:
            >>> state = BusinessAnalysisState(
            ...     topics_list=[topic1, topic2, topic3],
            ...     current_topic_index=1
            ... )
            >>> processed = state.processed_topics
            >>> len(processed)  # Returns 1 (topic1 is processed)
            >>> processed[0].topic  # Returns topic1.topic
        """
        if self.current_topic_index >= 0:
            return self.topics_list[: self.current_topic_index]
        return []

    @property
    def non_processed_topics(self) -> List[Topic]:
        """
        Get the list of topics that remain to be processed.

        This computed property returns a slice of the topics_list containing
        all topics that still need to be analyzed in the workflow. Topics are
        considered unprocessed when they appear at or after the current_topic_index.

        Returns:
            List[Topic]: Topics that still need analysis and report generation.
                Returns an empty list if all topics have been processed or if
                current_topic_index is out of bounds.

        Note:
            The property is computed dynamically based on current_topic_index.
            Topics at indices current_topic_index+1 and beyond are unprocessed.
            This property is read-only and cannot be set directly.

        Example:
            >>> state = BusinessAnalysisState(
            ...     topics_list=[topic1, topic2, topic3],
            ...     current_topic_index=1
            ... )
            >>> remaining = state.non_processed_topics
            >>> len(remaining)  # Returns 1 (topic3 is unprocessed)
            >>> remaining[0].topic  # Returns topic3.topic
        """
        if self.current_topic_index + 1 < len(self.topics_list):
            return self.topics_list[self.current_topic_index + 1 :]
        return []

    @property
    def current_topic(self) -> Optional[Topic]:
        """
        Get the currently active topic being analyzed.

        This computed property returns the Topic object at the current_topic_index
        if the index is within valid bounds, otherwise returns None. This represents
        the topic currently being processed by the analysis workflow.

        Returns:
            Optional[Topic]: The currently active topic for analysis, or None if
                no topic is currently active (index out of bounds or negative).

        Note:
            The property is computed dynamically based on current_topic_index.
            Bounds checking ensures safe access to the topics_list.
            This property is read-only and cannot be set directly.

        Example:
            >>> state = BusinessAnalysisState(
            ...     topics_list=[topic1, topic2, topic3],
            ...     current_topic_index=1
            ... )
            >>> current = state.current_topic
            >>> current.topic  # Returns topic2.topic
            >>>
            >>> # Out of bounds
            >>> state.current_topic_index = 5
            >>> state.current_topic  # Returns None
        """
        if 0 <= self.current_topic_index < len(self.topics_list):
            return self.topics_list[self.current_topic_index]
        return None

    @property
    def topic(self) -> Optional[Topic]:
        """
        Alias for current_topic property for backward compatibility.

        This property provides an alternative name for current_topic to maintain
        compatibility with existing code that may use the shorter 'topic' name.
        It returns the exact same value as current_topic.

        Returns:
            Optional[Topic]: The currently active topic for analysis, identical
                to current_topic property. Returns None if no topic is active.

        Note:
            This is a read-only alias property. Use current_topic_index to change
            which topic is active. This property exists for API compatibility.

        Example:
            >>> state = BusinessAnalysisState(
            ...     topics_list=[topic1, topic2],
            ...     current_topic_index=0
            ... )
            >>> state.topic.topic  # Same as state.current_topic.topic
            >>> state.topic is state.current_topic  # Returns True
        """
        return self.current_topic

    def print_structured(self, indent: int = 0):
        """
        Print the business analysis details in a readable, indented format.

        This method provides a comprehensive overview of the business analysis
        state, displaying key information for debugging and monitoring the
        analysis workflow. It shows business context, current processing status,
        completed work, and system state.

        Args:
            indent (int): Number of spaces to indent the entire output block.
                Useful for nesting within larger structures. Defaults to 0.

        Returns:
            None: This method prints directly to the logging system and doesn't
                return a value.

        Note:
            Output is sent to the logging system using log_info(), not printed
            to stdout. Reports are truncated to 100 characters for readability.
            The method provides a snapshot of the analysis state at the time
            of calling.

        Example:
            >>> state = BusinessAnalysisState(
            ...     business_type="Falooda",
            ...     location="Sweden",
            ...     current_topic_index=1,
            ...     topics_list=[topic1, topic2, topic3]
            ... )
            >>> state.print_structured()  # Logs formatted state information
            Business Type: Falooda
            Location: Sweden
            Current Topic: topic2.topic
              topic2.reason
            Topics Processed:
              - topic1.topic
                Description: topic1.reason
                Report: [first 100 chars of report]...
            Messages: 5 items
        """
        space = " " * indent
        log_info(f"{space}Business Type: {self.business_type}")
        log_info(f"{space}Location: {self.location}")

        if self.current_topic:
            log_info(f"{space}Current Topic: {self.current_topic.topic}")
            if self.current_topic.reason:
                log_info(f"{space}  {self.current_topic.reason}")

        if self.processed_topics:
            log_info(f"{space}Topics Processed:")
            for i, topic in enumerate(self.processed_topics):
                log_info(f"{space}  - {topic.topic}")
                if topic.reason:
                    log_info(f"{space}    Description: {topic.reason}")
                if i < len(self.raw_information):
                    report = self.raw_information[i]
                    log_info(
                        f"{space}    Report: {report[:100]}{'...' if len(report) > 100 else ''}"
                    )

        if self.messages:
            log_info(f"{space}Messages: {len(self.messages)} items")

    def save_text(self, filepath: str):
        """
        Save the business analysis details to a structured text file.

        This method creates a comprehensive, human-readable text report containing
        all aspects of the business analysis state. The report includes business
        context, executive summary, detailed topic breakdowns, and processing status.

        Args:
            filepath (str): File system path where the text file should be saved.
                Can be absolute or relative. Parent directories are created
                automatically if they don't exist.

        Returns:
            None: This method saves to disk and doesn't return a value.

        Raises:
            OSError: If the file cannot be written due to permission issues,
                disk space, or other file system errors.

        Note:
            The generated report includes:
            - Business type and location header
            - Current topic being processed (if any)
            - Complete executive report (if available)
            - Detailed breakdown of all processed topics with subtopics and reports
            - List of pending topics awaiting processing
            - Message history for debugging

            The report uses structured formatting with headers and bullet points
            for easy reading and professional presentation.

        Example:
            >>> state = BusinessAnalysisState(
            ...     business_type="Falooda",
            ...     location="Sweden",
            ...     report="Executive summary...",
            ...     processed_topics=[topic1, topic2]
            ... )
            >>> state.save_text("analysis_report.txt")
            >>> # File saved with complete formatted report
        """
        content = "BUSINESS ANALYSIS STATE\n"
        content += "=" * 50 + "\n\n"

        content += f"Business Type: {self.business_type}\n"
        content += f"Location: {self.location}\n\n"

        if self.current_topic:
            content += f"Current Topic: {self.current_topic.topic}\n"
            if self.current_topic.reason:
                content += f"  Description: {self.current_topic.reason}\n"
            content += "\n"

        if self.report:
            content += "EXECUTIVE REPORT:\n"
            content += "-" * 20 + "\n"
            content += f"{self.report}\n\n"

        if self.processed_topics:
            content += "TOPICS PROCESSED:\n"
            content += "-" * 20 + "\n"
            for i, topic in enumerate(self.processed_topics):
                content += f"• {topic.topic}\n"
                if topic.reason:
                    content += f"  Description: {topic.reason}\n"
                if topic.subtopics:
                    content += "  Subtopics:\n"
                    for subtopic in topic.subtopics:
                        content += f"    - {subtopic}\n"
                if topic.report:
                    content += f"  Report: {topic.report}\n"
                if i < len(self.raw_information):
                    content += f"  Raw Information: {self.raw_information[i]}\n"
                content += "\n"

        if self.non_processed_topics:
            content += "TOPICS PENDING PROCESSING:\n"
            content += "-" * 30 + "\n"
            for topic in self.non_processed_topics:
                content += f"• {topic.topic}\n"
                if topic.reason:
                    content += f"  Description: {topic.reason}\n"
                if topic.subtopics:
                    content += "  Subtopics:\n"
                    for subtopic in topic.subtopics:
                        content += f"    - {subtopic}\n"
                content += "\n"

        if self.messages:
            content += "MESSAGES:\n"
            content += "-" * 10 + "\n"
            for i, message in enumerate(self.messages, 1):
                content += f"{i}. {message}\n"

        # Import here to avoid circular import
        from src.utils.utils import save_text_file

        save_text_file(content, filepath)

    @model_validator(mode="after")
    def validate_state(self) -> "BusinessAnalysisState":
        """
        Validate that the state is internally consistent.

        This model validator ensures the BusinessAnalysisState maintains internal
        consistency, particularly around topic indexing and list bounds. It runs
        after all fields are set during model creation or updates.

        Validation checks performed:
        1. current_topic_index must be within valid bounds for topics_list
        2. If current_topic_index >= 0, it must be < len(topics_list)

        Returns:
            BusinessAnalysisState: The same instance after successful validation,
                allowing for method chaining.

        Raises:
            ValueError: If current_topic_index is out of bounds for the topics_list.
                Error message includes the invalid index and list length for debugging.

        Note:
            This validator runs automatically during model instantiation and field
            updates. It ensures the state remains in a valid configuration for
            workflow operations. Negative indices are allowed (indicating no
            active topic) but must not exceed the list bounds when non-negative.

        Example:
            >>> # Valid state
            >>> state = BusinessAnalysisState(
            ...     topics_list=[topic1, topic2],
            ...     current_topic_index=1
            ... )  # Valid - index 1 < len([topic1, topic2])
            >>>
            >>> # Invalid state - would raise ValueError
            >>> # BusinessAnalysisState(
            ... #     topics_list=[topic1, topic2],
            ... #     current_topic_index=2  # Invalid - index 2 >= len([topic1, topic2])
            ... # )
        """
        if self.current_topic_index >= 0 and self.current_topic_index >= len(
            self.topics_list
        ):
            raise ValueError(
                f"current_topic_index {self.current_topic_index} is out of range for topics_list of length {len(self.topics_list)}"
            )
        return self


# ============================================================================
# Expert System States
# ============================================================================


class ExpertDecision(BaseModel):
    """
    Router decision for expert-based query handling.

    Determines which domain expert (business analyst, marketing, operations, etc.)
    should process the current business plan topic or query.

    Attributes:
        expert (ExpertType): Selected expert type for query processing
    """

    expert: ExpertType = Field(
        description="Expert domain best suited to analyze this business plan topic"
    )
