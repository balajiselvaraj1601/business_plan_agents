# Test Pydantic models for business plan agents
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.states.state_collection import (
    BusinessAnalysisState,
    FeedbackResponse,
    PlanningResponse,
    Topic,
)
from src.utils.constants import DEFAULT_BUSINESS_TYPE, DEFAULT_LOCATION


def test_topic_model():
    """Test that Topic model can be created and parsed."""
    # Test creation
    topic = Topic(
        topic="Test Marketing Strategy",
        reason="A test marketing strategy",
        subtopics=["Define target audience", "Conduct research"],
        report=None,
    )
    assert topic.topic == "Test Marketing Strategy"
    assert len(topic.subtopics) == 2

    # Test parsing from dict
    data = {
        "topic": "Parsed Topic",
        "reason": "Parsed description",
        "subtopics": ["item1", "item2"],
        "report": None,
    }
    parsed_topic = Topic(**data)
    assert parsed_topic.topic == "Parsed Topic"


def test_business_plan_topics_response():
    """Test that BusinessPlanTopicsResponse can be created and parsed."""
    topic = Topic(
        topic="Marketing Strategy",
        reason="",
        subtopics=["Target audience"],
        report=None,
    )
    business_plan = PlanningResponse(topics=[topic])

    assert len(business_plan.topics) == 1
    assert business_plan.topics[0].topic == "Marketing Strategy"

    # Test parsing from dict
    data = {
        "topics": [
            {
                "topic": "Parsed Topic",
                "reason": "Parsed description",
                "subtopics": ["item1"],
                "report": None,
            }
        ]
    }
    parsed = PlanningResponse(**data)
    assert len(parsed.topics) == 1


def test_business_analysis_state():
    """Test that BusinessAnalysisState can be created and parsed."""
    # Create a Topic object for testing
    test_topic = Topic(topic="Pricing analysis", reason="", subtopics=[], report=None)

    analysis_state = BusinessAnalysisState(
        business_type=DEFAULT_BUSINESS_TYPE,
        location=DEFAULT_LOCATION,
        topics_list=[test_topic],
        current_topic_index=-1,
        description="Test analysis",
        raw_information=[],
        messages=[],
        report="",
    )

    assert analysis_state.current_topic is None  # No current topic since index is -1
    assert len(analysis_state.topics_list) == 1
    assert len(analysis_state.non_processed_topics) == 1

    # Test parsing from dict
    data = {
        "business_type": DEFAULT_BUSINESS_TYPE,
        "location": DEFAULT_LOCATION,
        "topics_list": [],
        "current_topic_index": -1,
        "description": "Test description",
        "raw_information": [],
        "messages": [],
        "report": "test",
    }
    parsed = BusinessAnalysisState(**data)
    assert parsed.current_topic is None


def test_validation_errors():
    """Test that models raise validation errors for invalid data."""
    with pytest.raises(ValidationError):
        Topic(topic="", reason="Missing title", subtopics=["item1"], report=None)

    with pytest.raises(ValidationError):
        BusinessAnalysisState(
            business_type="Test",
            location="Test",
            topics_list=[],
            current_topic_index=0,  # Invalid: index 0 but empty list
            description="Invalid index",
            raw_information=[],
            messages=[],
            report="",
        )


@pytest.mark.parametrize(
    "model_class,kwargs,expected_error_contains",
    [
        (
            Topic,
            {
                "topic": "",
                "reason": "Missing title",
                "subtopics": ["item1"],
                "report": None,
            },
            "Topic must not be empty",
        ),
        (
            Topic,
            {
                "topic": "   ",
                "reason": "Whitespace title",
                "subtopics": [],
                "report": None,
            },
            "Topic must not be empty",
        ),
        (
            PlanningResponse,
            {"topics": []},
            "Business plan must contain at least one topic",
        ),
    ],
)
def test_model_validation_errors_parameterized(
    model_class, kwargs, expected_error_contains
):
    """Test that models raise validation errors for invalid data with parameterization."""
    with pytest.raises(ValidationError) as exc_info:
        model_class(**kwargs)
    assert expected_error_contains in str(exc_info.value)


def test_topic_print_structured():
    """Test Topic's print_structured method."""
    topic = Topic(
        topic="Marketing",
        reason="Marketing strategy",
        subtopics=["Social media", "SEO"],
        report=None,
    )
    with patch("src.states.state_collection.log_info") as mock_log:
        topic.print_structured()
        assert mock_log.call_count >= 2  # Title and subtopics


@pytest.mark.parametrize(
    "model_class,method_name,expected_calls",
    [
        (Topic, "print_structured", 2),
        (PlanningResponse, "print_structured", 1),
        (BusinessAnalysisState, "print_structured", 3),
    ],
)
def test_print_structured_methods_parameterized(
    model_class, method_name, expected_calls
):
    """Test print_structured methods for different model classes."""
    if model_class == Topic:
        instance = Topic(
            topic="Test Topic",
            reason="Test description",
            subtopics=["sub1", "sub2"],
            report=None,
        )
    elif model_class == PlanningResponse:
        instance = PlanningResponse(
            topics=[Topic(topic="Topic 1", reason="", subtopics=[], report=None)]
        )
    else:  # BusinessAnalysisState
        instance = BusinessAnalysisState(
            business_type="Test",
            location="Test",
            topics_list=[Topic(topic="Topic 1", reason="", subtopics=[], report=None)],
            current_topic_index=0,
            raw_information=[],
            description=None,
            messages=[],
            report="",
        )

    with patch("src.states.state_collection.log_info") as mock_log:
        getattr(instance, method_name)()
        assert mock_log.call_count >= expected_calls


def test_planning_response_validation_empty_topics():
    """Test PlanningResponse validation with empty topics list."""
    with pytest.raises(ValidationError) as exc_info:
        PlanningResponse(topics=[])
    assert "Business plan must contain at least one topic" in str(exc_info.value)


def test_planning_response_validation_empty_topic_title():
    """Test PlanningResponse validation with empty topic title."""
    topic_with_empty_title = Topic(
        topic="Valid", reason="test", subtopics=[], report=None
    )
    # Manually set topic to empty to bypass Topic's own validation
    topic_dict = topic_with_empty_title.model_dump()
    topic_dict["topic"] = ""

    with pytest.raises(ValidationError) as exc_info:
        PlanningResponse(topics=[Topic(**topic_dict)])
    assert "topic must not be empty" in str(
        exc_info.value
    ).lower() or "Topic must not be empty" in str(exc_info.value)


def test_planning_response_save_json():
    """Test PlanningResponse save_json method."""
    topic = Topic(topic="Test", reason="desc", subtopics=["sub1"], report=None)
    response = PlanningResponse(topics=[topic])

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_plan.json"
        response.save_json(str(filepath))

        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert len(data["topics"]) == 1
        assert data["topics"][0]["topic"] == "Test"


def test_planning_response_load_json_success():
    """Test PlanningResponse load_json method with valid file."""
    topic = Topic(topic="Loaded", reason="test", subtopics=[], report=None)
    response = PlanningResponse(topics=[topic])

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.json"
        response.save_json(str(filepath))

        loaded = PlanningResponse.load_json(str(filepath))
        assert len(loaded.topics) == 1
        assert loaded.topics[0].topic == "Loaded"


def test_planning_response_load_json_file_not_found():
    """Test PlanningResponse load_json with non-existent file."""
    with pytest.raises(FileNotFoundError) as exc_info:
        PlanningResponse.load_json("/nonexistent/path/file.json")
    assert "not found" in str(exc_info.value).lower()


def test_planning_response_load_json_invalid_json():
    """Test PlanningResponse load_json with invalid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "invalid.json"
        filepath.write_text("{ invalid json }")

        with pytest.raises(ValueError) as exc_info:
            PlanningResponse.load_json(str(filepath))
        assert "Invalid JSON" in str(exc_info.value)


def test_planning_response_print_structured():
    """Test PlanningResponse print_structured method."""
    topic1 = Topic(topic="Topic1", reason="", subtopics=[], report=None)
    topic2 = Topic(topic="Topic2", reason="desc", subtopics=["sub1"], report=None)
    response = PlanningResponse(topics=[topic1, topic2])

    with patch("src.states.state_collection.log_info") as mock_log:
        response.print_structured()
        # Should call log_info for each topic plus spacing
        assert mock_log.call_count >= 2


def test_business_analysis_state_processed_topics():
    """Test BusinessAnalysisState processed_topics property."""
    topic1 = Topic(topic="T1", reason="", subtopics=[], report=None)
    topic2 = Topic(topic="T2", reason="", subtopics=[], report=None)
    topic3 = Topic(topic="T3", reason="", subtopics=[], report=None)

    state = BusinessAnalysisState(
        business_type="Bakery",
        location="SF",
        topics_list=[topic1, topic2, topic3],
        current_topic_index=1,  # At topic2, so topic1 is processed
    )

    assert len(state.processed_topics) == 1
    assert state.processed_topics[0].topic == "T1"


def test_business_analysis_state_processed_topics_negative_index():
    """Test processed_topics with negative current_topic_index."""
    topic = Topic(topic="T1", reason="", subtopics=[], report=None)

    state = BusinessAnalysisState(
        business_type="Bakery",
        location="SF",
        topics_list=[topic],
        current_topic_index=-1,
    )

    assert len(state.processed_topics) == 0


def test_business_analysis_state_non_processed_topics():
    """Test BusinessAnalysisState non_processed_topics property."""
    topic1 = Topic(topic="T1", reason="", subtopics=[], report=None)
    topic2 = Topic(topic="T2", reason="", subtopics=[], report=None)
    topic3 = Topic(topic="T3", reason="", subtopics=[], report=None)

    state = BusinessAnalysisState(
        business_type="Bakery",
        location="SF",
        topics_list=[topic1, topic2, topic3],
        current_topic_index=0,  # At topic1, so topic2 and topic3 are non-processed
    )

    assert len(state.non_processed_topics) == 2
    assert state.non_processed_topics[0].topic == "T2"
    assert state.non_processed_topics[1].topic == "T3"


def test_business_analysis_state_current_topic():
    """Test BusinessAnalysisState current_topic property."""
    topic1 = Topic(topic="T1", reason="", subtopics=[], report=None)
    topic2 = Topic(topic="T2", reason="", subtopics=[], report=None)

    state = BusinessAnalysisState(
        business_type="Bakery",
        location="SF",
        topics_list=[topic1, topic2],
        current_topic_index=1,
    )

    assert state.current_topic is not None
    assert state.current_topic.topic == "T2"
    # Test alias
    assert state.topic.topic == "T2"


def test_business_analysis_state_current_topic_out_of_bounds():
    """Test current_topic returns None when index is out of bounds."""
    topic = Topic(topic="T1", reason="", subtopics=[], report=None)

    # At last topic - current_topic should return the topic
    state = BusinessAnalysisState(
        business_type="Bakery",
        location="SF",
        topics_list=[topic],
        current_topic_index=0,
    )

    assert state.current_topic is not None
    assert state.current_topic.topic == "T1"


def test_business_analysis_state_print_structured_with_current_topic():
    """Test print_structured with current topic and description."""
    topic = Topic(topic="Marketing", reason="Marketing desc", subtopics=[], report=None)

    state = BusinessAnalysisState(
        business_type="Bakery",
        location="SF",
        topics_list=[topic],
        current_topic_index=0,
    )

    with patch("src.states.state_collection.log_info") as mock_log:
        state.print_structured()
        # Should log business_type, location, current topic with description
        assert mock_log.call_count >= 4


def test_business_analysis_state_print_structured_with_processed_topics():
    """Test print_structured with processed topics and raw information."""
    topic1 = Topic(topic="T1", reason="Desc1", subtopics=[], report=None)
    topic2 = Topic(topic="T2", reason="", subtopics=[], report=None)

    state = BusinessAnalysisState(
        business_type="Bakery",
        location="SF",
        topics_list=[topic1, topic2],
        current_topic_index=1,
        raw_information=["Analysis for T1"],
    )

    with patch("src.states.state_collection.log_info") as mock_log:
        state.print_structured()
        # Should include processed topics section
        call_args = [str(call) for call in mock_log.call_args_list]
        assert any("Topics Processed" in str(call) for call in call_args)


def test_business_analysis_state_print_structured_with_messages():
    """Test print_structured with messages."""
    topic = Topic(topic="T1", reason="", subtopics=[], report=None)

    state = BusinessAnalysisState(
        business_type="Bakery",
        location="SF",
        topics_list=[topic],
        current_topic_index=0,
        messages=["msg1", "msg2"],
    )

    with patch("src.states.state_collection.log_info") as mock_log:
        state.print_structured()
        # Should log messages count
        call_args = [str(call) for call in mock_log.call_args_list]
        assert any(
            "Messages" in str(call) and "2 items" in str(call) for call in call_args
        )


def test_business_analysis_state_save_text_full():
    """Test save_text with all sections populated."""
    topic1 = Topic(
        topic="T1", reason="Desc1", subtopics=["sub1", "sub2"], report="Report1"
    )
    topic2 = Topic(topic="T2", reason="Desc2", subtopics=["sub3"], report=None)
    topic3 = Topic(topic="T3", reason="", subtopics=[], report=None)

    state = BusinessAnalysisState(
        business_type="Bakery",
        location="SF",
        topics_list=[topic1, topic2, topic3],
        current_topic_index=1,  # At topic2, so topic1 is processed
        raw_information=["Raw info 1"],
        messages=["message1", "message2"],
        report="Executive Report Content",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_report.txt"
        with patch("src.utils.utils.save_text_file") as mock_save:
            state.save_text(str(filepath))

            # Verify save_text_file was called
            mock_save.assert_called_once()
            content = mock_save.call_args[0][0]

            # Verify content structure
            assert "BUSINESS ANALYSIS STATE" in content
            assert "Business Type: Bakery" in content
            assert "Location: SF" in content
            assert "Current Topic: T2" in content
            assert "EXECUTIVE REPORT:" in content
            assert "Executive Report Content" in content
            assert "TOPICS PROCESSED:" in content
            assert "• T1" in content
            assert "Description: Desc1" in content
            assert "Subtopics:" in content
            assert "- sub1" in content
            assert "Report: Report1" in content
            assert "Raw Information: Raw info 1" in content
            assert "TOPICS PENDING PROCESSING:" in content
            assert "• T3" in content
            assert "MESSAGES:" in content
            assert "1. message1" in content


def test_business_analysis_state_save_text_minimal():
    """Test save_text with minimal data."""
    topic = Topic(topic="T1", reason="", subtopics=[], report=None)

    state = BusinessAnalysisState(
        business_type="Bakery",
        location="SF",
        topics_list=[topic],
        current_topic_index=-1,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "minimal_report.txt"
        with patch("src.utils.utils.save_text_file") as mock_save:
            state.save_text(str(filepath))

            mock_save.assert_called_once()
            content = mock_save.call_args[0][0]

            # Verify basic content
            assert "BUSINESS ANALYSIS STATE" in content
            assert "Business Type: Bakery" in content
            assert "Location: SF" in content
            # Should have pending topic
            assert "TOPICS PENDING PROCESSING:" in content
            assert "• T1" in content


def test_business_analysis_state_validation_invalid_index():
    """Test BusinessAnalysisState validation with invalid current_topic_index."""
    topic = Topic(topic="T1", reason="", subtopics=[], report=None)

    with pytest.raises(ValidationError) as exc_info:
        BusinessAnalysisState(
            business_type="Bakery",
            location="SF",
            topics_list=[topic],
            current_topic_index=5,  # Out of bounds
        )
    assert "out of range" in str(exc_info.value).lower()


def test_feedback_model():
    """Test that Feedback model can be created, validated, and serialized with aliases."""
    # Test creation with valid data
    feedback = FeedbackResponse(
        assessment="Good plan with solid foundation",
        score=8,
        strength_list=["Clear objectives", "Realistic timeline"],
        weakness_list=["Budget constraints", "Risk assessment needed"],
        suggestion_list=["Add contingency plans", "Include stakeholder analysis"],
        recommendation_list=["Regular progress reviews", "Monthly budget tracking"],
    )

    assert feedback.assessment == "Good plan with solid foundation"
    assert feedback.score == 8
    assert len(feedback.strength_list) == 2
    assert len(feedback.weakness_list) == 2

    # Test JSON serialization
    json_str = feedback.model_dump_json()
    json_data = json.loads(json_str)

    expected_keys = [
        "assessment",
        "score",
        "strength_list",
        "weakness_list",
        "suggestion_list",
        "recommendation_list",
    ]

    for key in expected_keys:
        assert key in json_data

    # Test deserialization from JSON
    feedback_from_json = FeedbackResponse.model_validate_json(json_str)
    assert feedback_from_json.assessment == feedback.assessment
    assert feedback_from_json.score == feedback.score
    assert feedback_from_json.strength_list == feedback.strength_list

    # Test validation errors
    with pytest.raises(ValidationError):
        FeedbackResponse(
            assessment="Test",
            score=15,  # Invalid score > 10
            strength_list=[],
            weakness_list=[],
            suggestion_list=[],
            recommendation_list=[],
        )

    with pytest.raises(ValidationError):
        FeedbackResponse(
            assessment="Test",
            score=0,  # Invalid score < 1
            strength_list=[],
            weakness_list=[],
            suggestion_list=[],
            recommendation_list=[],
        )
