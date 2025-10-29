"""
Pytest configuration for the tests directory.
"""

import sys
from pathlib import Path

import pytest

# Add src to Python path for all tests
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def mock_log_debug():
    """Mock for log_debug function."""
    with pytest.mock.patch("src.utils.logging.log_debug") as mock:
        yield mock


@pytest.fixture
def mock_log_info():
    """Mock for log_info function."""
    with pytest.mock.patch("src.utils.logging.log_info") as mock:
        yield mock


@pytest.fixture
def mock_log_success():
    """Mock for log_success function."""
    with pytest.mock.patch("src.utils.logging.log_success") as mock:
        yield mock


@pytest.fixture
def mock_log_error():
    """Mock for log_error function."""
    with pytest.mock.patch("src.utils.logging.log_error") as mock:
        yield mock


@pytest.fixture
def mock_log_separator():
    """Mock for log_separator function."""
    with pytest.mock.patch("src.utils.logging.log_separator") as mock:
        yield mock


@pytest.fixture
def mock_log_data_processing():
    """Mock for log_data_processing function."""
    with pytest.mock.patch("src.utils.logging.log_data_processing") as mock:
        yield mock


@pytest.fixture
def sample_topic():
    """Sample Topic object for testing."""
    from src.states.state_collection import Topic

    return Topic(
        title="Sample Topic",
        description="Sample description",
        subtopics=["Subtopic 1", "Subtopic 2"],
        report=None,
    )


@pytest.fixture
def sample_business_analysis_state(sample_topic):
    """Sample BusinessAnalysisState for testing."""
    from src.states.state_collection import BusinessAnalysisState

    return BusinessAnalysisState(
        business_type="Test Business",
        location="Test City",
        topics_list=[sample_topic],
        current_topic_index=0,
        raw_information=[],
        description=None,
        messages=[],
        report="",
    )
