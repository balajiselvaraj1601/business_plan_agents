"""
Test suite for Business Plan Agents modules

This module contains tests to verify that all agent modules can be loaded
and run successfully. It tests the core functionality of:
- simple_planner: Business plan topic generation
- simple_agent: Basic business analysis agent
- smart_agent: Advanced business analysis agent
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.constants import DEFAULT_BUSINESS_TYPE, DEFAULT_LOCATION


def test_simple_planner_import():
    """Test that simple_planner module can be imported successfully."""
    try:
        from src.planner.simple_planner import planner

        assert callable(planner)
        print("✓ simple_planner imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import simple_planner: {e}")


def test_simple_agent_import():
    """Test that simple_agent module can be imported successfully."""
    try:
        from src.agent.simple_agent import analyst, report_generator

        assert callable(analyst)
        assert callable(report_generator)
        print("✓ simple_agent imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import simple_agent: {e}")


def test_smart_agent_import():
    """Test that smart_agent module can be imported successfully."""
    try:
        from src.agent.smart_agent import agent as smart_agent
        from src.agent.smart_agent import planner as smart_planner
        from src.agent.smart_agent import (
            supervisor,
        )

        assert callable(smart_planner)
        assert callable(smart_agent)
        assert callable(supervisor)
        print("✓ smart_agent imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import smart_agent: {e}")


def test_model_collection_import():
    """Test that model_collection can be imported and models are available."""
    try:
        from src.utils.model_collection import (
            llm_general,
            llm_reason,
            planner_llm,
        )

        assert llm_general is not None
        assert llm_reason is not None
        assert planner_llm is not None
        print("✓ model_collection imported successfully")
        print(f"  - llm_general: {type(llm_general).__name__}")
        print(f"  - llm_reason: {type(llm_reason).__name__}")
        print(f"  - planner_llm: {type(planner_llm).__name__}")
    except ImportError as e:
        pytest.fail(f"Failed to import model_collection: {e}")


def test_tools_import():
    """Test that tools can be imported successfully."""
    try:
        from langchain_core.tools import StructuredTool

        from src.tools.tool_collection import TOOLS, search_knowledge, search_web

        assert isinstance(TOOLS, list)
        assert len(TOOLS) > 0
        assert isinstance(search_knowledge, StructuredTool)
        assert isinstance(search_web, StructuredTool)
        print("✓ tools imported successfully")
        print(f"  - Number of tools: {len(TOOLS)}")
    except ImportError as e:
        pytest.fail(f"Failed to import tools: {e}")


def test_simple_planner_basic_functionality():
    """Test basic functionality of simple_planner."""
    try:
        from src.planner.simple_planner import planner
        from src.states.state_collection import PlanningResponse

        # Test with minimal parameters
        result = planner(business=DEFAULT_BUSINESS_TYPE, location=DEFAULT_LOCATION)
        assert isinstance(result, PlanningResponse)
        assert hasattr(result, "topics")
        assert len(result.topics) > 0
        print("✓ simple_planner basic functionality works")
    except Exception as e:
        pytest.fail(f"simple_planner basic functionality failed: {e}")


def test_simple_agent_setup():
    """Test that simple_agent graph can be set up."""
    try:
        from src.agent.simple_agent import setup_agent_graph
        from src.tools.tool_collection import TOOLS

        # Test graph setup
        app = setup_agent_graph(tools=TOOLS)
        assert app is not None
        print("✓ simple_agent graph setup works")
    except Exception as e:
        pytest.fail(f"simple_agent graph setup failed: {e}")


def test_smart_agent_state_creation():
    """Test that smart_agent state can be created."""
    try:
        from src.agent.smart_agent import BusinessAnalysisState

        # Test state creation
        state = BusinessAnalysisState(
            location=DEFAULT_LOCATION,
            business_type=DEFAULT_BUSINESS_TYPE,
            topics_list=[],
            current_topic_index=-1,
            description=None,
            raw_information=[],
            messages=[],
            report="",
        )
        assert state.location == DEFAULT_LOCATION
        assert state.business_type == DEFAULT_BUSINESS_TYPE
        assert isinstance(state.topics_list, list)
        assert state.current_topic_index == -1
        print("✓ smart_agent state creation works")
    except Exception as e:
        pytest.fail(f"smart_agent state creation failed: {e}")


def run_all_tests():
    """
    Execute all test functions manually for direct script execution.

    This function runs the complete test suite when the script is executed
    directly (not via pytest). It performs comprehensive validation of:

    - Module imports and basic functionality
    - Model collection availability
    - Tool collection setup
    - Agent initialization and basic operations
    - State creation and validation

    Returns:
        bool: True if all tests pass, False if any test fails

    Prints detailed progress and results to console.
    """
    print("Running Business Plan Agents Tests...")
    print("=" * 50)

    try:
        test_model_collection_import()
        test_tools_import()
        test_simple_planner_import()
        test_simple_agent_import()
        test_smart_agent_import()
        test_simple_planner_basic_functionality()
        test_simple_agent_setup()
        test_smart_agent_state_creation()

        print("=" * 50)
        print("🎉 All tests passed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
