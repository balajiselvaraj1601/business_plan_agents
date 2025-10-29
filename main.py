"""
Business Plan Agents - Main Entry Point

This module serves as the main entry point for the Business Plan Agents application,
a multi-agent system for automated business plan analysis and generation.

The application consists of:
- A planner agent that generates business plan topics and structure
- An analysis agent that performs detailed business analysis
- A smart agent that coordinates planning and analysis workflows

Usage:
    python main.py

This will display instructions for running the planner and agent components.
"""

from src.utils.constants import LOG_LEVEL
from src.utils.utils import log_info, setup_logging


def main():
    """
    Main application entry point.

    Initializes logging and displays usage instructions for running the
    business plan analysis system components.

    The system requires running components in sequence:
    1. Run the planner to generate business plan structure
    2. Run the agent to perform detailed analysis

    No arguments required - displays help information.
    """
    # Setup logging
    setup_logging(LOG_LEVEL)

    log_info("Business Plan Agents - Starting application...")
    log_info("Run the planner first: python -m src.planner.simple_planner")
    log_info("Then run the agent: python -m src.agent.simple_agent")


if __name__ == "__main__":
    main()
