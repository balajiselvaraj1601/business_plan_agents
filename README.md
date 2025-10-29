# Business Plan Agents 🤖

A comprehensive AI-powered multi-agent system for automated business plan generation and analysis. This system uses LangChain, LangGraph, and Ollama to create structured business plans and perform detailed strategic analysis.

## 🌟 Features

- **Intelligent Planning**: AI-generated business plan topics with hierarchical structure
- **Multi-Agent Analysis**: Specialized agents for different aspects of business analysis
- **Tool Integration**: Web search and knowledge retrieval capabilities
- **Expert Routing**: Automatic routing to domain-specific business experts
- **Comprehensive Testing**: 81+ test cases with 94% coverage
- **Modern Python Stack**: Type-safe, linted, and well-documented code

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Components](#components)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) with required models
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd business-plan-agents
   ```

2. **Set up Python environment**
   ```bash
   uv sync
   ```

3. **Install Ollama models**
   ```bash
   ollama pull granite3.3:8b
   ollama pull qwen3:8b
   ```

4. **Configure API keys**
   Create `meta_data.json`:
   ```json
   {
     "tavily_api_key": "your-tavily-api-key",
     "langsmith_api_key": "your-langsmith-api-key"
   }
   ```

### Running the System

1. **Generate Business Plan Structure**
   ```bash
   uv run python -m src.planner.simple_planner
   ```

2. **Run Business Analysis**
   ```bash
   uv run python -m src.agent.simple_agent
   ```

3. **Use Smart Agent (Workflow Orchestration)**
   ```bash
   uv run python -m src.agent.smart_agent
   ```

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:

```
business-plan-agents/
├── src/
│   ├── agent/           # Analysis agents (simple & smart)
│   ├── planner/         # Business plan topic generation
│   ├── prompts/         # AI prompt templates
│   ├── states/          # Pydantic state models
│   ├── tools/           # External tool integrations
│   └── utils/           # Shared utilities and constants
├── tests/               # Comprehensive test suite
└── reports/             # Generated analysis reports
```

### Data Flow

1. **Planning Phase**: Generate structured business plan topics
2. **Analysis Phase**: Deep-dive analysis of each topic using AI agents
3. **Tool Integration**: Web search and knowledge retrieval for current data
4. **Report Generation**: Comprehensive business analysis reports

## 🔧 Components

### Agents

- **Simple Agent**: Focused analysis agent using LangGraph workflows
- **Smart Agent**: Advanced orchestration with supervisor patterns

### Planner

- **Simple Planner**: AI-powered topic generation for business plans
- Generates hierarchical topic structures with subtopics and descriptions

### Tools

- **Web Search**: Tavily integration for market research
- **Knowledge Retrieval**: Internal knowledge base queries

### State Management

- **PlanningResponse**: Business plan topic structure
- **BusinessAnalysisState**: Analysis workflow state
- **Topic Models**: Individual topic and subtopic definitions

## ⚙️ Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# API Keys
TAVILY_API_KEY=your-tavily-key
LANGSMITH_API_KEY=your-langsmith-key

# Logging
LOG_LEVEL=INFO
FLAG_DEBUG=false
```

### Constants

Key configuration constants in `src/utils/constants.py`:

- Business types and locations
- Model configurations
- File paths and separators
- Debug and logging settings

## 📖 Usage

### Basic Usage

```python
from src.planner.simple_planner import generate_and_save_business_plan
from src.agent.simple_agent import setup_agent_graph

# Generate business plan structure
generate_and_save_business_plan()

# Set up and run analysis agent
graph = setup_agent_graph()
# Run analysis workflow...
```

### Expert Routing

```python
from src.utils.utils import choose_expert

expert = choose_expert("How should I price my new SaaS product?")
# Returns: "Financial Expert"
```

### Custom Analysis

```python
from src.agent.smart_agent import create_and_run_workflow

# Run complete planning and analysis workflow
result = create_and_run_workflow(
    business_type="Technology Startup",
    location="San Francisco, CA"
)
```

## 🧪 Testing

### Run All Tests

```bash
uv run pytest
```

### Run with Coverage

```bash
uv run pytest --cov=src --cov-report=html
```

### Run Specific Test Categories

```bash
# Agent tests
uv run pytest tests/agent/

# Utility tests
uv run pytest tests/utils/

# Integration tests
uv run pytest tests/test_pydantic_parsers.py
```

### CI Pipeline

Run the complete CI pipeline:

```bash
uv run tox -e ci
```

This includes:
- Linting (ruff)
- Code formatting (black, isort)
- Type checking (mypy)
- Testing (pytest)
- Coverage analysis
- Pre-commit hooks

## 🛠️ Development

### Code Quality Tools

- **Linting**: `ruff` for fast, comprehensive Python linting
- **Formatting**: `black` for consistent code formatting
- **Import Sorting**: `isort` for organized imports
- **Type Checking**: `mypy` for static type analysis

### Development Workflow

1. **Install development dependencies**
   ```bash
   uv sync --group dev
   ```

2. **Run pre-commit hooks**
   ```bash
   pre-commit install
   pre-commit run --all-files
   ```

3. **Run type checking**
   ```bash
   uv run mypy --explicit-package-bases src
   ```

### Project Structure Guidelines

- **Imports**: Organized at module top-level
- **Logging**: Comprehensive logging with structured messages
- **Error Handling**: Specific exception handling with meaningful messages
- **Documentation**: Docstrings for all public functions and classes
- **Type Hints**: Full type annotations throughout

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes with proper tests
4. Run the full CI pipeline: `uv run tox -e ci`
5. Ensure all tests pass and coverage remains high
6. Submit a pull request

### Code Standards

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new functionality
- Update documentation for API changes
- Maintain test coverage above 90%

## 📊 Project Metrics

- **Test Coverage**: 94%
- **Test Cases**: 81+
- **Python Version**: 3.11+
- **Dependencies**: 25+ packages
- **Lines of Code**: ~1400+ statements

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the LLM framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [Ollama](https://ollama.ai/) for local LLM hosting
- [Tavily](https://tavily.com/) for web search capabilities

---

**Built with ❤️ using modern Python and AI technologies**
