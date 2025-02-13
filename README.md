# Integration Agent

An AI-powered tool that automatically creates service integrations from natural language descriptions. The agent searches for official documentation, analyzes requirements, and generates production-ready integration code.

## Features

- 🔍 Intelligent documentation search and analysis
- 📚 Focuses on official documentation sources
- 🛠️ Generates production-ready integration code
- 🔒 Implements security best practices
- 📝 Provides comprehensive documentation and examples

## Installation

### Using Poetry (Recommended)
```bash
# Install poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

### Using pip
```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```bash
TAVILY_API_KEY=your_tavily_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

Get your API keys:
- [Tavily API Key](https://tavily.com/)
- [Anthropic API Key](https://www.anthropic.com/)

## Usage

### Basic Example
```python
import asyncio
from agent.agent import IntegrationAgent, IntegrationRequest

async def main():
    # Initialize the agent
    agent = IntegrationAgent(
        tavily_api_key="your_tavily_key",
        anthropic_api_key="your_anthropic_key"
    )
    
    # Create an integration request
    request = IntegrationRequest(
        service_name="GitHub",
        integration_type="REST API",
        description="Create an integration that can list repositories and create issues"
    )
    
    # Generate the integration
    result = await agent.create_integration(request)
    
    if result["status"] == "success":
        print("Integration created successfully!")
        print(result["integration"])

if __name__ == "__main__":
    asyncio.run(main())
```

### Run Example Script
```bash
poetry run python examples/test_integration.py
```

## Project Structure

```
integration-agent/
├── agent/                   # Main package
│   ├── tools/              # Integration tools
│   │   ├── search.py       # Documentation search
│   │   └── parser.py       # Documentation parser
│   └── utils/              # Utility functions
├── examples/               # Example scripts
├── tests/                  # Test suite
└── pyproject.toml         # Project configuration
```

## Development

### Running Tests
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=agent
```

### Code Style
The project uses:
- Black for formatting
- isort for import sorting
- mypy for type checking
- ruff for linting

Run formatters:
```bash
poetry run black .
poetry run isort .
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the agent framework
- [Tavily](https://tavily.com/) for the documentation search API
- [Anthropic](https://www.anthropic.com/) for the Claude LLM