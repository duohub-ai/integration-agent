# tests/test_agent.py
import pytest
from unittest.mock import Mock, patch
from agent.agent import IntegrationAgent, IntegrationRequest
from agent.tools.search import SearchConfig

@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

@pytest.fixture
def agent(mock_env_vars):
    return IntegrationAgent(
        tavily_api_key="test-tavily-key",
        anthropic_api_key="test-anthropic-key"
    )

@pytest.fixture
def sample_request():
    return IntegrationRequest(
        service_name="TestAPI",
        integration_type="REST",
        description="Create a simple REST API integration for user management"
    )

@pytest.mark.asyncio
async def test_find_documentation(agent, sample_request):
    # Mock search results
    mock_results = [
        {
            "url": "https://api.test.com/docs",
            "content": "Official API Documentation",
            "is_documentation": True,
            "relevance_score": 0.9
        }
    ]
    
    with patch.object(agent.search_tool, 'search', return_value=mock_results):
        results = await agent._find_documentation(sample_request)
        assert len(results) > 0
        assert results[0].url == "https://api.test.com/docs"
        assert results[0].is_documentation

@pytest.mark.asyncio
async def test_analyze_requirements(agent, sample_request):
    mock_search_results = [
        {
            "url": "https://api.test.com/docs",
            "content": "API requires API key authentication",
            "is_documentation": True,
            "relevance_score": 0.9
        }
    ]
    
    analysis = await agent._analyze_requirements(sample_request, mock_search_results)
    assert isinstance(analysis, dict)
    assert "authentication" in analysis

@pytest.mark.asyncio
async def test_generate_integration(agent, sample_request):
    mock_analysis = {
        "authentication": "API Key",
        "endpoints": ["/api/v1/users"],
        "requirements": ["Python 3.8+"]
    }
    
    result = await agent._generate_integration(sample_request, mock_analysis)
    assert isinstance(result, dict)
    assert "code" in result or "integration" in result

@pytest.mark.asyncio
async def test_create_integration_end_to_end(agent, sample_request):
    # Test the full integration creation process
    mock_search_results = [
        {
            "url": "https://api.test.com/docs",
            "content": "Official API Documentation",
            "is_documentation": True,
            "relevance_score": 0.9
        }
    ]
    
    with patch.object(agent.search_tool, 'search', return_value=mock_search_results):
        result = await agent.create_integration(sample_request)
        assert result["status"] == "success"
        assert "integration" in result

def test_agent_initialization_with_invalid_config():
    with pytest.raises(ValueError):
        IntegrationAgent(tavily_api_key="", anthropic_api_key="test")
    
    with pytest.raises(ValueError):
        IntegrationAgent(tavily_api_key="test", anthropic_api_key="")

@pytest.mark.asyncio
async def test_error_handling(agent, sample_request):
    # Test error handling when search fails
    with patch.object(agent.search_tool, 'search', side_effect=Exception("Search failed")):
        result = await agent.create_integration(sample_request)
        assert result["status"] == "error"
        assert "error" in result

def test_calculate_relevance(agent):
    test_result = {
        "url": "https://docs.test.com/api",
        "content": "API documentation with examples"
    }
    
    test_request = IntegrationRequest(
        service_name="Test",
        integration_type="API",
        description="Test integration"
    )
    
    score = agent._calculate_relevance(test_result, test_request)
    assert 0 <= score <= 1.0