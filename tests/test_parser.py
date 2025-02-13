# tests/test_parser.py
import pytest
from bs4 import BeautifulSoup
from agent.tools.parser import DocumentationParser, CodeExample, Endpoint

@pytest.fixture
def parser():
    return DocumentationParser()

@pytest.fixture
def sample_html():
    return """
    <html>
        <head>
            <title>API Documentation</title>
        </head>
        <body>
            <h1>API Reference</h1>
            <div class="introduction">
                This is the API documentation.
            </div>
            
            <div class="authentication">
                <h2>Authentication</h2>
                <p>Use API Key authentication</p>
                <code>
                    Authorization: Bearer YOUR_API_KEY
                </code>
            </div>
            
            <div class="endpoint">
                <h3>GET /api/v1/users</h3>
                <p class="description">List all users</p>
                <table class="parameters">
                    <tr>
                        <th>Parameter</th>
                        <th>Description</th>
                    </tr>
                    <tr>
                        <td>limit</td>
                        <td>Maximum number of records</td>
                    </tr>
                </table>
                <pre class="language-python">
                    import requests
                    response = requests.get('/api/v1/users')
                </pre>
            </div>
        </body>
    </html>
    """

def test_extract_title(parser, sample_html):
    soup = BeautifulSoup(sample_html, 'html.parser')
    title = parser._extract_title(soup)
    assert title == "API Documentation"

def test_extract_overview(parser, sample_html):
    soup = BeautifulSoup(sample_html, 'html.parser')
    overview = parser._extract_overview(soup)
    assert "This is the API documentation" in overview

def test_extract_authentication(parser, sample_html):
    soup = BeautifulSoup(sample_html, 'html.parser')
    auth_info = parser._extract_authentication(soup)
    assert "API Key" in auth_info.get("instructions", "")

def test_extract_endpoints(parser, sample_html):
    soup = BeautifulSoup(sample_html, 'html.parser')
    endpoints = parser._extract_endpoints(soup)
    assert len(endpoints) > 0
    endpoint = endpoints[0]
    assert endpoint.method == "GET"
    assert endpoint.path == "/api/v1/users"

def test_extract_code_examples(parser, sample_html):
    soup = BeautifulSoup(sample_html, 'html.parser')
    examples = parser._extract_code_examples(soup)
    assert len(examples) > 0
    assert isinstance(examples[0], CodeExample)
    assert "requests.get" in examples[0].code

def test_detect_language(parser):
    code_blocks = {
        'python': BeautifulSoup('<code class="language-python">import requests</code>', 'html.parser').code,
        'json': BeautifulSoup('<code class="language-json">{"key": "value"}</code>', 'html.parser').code,
        'unknown': BeautifulSoup('<code>some generic code</code>', 'html.parser').code
    }
    
    assert parser._detect_language(code_blocks['python']) == 'python'
    assert parser._detect_language(code_blocks['json']) == 'json'
    assert parser._detect_language(code_blocks['unknown']) == 'unknown'

@pytest.mark.asyncio
async def test_parse_documentation(parser, requests_mock):
    # Mock a documentation page
    url = "https://api.example.com/docs"
    requests_mock.get(url, text="""
    <html>
        <head>
            <title>API Documentation</title>
        </head>
        <body>
            <h1>API Reference</h1>
            <div class="introduction">
                This is the API documentation.
            </div>
        </body>
    </html>
    """)
    
    result = await parser.parse_documentation(url)
    
    assert result["title"] == "API Documentation"
    assert "endpoints" in result
    assert "authentication" in result
    assert "examples" in result

@pytest.mark.asyncio
async def test_parser_error_handling(parser, requests_mock):
    url = "https://api.example.com/docs"
    requests_mock.get(url, status_code=404)
    
    with pytest.raises(Exception):
        await parser.parse_documentation(url)