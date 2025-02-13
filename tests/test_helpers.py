# tests/test_helpers.py
import pytest
from agent.utils.helpers import (
    extract_code_blocks,
    validate_integration_config,
    parse_llm_response,
    get_file_extension
)

def test_extract_code_blocks():
    markdown_text = '''
    Here's some Python code:
    ```python
    def hello():
        print("Hello, world!")
    ```
    
    And some JSON:
    ```json
    {
        "key": "value"
    }
    ```
    '''
    
    blocks = extract_code_blocks(markdown_text)
    assert len(blocks) == 2
    assert blocks[0]["language"] == "python"
    assert "hello()" in blocks[0]["code"]
    assert blocks[1]["language"] == "json"
    assert "key" in blocks[1]["code"]

def test_validate_integration_config():
    # Test valid config
    valid_config = {
        "service_name": "TestAPI",
        "integration_type": "REST",
        "description": "Test integration"
    }
    result = validate_integration_config(valid_config)
    assert result["service_name"] == "TestAPI"
    assert result["integration_type"] == "rest"
    
    # Test missing required field
    invalid_config = {
        "service_name": "TestAPI"
    }
    with pytest.raises(ValueError):
        validate_integration_config(invalid_config)

def test_parse_llm_response():
    response = '''
    Here's the integration code:
    ```python
    def test():
        pass
    ```
    
    Requirements:
    - Python 3.8+
    - Requests library
    
    Configuration:
    ```json
    {
        "api_key": "required",
        "base_url": "optional"
    }
    ```
    '''
    
    result = parse_llm_response(response)
    assert "code_blocks" in result
    assert len(result["code_blocks"]) == 2

def test_get_file_extension():
    assert get_file_extension("python") == ".py"
    assert get_file_extension("javascript") == ".js"
    assert get_file_extension("unknown") == ".txt"
    assert get_file_extension("PYTHON") == ".py"  # Case insensitive