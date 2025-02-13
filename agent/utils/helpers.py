from typing import Dict, List, Optional, Any
import re
import json
from pathlib import Path
import logging
from dataclasses import asdict

logger = logging.getLogger(__name__)

def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """
    Extract code blocks from markdown-formatted text.
    
    Args:
        text: Markdown text containing code blocks
        
    Returns:
        List of dictionaries containing language and code
    """
    code_blocks = []
    pattern = r"```(\w*)\n([\s\S]*?)```"
    
    matches = re.finditer(pattern, text)
    for match in matches:
        language = match.group(1) or "text"
        code = match.group(2).strip()
        code_blocks.append({
            "language": language,
            "code": code
        })
    
    return code_blocks

def validate_integration_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize integration configuration.
    
    Args:
        config: Integration configuration dictionary
        
    Returns:
        Validated and normalized configuration
    """
    required_fields = ['service_name', 'integration_type']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    # Normalize fields
    normalized = {
        'service_name': config['service_name'].strip(),
        'integration_type': config['integration_type'].lower().strip(),
        'description': config.get('description', '').strip(),
        'authentication_type': config.get('authentication_type', 'none').lower(),
        'specific_endpoints': config.get('specific_endpoints', [])
    }
    
    return normalized

def parse_llm_response(response: str) -> Dict[str, Any]:
    """
    Parse structured information from LLM response.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Dictionary containing parsed information
    """
    result = {
        'code_blocks': extract_code_blocks(response),
        'metadata': {}
    }
    
    # Try to extract JSON blocks
    json_pattern = r"\{[\s\S]*?\}"
    json_matches = re.finditer(json_pattern, response)
    
    for match in json_matches:
        try:
            json_str = match.group(0)
            json_data = json.loads(json_str)
            if isinstance(json_data, dict):
                result['metadata'].update(json_data)
        except json.JSONDecodeError:
            continue
    
    # Extract requirements if present
    req_pattern = r"Requirements:(.*?)(?:\n\n|\Z)"
    req_match = re.search(req_pattern, response, re.DOTALL)
    if req_match:
        result['requirements'] = [
            r.strip() for r in req_match.group(1).split('\n') if r.strip()
        ]
    
    return result

def save_integration(
    integration_data: Dict[str, Any],
    output_dir: str = "generated_integrations"
) -> Dict[str, str]:
    """
    Save generated integration files to disk.
    
    Args:
        integration_data: Dictionary containing integration code and metadata
        output_dir: Directory to save files
        
    Returns:
        Dictionary mapping file names to their paths
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save code files
        for idx, code_block in enumerate(integration_data.get('code_blocks', [])):
            language = code_block['language']
            extension = get_file_extension(language)
            
            if idx == 0:
                filename = f"integration{extension}"
            else:
                filename = f"integration_{idx}{extension}"
            
            file_path = output_path / filename
            with open(file_path, 'w') as f:
                f.write(code_block['code'])
            
            saved_files[filename] = str(file_path)
        
        # Save metadata
        if integration_data.get('metadata'):
            meta_path = output_path / 'metadata.json'
            with open(meta_path, 'w') as f:
                json.dump(integration_data['metadata'], f, indent=2)
            saved_files['metadata.json'] = str(meta_path)
        
        return saved_files
        
    except Exception as e:
        logger.error(f"Error saving integration: {str(e)}")
        raise

def get_file_extension(language: str) -> str:
    """Get appropriate file extension for a programming language."""
    extensions = {
        'python': '.py',
        'javascript': '.js',
        'typescript': '.ts',
        'java': '.java',
        'ruby': '.rb',
        'php': '.php',
        'go': '.go',
        'rust': '.rs',
        'c': '.c',
        'cpp': '.cpp',
        'csharp': '.cs',
        'swift': '.swift',
        'kotlin': '.kt',
        'scala': '.scala',
        'r': '.r',
        'julia': '.jl',
        'shell': '.sh',
        'sql': '.sql',
        'html': '.html',
        'css': '.css',
        'json': '.json',
        'yaml': '.yaml',
        'xml': '.xml',
        'markdown': '.md',
        'text': '.txt'
    }
    return extensions.get(language.lower(), '.txt')

def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass instance to a dictionary, handling nested dataclasses."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj