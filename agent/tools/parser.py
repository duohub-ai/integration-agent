# agent/tools/parser.py
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from bs4 import BeautifulSoup
import requests
import re
import logging
from urllib.parse import urljoin

@dataclass
class CodeExample:
    """Represents a code example from documentation."""
    language: str
    code: str
    description: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

@dataclass
class Endpoint:
    """Represents an API endpoint."""
    path: str
    method: str
    description: Optional[str] = None
    parameters: Optional[List[Dict[str, Any]]] = None
    response: Optional[Dict[str, Any]] = None
    authentication: Optional[str] = None

class DocumentationParser:
    """Parser for API and integration documentation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # Set default log level
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Documentation Parser Bot/1.0'
        })
    
    async def parse_documentation(self, url: str) -> Dict[str, Any]:
        """Parse a documentation page to extract structured information."""
        self.logger.info(f"Starting documentation parsing for URL: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            self.logger.debug(f"Successfully fetched documentation from {url}")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract overview from introduction div
            overview = None
            intro_div = soup.find('div', class_='introduction')
            if intro_div:
                overview = intro_div.text.strip()
                self.logger.debug("Found introduction section")
            else:
                self.logger.debug("No introduction section found")
            
            examples = self._extract_code_examples(soup)
            self.logger.info(f"Found {len(examples)} code examples")
            
            result = {
                "title": soup.title.string if soup.title else None,
                "overview": overview,
                "endpoints": self._extract_endpoints(soup),
                "authentication": self._extract_authentication(soup),
                "examples": examples
            }
            self.logger.info("Documentation parsing completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing documentation: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract the page title."""
        if soup.title:
            return soup.title.string.strip()
        
        # Try common title elements
        for selector in ['h1', '.page-title', '.documentation-title']:
            element = soup.select_one(selector)
            if element:
                return element.text.strip()
        return None
    
    def _extract_overview(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract documentation overview."""
        # Look for common overview sections
        selectors = [
            '.introduction', '.overview', '#overview',
            'section[role="main"] > p:first-of-type'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.text.strip()
        return None
    
    def _extract_authentication(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract authentication information."""
        self.logger.debug("Extracting authentication information")
        auth_info = {}
        
        auth_section = soup.find('div', class_='authentication')
        if auth_section:
            auth_info['instructions'] = auth_section.get_text(strip=True)
            auth_info['type'] = 'API Key' if 'API Key' in auth_section.get_text() else None
            code_block = auth_section.find('code')
            if code_block:
                auth_info['example'] = code_block.get_text(strip=True)
            self.logger.debug(f"Found authentication info of type: {auth_info.get('type')}")
        else:
            self.logger.debug("No authentication section found")
        
        return auth_info
    
    def _extract_endpoints(self, soup: BeautifulSoup) -> List[Endpoint]:
        """Extract API endpoints."""
        self.logger.debug("Extracting API endpoints")
        endpoints = []
        endpoint_div = soup.find('div', class_='endpoint')
        
        if endpoint_div:
            endpoint_title = endpoint_div.find('h3')
            if endpoint_title:
                method_path = endpoint_title.get_text(strip=True).split(' ')
                if len(method_path) >= 2:
                    endpoints.append(Endpoint(
                        method=method_path[0],
                        path=method_path[1],
                        description=endpoint_div.find('p', class_='description').get_text(strip=True)
                        if endpoint_div.find('p', class_='description') else None
                    ))
                    self.logger.debug(f"Found endpoint: {method_path[0]} {method_path[1]}")
        
        self.logger.info(f"Extracted {len(endpoints)} endpoints")
        return endpoints
    
    def _extract_code_examples(self, soup: BeautifulSoup) -> List[CodeExample]:
        """Extract code examples."""
        self.logger.debug("Extracting code examples")
        examples = []
        pre_blocks = soup.find_all('pre', class_=lambda x: x and 'language-' in x)
        
        for block in pre_blocks:
            language = block.get('class', [''])[0].replace('language-', '')
            code = block.get_text(strip=True)
            description = None
            
            prev_elem = block.find_previous(['p', 'h3', 'h4'])
            if prev_elem:
                description = prev_elem.get_text(strip=True)
            
            examples.append(CodeExample(
                language=language,
                code=code,
                description=description
            ))
            self.logger.debug(f"Found code example in language: {language}")
        
        self.logger.info(f"Extracted {len(examples)} code examples")
        return examples
    
    def _extract_requirements(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract integration requirements."""
        requirements = {}
        
        # Look for requirements sections
        req_sections = soup.find_all(['div', 'section'], 
            string=re.compile(r'requirements|prerequisites', re.I))
        
        for section in req_sections:
            # Extract version requirements
            version_match = re.search(r'version (\d+\.[\d\.x]+)', section.text)
            if version_match:
                requirements['version'] = version_match.group(1)
            
            # Extract dependencies
            dependencies = []
            lists = section.find_all(['ul', 'ol'])
            for lst in lists:
                dependencies.extend([item.text.strip() for item in lst.find_all('li')])
            
            if dependencies:
                requirements['dependencies'] = dependencies
        
        return requirements
    
    @staticmethod
    def _detect_language(code_block: BeautifulSoup) -> str:
        """Detect programming language of code block."""
        # Check class attributes
        classes = code_block.get('class', [])
        for class_ in classes:
            if 'language-' in class_:
                return class_.replace('language-', '')
            
        # Try to detect from content
        content = code_block.text.lower()
        if 'import' in content and ('def' in content or 'class' in content):
            return 'python'
        elif '{' in content and (':' in content or '=' in content):
            return 'json'
        elif '<' in content and '>' in content:
            return 'xml'
        
        return 'unknown'
    
    @staticmethod
    def _extract_description(section: BeautifulSoup) -> Optional[str]:
        """Extract description from a section."""
        desc_elem = section.find(['p', 'div'], class_=re.compile(r'description'))
        return desc_elem.text.strip() if desc_elem else None
    
    @staticmethod
    def _extract_parameters(section: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract parameters from a section."""
        params = []
        param_tables = section.find_all('table', 
            class_=re.compile(r'parameters|params'))
        
        for table in param_tables:
            rows = table.find_all('tr')
            for row in rows[1:]:  # Skip header row
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    params.append({
                        'name': cells[0].text.strip(),
                        'description': cells[1].text.strip(),
                        'required': 'required' in row.text.lower()
                    })
        
        return params
    
    @staticmethod
    def _extract_response(section: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Extract response information from a section."""
        response = {}
        
        # Look for response examples
        response_blocks = section.find_all(['pre', 'code'], 
            class_=re.compile(r'response|example'))
        
        if response_blocks:
            response['examples'] = [block.text.strip() for block in response_blocks]
            
        # Look for response schema
        schema_blocks = section.find_all(['pre', 'code'], 
            class_=re.compile(r'schema'))
            
        if schema_blocks:
            response['schema'] = schema_blocks[0].text.strip()
            
        return response if response else None