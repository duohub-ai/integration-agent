# agent/agent.py
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from langchain_community.tools import TavilySearchResults
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_react_agent
from langchain_core.tools import Tool
from bs4 import BeautifulSoup
import requests
import logging
import json
from langchain_core.prompts import PromptTemplate
from pathlib import Path

@dataclass
class IntegrationRequest:
    """Data class for integration requests."""
    service_name: str
    integration_type: str
    description: str
    authentication_type: Optional[str] = None
    specific_endpoints: Optional[List[str]] = None

@dataclass
class SearchResult:
    """Data class for processed search results."""
    url: str
    content: str
    is_documentation: bool
    relevance_score: float

class IntegrationAgent:
    """Agent that helps create integrations based on natural language requests."""
    
    def __init__(
        self,
        tavily_api_key: str,
        anthropic_api_key: str,
        search_depth: str = "advanced",
        max_search_results: int = 5
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing IntegrationAgent with search_depth={search_depth}, max_results={max_search_results}")
        self.setup_tools(tavily_api_key, search_depth, max_search_results)
        self.setup_llm(anthropic_api_key)
        self.logger.info("IntegrationAgent initialization complete")
        
    def setup_tools(self, tavily_api_key: str, search_depth: str, max_results: int):
        """Initialize search and documentation tools."""
        self.logger.debug("Setting up integration tools")
        self.search_tool = TavilySearchResults(
            max_results=max_results,
            search_depth=search_depth,
            include_raw_content=True,
            include_domains=[".io", ".com", ".dev", ".org"],  # Focus on documentation domains
        )
        
        self.doc_parser_tool = Tool(
            name="documentation_parser",
            description="Parse and extract information from documentation pages",
            func=self.parse_documentation
        )
        
        self.tools = [self.search_tool, self.doc_parser_tool]
        self.logger.debug(f"Tools setup complete. Configured {len(self.tools)} tools")

    def setup_llm(self, anthropic_api_key: str):
        """Initialize the LLM with appropriate system prompts."""
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key=anthropic_api_key,
            temperature=0.5
        )
        
        system_prompt = """You are an expert system integration engineer. Your task is to:
        1. Analyze integration requirements
        2. Find and understand official documentation
        3. Generate working integration code
        4. Provide clear usage examples
        
        Follow these principles:
        - Always prefer official documentation over third-party sources
        - Focus on security best practices
        - Generate well-documented, production-ready code
        - Include error handling and logging
        - Explain your reasoning clearly
        """
        
        prompt_template = """Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        self.agent_executor = create_react_agent(
            llm=self.llm.bind(system=system_prompt),
            tools=self.tools,
            prompt=prompt
        )

    async def create_integration(self, request: IntegrationRequest) -> Dict[str, Any]:
        """Creates an integration based on the service requirements."""
        self.logger.info(f"Starting integration creation for service: {request.service_name}")
        try:
            # Step 1: Search for documentation
            self.logger.debug("Finding documentation")
            search_results = await self._find_documentation(request)
            self.logger.info(f"Found {len(search_results)} relevant documentation sources")
            
            # Step 2: Analyze documentation and requirements
            self.logger.debug("Analyzing requirements and documentation")
            analysis = await self._analyze_requirements(request, search_results)
            self.logger.info("Requirements analysis complete")
            
            # Step 3: Generate integration code
            self.logger.debug("Generating integration code")
            integration = await self._generate_integration(request, analysis)
            self.logger.info("Integration generation complete")
            
            # Step 4: Save the generated files
            saved_files = self.save_integration(integration, request.service_name)
            self.logger.info(f"Integration files saved to: {saved_files}")
            
            return {
                "status": "success",
                "integration": integration,
                "documentation_sources": search_results,
                "analysis": analysis,
                "saved_files": saved_files
            }
            
        except Exception as e:
            self.logger.error(f"Error creating integration: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "partial_results": locals().get("integration", None)
            }

    async def _find_documentation(self, request: IntegrationRequest) -> List[SearchResult]:
        """Find and validate documentation for the service."""
        self.logger.debug(f"Searching documentation for {request.service_name}")
        search_queries = [
            f"{request.service_name} {request.integration_type} documentation",
            f"{request.service_name} API reference {request.integration_type}",
            f"{request.service_name} developer guides {request.integration_type}"
        ]
        
        results = []
        for query in search_queries:
            self.logger.debug(f"Executing search query: {query}")
            search_response = await self.search_tool.ainvoke({"query": query})
            
            # Process and filter results
            for result in search_response:
                relevance = self._calculate_relevance(result, request)
                if relevance > 0.6:  # Threshold for relevance
                    results.append(SearchResult(
                        url=result["url"],
                        content=result["content"],
                        is_documentation=self._is_documentation_url(result["url"]),
                        relevance_score=relevance
                    ))
        
        self.logger.debug(f"Found {len(results)} results above relevance threshold")
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)

    async def _analyze_requirements(
        self, 
        request: IntegrationRequest,
        search_results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Analyze the integration requirements and documentation."""
        analysis_prompt = f"""
        Analyze the integration requirements for {request.service_name}:
        1. Integration Type: {request.integration_type}
        2. Requirements: {request.description}
        3. Authentication: {request.authentication_type or 'Not specified'}
        
        Based on the documentation found, provide:
        1. Required authentication steps
        2. Key endpoints or features needed
        3. Any rate limits or restrictions
        4. Dependencies required
        5. Potential implementation challenges
        
        Documentation context:
        {json.dumps([result.__dict__ for result in search_results[:3]])}
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
        return self._parse_analysis_response(response.content)

    async def _generate_integration(
        self,
        request: IntegrationRequest,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate the integration code and documentation."""
        # Generate the complete integration code in one go, utilizing the full token limit
        generation_prompt = f"""
        Generate a complete, production-ready integration for {request.service_name}.
        The code must be complete and include ALL of the following:
        
        1. All necessary imports
        2. Complete class definition with proper type hints
        3. Full authentication implementation
        4. All core API methods and functionality
        5. Comprehensive error handling and logging
        6. Helper utilities and support functions
        
        Requirements:
        {request.description}
        
        Analysis context:
        {json.dumps(analysis)}
        
        Important:
        - Include complete docstrings for all classes and methods
        - Implement proper error handling for all API calls
        - Add logging for important operations
        - Follow PEP 8 style guidelines
        - Include type hints for all methods
        - Do not include any other text or comments before or after the code
        - Do not create a dependencies file, this will be created later
        - Do not wrap the file in ```python or ```, return just the code.
        
        Return ONLY the complete Python code without any explanation or markdown.
        The code should be fully functional and ready to use.
        """
        
        # Use the maximum available tokens for the code generation
        code_response = await self.llm.ainvoke(
            [HumanMessage(content=generation_prompt)],
            max_tokens=8000  # Leave some tokens for system messages
        )
        
        integration_code = code_response.content
        
        # Generate README with remaining context
        documentation_prompt = f"""
        Create a comprehensive README.md for the {request.service_name} integration.
        Include:

        1. Overview of the integration
        2. Prerequisites and dependencies
        3. Installation instructions
        4. Configuration steps (including environment variables)
        5. Usage examples with code snippets
        6. API Reference for all public methods

        Base this on:
        - Service: {request.service_name}
        - Type: {request.integration_type}
        - Description: {request.description}
        - Authentication: {request.authentication_type}
        
        Return ONLY the markdown content for README.md.
        Do NOT include license documentation. That is not needed.

        """
        
        docs_response = await self.llm.ainvoke(
            [HumanMessage(content=documentation_prompt)],
            max_tokens=4000  # Sufficient for README generation
        )

        # Generate pyproject.toml with dependencies
        pyproject_prompt = f"""
        Create a pyproject.toml file for the {request.service_name} integration with:
        1. Python version requirement (3.12 - 4.0)
        2. All necessary dependencies based on the integration code
        3. Project metadata and configuration
        4. Build system requirements
        - Do not wrap the file in ```, return just the code.
        
        Integration code context for dependencies:
        {integration_code}
        
        Return ONLY the pyproject.toml content.
        """

        pyproject_response = await self.llm.ainvoke(
            [HumanMessage(content=pyproject_prompt)],
            max_tokens=2000  # Sufficient for pyproject.toml
        )
        
        return {
            "files": {
                "integration.py": integration_code,
                "README.md": docs_response.content,
                "pyproject.toml": pyproject_response.content
            },
            "description": request.description
        }

    def _calculate_relevance(self, result: Dict[str, str], request: IntegrationRequest) -> float:
        """Calculate relevance score for a search result."""
        relevance = 0.0
        content = result["content"].lower()
        
        # Check for official documentation indicators
        if self._is_documentation_url(result["url"]):
            relevance += 0.4
            
        # Check for specific keywords
        keywords = [
            request.service_name.lower(),
            request.integration_type.lower(),
            "api",
            "integration",
            "documentation",
            "guide",
            "reference"
        ]
        
        for keyword in keywords:
            if keyword in content:
                relevance += 0.1
                
        return min(relevance, 1.0)

    @staticmethod
    def _is_documentation_url(url: str) -> bool:
        """Check if URL is likely to be official documentation."""
        doc_indicators = ["docs.", "developer.", "api.", "developers."]
        return any(indicator in url.lower() for indicator in doc_indicators)

    def _parse_analysis_response(self, content: str) -> Dict[str, Any]:
        """Parse the LLM's analysis response into structured data."""
        try:
            self.logger.debug("Parsing analysis response")
            return {
                "raw_analysis": content,
                "parsed_sections": {
                    "authentication": self._extract_section(content, "authentication"),
                    "endpoints": self._extract_section(content, "endpoints"),
                    "rate_limits": self._extract_section(content, "rate limits"),
                    "dependencies": self._extract_section(content, "dependencies"),
                    "challenges": self._extract_section(content, "challenges")
                }
            }
        except Exception as e:
            self.logger.error(f"Error parsing analysis response: {str(e)}")
            return {"raw_analysis": content}

    def _parse_generation_response(self, content: str) -> Dict[str, Any]:
        """Parse the LLM's generation response into structured data."""
        try:
            self.logger.debug("Parsing generation response")
            return {
                "raw_generation": content,
                "parsed_sections": {
                    "setup": self._extract_section(content, "setup"),
                    "authentication": self._extract_section(content, "authentication"),
                    "main_code": self._extract_section(content, "main integration"),
                    "error_handling": self._extract_section(content, "error handling"),
                    "examples": self._extract_section(content, "examples"),
                    "testing": self._extract_section(content, "testing")
                }
            }
        except Exception as e:
            self.logger.error(f"Error parsing generation response: {str(e)}")
            return {"raw_generation": content}

    def _extract_section(self, content: str, section_name: str) -> str:
        """Helper method to extract sections from the LLM response."""
        try:
            # Simple section extraction - can be made more sophisticated
            lower_content = content.lower()
            section_start = lower_content.find(section_name.lower())
            if section_start == -1:
                return ""
            
            # Find the next section or end of content
            next_section = float('inf')
            for section in ["setup", "authentication", "main integration", 
                          "error handling", "examples", "testing"]:
                pos = lower_content.find(section.lower(), section_start + len(section_name))
                if pos != -1 and pos < next_section:
                    next_section = pos
            
            if next_section == float('inf'):
                section_content = content[section_start:]
            else:
                section_content = content[section_start:next_section]
                
            return section_content.strip()
        except Exception as e:
            self.logger.error(f"Error extracting section {section_name}: {str(e)}")
            return ""

    async def parse_documentation(self, url: str) -> Dict[str, Any]:
        """Parse a documentation page to extract structured information."""
        self.logger.debug(f"Parsing documentation from URL: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            result = {
                "title": soup.title.string if soup.title else None,
                "endpoints": self._extract_endpoints(soup),
                "authentication": self._extract_authentication(soup),
                "examples": self._extract_code_examples(soup)
            }
            self.logger.debug(f"Successfully parsed documentation from {url}")
            return result
        except Exception as e:
            self.logger.error(f"Error parsing documentation from {url}: {str(e)}", exc_info=True)
            return {"error": str(e)}

    def _extract_endpoints(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract API endpoints from documentation."""
        # Implementation for extracting endpoints
        pass

    def _extract_authentication(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract authentication information from documentation."""
        # Implementation for extracting authentication info
        pass

    def _extract_code_examples(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract code examples from documentation."""
        # Implementation for extracting code examples
        pass

    def _generate_action_path(self, description: str) -> str:
        """
        Convert an action description into a valid directory name.
        
        Args:
            description: The action description (e.g., "Log a call to a Company in Hubspot")
            
        Returns:
            A sanitized path name (e.g., "log_call_company")
        """
        # Convert to lowercase and remove service name references
        action = description.lower()
        for service in ["hubspot", "intercom", "salesforce"]:  # Add more services as needed
            action = action.replace(f" in {service}", "")
            action = action.replace(f" to {service}", "")
        
        # Remove common filler words
        filler_words = ["a", "the", "to", "and", "or", "in"]
        words = action.split()
        words = [w for w in words if w not in filler_words]
        
        # Join remaining words with underscores
        action_path = "_".join(words)
        
        # Remove any special characters and ensure it's a valid directory name
        action_path = "".join(c if c.isalnum() or c == "_" else "_" for c in action_path)
        action_path = action_path.strip("_")
        
        return action_path

    def save_integration(
        self,
        integration_data: Dict[str, Any],
        service_name: str,
        output_dir: str = "integrations"
    ) -> Dict[str, str]:
        """
        Save generated integration files to disk.
        
        Args:
            integration_data: Dictionary containing integration code and documentation
            service_name: Name of the service for directory structure
            output_dir: Base directory for integrations
            
        Returns:
            Dictionary mapping file names to their paths
        """
        try:
            # Generate action path from the description
            action_path = self._generate_action_path(integration_data.get('description', 'default'))
            
            # Create service and action-specific directory
            service_dir = Path(output_dir) / service_name.lower() / action_path
            service_dir.mkdir(parents=True, exist_ok=True)
            
            saved_files = {}
            
            # Save all generated files
            for filename, content in integration_data['files'].items():
                file_path = service_dir / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                saved_files[filename] = str(file_path)
                self.logger.info(f"Saved {filename} to {file_path}")
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Error saving integration: {str(e)}")
            raise