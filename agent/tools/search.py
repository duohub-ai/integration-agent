# agent/tools/search.py
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

@dataclass
class SearchConfig:
    """Configuration for the enhanced search tool."""
    search_depth: str = "advanced"
    max_results: int = 5
    include_domains: List[str] = None
    exclude_domains: List[str] = None
    api_key: Optional[str] = None

    def __post_init__(self):
        if self.include_domains is None:
            self.include_domains = [".io", ".com", ".dev", ".org"]
        if self.exclude_domains is None:
            self.exclude_domains = []

class EnhancedSearchTool:
    """Enhanced search tool with documentation focus and result processing."""
    
    def __init__(self, config: SearchConfig):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing EnhancedSearchTool with config: {config}")
        self.config = config
        
        # Initialize Tavily search
        self.search_tool = TavilySearchResults(
            api_key=config.api_key,
            max_results=config.max_results,
            search_depth=config.search_depth,
            include_raw_content=True,
            include_domains=config.include_domains,
            exclude_domains=config.exclude_domains
        )
    
    async def search(self, query: str, context: Optional[Dict] = None) -> List[Dict]:
        """
        Perform an enhanced search with post-processing of results.
        
        Args:
            query: Search query string
            context: Optional context for result filtering/ranking
            
        Returns:
            List of processed search results
        """
        self.logger.debug(f"Performing search with query: {query}, context: {context}")
        try:
            raw_results = await self.search_tool.ainvoke({"query": query})
            self.logger.debug(f"Received {len(raw_results)} raw results")
            processed_results = self._process_results(raw_results, context)
            self.logger.info(f"Successfully processed {len(processed_results)} results")
            return processed_results
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}", exc_info=True)
            return []
    
    def _process_results(self, results: List[Dict], context: Optional[Dict]) -> List[Dict]:
        """Process and enhance search results."""
        self.logger.debug(f"Processing {len(results)} results with context: {context}")
        processed = []
        
        for result in results:
            # Add metadata and scoring
            processed_result = {
                **result,
                "is_documentation": self._is_documentation_url(result.get("url", "")),
                "relevance_score": self._calculate_relevance(result, context)
            }
            processed.append(processed_result)
        
        # Sort by relevance
        return sorted(processed, key=lambda x: x["relevance_score"], reverse=True)
    
    @staticmethod
    def _is_documentation_url(url: str) -> bool:
        """Check if URL is likely to be official documentation."""
        doc_indicators = [
            "docs.", "developer.", "api.", 
            "developers.", "documentation.",
            "/docs/", "/api/", "/developer/"
        ]
        return any(indicator in url.lower() for indicator in doc_indicators)
    
    def _calculate_relevance(self, result: Dict, context: Optional[Dict]) -> float:
        """Calculate relevance score for a result."""
        score = 0.0
        
        # Base score for documentation URLs
        if self._is_documentation_url(result.get("url", "")):
            score += 0.4
        
        # Context-based scoring
        if context:
            score += self._context_based_score(result, context)
        
        # Content-based scoring
        content = result.get("content", "").lower()
        if "api reference" in content or "developer guide" in content:
            score += 0.2
        if "example" in content or "tutorial" in content:
            score += 0.1
            
        return min(score, 1.0)
    
    def _context_based_score(self, result: Dict, context: Dict) -> float:
        """Calculate context-based relevance score."""
        score = 0.0
        content = result.get("content", "").lower()
        
        # Check for service name
        if context.get("service_name", "").lower() in content:
            score += 0.2
            
        # Check for integration type
        if context.get("integration_type", "").lower() in content:
            score += 0.1
            
        # Check for specific requirements
        for keyword in context.get("keywords", []):
            if keyword.lower() in content:
                score += 0.05
                
        return score

    def as_tool(self) -> Tool:
        """Convert to a LangChain tool."""
        return Tool(
            name="enhanced_search",
            description="Search for documentation and technical information",
            func=self.search
        )

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
        self.logger.info("Initializing IntegrationAgent")
        self.setup_search_tool(tavily_api_key, search_depth, max_search_results)
        self.setup_llm(anthropic_api_key)
        
    def setup_search_tool(self, tavily_api_key: str, search_depth: str, max_results: int):
        """Initialize enhanced search tool."""
        config = SearchConfig(
            search_depth=search_depth,
            max_results=max_results,
            api_key=tavily_api_key
        )
        self.enhanced_search = EnhancedSearchTool(config)
        self.doc_parser_tool = Tool(
            name="documentation_parser",
            description="Parse and extract information from documentation pages",
            func=self.parse_documentation
        )
        self.tools = [self.enhanced_search.as_tool(), self.doc_parser_tool]

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
        
        self.agent_executor = create_react_agent(
            llm=self.llm.bind(system=system_prompt),
            tools=self.tools
        )

    async def create_integration(self, request: IntegrationRequest) -> Dict[str, Any]:
        """
        Creates an integration based on the service requirements.
        
        Args:
            request: IntegrationRequest object containing service details
            
        Returns:
            Dict containing the integration code, documentation, and examples
        """
        self.logger.info(f"Creating integration for service: {request.service_name}")
        try:
            self.logger.debug("Starting documentation search")
            search_results = await self._find_documentation(request)
            self.logger.debug(f"Found {len(search_results)} documentation results")
            
            self.logger.debug("Analyzing requirements")
            analysis = await self._analyze_requirements(request, search_results)
            self.logger.debug("Requirements analysis complete")
            
            self.logger.debug("Generating integration code")
            integration = await self._generate_integration(request, analysis)
            self.logger.info("Integration generation complete")
            
            return {
                "status": "success",
                "integration": integration,
                "documentation_sources": search_results,
                "analysis": analysis
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
        search_queries = [
            f"{request.service_name} {request.integration_type} documentation",
            f"{request.service_name} API reference {request.integration_type}",
            f"{request.service_name} developer guides {request.integration_type}"
        ]
        
        results = []
        for query in search_queries:
            search_response = await self.enhanced_search.search(query)
            
            # Convert to SearchResult objects
            for result in search_response:
                results.append(SearchResult(
                    url=result["url"],
                    content=result["content"],
                    is_documentation=result["is_documentation"],
                    relevance_score=result["relevance_score"]
                ))
        
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
        generation_prompt = f"""
        Based on the analysis of {request.service_name} integration requirements,
        generate a complete integration solution including:
        
        1. Installation and setup instructions
        2. Authentication implementation
        3. Main integration class/module
        4. Error handling
        5. Usage examples
        6. Testing approach
        
        Analysis context:
        {json.dumps(analysis)}
        
        Requirements:
        {request.description}
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=generation_prompt)])
        return self._parse_generation_response(response.content)

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
        # Implementation for parsing analysis response
        pass

    def _parse_generation_response(self, content: str) -> Dict[str, Any]:
        """Parse the LLM's generation response into structured data."""
        # Implementation for parsing generation response
        pass

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
            self.logger.info(f"Successfully parsed documentation from {url}")
            return result
        except Exception as e:
            self.logger.error(f"Error parsing documentation: {str(e)}", exc_info=True)
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