from dataclasses import dataclass
from typing import Optional
import logging
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

@dataclass
class IntegrationRow:
    """Data class for integration spreadsheet rows."""
    row_number: int
    integration_name: str
    current_type: Optional[str]
    action: str

class IntegrationTypeDetector:
    """Tool for determining integration types based on documentation searches."""
    
    def __init__(self, tavily_api_key: str, anthropic_api_key: str):
        self.logger = logging.getLogger(__name__)
        self.setup_tools(tavily_api_key)
        self.setup_llm(anthropic_api_key)
        
    def setup_tools(self, tavily_api_key: str):
        """Initialize search tools."""
        self.search_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_raw_content=True,
            include_domains=[".io", ".com", ".dev", ".org"],
        )
        
    def setup_llm(self, anthropic_api_key: str):
        """Initialize the LLM with appropriate system prompts."""
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key=anthropic_api_key,
            temperature=0.1
        )
        
        system_prompt = """You are an expert at analyzing software integrations and APIs.
        Your task is to determine the most appropriate integration type for a given service
        based on its documentation and requirements.
        
        Common integration types include:
        - REST API
        - GraphQL API
        - SOAP API
        - Webhook
        - SDK
        - OAuth
        - Event-driven
        - Batch Processing
        - File-based
        - Database
        
        Provide specific integration types, not generic descriptions.
        If multiple types apply, list the primary one first."""
        
        self.llm = self.llm.bind(system=system_prompt)

    async def determine_type(self, integration_name: str, action: str) -> str:
        """Determine the integration type based on service name and action."""
        search_query = f"{integration_name} API integration documentation"
        
        try:
            # Search for documentation
            search_results = await self.search_tool.ainvoke({"query": search_query})
            
            # Analyze results with LLM
            analysis_prompt = f"""
            Determine the most appropriate integration type for {integration_name}.
            
            Integration requirements:
            {action}
            
            Documentation found:
            {search_results[:3]}
            
            Based on the documentation and requirements, what is the primary integration type?
            Respond with ONLY the integration type (e.g., "REST API", "GraphQL API", "Webhook", etc.).
            Do not include any explanation or additional text.
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error determining integration type: {str(e)}")
            raise