import asyncio
import os
from dotenv import load_dotenv
from agent.agent import IntegrationAgent, IntegrationRequest
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

async def main():
    # Load environment variables
    load_dotenv()
    logger = logging.getLogger(__name__)
    logger.info("Starting integration test")
    
    # Get API keys
    tavily_key = os.getenv("TAVILY_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not tavily_key or not anthropic_key:
        logger.error("Missing required API keys")
        raise ValueError("Please set TAVILY_API_KEY and ANTHROPIC_API_KEY in .env file")
    
    logger.info("API keys loaded successfully")
    
    # Initialize the agent
    agent = IntegrationAgent(
        tavily_api_key=tavily_key,
        anthropic_api_key=anthropic_key
    )
    logger.info("Integration agent initialized")
    
    # Create a test integration request
    request = IntegrationRequest(
        service_name="GitHub",
        integration_type="REST API",
        description="Create an integration that can list repositories and create issues with labels"
    )
    logger.info(f"Created integration request for {request.service_name}")
    
    # Generate the integration
    print(f"\nGenerating integration for {request.service_name}...")
    try:
        result = await agent.create_integration(request)
        logger.info("Integration generation completed")
        
        if result["status"] == "success":
            logger.info("Integration created successfully")
            print("\n✅ Integration created successfully!")
            
            print("\n📚 Documentation Sources:")
            for source in result["documentation_sources"][:3]:
                print(f"- {source.url} (Relevance: {source.relevance_score:.2f})")
            
            print("\n🔍 Analysis:")
            print(result["analysis"])
            
            print("\n💻 Integration Code:")
            print(result["integration"])
        else:
            logger.error(f"Integration creation failed: {result['error']}")
            print(f"\n❌ Error creating integration: {result['error']}")
            if result.get("partial_results"):
                logger.warning("Partial results available")
                print("\nPartial results:")
                print(result["partial_results"])
    except Exception as e:
        logger.exception("Unexpected error during integration creation")
        raise

if __name__ == "__main__":
    asyncio.run(main())