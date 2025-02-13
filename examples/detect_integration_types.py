import asyncio
import os
import logging
from dotenv import load_dotenv
from agent.tools.type_detector import IntegrationTypeDetector
from agent.utils.sheets import GoogleSheetsHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Load environment variables
    load_dotenv()
    
    # Get required credentials
    tavily_key = os.getenv("TAVILY_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    spreadsheet_id = os.getenv("SPREADSHEET_ID")
    
    if not all([tavily_key, anthropic_key, spreadsheet_id]):
        raise ValueError("Missing required environment variables")
    
    # Initialize components
    sheets_handler = GoogleSheetsHandler(spreadsheet_id)
    detector = IntegrationTypeDetector(tavily_key, anthropic_key)
    
    # Get rows without integration types
    rows = sheets_handler.get_rows_without_type('A1:D100')  # Adjust range as needed
    
    # Process each row
    for row in rows:
        logger.info(f"Processing integration: {row.integration_name}")
        
        try:
            integration_type = await detector.determine_type(
                row.integration_name,
                row.action
            )
            
            # Update the spreadsheet
            sheets_handler.update_integration_type(row.row_number, integration_type)
            
            print(f"✅ Updated {row.integration_name}: {integration_type}")
            
        except Exception as e:
            logger.error(f"Failed to process {row.integration_name}: {str(e)}")
            continue

if __name__ == "__main__":
    asyncio.run(main())