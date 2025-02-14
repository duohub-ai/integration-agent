import asyncio
import os
import pickle
from dotenv import load_dotenv
from agent.agent import IntegrationAgent, IntegrationRequest
import logging
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleSheetsProcessor:
    def __init__(self, spreadsheet_id):
        """
        Initialize the Google Sheets processor
        
        Args:
            spreadsheet_id (str): ID of the Google Sheet to process
        """
        self.spreadsheet_id = spreadsheet_id
        self.scopes = ['https://www.googleapis.com/auth/spreadsheets']
        
        # Get credentials and create service
        self.credentials = self.get_credentials_with_refresh_token()
        self.service = build('sheets', 'v4', credentials=self.credentials)
        self.sheet = self.service.spreadsheets()

    def get_credentials_with_refresh_token(self):
        """Gets credentials using an existing refresh token"""
        refresh_token = os.getenv("GOOGLE_REFRESH_TOKEN")
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        
        if not all([refresh_token, client_id, client_secret]):
            raise ValueError("Missing required Google OAuth credentials in environment variables")
        
        credentials = Credentials(
            None,  # No access token initially
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=self.scopes
        )
        
        # Force a refresh to get a valid access token
        credentials.refresh(Request())
        return credentials

    def get_unprocessed_rows(self, range_name):
        """Get all rows that haven't been marked as completed"""
        try:
            result = self.sheet.values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            
            rows = result.get('values', [])
            unprocessed_rows = []
            
            for idx, row in enumerate(rows[1:], start=2):  # Skip header row
                # Check if row exists and has enough columns
                if len(row) >= 4:  # Ensure we have all required columns
                    # Check if first column is FALSE or empty (unchecked checkbox)
                    if not row[0] or row[0] == 'FALSE':
                        unprocessed_rows.append({
                            'row_number': idx,
                            'integration_name': row[1],
                            'type': row[2],
                            'instruction': row[3]
                        })
            
            return unprocessed_rows
        except HttpError as error:
            logger.error(f"Error reading from Google Sheets: {error}")
            raise

    def mark_row_complete(self, row_number):
        """Mark a row as completed by checking the checkbox"""
        try:
            range_name = f'A{row_number}'
            body = {
                'values': [[True]]
            }
            self.sheet.values().update(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                body=body
            ).execute()
            logger.info(f"Marked row {row_number} as complete")
        except HttpError as error:
            logger.error(f"Error updating Google Sheets: {error}")
            raise

async def main():
    # Load environment variables
    load_dotenv()
    
    # Get API keys and configuration
    tavily_key = os.getenv("TAVILY_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    spreadsheet_id = os.getenv("SPREADSHEET_ID")
    
    # Add API keys for testing
    api_keys = {
        'hubspot': os.getenv('HUBSPOT_API_KEY'),
        'intercom': os.getenv('INTERCOM_API_KEY'),
        'opentable': os.getenv('OPENTABLE_API_KEY'),
        'google': os.getenv('GOOGLE_REFRESH_TOKEN')
    }
    
    # Log environment variable status
    logger.info("Checking environment variables:")
    logger.info(f"Tavily API Key present: {bool(tavily_key)}")
    logger.info(f"Anthropic API Key present: {bool(anthropic_key)}")
    logger.info(f"Spreadsheet ID present: {bool(spreadsheet_id)}")
    logger.info("Available testing credentials:")
    for service, key in api_keys.items():
        logger.info(f"- {service.title()}: {bool(key)}")
    
    if not all([tavily_key, anthropic_key, spreadsheet_id]):
        raise ValueError("Missing required environment variables")
    
    try:
        # Initialize the sheets processor
        logger.info("Initializing Google Sheets processor...")
        sheets_processor = GoogleSheetsProcessor(
            spreadsheet_id=spreadsheet_id
        )
        logger.info("Google Sheets processor initialized successfully")
        
        # Initialize the integration agent
        logger.info("Initializing Integration Agent...")
        agent = IntegrationAgent(
            tavily_api_key=tavily_key,
            anthropic_api_key=anthropic_key
        )
        logger.info("Integration Agent initialized successfully")
        
        # Create integrations directory
        integrations_dir = Path("integrations")
        integrations_dir.mkdir(exist_ok=True)
        logger.info(f"Created/verified integrations directory at: {integrations_dir.absolute()}")
        
    except Exception as e:
        logger.exception("Failed during initialization:")
        raise

    # Get unprocessed rows
    try:
        logger.info("Fetching unprocessed rows...")
        unprocessed_rows = sheets_processor.get_unprocessed_rows('A1:D100')  # Adjust range as needed
        logger.info(f"Found {len(unprocessed_rows)} unprocessed rows")
    except Exception as e:
        logger.exception("Failed to fetch unprocessed rows:")
        raise

    # Process each row
    for row in unprocessed_rows:
        logger.info(f"Processing integration: {row['integration_name']}")
        
        request = IntegrationRequest(
            service_name=row['integration_name'],
            integration_type=row['type'],
            description=row['instruction']
        )
        
        try:
            result = await agent.create_integration(request)
            
            if result["status"] == "success":
                logger.info(f"Integration {row['integration_name']} created successfully")
                
                # Test integration if credentials are available
                service_name = row['integration_name'].lower()
                if any(service in service_name and key for service, key in api_keys.items()):
                    logger.info(f"Found credentials for {service_name}, running integration test...")
                    try:
                        # Import and test the integration
                        import importlib.util
                        import sys
                        
                        # Get the main integration file path
                        integration_file = Path(result['saved_files']['integration.py'])
                        
                        # Import the module
                        spec = importlib.util.spec_from_file_location(
                            f"integration_test_{service_name}",
                            integration_file
                        )
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[spec.name] = module
                        spec.loader.exec_module(module)
                        
                        # Run the test method if it exists
                        if hasattr(module, 'test_integration'):
                            test_result = await module.test_integration()
                            logger.info(f"Integration test result: {test_result}")
                            print(f"\n🧪 Integration test completed: {test_result}")
                        else:
                            logger.warning("No test_integration method found in the module")
                    except Exception as test_error:
                        logger.error(f"Integration test failed: {str(test_error)}")
                        print(f"\n⚠️ Integration test failed: {str(test_error)}")
                
                # Mark row as complete
                sheets_processor.mark_row_complete(row['row_number'])
                
                # Log success details
                print(f"\n✅ Integration {row['integration_name']} created successfully!")
                print(f"\n💾 Integration saved to: {result['saved_files']}")
                
                print("\n📚 Documentation Sources:")
                for source in result["documentation_sources"][:3]:
                    print(f"- {source.url} (Relevance: {source.relevance_score:.2f})")
            else:
                logger.error(f"Failed to create integration {row['integration_name']}: {result['error']}")
                print(f"\n❌ Error creating integration: {result['error']}")
        
        except Exception as e:
            logger.exception(f"Unexpected error processing {row['integration_name']}")
            continue

if __name__ == "__main__":
    asyncio.run(main())