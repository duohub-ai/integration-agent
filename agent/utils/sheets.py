import os
import logging
from typing import List
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from ..tools.type_detector import IntegrationRow

logger = logging.getLogger(__name__)

class GoogleSheetsHandler:
    """Handles interactions with Google Sheets."""
    
    def __init__(self, spreadsheet_id: str):
        self.spreadsheet_id = spreadsheet_id
        self.scopes = ['https://www.googleapis.com/auth/spreadsheets']
        self.credentials = self._get_credentials()
        self.service = build('sheets', 'v4', credentials=self.credentials)
        self.sheet = self.service.spreadsheets()
        
    def _get_credentials(self) -> Credentials:
        """Gets credentials using refresh token from environment."""
        refresh_token = os.getenv("GOOGLE_REFRESH_TOKEN")
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        
        if not all([refresh_token, client_id, client_secret]):
            raise ValueError("Missing required Google OAuth credentials")
        
        credentials = Credentials(
            None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=self.scopes
        )
        
        credentials.refresh(Request())
        return credentials
        
    def get_rows_without_type(self, range_name: str) -> List[IntegrationRow]:
        """Get rows that don't have an integration type specified."""
        try:
            result = self.sheet.values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            
            rows = result.get('values', [])
            rows_without_type = []
            
            for idx, row in enumerate(rows[1:], start=2):  # Skip header
                if len(row) >= 4 and (len(row) < 3 or not row[2].strip()):
                    rows_without_type.append(IntegrationRow(
                        row_number=idx,
                        integration_name=row[1],
                        current_type=None if len(row) < 3 else row[2],
                        action=row[3]
                    ))
            
            return rows_without_type
            
        except HttpError as error:
            logger.error(f"Error reading from Google Sheets: {error}")
            raise
            
    def update_integration_type(self, row_number: int, integration_type: str):
        """Update the integration type for a specific row."""
        try:
            range_name = f'C{row_number}'
            body = {
                'values': [[integration_type]]
            }
            self.sheet.values().update(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                body=body
            ).execute()
            logger.info(f"Updated integration type for row {row_number}")
            
        except HttpError as error:
            logger.error(f"Error updating Google Sheets: {error}")
            raise