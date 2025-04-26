import os
import json
import ee

def initialize_earth_engine():
    try:
        # Get the service account credentials from environment variable
        credentials = os.getenv('GEE_SERVICE_ACCOUNT')
        if credentials:
            # Create a JSON file with the credentials
            service_account = json.loads(credentials)
            credentials_path = 'gee-credentials.json'
            with open(credentials_path, 'w') as f:
                json.dump(service_account, f)
            
            # Initialize Earth Engine with service account
            credentials = ee.ServiceAccountCredentials(
                service_account['client_email'],
                credentials_path
            )
            ee.Initialize(credentials)
            
            # Clean up the credentials file
            os.remove(credentials_path)
        else:
            # Try to initialize without credentials (for local development)
            ee.Initialize()
    except Exception as e:
        print(f"Earth Engine initialization warning: {str(e)}")
        # Continue without Earth Engine if initialization fails
        pass 