import os
import json
import ee

def initialize_earth_engine():
    """Initialize Earth Engine with error handling"""
    try:
        # Get the service account credentials from environment variable
        credentials = os.getenv('GEE_SERVICE_ACCOUNT')
        if not credentials:
            print("Warning: GEE_SERVICE_ACCOUNT environment variable not found")
            return False

        try:
            # Parse credentials JSON
            service_account = json.loads(credentials)
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in GEE_SERVICE_ACCOUNT: {str(e)}")
            return False

        try:
            # Create a temporary credentials file
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
            print("Earth Engine initialized successfully")
            return True

        except Exception as e:
            print(f"Warning: Earth Engine initialization failed: {str(e)}")
            # Clean up credentials file if it exists
            if os.path.exists(credentials_path):
                os.remove(credentials_path)
            return False

    except Exception as e:
        print(f"Warning: Earth Engine initialization error: {str(e)}")
        return False 