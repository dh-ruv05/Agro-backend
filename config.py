import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # reCAPTCHA Configuration
    RECAPTCHA_SITE_KEY = os.getenv('RECAPTCHA_SITE_KEY', 'YOUR_SITE_KEY')
    RECAPTCHA_SECRET_KEY = os.getenv('RECAPTCHA_SECRET_KEY', 'YOUR_SECRET_KEY')
    
    # Database Configuration
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///agro_nexus.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Session Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    
    # Other Settings
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true' 