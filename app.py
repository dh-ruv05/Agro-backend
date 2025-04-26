from flask import Flask, request, render_template, Blueprint, session
from flask_cors import CORS
import os
from dotenv import load_dotenv
from backend.config import Config

def create_app():
    app = Flask(__name__)
    load_dotenv()
    
    # Load configuration
    app.config.from_object(Config)
    
    # Configure CORS
    CORS(app, resources={
        r"/*": {
            "origins": [
                "http://localhost:3000",  # Local frontend
                "https://your-frontend-domain.vercel.app",  # Vercel domain
                "https://*.onrender.com"  # Render domains
            ],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    @app.route('/test')
    def test():
        return {'status': 'success', 'message': 'Application is running!'}
    
    # Lazy loading of blueprints to reduce initial memory usage
    def register_blueprints():
        from backend.test import bp2
        from backend.test2 import bp1
        app.register_blueprint(bp1, url_prefix='/')
        app.register_blueprint(bp2, url_prefix='/')
    
    register_blueprints()
    
    return app

# Create the app instance
app = create_app()

# Only run the app if running directly
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)