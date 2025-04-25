from flask import Flask, request, render_template, Blueprint, session
import os
from dotenv import load_dotenv
from backend.config import Config

def create_app():
    app = Flask(__name__)
    load_dotenv()
    
    # Load configuration
    app.config.from_object(Config)
    
    @app.route('/test')
    def test():
        return 'Application is running!'
    
    # Lazy loading of blueprints to reduce initial memory usage
    def register_blueprints():
        from backend.test import bp2
        from backend.test2 import bp1
        app.register_blueprint(bp1, url_prefix='/')
        app.register_blueprint(bp2, url_prefix='/')
    
    register_blueprints()
    
    return app

# Only create the app instance if running directly
if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)