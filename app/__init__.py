import os
import logging
from flask import Flask
from app.routes import api_bp

def create_app(config=None):
    """
    Create and configure the Flask application
    
    Args:
        config (object, optional): Configuration object
        
    Returns:
        Flask application
    """
    app = Flask(__name__)
    
    # Load default configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev_key_change_this_in_production'),
        UPLOAD_FOLDER=os.path.join(os.getcwd(), 'storage/schedules'),
        ALLOWED_EXTENSIONS={'pdf'},
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max upload
    )
    
    # Override with provided config if any
    if config:
        app.config.from_object(config)
    
    # Setup logging
    setup_logging(app)
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    # Ensure directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Log application startup
    app.logger.info('Application initialized successfully')
    
    return app

def setup_logging(app):
    """Configure logging for the application"""
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'app.log'))
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the app
    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.INFO)