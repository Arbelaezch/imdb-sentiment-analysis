import os
import sys
import logging

# Add src directory to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'api'))

from src.api.app import app, load_model

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def create_app():
    """Create and configure the Flask application"""
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Check if model exists
    model_path = "models/sentiment_model.pkl"
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}")
        logging.error("Please run training first: python src/training/train_model.py")
        raise FileNotFoundError("Model file not found")
    
    # Load model
    if not load_model():
        logging.error("Failed to load model")
        raise RuntimeError("Failed to load model")
    
    logging.info("Application initialized successfully")
    return app

if __name__ == '__main__':
    # For development
    try:
        app = create_app()
        logging.info("Starting development server...")
        logging.info("Open your browser to: http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        sys.exit(1)
else:
    # For production (gunicorn)
    app = create_app()