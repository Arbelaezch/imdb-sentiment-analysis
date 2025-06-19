import os
import sys

# Add src directory to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'api'))

from src.api.app import app, load_model
import logging

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Check if model exists
    model_path = "models/sentiment_model.pkl"
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        logger.error("Please run training first: python src/training/train_model.py")
        sys.exit(1)
    
    # Load model
    if load_model():
        logger.info("Model loaded successfully!")
        logger.info("Starting Flask application...")
        logger.info("Open your browser to: http://localhost:5000")
        
        # Run the app
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Failed to load model. Exiting.")
        sys.exit(1)