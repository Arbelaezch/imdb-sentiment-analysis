from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import json
from datetime import datetime
import os
import logging

from model_utils import SentimentPredictor

# Initialize Flask app
app = Flask(__name__, template_folder='../../frontend/templates', static_folder='../../frontend/static')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize predictor
predictor = None

def load_model():
    """Load the trained model"""
    global predictor
    try:
        # Try multiple possible paths
        possible_paths = [
            "models/sentiment_model.pkl",
            "../../models/sentiment_model.pkl",
            "../models/sentiment_model.pkl"
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path:
            predictor = SentimentPredictor(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        else:
            logger.error(f"Model file not found. Tried paths: {possible_paths}")
            logger.error("Please ensure training completed and model was saved")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False
    return True

@app.route('/')
def home():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if predictor is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Model not loaded',
            'timestamp': datetime.now().isoformat()
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'message': 'Service is running',
        'model_loaded': True,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information and performance metrics"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        # Try multiple possible paths for performance metrics
        possible_paths = [
            "models/model_performance.json",
            "../../models/model_performance.json",
            "../models/model_performance.json"
        ]
        
        performance = {"message": "Performance metrics not available"}
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    performance = json.load(f)
                break
        
        return jsonify({
            'model_info': predictor.get_model_info(),
            'performance': performance,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': 'Failed to get model info'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment for a given review"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'review_text' not in data:
            return jsonify({'error': 'Missing review_text in request'}), 400
        
        review_text = data['review_text']
        
        # Validate input
        if not review_text or not review_text.strip():
            return jsonify({'error': 'Empty review text'}), 400
        
        if len(review_text) > 10000:  # Reasonable limit
            return jsonify({'error': 'Review text too long (max 10,000 characters)'}), 400
        
        # Make prediction
        result = predictor.predict(review_text)
        
        # Log prediction (for monitoring)
        log_prediction(review_text, result)
        
        return jsonify({
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'prediction_time': datetime.now().isoformat(),
            'review_length': len(review_text)
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict sentiment for multiple reviews"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'reviews' not in data:
            return jsonify({'error': 'Missing reviews array in request'}), 400
        
        reviews = data['reviews']
        
        if not isinstance(reviews, list):
            return jsonify({'error': 'Reviews must be an array'}), 400
        
        if len(reviews) > 100:  # Batch size limit
            return jsonify({'error': 'Too many reviews (max 100 per batch)'}), 400
        
        # Make predictions
        results = []
        for i, review_text in enumerate(reviews):
            if not review_text or not review_text.strip():
                results.append({'error': f'Empty review at index {i}'})
                continue
            
            try:
                result = predictor.predict(review_text)
                results.append({
                    'sentiment': result['sentiment'],
                    'confidence': result['confidence']
                })
            except Exception as e:
                results.append({'error': f'Prediction failed for review {i}: {str(e)}'})
        
        return jsonify({
            'predictions': results,
            'prediction_time': datetime.now().isoformat(),
            'total_reviews': len(reviews)
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': 'Batch prediction failed'}), 500

def log_prediction(review_text, result):
    """Log prediction for monitoring purposes"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'review_length': len(review_text),
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'review_preview': review_text[:100] + '...' if len(review_text) > 100 else review_text
        }
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Append to daily log file
        log_file = f"logs/predictions_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    except Exception as e:
        logger.error(f"Failed to log prediction: {str(e)}")

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Starting Flask application...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Failed to load model. Exiting.")
        exit(1)