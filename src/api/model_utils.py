import joblib
import numpy as np
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class SentimentPredictor:
    def __init__(self, model_path):
        """Initialize predictor with trained model"""
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.vectorizer = self.model_data['vectorizer']
        self.performance = self.model_data.get('performance', {})
        self.training_date = self.model_data.get('training_date', 'Unknown')
        
        # Initialize text preprocessor
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            # Download stopwords if not available
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text (same as training)"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def predict(self, review_text):
        """
        Predict sentiment for a single review
        
        Args:
            review_text (str): Raw review text
            
        Returns:
            dict: Prediction result with sentiment and confidence
        """
        # Preprocess text
        cleaned_text = self.clean_text(review_text)
        
        # Vectorize
        text_vector = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(text_vector)[0]
        prediction_proba = self.model.predict_proba(text_vector)[0]
        
        # Calculate confidence (probability of predicted class)
        confidence = float(max(prediction_proba))
        
        # Convert prediction to sentiment label
        sentiment = 'positive' if prediction == 1 else 'negative'
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'negative': float(prediction_proba[0]),
                'positive': float(prediction_proba[1])
            }
        }
    
    def predict_batch(self, review_texts):
        """
        Predict sentiment for multiple reviews
        
        Args:
            review_texts (list): List of raw review texts
            
        Returns:
            list: List of prediction results
        """
        results = []
        for text in review_texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': f'Prediction failed: {str(e)}',
                    'sentiment': None,
                    'confidence': None
                })
        
        return results
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_type': self.performance.get('selected_model', 'Unknown'),
            'training_date': self.training_date,
            'test_accuracy': self.performance.get('test_accuracy', 'Unknown'),
            'cross_val_score': self.performance.get('cross_val_score', 'Unknown'),
            'best_params': self.performance.get('best_params', {}),
            'vectorizer_features': getattr(self.vectorizer, 'max_features', 'Unknown')
        }
    
    def explain_prediction(self, review_text, top_n=10):
        """
        Get top features that influenced the prediction
        
        Args:
            review_text (str): Raw review text
            top_n (int): Number of top features to return
            
        Returns:
            dict: Explanation with top positive and negative features
        """
        # This only works with linear models (LogisticRegression)
        if not hasattr(self.model, 'coef_'):
            return {'error': 'Feature explanation not available for this model type'}
        
        # Preprocess and vectorize
        cleaned_text = self.clean_text(review_text)
        text_vector = self.vectorizer.transform([cleaned_text])
        
        # Get feature names and coefficients
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.model.coef_[0]
        
        # Get features present in this text
        feature_indices = text_vector.nonzero()[1]
        active_features = [(feature_names[i], coefficients[i], text_vector[0, i]) 
                          for i in feature_indices]
        
        # Sort by influence (coefficient * feature value)
        active_features.sort(key=lambda x: abs(x[1] * x[2]), reverse=True)
        
        # Separate positive and negative influences
        positive_features = [(feat, coef, val) for feat, coef, val in active_features 
                           if coef > 0][:top_n]
        negative_features = [(feat, coef, val) for feat, coef, val in active_features 
                           if coef < 0][:top_n]
        
        return {
            'positive_features': [
                {'word': feat, 'influence': float(coef * val)} 
                for feat, coef, val in positive_features
            ],
            'negative_features': [
                {'word': feat, 'influence': float(abs(coef * val))} 
                for feat, coef, val in negative_features
            ]
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the predictor if model exists
    import os
    
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
        
        # Test predictions
        test_reviews = [
            "This movie was absolutely fantastic! Great acting and plot.",
            "Terrible movie, waste of time. Very boring and poorly made.",
            "It was okay, nothing special but not bad either."
        ]
        
        print("Testing predictions:")
        for review in test_reviews:
            result = predictor.predict(review)
            print(f"Review: {review[:50]}...")
            print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
            print()
    else:
        print(f"Model not found. Tried paths: {possible_paths}")
        print("Run training first: python src/training/train_model.py")