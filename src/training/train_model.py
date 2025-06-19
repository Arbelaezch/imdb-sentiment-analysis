import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import load_and_preprocess_data

class SentimentModelTrainer:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.model_performance = {}
        
    def create_features(self, X_train, X_test):
        """Create TF-IDF features"""
        print("Creating TF-IDF features...")
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Keep top 10k features
            min_df=2,           # Word must appear in at least 2 documents
            max_df=0.8,         # Ignore words that appear in >80% of documents
            ngram_range=(1, 2), # Use unigrams and bigrams
            stop_words='english'
        )
        
        # Fit and transform training data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
        return X_train_tfidf, X_test_tfidf
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train and tune Logistic Regression model"""
        print("Training Logistic Regression...")
        
        # Hyperparameter tuning
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'max_iter': [1000]
        }
        
        lr = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_lr = grid_search.best_estimator_
        
        # Predictions
        y_pred_train = best_lr.predict(X_train)
        y_pred_test = best_lr.predict(X_test)
        
        # Performance metrics
        performance = {
            'model_type': 'Logistic Regression',
            'best_params': grid_search.best_params_,
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
            'cross_val_score': cross_val_score(best_lr, X_train, y_train, cv=5).mean()
        }
        
        return best_lr, performance
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train and tune Random Forest model"""
        print("Training Random Forest...")
        
        # Convert sparse matrix to dense for Random Forest
        X_train_dense = X_train.toarray()
        X_test_dense = X_test.toarray()
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_dense, y_train)
        
        # Best model
        best_rf = grid_search.best_estimator_
        
        # Predictions
        y_pred_train = best_rf.predict(X_train_dense)
        y_pred_test = best_rf.predict(X_test_dense)
        
        # Performance metrics
        performance = {
            'model_type': 'Random Forest',
            'best_params': grid_search.best_params_,
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
            'cross_val_score': cross_val_score(best_rf, X_train_dense, y_train, cv=3).mean()
        }
        
        return best_rf, performance
    
    def compare_models(self, X_train, y_train, X_test, y_test):
        """Train and compare multiple models"""
        models_performance = {}
        
        # Logistic Regression
        lr_model, lr_perf = self.train_logistic_regression(X_train, y_train, X_test, y_test)
        models_performance['logistic_regression'] = {
            'model': lr_model,
            'performance': lr_perf
        }
        
        # Random Forest (only if dataset is not too large)
        if X_train.shape[0] < 40000:  # Skip RF for very large datasets
            rf_model, rf_perf = self.train_random_forest(X_train, y_train, X_test, y_test)
            models_performance['random_forest'] = {
                'model': rf_model,
                'performance': rf_perf
            }
        
        # Select best model based on test accuracy
        best_model_name = max(
            models_performance.keys(),
            key=lambda k: models_performance[k]['performance']['test_accuracy']
        )
        
        self.model = models_performance[best_model_name]['model']
        self.model_performance = models_performance[best_model_name]['performance']
        self.model_performance['selected_model'] = best_model_name
        
        print(f"\nBest model: {best_model_name}")
        print(f"Test accuracy: {self.model_performance['test_accuracy']:.4f}")
        
        return models_performance
    
    def save_model(self, model_path="models/sentiment_model.pkl"):
        """Save trained model and vectorizer"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'performance': self.model_performance,
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
        
        # Save performance metrics as JSON
        with open('models/model_performance.json', 'w') as f:
            json.dump(self.model_performance, f, indent=2)
    
    def plot_confusion_matrix(self, X_test, y_test, save_path="models/confusion_matrix.png"):
        """Plot and save confusion matrix"""
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'], 
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")

def main():
    """Main training pipeline"""
    print("Starting IMDB Sentiment Analysis Training Pipeline")
    print("=" * 50)
    
    # Load and preprocess data
    data_path = "data/raw"  
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    # Initialize trainer
    trainer = SentimentModelTrainer()
    
    # Create features
    X_train_tfidf, X_test_tfidf = trainer.create_features(X_train, X_test)
    
    # Train and compare models
    models_performance = trainer.compare_models(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # Save model
    trainer.save_model()
    
    # Plot confusion matrix
    trainer.plot_confusion_matrix(X_test_tfidf, y_test)
    
    # Print final results
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Best Model: {trainer.model_performance['selected_model']}")
    print(f"Test Accuracy: {trainer.model_performance['test_accuracy']:.4f}")
    print(f"Cross-validation Score: {trainer.model_performance['cross_val_score']:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    report = trainer.model_performance['classification_report']
    print(f"Precision (Negative): {report['0']['precision']:.4f}")
    print(f"Recall (Negative): {report['0']['recall']:.4f}")
    print(f"F1-score (Negative): {report['0']['f1-score']:.4f}")
    print(f"Precision (Positive): {report['1']['precision']:.4f}")
    print(f"Recall (Positive): {report['1']['recall']:.4f}")
    print(f"F1-score (Positive): {report['1']['f1-score']:.4f}")
    
    return trainer

if __name__ == "__main__":
    trainer = main()