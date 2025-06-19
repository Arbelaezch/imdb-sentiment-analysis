import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import joblib

class IMDBDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self):
        """
        Load IMDB dataset from various formats
        Returns: pandas DataFrame with 'review' and 'sentiment' columns
        """
        # Try to detect dataset format
        if os.path.exists(os.path.join(self.data_path, 'train')):
            # Folder structure format (train/pos, train/neg, test/pos, test/neg)
            return self._load_folder_format()
        elif any(f.endswith('.csv') for f in os.listdir(self.data_path)):
            return self._load_csv_format()
        else:
            raise ValueError("Dataset format not recognized. Expected folder structure or CSV file.")
    
    def _load_folder_format(self):
        """Load from folder structure (original IMDB format)"""
        reviews = []
        sentiments = []
        
        # Load training data
        for sentiment in ['pos', 'neg']:
            folder_path = os.path.join(self.data_path, 'train', sentiment)
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith('.txt'):
                        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                            reviews.append(f.read())
                            sentiments.append(1 if sentiment == 'pos' else 0)
        
        # Load test data if available
        for sentiment in ['pos', 'neg']:
            folder_path = os.path.join(self.data_path, 'test', sentiment)
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith('.txt'):
                        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                            reviews.append(f.read())
                            sentiments.append(1 if sentiment == 'pos' else 0)
        
        return pd.DataFrame({
            'review': reviews,
            'sentiment': sentiments
        })
    
    def _load_csv_format(self):
        """Load from CSV file"""
        csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError("No CSV files found in the data directory")
        
        csv_path = os.path.join(self.data_path, csv_files[0])
        print(f"Loading dataset from: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # For IMDB dataset, we expect 'review' and 'sentiment' columns
        if 'review' in df.columns and 'sentiment' in df.columns:
            df_clean = df[['review', 'sentiment']].copy()
        else:
            # Try to identify review and sentiment columns
            possible_review_cols = ['review', 'text', 'comment', 'content']
            possible_sentiment_cols = ['sentiment', 'label', 'rating', 'polarity']
            
            review_col = None
            sentiment_col = None
            
            for col in df.columns:
                if col.lower() in possible_review_cols:
                    review_col = col
                elif col.lower() in possible_sentiment_cols:
                    sentiment_col = col
            
            if not review_col or not sentiment_col:
                print("Available columns:", df.columns.tolist())
                raise ValueError("Could not identify review and sentiment columns")
            
            df_clean = df[[review_col, sentiment_col]].copy()
            df_clean.columns = ['review', 'sentiment']
        
        # Convert sentiment to binary if needed
        if df_clean['sentiment'].dtype == 'object':
            sentiment_map = {'positive': 1, 'negative': 0, 'pos': 1, 'neg': 0}
            df_clean['sentiment'] = df_clean['sentiment'].map(sentiment_map)
            print("Converted sentiment labels to binary (positive=1, negative=0)")
        
        print(f"Sentiment distribution:")
        print(df_clean['sentiment'].value_counts())
        
        # Remove any NaN values
        df_clean = df_clean.dropna()
        print(f"Final dataset shape after removing NaN: {df_clean.shape}")
        
        return df_clean

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text"""
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
    
    def preprocess_dataframe(self, df):
        """Preprocess entire dataframe"""
        print("Preprocessing text data...")
        df_clean = df.copy()
        df_clean['review_clean'] = df_clean['review'].apply(self.clean_text)
        
        # Remove empty reviews
        df_clean = df_clean[df_clean['review_clean'].str.len() > 0]
        
        print(f"Dataset shape after preprocessing: {df_clean.shape}")
        return df_clean

def load_and_preprocess_data(data_path, test_size=0.2, random_state=42):
    """
    Complete data loading and preprocessing pipeline
    """
    # Load data
    loader = IMDBDataLoader(data_path)
    df = loader.load_data()
    print(f"Loaded {len(df)} reviews")
    print(f"Sentiment distribution: {df['sentiment'].value_counts()}")
    
    # Preprocess text
    preprocessor = TextPreprocessor()
    df_clean = preprocessor.preprocess_dataframe(df)
    
    # Split data
    X = df_clean['review_clean']
    y = df_clean['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Save preprocessor for later use
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    data_path = "data/raw"
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    # Save processed data
    pd.DataFrame({
        'review': X_train,
        'sentiment': y_train
    }).to_csv('data/processed/train.csv', index=False)
    
    pd.DataFrame({
        'review': X_test,
        'sentiment': y_test
    }).to_csv('data/processed/test.csv', index=False)