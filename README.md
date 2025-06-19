# 🎬 IMDB Movie Review Sentiment Analyzer

A production-ready machine learning application that classifies movie reviews as positive or negative sentiment using natural language processing. Built with scikit-learn, Flask, and deployed on Render.

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.3+-green.svg)
![Docker](https://img.shields.io/badge/docker-supported-blue.svg)
![Render](https://img.shields.io/badge/deployed%20on-Render-46E3B7.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Live Demo

**[Try the live application →](https://imdb-sentiment-analysis-nbmd.onrender.com/)**

Test it with your own movie reviews or use the built-in examples to see real-time sentiment analysis in action!

## 📊 Model Performance

- **Accuracy**: 89.36%
- **Model**: Logistic Regression with TF-IDF
- **Training Data**: 50,000 IMDB movie reviews
- **Features**: 10,000 TF-IDF vectors (unigrams + bigrams)

| Metric | Negative Reviews | Positive Reviews |
|--------|------------------|------------------|
| Precision | 90.16% | 88.59% |
| Recall | 88.36% | 90.36% |
| F1-Score | 89.25% | 89.47% |

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Web Interface │────│  Flask API   │────│  ML Pipeline    │
│   (HTML/JS/CSS) │    │  (REST API)  │    │  (scikit-learn) │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │
                       ┌──────────────┐
                       │   Monitoring │    🌐 Deployed on Render
                       │   & Logging  │    📊 89.36% Accuracy
                       └──────────────┘    ⚡ <100ms Response
```

## 🛠️ Technologies Used

- **Machine Learning**: scikit-learn, pandas, nltk, matplotlib
- **Web Framework**: Flask, Gunicorn
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Deployment**: Render, Docker-ready
- **Data Processing**: TF-IDF vectorization, text preprocessing
- **Monitoring**: Custom logging, health checks, prediction tracking

## 📋 Prerequisites

- Python 3.11+
- Docker (for deployment)
- AWS Account (for cloud deployment)
- 4GB+ RAM recommended for training

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### 3. Get the Dataset
Download the IMDB dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place `IMDB-dataset.csv` in the `data/raw/` folder.

### 4. Train the Model
```bash
python src/training/train_model.py
```

### 5. Run the Application
```bash
python run_app.py
```

Visit `http://localhost:5000` to use the web interface.

## 🐳 Docker Deployment

### Build and Run Locally
```bash
# Build Docker image
docker build -t imdb-sentiment-api .

# Run container
docker run -d \
  --name sentiment-api \
  -p 5000:5000 \
  -v $(pwd)/logs:/app/logs \
  imdb-sentiment-api
```

### Using Docker Compose
```bash
docker-compose up -d
```

## ☁️ Deployment

### Live Production Deployment
- **Platform**: [Render](https://render.com) (Free Tier)
- **URL**: https://imdb-sentiment-analysis-nbmd.onrender.com/
- **Features**: Automatic HTTPS, Health monitoring, Auto-scaling
- **Performance**: ~100ms response time, 99%+ uptime

### Deployment Architecture
```yaml
# render.yaml - Production configuration
services:
  - type: web
    name: imdb-sentiment-api
    buildCommand: |
      pip install -r requirements.txt
      python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
    startCommand: gunicorn --bind 0.0.0.0:$PORT --workers 1 run_app:app
    healthCheckPath: /health
```

### Local Development with Docker
```bash
# Build and run locally
docker build -t imdb-sentiment-api .
docker run -p 5000:5000 imdb-sentiment-api
```

### Alternative Deployment Options
- **AWS EC2**: Production-scale deployment (~$15-20/month)
- **Railway**: Alternative free tier with generous limits
- **Docker Compose**: Multi-service local development

## 📡 API Endpoints

Base URL: `https://imdb-sentiment-analysis-nbmd.onrender.com`

### Health Check
```bash
GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-06-18T10:30:00Z"
}
```

### Predict Sentiment
```bash
POST /predict
Content-Type: application/json

{
  "review_text": "This movie was absolutely fantastic!"
}

Response:
{
  "sentiment": "positive",
  "confidence": 0.87,
  "prediction_time": "2025-06-18T10:30:00Z"
}
```

### Batch Predictions
```bash
POST /batch_predict
Content-Type: application/json

{
  "reviews": [
    "Great movie!",
    "Terrible film, waste of time."
  ]
}

Response:
{
  "predictions": [
    {"sentiment": "positive", "confidence": 0.92},
    {"sentiment": "negative", "confidence": 0.88}
  ],
  "total_reviews": 2
}
```

### Model Information
```bash
GET /model_info

Response:
{
  "model_info": {
    "model_type": "logistic_regression",
    "test_accuracy": 0.8936,
    "training_date": "2025-06-18T10:00:00Z"
  },
  "performance": {
    "test_accuracy": 0.8936,
    "cross_val_score": 0.8861
  }
}
```

## 📊 Project Structure

```
imdb-sentiment-classifier/
├── data/
│   ├── raw/                    # Raw dataset files
│   └── processed/              # Processed data
├── src/
│   ├── training/
│   │   ├── train_model.py      # Model training pipeline
│   │   └── preprocess.py       # Data preprocessing
│   ├── api/
│   │   ├── app.py              # Flask application
│   │   └── model_utils.py      # Model utilities
│   └── monitoring/
│       └── monitor.py          # Monitoring utilities
├── models/
│   ├── sentiment_model.pkl     # Trained model
│   ├── preprocessor.pkl        # Text preprocessor
│   └── model_performance.json  # Performance metrics
├── frontend/
│   ├── templates/
│   │   └── index.html          # Web interface
│   └── static/                 # CSS/JS assets
├── logs/                       # Application logs
├── tests/                      # Unit tests
├── Dockerfile                  # Container configuration
├── docker-compose.yml          # Multi-service deployment
├── requirements.txt            # Python dependencies
└── run_app.py                  # Application entry point
```

## 📧 Contact

**Christian Arbelaez** - arbelaezch@gmail.com

Project Link: [https://github.com/Arbelaezch/imdb-sentiment-analysis](https://github.com/Arbelaezch/imdb-sentiment-analysis)

---

## 📈 Performance Metrics

### Response Times
- **Average**: <100ms
- **95th percentile**: <200ms
- **Health check**: <50ms

### Scalability
- **Concurrent users**: 50+ (single instance)
- **Requests per second**: 10-20
- **Memory usage**: ~200MB

### Model Metrics
```python
                precision    recall  f1-score   support

    negative       0.90      0.88      0.89      4961
    positive       0.89      0.90      0.89      5039

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000
```

---

*Built with ❤️ for demonstrating production ML skills*