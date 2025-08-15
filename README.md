# IMDB Movie Review Sentiment Analysis

## Project Overview
This project predicts the sentiment of IMDB movie reviews as **positive** or **negative** using both traditional machine learning and deep learning approaches. The workflow involves **data preprocessing, model training, evaluation, and models ready for real-time predictions**.

## Approach

### 1. Data Preprocessing
- Cleaned text by removing HTML tags, punctuation, numbers, and stopwords.
- Converted text to lowercase and normalized spaces.

### 2. Machine Learning Models
- Multinomial Naive Bayes (NB)
- Logistic Regression (LR)
- Random Forest (RF)
- Used `TfidfVectorizer` for feature extraction.
- Implemented **sklearn Pipelines** for clean workflows and easy deployment.

### 3. Deep Learning Model
- Built a **TensorFlow model** with `TextVectorization`, `Embedding`, and `Dense` layers.
- Applied `GlobalAveragePooling1D` and ReLU activation before the final sigmoid output.

### 4. Evaluation
- Metrics: Accuracy, Precision, Recall, F1 Score
- Confusion matrices and accuracy comparison plots for all models.
- Observed the best ML model: Logistic Regression; the DL model achieved comparable results.

## Tools & Libraries
- **Python:** Core language for development
- **Pandas, NumPy, Matplotlib:** Data handling and visualization
- **NLTK:** Stopwords and text preprocessing
- **Scikit-learn:** ML models, pipelines, evaluation metrics
- **TensorFlow/Keras:** Deep learning model and TextVectorization
- **Joblib:** Save/load ML pipelines
- **Google Colab:** Environment for execution

## Results
- Logistic Regression (ML) achieved high accuracy on test data (**89.52%**)
- Deep Learning model also performed well (**88.92% accuracy**)
