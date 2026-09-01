## Sentiment Analysis Web App#
A three-class sentiment classifier — Positive / Neutral / Negative — trained with scikit-learn and served through an interactive Streamlit interface that returns a label and a confidence score for any sentence you type.

Built with Python, scikit-learn and Streamlit by Noureen Siraj.

## Screenshots#
# The app

<img width="1836" height="851" alt="sentimentanalysis screenshot" src="https://github.com/user-attachments/assets/9a7b4473-3561-45e9-9d28-1a816f4794e9" />


# A positive prediction

<img width="1836" height="851" alt="sentimentanalysis screenshot" src="https://github.com/user-attachments/assets/f32e6ccb-b56a-4015-8a59-841f9472aee4" />


# A neutral prediction

<img width="1836" height="851" alt="sentimentanalysis screenshot" src="https://github.com/user-attachments/assets/37c7841d-fb48-4f4e-a1e0-9223438facf6" />

# A negative Prediction

<img width="1561" height="892" alt="ML" src="https://github.com/user-attachments/assets/5ad99b21-c1bf-45aa-81c0-06ad2bf75c45" />


# What it does
Type a sentence — a product review, a tweet, a comment — and the model returns one of three labels along with how confident it is. Three example sentences are built into the interface so you can try it without thinking of your own.

The confidence percentage is the model's own probability for the predicted class, so a genuinely ambiguous sentence like "It arrived on time, nothing special either way" correctly lands on Neutral with lower confidence, while a clearly enthusiastic sentence scores much higher on Positive.

## How it works
# 1. Text cleaning

Both training and prediction run through the same cleaning function, so the model never sees a differently-shaped input at inference time than it did during training:

lowercase everything
strip URLs and @mentions
strip HTML tags and numbers
remove punctuation and collapse extra whitespace
# 2. Feature extraction — TF-IDF

TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),   # unigrams and bigrams
    sublinear_tf=True,
    min_df=2,
)
Bigrams matter here: "not good" carries the opposite meaning of "good", and a unigram-only model would miss that.

# 3. Model — Logistic Regression

LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
class_weight="balanced" was chosen because sentiment datasets are usually skewed — negative reviews tend to outnumber neutral ones, and without balancing the model learns to ignore the smallest class.

# 4. Evaluation

An 80/20 stratified train-test split, scored with accuracy, a full classification report (precision, recall and F1 per class) and a confusion matrix — not accuracy alone, which can look healthy while a whole class is being missed.

# 5. Persistence

The trained model and the fitted vectorizer are saved with joblib, so the app loads them instantly instead of retraining on every launch.

# Project structure 
sentiment_project/
├── train.py               training pipeline: clean → vectorize → train → evaluate → save
├── app.py                 Streamlit interface
├── sentiment_model.pkl    trained logistic regression model
├── vectorizer.pkl         fitted TF-IDF vectorizer
├── requirements.txt       dependencies
└── screenshots/           images used in this README
# Running it locally
Requires Python 3.9 or newer.

# cd sentiment_project
pip install -r requirements.txt
streamlit run app.py
The app opens at http://localhost:8501. The trained model is already committed, so it works straight away — no training needed.

# Retraining on your own data
train.py expects a CSV named sample_data.csv in the same folder. The dataset itself is not committed to keep the repository small. The Twitter US Airline Sentiment dataset from Kaggle works without any changes.

The script auto-detects the column names, so any CSV with a text column named text, review or tweet and a label column named sentiment, label or airline_sentiment will train correctly.

# python train.py
This overwrites sentiment_model.pkl and vectorizer.pkl and prints the accuracy, classification report and confusion matrix.

## Built with
Tool	Used for
Python	Core language
pandas	Loading and filtering the dataset
scikit-learn	TF-IDF vectorizer, logistic regression, evaluation metrics
joblib	Saving and loading the trained model
Streamlit	The web interface

# What I would improve next#
Swap logistic regression for a fine-tuned transformer and compare F1 per class
Add a calibration step so the confidence percentage is better aligned with real accuracy
Handle negation more explicitly rather than relying on bigrams alone
Show the top contributing words for each prediction so the output is explainable

### Noureen Siraj · github.com/NOUREENSIRAJ · noureensiraj30@gmail.com
