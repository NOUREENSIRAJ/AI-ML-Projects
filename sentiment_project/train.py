"""
Sentiment Analysis - Model Training Script
Data cleaning, feature extraction, training, evaluation, saving
"""

import pandas as pd
import re
import string
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------- Step 1: Load Data ----------
df = pd.read_csv("sample_data.csv")
print("Dataset loaded:", df.shape)

# ---------- Step 2: Data Cleaning ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove links
    text = re.sub(r"<.*?>", "", text)                     # remove HTML tags
    text = re.sub(r"\d+", "", text)                       # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()               # remove extra spaces
    return text

df["clean_text"] = df["review"].apply(clean_text)

# ---------- Step 3: Feature Extraction (TF-IDF) ----------
# - unigrams + bigrams (ngram_range) so phrases like "blew me away" or "not good"
#   are captured, not just single words in isolation
# - no stop-word removal: words like "not", "never", "no" carry sentiment meaning
#   and removing them was hurting accuracy on negations
# - sublinear_tf dampens the effect of very frequent words
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)
X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]

# ---------- Step 4: Train/Test Split + Model Training ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression generally outperforms Naive Bayes for TF-IDF text
# classification, especially with bigrams and a larger vocabulary.
model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train, y_train)

# ---------- Step 5: Evaluation ----------
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- Step 6: Save Model + Vectorizer ----------
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nModel and vectorizer saved: sentiment_model.pkl, vectorizer.pkl")
