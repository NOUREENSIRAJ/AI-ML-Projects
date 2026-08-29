"""
Sentiment Analysis - Model Training Script
Step 2-6: Data cleaning, feature extraction, training, evaluation, saving
"""

import pandas as pd
import re
import string
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------- Step 1: Load Data ----------
df = pd.read_csv("sample_data.csv")
print("Dataset loaded:", df.shape)

# ---------- Step 2: Data Cleaning ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove links
    text = re.sub(r"\d+", "", text)                       # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()               # remove extra spaces
    return text

df["clean_text"] = df["review"].apply(clean_text)

# ---------- Step 3: Feature Extraction (TF-IDF) ----------
vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]

# ---------- Step 4: Train/Test Split + Model Training ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

# ---------- Step 5: Evaluation ----------
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- Step 6: Save Model + Vectorizer ----------
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nModel aur vectorizer save ho gaye: sentiment_model.pkl, vectorizer.pkl")
