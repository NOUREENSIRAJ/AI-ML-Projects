"""
Sentiment Analysis - Model Training Script (3-class: Positive / Neutral / Negative)
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
# Expects a dataset with a text column and a 3-class sentiment column
# (positive / neutral / negative). The Twitter US Airline Sentiment dataset
# from Kaggle works directly: columns "text" and "airline_sentiment".
df = pd.read_csv("sample_data.csv")
print("Dataset loaded:", df.shape)

# Auto-detect the text and label columns so this works with slightly
# different column names across datasets.
text_col_candidates = ["text", "review", "tweet"]
label_col_candidates = ["airline_sentiment", "sentiment", "label"]

text_col = next((c for c in text_col_candidates if c in df.columns), None)
label_col = next((c for c in label_col_candidates if c in df.columns), None)

if text_col is None or label_col is None:
    raise ValueError(
        f"Could not find expected columns. Found columns: {list(df.columns)}. "
        f"Expected a text column (one of {text_col_candidates}) and a label "
        f"column (one of {label_col_candidates})."
    )

df = df[[text_col, label_col]].dropna()
df.columns = ["text", "sentiment"]
df["sentiment"] = df["sentiment"].str.lower().str.strip()
df = df[df["sentiment"].isin(["positive", "neutral", "negative"])]
print("Class distribution:\n", df["sentiment"].value_counts())

# ---------- Step 2: Data Cleaning ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove links
    text = re.sub(r"@\w+", "", text)                      # remove @mentions (tweets)
    text = re.sub(r"<.*?>", "", text)                     # remove HTML tags
    text = re.sub(r"\d+", "", text)                       # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()               # remove extra spaces
    return text

df["clean_text"] = df["text"].apply(clean_text)

# ---------- Step 3: Feature Extraction (TF-IDF) ----------
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

model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
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
