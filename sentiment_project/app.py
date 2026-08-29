"""
Sentiment Analysis - Web App (Streamlit)
Step 7-8: UI + Deployment ready
Run with: streamlit run app.py
"""

import streamlit as st
import joblib
import re
import string
import os

# ---------- Load saved model + vectorizer ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- UI ----------
st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬")
st.title("💬 Sentiment Analysis App")
st.write("Enter any review or sentence, and the model will predict its sentiment.")

user_input = st.text_area("Enter your text here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec).max()

        if prediction == "positive":
            st.success(f"Sentiment: Positive 😊 (confidence: {proba:.2f})")
        else:
            st.error(f"Sentiment: Negative 😞 (confidence: {proba:.2f})")
