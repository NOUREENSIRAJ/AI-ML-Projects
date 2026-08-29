"""
Sentiment Analysis - Web App (Streamlit)
Step 7-8: UI + Deployment ready
Run with: streamlit run app.py
"""

import streamlit as st
import joblib
import re
import string

# ---------- Load saved model + vectorizer ----------
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

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
st.write("Koi bhi review ya sentence likhen, model uska sentiment predict karega.")

user_input = st.text_area("Apna text yahan likhen:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Pehle kuch text likhen.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec).max()

        if prediction == "positive":
            st.success(f"Sentiment: Positive 😊 (confidence: {proba:.2f})")
        else:
            st.error(f"Sentiment: Negative 😞 (confidence: {proba:.2f})")
