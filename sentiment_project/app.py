"""
Sentiment Analysis - Web App (Streamlit)
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


# ---------- Page config ----------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
)

# ---------- Custom styling ----------
st.markdown("""
    <style>
        .main { background-color: #0E1117; }
        .block-container { padding-top: 3rem; max-width: 720px; }
        .app-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .app-header h1 {
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .app-header p {
            color: #9CA3AF;
            font-size: 0.95rem;
        }
        .result-card {
            border-radius: 14px;
            padding: 1.5rem;
            margin-top: 1.5rem;
            text-align: center;
        }
        .result-positive {
            background: rgba(34, 197, 94, 0.12);
            border: 1px solid rgba(34, 197, 94, 0.35);
        }
        .result-negative {
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.35);
        }
        .result-label {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .result-sub {
            color: #9CA3AF;
            font-size: 0.85rem;
        }
        .stTextArea textarea {
            border-radius: 10px;
            font-size: 0.95rem;
        }
        footer {visibility: hidden;}
        .footer-note {
            text-align: center;
            color: #6B7280;
            font-size: 0.8rem;
            margin-top: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
    <div class="app-header">
        <h1>🎬 Sentiment Analysis</h1>
        <p>A machine learning model trained on 50,000 IMDB movie reviews (TF-IDF + Naive Bayes).<br>
        Enter any review or sentence below to see its predicted sentiment.</p>
    </div>
""", unsafe_allow_html=True)

# ---------- Example chips ----------
examples = [
    "This movie completely blew me away, one of the best I've seen.",
    "Waste of time, the plot made no sense at all.",
    "It was okay, nothing memorable but not terrible either.",
]

st.caption("Try an example:")
cols = st.columns(len(examples))
selected_example = None
for i, col in enumerate(cols):
    with col:
        if st.button(f"Example {i+1}", use_container_width=True):
            selected_example = examples[i]

# ---------- Input ----------
default_text = selected_example if selected_example else st.session_state.get("last_input", "")
user_input = st.text_area(
    "Your text",
    value=default_text,
    height=120,
    placeholder="e.g. The acting was brilliant and the story kept me hooked till the end...",
    label_visibility="collapsed",
)
st.session_state["last_input"] = user_input

analyze = st.button("Analyze Sentiment", type="primary", use_container_width=True)

# ---------- Prediction ----------
if analyze:
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec).max()
        confidence_pct = round(proba * 100, 1)

        if prediction == "positive":
            st.markdown(f"""
                <div class="result-card result-positive">
                    <div class="result-label">😊 Positive</div>
                    <div class="result-sub">Confidence: {confidence_pct}%</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card result-negative">
                    <div class="result-label">😞 Negative</div>
                    <div class="result-sub">Confidence: {confidence_pct}%</div>
                </div>
            """, unsafe_allow_html=True)

        st.progress(proba)

# ---------- Footer ----------
st.markdown("""
    <div class="footer-note">
        Built with Python, scikit-learn & Streamlit · by Noreen Siraj
    </div>
""", unsafe_allow_html=True)
