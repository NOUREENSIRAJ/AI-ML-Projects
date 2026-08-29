# Sentiment Analysis Project

## Files
- `sample_data.csv` — chota demo dataset (25 rows). **Real project ke liye Kaggle se "IMDB Movie Reviews" ya "Amazon Reviews" dataset download karen (1000+ rows), aur isi filename se replace karen.**
- `train.py` — data clean karta hai, model train karta hai, `sentiment_model.pkl` aur `vectorizer.pkl` save karta hai
- `app.py` — Streamlit web app, jisme user text likh kar prediction dekh sakta hai
- `requirements.txt` — zaroori libraries

## Run karne ka tareeqa
```bash
pip install -r requirements.txt
python train.py          # model train + save
streamlit run app.py     # web app chalayen
```

## Note
Demo dataset sirf 25 rows ka hai isliye accuracy low aayegi — ye sirf structure test karne ke liye hai.
Real dataset (Kaggle) laga kar accuracy 85%+ tak aa sakti hai.

## Deploy (Step 8)
1. GitHub par code push karen
2. https://share.streamlit.io par jaa kar GitHub repo link se free deploy karen
3. Live link Upwork/LinkedIn portfolio mein add karen
