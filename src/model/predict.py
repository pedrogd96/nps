import joblib
from src.pipeline.preprocessing_pipeline import preprocess_text


model = joblib.load("models/naive_bayes.pkl")
vectorizer = joblib.load("models/ngram_vectorizer.pkl")
encoder = joblib.load("models/label_encoder.pkl")

def predict(comment):
    processed = preprocess_text(comment)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    responsible = encoder.inverse_transform([prediction])[0]
    return responsible