import pandas as pd
from src.vectorizers.ngram_vectorizer import train_tfidf_bigram

INPUT_FILE = ("data/processed/nps_training.csv")
OUTPUT_FILE = ("data/processed/nps_training_vectorized.csv")
MODEL_FILE = ("models/tfidf_bigram_vectorizer.pkl")

def build_data():
    print("Carregando dataset nps_training...")

    df = pd.read_csv(INPUT_FILE)
    corpus = (df["comentario"].tolist())
    X, vectorizer = (train_tfidf_bigram(corpus))
    vectorized_df = pd.DataFrame(
        X.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    vectorized_df["responsavel"] = df["responsavel"]
    vectorized_df.to_csv(OUTPUT_FILE, index=False)
    print("Pipeline finalizado.")