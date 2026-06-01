import nltk
from nltk.corpus import stopwords
from .normalizer import normalize_text
from .base_preprocessor import load_dataset, save_dataset
nltk.download("stopwords")

STOPWORDS = set(stopwords.words("portuguese"))
CUSTOM_STOPWORDS = {
    "app",
    "sistema",
    "plataforma"
}
STOPWORDS.update(CUSTOM_STOPWORDS)

def remove_stopwords(text):
    words = normalize_text(text).split()
    words = [
        word
        for word in words
        if word not in STOPWORDS
    ]

    return " ".join(words)


def remove_stopwords_comments():
    df = load_dataset()
    df["comentario_sem_stopwords"] = (df["comentario"].apply(remove_stopwords))
    save_dataset(df, "data/processed/preprocessors", "stopwords_removed.csv")
    return df