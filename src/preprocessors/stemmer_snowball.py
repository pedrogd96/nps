from nltk.stem import SnowballStemmer
from .normalizer import normalize_text
from .base_preprocessor import load_dataset, save_dataset

stemmer = SnowballStemmer("portuguese")

def stem_text(text):
    words = normalize_text(text).split()
    stems = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(stems)


def stem_snowball_comments():
    df = load_dataset()
    df["comentario_stem_snowball"] = (df["comentario"].astype(str).apply(stem_text))
    save_dataset(df, "data/processed/preprocessors", "stemming_snowball.csv")
    return df