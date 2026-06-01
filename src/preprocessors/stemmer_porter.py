from nltk.stem import PorterStemmer
from .normalizer import normalize_text
from .base_preprocessor import load_dataset, save_dataset

stemmer = PorterStemmer()

def stem_text(text):
    words = normalize_text(text).split()
    stems = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(stems)


def stem_porter_comments():
    df = load_dataset()
    df["comentario_stem_porter"] = (df["comentario"].astype(str).apply(stem_text))
    save_dataset(df, "stemming_porter.csv")
    return df