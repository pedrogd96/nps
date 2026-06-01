import nltk
from nltk.tokenize import word_tokenize
from .base_preprocessor import load_dataset,save_dataset
nltk.download("punkt_tab")

def tokenize_comments():
    df = load_dataset()
    df["comentario_tokenizado"] = df["comentario"].astype(str).apply(
        lambda text: word_tokenize(text, language="portuguese")
    )

    save_dataset(df, "tokenization.csv")
    return df