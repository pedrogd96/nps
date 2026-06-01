import re
import unidecode
from .base_preprocessor import load_dataset, save_dataset

def normalize_text(text):
    text = str(text)
    text = text.lower()
    text = unidecode.unidecode(text)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_comments():
    df = load_dataset()
    df["comentario_normalizado"] = (df["comentario"].apply(normalize_text))
    save_dataset(df, "data/processed/preprocessors", "normalized.csv")
    return df