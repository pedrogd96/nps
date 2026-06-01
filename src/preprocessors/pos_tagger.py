import spacy
from .base_preprocessor import load_dataset, save_dataset
nlp = spacy.load("pt_core_news_sm")

def extract_pos(text):
    doc = nlp(str(text))
    return [
        (token.text, token.pos_)
        for token in doc
    ]


def pos_tag_comments():
    df = load_dataset()
    df["pos_tags"] = (df["comentario"].apply(extract_pos))
    save_dataset(df, "data/processed/preprocessors", "pos_tagging.csv")
    return df