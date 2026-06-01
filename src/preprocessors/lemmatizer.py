import spacy
from .base_preprocessor import load_dataset, save_dataset
nlp = spacy.load("pt_core_news_sm")

def lemmatize_text(text):
    doc = nlp(str(text))
    lemmas = [
        token.lemma_
        for token in doc
    ]

    return " ".join(lemmas)


def lemmatize_comments():
    df = load_dataset()
    df["comentario_lemma"] = (df["comentario"].apply(lemmatize_text))
    save_dataset(df, "lemmatization.csv")
    return df