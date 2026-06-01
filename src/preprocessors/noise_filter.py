import spacy
from .base_preprocessor import load_dataset, save_dataset
nlp = spacy.load("pt_core_news_sm")

def is_noise(text: str) -> bool:
    """
    Retorna True quando:
    - comentário possui <= 4 palavras
    OU
    - não possui nenhum substantivo
    """

    text = str(text).strip()

    if not text:
        return True

    doc = nlp(text)
    tokens = [
        token
        for token in doc
        if not token.is_space and not token.is_punct
    ]

    if len(tokens) <= 4:
        return True

    nouns = [
        token
        for token in doc
        if token.pos_ in ["NOUN", "PROPN"]
    ]

    if len(nouns) == 0:
        return True

    return False


def filter_noise_comments():
    df = load_dataset()
    df["comentario_original"] = df["comentario"]
    df["is_noise"] = (df["comentario"].apply(is_noise))
    df["comentario_filtrado"] = (
        df.apply(
            lambda row: ""
            if row["is_noise"]
            else row["comentario"],
            axis=1
        )
    )

    save_dataset(df, "noise_filter.csv")
    return df