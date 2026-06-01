from src.preprocessors.stopwords_remover import remove_stopwords
from src.preprocessors.normalizer import normalize_text
from src.preprocessors.stemmer_snowball import stem_text
from src.preprocessors.noise_filter import is_noise
from src.preprocessors.base_preprocessor import load_dataset, save_dataset

def preprocess_text(text: str) -> str:
    """
    Pipeline oficial utilizado em:
    - Treinamento
    - Teste
    - Inferência da API

    Ordem:
    1. Remoção de Stop Words
    2. Normalização
    3. Stemming Snowball
    4. Noise Filter
    """

    if text is None:
        return ""

    text = str(text)
    text = remove_stopwords(text)
    text = normalize_text(text)
    text = stem_text(text)
    if is_noise(text):
        return ""

    return text.strip()

def build_data():
    df = load_dataset()
    df["comentario"] = (df["comentario"].apply(preprocess_text))
    df = df[df["comentario"] != ""]
    save_dataset(df, "data/processed", "nps_training.csv")
    return df