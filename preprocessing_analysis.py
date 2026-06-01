from src.preprocessors.tokenizer import tokenize_comments
from src.preprocessors.normalizer import normalize_comments
from src.preprocessors.stopwords_remover import remove_stopwords_comments
from src.preprocessors.stemmer_porter import stem_porter_comments
from src.preprocessors.stemmer_snowball import stem_snowball_comments
from src.preprocessors.lemmatizer import lemmatize_comments
from src.preprocessors.pos_tagger import pos_tag_comments
from src.preprocessors.noise_filter import filter_noise_comments
from src.pipeline.preprocessing_pipeline import build_data


def run():
    tokenize_comments()
    normalize_comments()
    remove_stopwords_comments()
    stem_porter_comments()
    stem_snowball_comments()
    lemmatize_comments()
    pos_tag_comments()
    filter_noise_comments()


if __name__ == "__main__":
    run()
    build_data()