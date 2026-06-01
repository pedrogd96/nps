from sklearn.feature_extraction.text import TfidfVectorizer

def train_tfidf_bigram(corpus):
    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        max_features=10000,
        min_df=2,
        max_df=0.95
    )

    X = vectorizer.fit_transform(corpus)
    return X, vectorizer