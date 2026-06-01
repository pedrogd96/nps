from sklearn.feature_extraction.text import TfidfVectorizer

def train_tfidf(corpus):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer