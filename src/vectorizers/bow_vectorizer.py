from sklearn.feature_extraction.text import CountVectorizer

def train_bow(corpus):
    vectorizer = CountVectorizer(max_features=5000)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer