from gensim.models import Word2Vec

def train_word2vec(corpus):

    tokenized = [
        text.split()
        for text in corpus
    ]

    model = Word2Vec(
        sentences=tokenized,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4
    )

    model.save("models/word2vec.model")
    return model