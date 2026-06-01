from sklearn.metrics.pairwise import cosine_similarity

class SimilaritySearch:

    def __init__(self, vectorizer, matrix, documents):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.documents = documents

    def search(self, query, top_k=5):
        query_vector = (
            self.vectorizer.transform(
                [query]
            )
        )

        similarities = (cosine_similarity(query_vector, self.matrix).flatten())
        indexes = (similarities.argsort()[::-1][:top_k])
        results = []

        for idx in indexes:
            results.append(
                (
                    self.documents[idx],
                    similarities[idx]
                )
            )

        return results