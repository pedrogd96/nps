from pathlib import Path
import pandas as pd
from src.vectorizers.bow_vectorizer import train_bow
from src.vectorizers.tfidf_vectorizer import train_tfidf
from src.vectorizers.ngram_vectorizer import train_tfidf_bigram
from src.vectorizers.word2vec_vectorizer import train_word2vec
from src.vectorizers.similarity_search import SimilaritySearch
from src.vectorizers.tsne_visualization import plot_tsne
from src.pipeline.vectorizer_pipeline import build_data

INPUT_FILE = ("data/processed/nps_training.csv")
OUTPUT_PATH = Path("data/processed/vectorizers")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

def run():
    print("Carregando dataset...")

    df = pd.read_csv(INPUT_FILE)
    corpus = (df["comentario"].fillna("").tolist())

    print("Executando Bag of Words...")

    X_bow, bow = train_bow(corpus)
    print(f"Tamanho do bow - {len(bow.get_feature_names_out())}")
    bow_df = pd.DataFrame(X_bow.toarray(), columns=bow.get_feature_names_out())
    bow_df.to_csv(OUTPUT_PATH / "bow.csv", index=False)
    print("*" * 50)
    print(bow_df.sum().sort_values(ascending=False).head(20))

    print("*" * 50)
    print("*" * 50)
    print("Executando TF-IDF...")

    X_tfidf, tfidf = (train_tfidf(corpus))
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_df.to_csv(OUTPUT_PATH / "tfidf.csv", index=False)
    print(tfidf_df.mean().sort_values(ascending=False).head(20))

    print("*" * 50)
    print("*" * 50)
    print("Executando TF-IDF Bigram...")

    X_bigram, bigram = (train_tfidf_bigram(corpus))
    pd.DataFrame(X_bigram.toarray(), columns=bigram.get_feature_names_out()).to_csv(OUTPUT_PATH / "tfidf_bigram.csv", index=False)

    print([col for col in bigram.get_feature_names_out() if " " in col][:20])
    print("*" * 50)
    print("*" * 50)
    print("Treinando Word2Vec...")

    model = train_word2vec(corpus)
    words = list(model.wv.index_to_key)
    vectors = [model.wv[word]for word in words]
    pd.DataFrame(vectors, index=words).to_csv(OUTPUT_PATH / "word2vec_vectors.csv")

    print("Palavra demor")
    print(model.wv.most_similar("demor"))
    print("*" * 50)
    print("Palavra erro")
    print(model.wv.most_similar("erro"))
    print("*" * 50)
    print("Palavra tent")
    print(model.wv.most_similar("tent"))
    print("*" * 50)
    print("*" * 50)
    print("Executando Similarity Search...")

    search = SimilaritySearch(tfidf, X_tfidf, corpus)

    test_queries = [
        "erro integr",
        "suport demor",
        "integraca api",
        "performanc carreg",
        "atualizaca simpl"
    ]

    search_results = []
    for query in test_queries:
        results = search.search(query, top_k=5)

        for doc, score in results:
            search_results.append(
                {
                    "query": query,
                    "score": score,
                    "documento": doc
                }
            )

    pd.DataFrame(search_results).to_csv(OUTPUT_PATH / "similarity_search.csv", index=False)

    print("*" * 50)
    print("*" * 50)
    print("Gerando t-SNE...")

    plot_tsne(X_tfidf.toarray(), df["responsavel"])
    print("Finalizado.")


if __name__ == "__main__":
    run()
    build_data()