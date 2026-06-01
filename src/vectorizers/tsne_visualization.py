import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(X, labels):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = (tsne.fit_transform(X))
    plt.figure(figsize=(12,8))

    categories = labels.unique()
    for category in categories:
        mask = labels == category

        plt.scatter(
            X_embedded[mask, 0],
            X_embedded[mask, 1],
            label=category,
            alpha=0.7
        )

    scatter = plt.scatter(
        X_embedded[:,0],
        X_embedded[:,1],
        c=labels.astype("category").cat.codes
    )

    plt.title("Visualização t-SNE dos comentários NPS")
    plt.legend(title="Responsável")
    plt.tight_layout()
    plt.show()