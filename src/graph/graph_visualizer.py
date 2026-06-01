import pickle
import networkx as nx
import matplotlib.pyplot as plt


GRAPH_FILE = "data/graphs/knowledge_graph.gpickle"
OUTPUT_IMAGE = "data/graphs/knowledge_graph.png"


def main():

    print("Carregando grafo...")
    with open(GRAPH_FILE, "rb") as f:
        G = pickle.load(f)
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, seed=42, k=0.7)

    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("Grafo de Conhecimento - Pesquisa NPS")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    plt.show()
    print(f"Grafo salvo em {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()