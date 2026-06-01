import pickle
import pandas as pd
import networkx as nx
from itertools import combinations


INPUT_FILE = "data/processed/ner/levenshtein_entities.csv"
GRAPH_FILE = "data/graphs/knowledge_graph.gpickle"
CENTRALITY_FILE = "data/graphs/graph_centrality.csv"


def build_graph():
    print("Carregando entidades...")
    df = pd.read_csv(INPUT_FILE)
    G = nx.Graph()
    grouped = df.groupby("id_comentario")
    print("Construindo grafo...")

    for _, group in grouped:
        entities = list(set(group["entidade_aproximada"].dropna().tolist()))

        if len(entities) < 2:
            continue

        for source, target in combinations(entities, 2):
            if G.has_edge(source, target):
                G[source][target]["weight"] += 1
            else:
                G.add_edge(source, target, weight=1)

    print(f"Nós: {G.number_of_nodes()}")
    print(f"Arestas: {G.number_of_edges()}")
    with open(GRAPH_FILE, "wb") as f:
        pickle.dump(G, f)
    print(f"Grafo salvo em: {GRAPH_FILE}")

    return G


def calculate_centrality(G):
    print("Calculando centralidade...")
    centrality = nx.degree_centrality(G)
    result = pd.DataFrame(
        {
            "entidade": centrality.keys(),
            "centralidade": centrality.values()
        }
    )

    result = result.sort_values(by="centralidade", ascending=False)
    result.to_csv(CENTRALITY_FILE, index=False)
    print(result.head(20))


if __name__ == "__main__":
    graph = build_graph()
    calculate_centrality(graph)