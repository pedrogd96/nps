import pandas as pd
import spacy
from collections import Counter


INPUT_FILE = "data/raw/nps_data.csv"
OUTPUT_FILE = "data/processed/ner/nps_ner_entities.csv"
nlp = spacy.load("pt_core_news_sm")


def extract_entities(text):
    if pd.isna(text):
        return []

    doc = nlp(str(text))
    entities = []

    for ent in doc.ents:
        entities.append({
            "texto": ent.text,
            "tipo": ent.label_
        })

    return entities


def main():

    print("Carregando dataset...")
    df = pd.read_csv(INPUT_FILE)
    all_entities = []
    print("Extraindo entidades...")

    for idx, row in df.iterrows():
        entities = extract_entities(row["comentario"])

        for entity in entities:
            all_entities.append({
                "id_comentario": idx,
                "responsavel": row["responsavel"],
                "entidade": entity["texto"],
                "tipo": entity["tipo"]
            })

    entities_df = pd.DataFrame(all_entities)
    entities_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Arquivo salvo em: {OUTPUT_FILE}")
    print("\nTop entidades encontradas:")

    counter = Counter(entities_df["entidade"])

    for entity, count in counter.most_common(20):
        print(f"{entity}: {count}")


if __name__ == "__main__":
    main()