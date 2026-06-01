import pandas as pd
import Levenshtein
import string
from collections import Counter
from nltk.corpus import stopwords


INPUT_FILE = "data/raw/nps_data.csv"
OUTPUT_FILE = "data/processed/ner/levenshtein_entities.csv"

VALID_ENTITIES = [
    "problema",
    "plataforma",
    "sistema",
    "integracao",
    "relatorio",
    "notificacao",
    "transferencia",
    "performance",
    "suporte",
    "atendimento",
    "dashboard",
    "login",
    "cadastro",
    "api",
    "onboarding",
    "boleto",
    "extrato",
    "pagamento",
    "pix",
    "cartao",
    "usuario",
    "cliente",
    "consultor",
    "implantacao",
    "treinamento",
    "financeiro",
    "saldo",
    "limite",
    "erro",
    "bug",
    "usabilidade",
    "instabilidade",
    "carregamento",
    "acesso",
    "seguranca"
]

STOP_WORDS = set(stopwords.words("portuguese"))

def clean_word(word):
    word = word.lower()
    word = word.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    return word.strip()

def find_closest_entity(word, max_distance=3):
    best_match = None
    best_distance = 999

    for entity in VALID_ENTITIES:
        distance = Levenshtein.distance(
            word.lower(),
            entity.lower()
        )

        if distance < best_distance:
            best_distance = distance
            best_match = entity

    if best_distance <= max_distance:
        return best_match, best_distance

    return None, None


def extract_entities(text):
    results = []
    words = str(text).split()

    for word in words:
        word = clean_word(word)

        if not word:
            continue

        if word in STOP_WORDS:
            continue

        if len(word) < 4:
            continue

        match, distance = find_closest_entity(word)

        if match:
            results.append({
                "termo_original": word,
                "entidade_aproximada": match,
                "distancia": distance
            })

    return results


def main():
    print("Carregando dataset...")
    df = pd.read_csv(INPUT_FILE)
    all_matches = []
    print("Executando Levenshtein...")

    for idx, row in df.iterrows():
        matches = extract_entities(row["comentario"])

        for match in matches:
            all_matches.append({
                "id_comentario": idx,
                "responsavel": row["responsavel"],
                **match
            })

    result_df = pd.DataFrame(all_matches)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nArquivo salvo em:")
    print(OUTPUT_FILE)
    print("\nExemplos encontrados:")
    print(result_df.head(20))
    print("\nCorreções mais frequentes:")

    corrections = Counter(
        zip(
            result_df["termo_original"],
            result_df["entidade_aproximada"]
        )
    )

    for (original, corrected), count in corrections.most_common(20):
        print(f"{original} -> {corrected}: {count}")


if __name__ == "__main__":
    main()