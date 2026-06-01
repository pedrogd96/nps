from pathlib import Path
import joblib
import pandas as pd
import mlflow
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from src.pipeline.preprocessing_pipeline import build_data
from src.vectorizers.ngram_vectorizer import train_tfidf_bigram


TRAINING_DATASET = "data/processed/nps_training.csv"
MODEL_PATH = "models/naive_bayes.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
VECTORIZER_PATH = "models/ngram_vectorizer.pkl"

def train():
    RANDOM_STATE = 42

    ensure_training_dataset()
    print("Carregando dataset processado...")
    df = pd.read_csv(TRAINING_DATASET)
    X = df["comentario"]
    y = df["responsavel"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    X_vectorized, vectorizer = train_tfidf_bigram(X, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    mlflow.set_experiment("nps_responsavel_classifier")
    with mlflow.start_run(run_name="naive_bayes"):
        model = MultinomialNB()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=encoder.classes_)
        print(report)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        mlflow.log_param("model_type", "MultinomialNB")
        mlflow.log_param("vectorizer", "TF-IDF Bigram")
        mlflow.log_param("dataset_rows", len(df))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        joblib.dump(encoder, ENCODER_PATH)

    print("Modelo salvo")

def ensure_training_dataset():
    training_file = Path(TRAINING_DATASET)
    if training_file.exists():
        print("Dataset processado encontrado.")
        return

    print("Dataset processado não encontrado.")
    print("Executando pipeline de pré-processamento...")

    build_data()

    if not training_file.exists():
        raise FileNotFoundError(
            f"{TRAINING_DATASET} não foi gerado."
        )

    print("Dataset processado criado com sucesso.")