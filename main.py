import time
import requests
from pathlib import Path
from src.model.naive_bayes import train

MODEL_FILE = Path("models/naive_bayes.pkl")

def wait_for_mlflow(url="http://mlflow:5000/health", timeout=180):

    for i in range(timeout):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                print("MLflow está pronto!")
                return
        except Exception as e:
            pass

        time.sleep(1)

    raise Exception("MLflow não iniciou a tempo")

def main():
    wait_for_mlflow()

    if not MODEL_FILE.exists():
        print("Modelo não encontrado.")
        print("Executando treinamento...")
        train()
    else:
        print("Modelo já existe.")

if __name__ == "__main__":
    main()