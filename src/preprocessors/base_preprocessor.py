from pathlib import Path
import pandas as pd

from src.data.load_data import load_data
from src.utils.config_loader import load_config

def load_dataset():
    config = load_config("configs/data.yaml")
    return load_data(config["data"]["path"])


def save_dataset(df: pd.DataFrame, filename: str):
    PROCESSED_DATA = Path("data/processed/preprocessors")
    output_path = PROCESSED_DATA / filename
    df.to_csv(output_path, index=False)
    print(f"Arquivo salvo: {output_path}")