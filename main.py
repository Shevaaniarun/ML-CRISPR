from src.load_data import load_dataset
from src.train_catboost import train_model
from src.config import DATA_PATHS, TARGET_COLUMN, SEQUENCE_COLUMN

if __name__ == "__main__":

    print("Loading dataset...")
    df = load_dataset(DATA_PATHS["WT"])

    print("Training CatBoost model...")
    model = train_model(df, SEQUENCE_COLUMN, TARGET_COLUMN)
