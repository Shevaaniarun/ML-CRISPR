import pandas as pd

def load_dataset(filepath):
    """
    Loads CRISPR sgRNA dataset from CSV file.
    """
    df = pd.read_csv(filepath)

    # Drop empty rows
    df = df.dropna()

    return df
