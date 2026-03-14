import os
import pandas as pd

# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import feature_engineering

DATA_PATH="/Users/kamil/PycharmProjects/Churn_model/data/raw_data/processed_data/external/Churn_Modelling.csv"
TARGET_COL='Exited'

def main():
    # Load data
    df = load_data(DATA_PATH)

    # Preprocess data
    df = preprocess_data(df)

    # Feature engineering
    df = feature_engineering(df)

    # Check if target column exists
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' is missing from the dataset.")

    print("Data loading, preprocessing, and feature engineering completed successfully.")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

if __name__ == "__main__":
    main()
