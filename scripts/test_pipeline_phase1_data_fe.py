from prefect import flow, task

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import feature_engineering
from src.utils.validate_data import validate_churn_data

DATA_PATH="/Users/kamil/PycharmProjects/Churn_model/data/raw_data/processed_data/external/Churn_Modelling.csv"
TARGET_COL='Exited'

@task
def load_task(path: str):
    return load_data(path)
@task
def preprocess_task(df):
    return preprocess_data(df)
@task
def feature_engineering_task(df):
    return feature_engineering(df)
@task
def validate_data_task(df):
    from src.utils.validate_data import validate_churn_data
    is_valid, failed_expectations = validate_churn_data(df)
    if not is_valid:
        raise ValueError(f"Data validation failed for: {failed_expectations}")

@flow
def main():
    df = load_task(DATA_PATH)
    df = preprocess_task(df)
    df = feature_engineering_task(df)
    validate_data_task(df)

    print("Data pipeline executed successfully.")
    print(df.head())

if __name__ == "__main__":
    main()
