import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], axis=1)
    return df
