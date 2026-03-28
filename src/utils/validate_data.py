import pandera as pa
from pandera import Column, DataFrameSchema
from typing import Tuple, List
import pandas as pd

def validate_churn_data_pred(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    print("Starting data validation...")

    schema = DataFrameSchema({
        "RowNumber": Column(int, nullable=False, checks=pa.Check.ge(0)),
        "CustomerId": Column(int, nullable=False, checks=pa.Check.ge(0)),
        "Surname": Column(str, nullable=False),
        "CreditScore": Column(int, nullable=False, checks=pa.Check.ge(0)),
        "Geography": Column(str, nullable=False, checks=pa.Check.isin(["France","Spain","Germany"])),
        "Gender": Column(str, nullable=False, checks=pa.Check.isin(["Female","Male"])),
        "Age": Column(int, nullable=False, checks=pa.Check.ge(18)),
        "Tenure": Column(int, nullable=False, checks=pa.Check.ge(0)),
        "Balance": Column(float, nullable=False, checks=pa.Check.ge(0)),
        "NumOfProducts": Column(int, nullable=False, checks=pa.Check.ge(0)),
        "HasCrCard": Column(int, nullable=False, checks=pa.Check.isin([0,1])),
        "IsActiveMember": Column(int, nullable=False, checks=pa.Check.isin([0,1])),
        "EstimatedSalary": Column(float, nullable=False, checks=pa.Check.ge(0))
    })

    try:
        schema.validate(df)
        print("Data validation passed successfully.")
        return True, []
    except pa.errors.SchemaErrors as e:
        failed_columns = list(e.failure_cases["column"])
        print(f"Data validation failed for: {failed_columns}")
        return False, failed_columns
    
def validate_training_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    print("Starting training data validation...")

    schema = DataFrameSchema({
        "RowNumber": Column(int, nullable=False, checks=pa.Check.ge(0)),
        "CustomerId": Column(int, nullable=False, checks=pa.Check.ge(0)),
        "Surname": Column(str, nullable=False),
        "CreditScore": Column(int, nullable=False, checks=pa.Check.ge(0)),
        "Geography": Column(str, nullable=False, checks=pa.Check.isin(["France","Spain","Germany"])),
        "Gender": Column(str, nullable=False, checks=pa.Check.isin(["Female","Male"])),
        "Age": Column(int, nullable=False, checks=pa.Check.ge(18)),
        "Tenure": Column(int, nullable=False, checks=pa.Check.ge(0)),
        "Balance": Column(float, nullable=False, checks=pa.Check.ge(0)),
        "NumOfProducts": Column(int, nullable=False, checks=pa.Check.ge(0)),
        "HasCrCard": Column(int, nullable=False, checks=pa.Check.isin([0,1])),
        "IsActiveMember": Column(int, nullable=False, checks=pa.Check.isin([0,1])),
        "EstimatedSalary": Column(float, nullable=False, checks=pa.Check.ge(0)),
        "Exited": Column(int, nullable=False, checks=pa.Check.isin([0,1]))
    })
    try:
        schema.validate(df)
        print("Training data validation passed successfully.")
        return True, []
    except pa.errors.SchemaErrors as e:
        failed_columns = list(e.failure_cases["column"])
        print(f"Training data validation failed for: {failed_columns}")
        return False, failed_columns