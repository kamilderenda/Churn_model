import pandera as pa
from pandera import Column, DataFrameSchema
from typing import Tuple, List
import pandas as pd

def validate_churn_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    print("Starting data validation...")

    schema = DataFrameSchema({
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