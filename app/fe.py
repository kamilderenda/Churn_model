import pandas as pd
import numpy as np

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['CreditScoreClass']=pd.cut(df['CreditScore'], bins=[350, 420, 600, 725, np.inf], labels=['Low', 'Medium', 'High', 'Very High'])
    df['AgeGroup']=pd.cut(df['Age'], bins=[18, 30, 45, 60, np.inf], labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])
    df['TenureGroup']=pd.cut(df['Tenure'], bins=[0, 4, 8,  np.inf], labels=['New', 'Established', 'Loyal'])
    df['BalanceGroup']=pd.cut(df['Balance'], bins=[-1, 0, 50000, 100000, np.inf], labels=['No Balance', 'Low Balance', 'Medium Balance', 'High Balance'])
    df['EstimatedSalaryGroup']=pd.cut(df['EstimatedSalary'], bins=[0, 50000, 100000, np.inf], labels=['Low Salary', 'Medium Salary', 'High Salary'])
    return df

def prepare_request(data: dict) -> dict:
    df = pd.DataFrame([data])
    df = feature_engineering(df)
    return df.to_dict(orient="records")[0]