from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from app.fe import feature_engineering

app = FastAPI(title='Churn Prediction API')

model=joblib.load('model/Churn_model.pkl')

class ChurnRequest(BaseModel):
    CreditScore: int
    Geography: object
    Gender: object
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    BalanceGroup: object
    EstimatedSalaryGroup: object
    TenureGroup: object
    AgeGroup: object
    CreditScoreClass: object

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API!"}

@app.post("/predict")
def predict_churn(data: ChurnRequest):
    input_df= pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return {"churn_prediction": int(prediction), "churn_probability": float(probability)}