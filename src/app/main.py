from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict


app = FastAPI(title='Churn Prediction API',
                  description='API for predicting customer churn using a pre-trained model.',
                  version='1.0.0')


@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API!"}

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


@app.post("/predict")
def predict_churn(data: ChurnRequest):
    try:
        prediction = predict(data.dict())
        return {"churn_prediction": prediction}
    except Exception as e:
        return {"error": str(e)}

def gradio_interface(
    gender, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    payload = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }
    out = predict(payload)
    return str(out)

def gradio_interface(
        geography, gender, credit_score, age, tenure, balance, num_of_products,
        has_credit_card, is_active_member, estimated_salary
):
    payload = {
        "Geography": geography,
        "Gender": gender,
        "CreditScore": int(credit_score),
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(num_of_products),
        "HasCrCard": int(has_credit_card),
        "IsActiveMember": int(is_active_member),
        "EstimatedSalary": float(estimated_salary)
    }
    out= predict(payload)
    return str(out)

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(['France', 'Spain', 'Germany'], label="Geography"),
        gr.Dropdown(['Male', 'Female'], label='Gender'),
        gr.Number(label='Credit Score'),
        gr.Number(label='Age'),
        gr.Number(label='Tenure'),
        gr.Number(label='Balance'),
        gr.Number(label='Num of Products'),
        gr.Dropdown([0, 1], label='Has Credit Card'),
        gr.Dropdown([0, 1], label='Is Active Member'),
        gr.Number(label='Estimated Salary')
    ],
    outputs="text",
    title="Churn Prediction",
    description="Fill in the customer details to get a churn prediction.",
)

app = gr.mount_gradio_app(app, demo, path="/ui")