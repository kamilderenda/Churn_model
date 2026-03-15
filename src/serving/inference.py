import mlflow
import pandas as pd
from pathlib import Path
import os
from src.features.build_features import feature_engineering

models_dir = Path("src/serving/models")
latest = max(models_dir.iterdir(), key=os.path.getmtime)
model_path = latest / "artifacts"

try:
    model = mlflow.pyfunc.load_model(str(model_path))
    print("Model loaded successfully.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")

def predict(data: dict) -> str:
    try:
        input_data = feature_engineering(pd.DataFrame([data]))
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            return "The customer is likely to churn."
        else:
            return "The customer is unlikely to churn."
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return -1