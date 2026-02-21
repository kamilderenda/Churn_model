import requests
from app.fe import prepare_request

url = "http://127.0.0.1:8000/predict"

data_0 = {
    "CreditScore": 525,
    "Geography": "Germany",
    "Gender": "Male",
    "Age": 33,
    "Tenure": 4,
    "Balance": 131023.76,
    "NumOfProducts": 2,
    "HasCrCard": 0,
    "IsActiveMember": 0,
    "EstimatedSalary": 55072.93
}

data_prepared = prepare_request(data_0)
response = requests.post(url, json=data_prepared)
print(response.json())

data_1 = {
  "CreditScore": 699,
  "Geography": "Germany",
  "Gender": "Female",
  "Age": 54,
  "Tenure": 3,
  "Balance": 111009.32,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 155905.79
}


data_prepared = prepare_request(data_1)
response = requests.post(url, json=data_prepared)
print(response.json())
