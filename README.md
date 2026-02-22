# 🧠 Customer Churn Classification – End-to-End ML Project

Production-oriented machine learning project for customer churn prediction, developed with a strong focus on recall optimization and full model deployment lifecycle.

---

## 📌 Business Problem

Customer churn prediction is a **cost-asymmetric classification problem**.

Failing to detect a churner (False Negative) is significantly more expensive than incorrectly flagging a non-churner (False Positive).  

Therefore, the primary optimization objective was:

> **Maximize Recall while maintaining acceptable Precision**

To align with business objectives, recall-weighted evaluation metrics (F-beta, β > 1) were introduced.

---

# 🏗 Project Structure

The project consists of two main stages:

### 1️⃣ Research & Model Development (Jupyter Notebook)

- Exploratory Data Analysis
- Feature Engineering
- Model Benchmarking
- Hyperparameter Optimization
- Feature Importance Analysis
- Threshold & Probability Analysis

### 2️⃣ Productionization (FastAPI + Docker)

- Model serialization (.pkl)
- FastAPI for inference
- Input validation
- Docker containerization
- Reproducible deployment

---

# 📊 Exploratory Data Analysis (EDA)

Comprehensive analysis of:

- Numerical features
- Categorical features
- Feature distributions vs target (Churn)
- Correlations
- Class imbalance
- Informative vs weak predictors

### Removed Features

The following identifier-like features were removed:

- `RowNumber`
- `CustomerId`
- `Surname`

They did not provide predictive value.

---

# 🛠 Feature Engineering

A key part of the project involved transforming selected numerical variables into categorical bins.

### Motivation

Some numerical variables did not clearly separate churn classes in raw form.  
To improve separability:

- Strategic discretization was applied
- Behavior-driven binning was designed
- Class contrast was increased

### Result

Engineered features:

- Improved churn/non-churn separation
- Increased model sensitivity
- Positively influenced recall-focused metrics

---

# 🤖 Model Development

Model development was conducted in two main phases.

---

## Phase 1. Model & Metric Benchmarking

Multiple combinations were evaluated:

### Encoders
- One-Hot Encoding
- Target Encoding

### Models
- Random Forest
- XGBoost
- CatBoost
- LightGBM

### Evaluation Metrics

To align with business goals:

- F1 Score
- Recall
- F-beta (β = 5, 6, 7)

Higher β values emphasized recall over precision.

Initial training used default hyperparameters to:

- Identify promising configurations
- Understand baseline behavior
- Select top-performing pipelines

---

## Phase 2. Hyperparameter Optimization

The top-performing configurations were optimized using **Optuna**.

Optimization objective:

- Maximize recall-focused metric
- Maintain acceptable precision
- Ensure model stability

---

# 🏆 Final Model Performance

      precision    recall  f1-score   support

0       0.94      0.80      0.86      1593
1       0.50      0.79      0.61       407

    accuracy                           0.80      2000
   macro avg       0.72      0.79      0.74      2000
weighted avg       0.85      0.80      0.81      2000


Additional evaluation:

- Threshold behavior analysis
![shap.png](..%2F..%2FDesktop%2Fkaggle_bank_project%2Fshap.png)
- Probability distribution inspection
![predict analys.png](..%2F..%2FDesktop%2Fkaggle_bank_project%2Fpredict%20analys.png)


---

# 📈 Feature Importance & Model Insights

The selected model was analyzed for:

- Feature importance ranking
- Contribution of engineered features
- Stability of top predictors
- Impact of binning strategy

Key findings:

- Categorical features was more influential than raw numerical features
- Engineered features significantly contributed to model performance
- Age was the most important numeric predictor

---

# 🚀 Production Deployment

The final model was deployed as a REST API using:

- **FastAPI**
- **Docker**

### Production Flow

Client → API → Feature Engineering → Model → JSON Response

### API Output

- Predicted class (0/1)
- Churn probability

### Example Response

#### json
{
  "churn_prediction": 0,
  "churn_probability": 0.23
}

🧰 Tech Stack

Python

Pandas / NumPy

Scikit-learn

LightGBM

XGBoost

CatBoost

Optuna

FastAPI

Docker

📦 How to Run
### Run locally
uvicorn app.main:app --reload

Open:
http://localhost:8000/docs

### Run with Docker

Build:

docker build -t churn-api .

Run:

docker run -p 8000:8000 churn-api

Open:
http://localhost:8000/docs