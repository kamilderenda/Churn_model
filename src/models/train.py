import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cat_cols=['Geography',
 'Gender',
 'NumOfProducts',
 'HasCrCard',
 'IsActiveMember',
 'CreditScoreClass',
 'AgeGroup',
 'TenureGroup',
 'BalanceGroup',
 'EstimatedSalaryGroup']

num_cols=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']

def train_model(df: pd.DataFrame) -> Pipeline:
    params={'num_leaves': 216, 'max_depth': 3, 'learning_rate': 0.026651486106472815, 'num_boost_round': 343, 'min_data_in_leaf': 31, 'feature_fraction': 0.44986320323360535, 'bagging_fraction': 0.7326343382898357, 'bagging_freq': 2, 'lambda_l2': 0.04910513025563936, 'lambda_l1': 0.008018648951855945,
            'random_state': 42, 'scale_pos_weight': 3.9, "objective": "binary", "boosting_type": "gbdt","verbosity": -1}
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Exited', axis=1), df['Exited'], test_size=0.2,random_state=42, stratify=df['Exited'])
    cat_process=Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    num_process=Pipeline(steps=[
        ('scaler', PowerTransformer())
    ])

    preprocessor=ColumnTransformer(transformers=[
        ('cat', cat_process, cat_cols),
        ('num', num_process, num_cols)
    ])
    model=lgb.LGBMClassifier(**params)
    pipeline=Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    
    report_dict=classification_report(y_test, pipeline.predict(X_test), output_dict=True)
    mlflow.set_experiment("churn_model_retraining_1")
    mlflow.set_tracking_uri("http://localhost:5000")
    # mlflow.lightgbm.autolog()
    with mlflow.start_run(run_name="retrain_run_1"):
        
        for label, metrics in report_dict.items():
            clean_label = str(label).replace(" ", "_") 
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    clean_metric = metric_name.replace(" ", "_")
                    mlflow.log_metric(f"{clean_label}_{clean_metric}", value)
            else:
                mlflow.log_metric(clean_label, metrics)

        mlflow.log_params(params)

        feature_importances = pipeline.named_steps['model'].feature_importances_
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

        sorted_indices = np.argsort(feature_importances)[::-1]
        sorted_importances = feature_importances[sorted_indices]
        sorted_names = feature_names[sorted_indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=sorted_importances, y=sorted_names, ax=ax)
        ax.set_title("Feature Importances")
        plt.tight_layout()
        fig.canvas.draw()

        mlflow.log_figure(fig, "feature_importances.png")
        plt.close(fig)

        mlflow.sklearn.log_model(pipeline, "model", registered_model_name="churn_model")
    return pipeline