import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, classification_report

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

def train_model(df: pd.DataFrame, target_col: str) -> Pipeline:
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

    with mlflow.start_run():
        f1= f1_score(y_test, pipeline.predict(X_test))
        precision=precision_score(y_test, pipeline.predict(X_test))
        recall=recall_score(y_test, pipeline.predict(X_test))

        mlflow.log_params(params)
        mlflow.sklearn.log_model(pipeline, "model")
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        train_ds=mlflow.data.from_pandas(X_train, source="train_data")
        mlflow.log_data(train_ds, "train_data")

    return pipeline