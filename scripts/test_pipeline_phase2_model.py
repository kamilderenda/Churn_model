from src.models.train import train_model
import pandas as pd

df=pd.read_csv("/Users/kamil/PycharmProjects/Churn_model/data/raw_data/processed_data/external/Churn_Modelling_Feature_Engineered.csv")

def main():
    try:
        model = train_model(df)
        print("Model training completed successfully.")
    except Exception as e:
        print(f"An error occurred during model training: {e}")

if __name__ == "__main__":
    main()