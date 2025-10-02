
import mlflow
from mlflow.tracking import MlflowClient
from feature_engineer import add_features
from etl import Dataloader
import pandas as pd

def get_model():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()

    model_name = "Mejor modelo"

    try:
        # 🔹 Intentar primero con alias (nuevo enfoque recomendado)
        model_uri = f"models:/{model_name}@production"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Modelo cargado con alias: {model_uri}")
        return model
    except Exception as e_alias:
        print(f"⚠️ No se encontró alias 'production': {e_alias}")
        print("🔄 Intentando cargar con stage 'Production'...")

        try:
            # 🔹 Intentar con stage clásico (modo legacy)
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.sklearn.load_model(model_uri)
            print(f"✅ Modelo cargado con stage: {model_uri}")
            return model
        except Exception as e_stage:
            raise RuntimeError(
                f"❌ No se encontró el modelo '{model_name}' ni en alias 'production' "
                f"ni en stage 'Production'. Error original: {e_stage}"
            )

def get_etl_data():
    dataloader = Dataloader(path_to_save="data", n_samples=100000)
    dataset_path = dataloader.download_data()
    df = dataloader.load_data()
    return df

def get_features(df):
    feature_engineer = add_features(df)
    df = feature_engineer.preprocess_data()
    return df

def get_data():
    df = get_etl_data()
    feature_engineer = add_features(df)
    df = feature_engineer.preprocess_data()
    return df

def predict(model, df):
    return model.predict_proba(df)

def save_prediction(df, prediction):
    df["prediction"] = prediction[:, 1]
    df["prediction"] = df["prediction"].astype(int)
    df["prediction"] = df["prediction"].map({0: "No", 1: "Si"})
    df.to_csv('predictions/predictions.csv', index=False)
    return df

def save_prediction_proba(df, prediction):
    df["prediction_0"] = prediction[:, 0]
    df["prediction_1"] = prediction[:, 1]
    df["prediction_0"] = df["prediction_0"].astype(int)
    df["prediction_1"] = df["prediction_1"].astype(int)
    df["prediction_0"] = df["prediction_0"].map({0: "No", 1: "Si"})
    df["prediction_1"] = df["prediction_1"].map({0: "No", 1: "Si"})
    df.to_csv('predictions/predictions_proba.csv', index=False)
    return df

def main():
    model = get_model()
    df = get_etl_data()
    prediction = predict(model, df)
    #save_prediction(df, prediction)
    save_prediction_proba(df, prediction)

if __name__ == "__main__":
    main()