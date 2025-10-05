from mlflow.tracking import MlflowClient

client = MlflowClient()
for mv in client.search_model_versions("name='Mejor modelo'"):
    print(f"Version: {mv.version}, Stage: {mv.current_stage}, Run ID: {mv.run_id}")

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
    dataloader = Dataloader(path_to_save="data/pred_batch.csv", n_samples=100000)
    dataset_path = dataloader.download_data()
    df = dataloader.load_data()
    return df

def get_features(df):
    feature_engineer = add_features(df)
    df = feature_engineer.preprocess_data()
    return df

def predict(model, df):
    return model.predict_proba(df)

def save_prediction(df, prediction):
    df["prediction"] = prediction[:, 1]
    df["prediction"] = df["prediction"].astype(int)
    df["prediction"] = df["prediction"].map({0: "No", 1: "Si"})
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "predictions")

    file_path = os.path.join(save_dir, "predictions.csv")
    df.to_csv(file_path, index=False)

    print(f"✅ Predicciones guardadas en: {file_path}")
    return df 

from etl import Dataloader

def get_etl_data():
    dataloader = Dataloader(path_to_save="C:/Users/Valentina Molina/Documents/Repositorios/Proyecto_Final_MLOps/data/PS_20174392719_1491204439457_log.csv")
    df = dataloader.load_data()
    dataloader.drop_name_columns()
    return df

import os

base_dir = os.path.dirname(os.path.abspath(__file__))
predictions_dir = os.path.join(base_dir, "predictions")

print("Archivos en carpeta predictions:",
      os.listdir(predictions_dir) if os.path.exists(predictions_dir) else "Carpeta no existe")

model = get_model()

model

df = get_etl_data()

from feature_engineer import add_features
df = add_features(df)
df.head()

prediction = predict(model, df)
prediction

prediction_array = prediction[:, 1]

save_prediction(df, prediction)