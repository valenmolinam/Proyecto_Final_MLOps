import os
import logging
import traceback
from datetime import datetime
from flask import Flask, request, jsonify

import pandas as pd
import numpy as np

from src.app.pred_online.etl import Dataloader
from src.app.pred_online.feature_engineer import add_features
from src.app.pred_online.prediction import get_model, predict

# Configuración de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Cargar modelo en el arranque
try:
    model = get_model()
    logger.info("✅ Modelo cargado exitosamente en el arranque")
except Exception as e:
    logger.error(f"❌ Error cargando el modelo: {e}")
    model = None

# Crear la app Flask
app = Flask("fraud-detection")


@app.route("/health", methods=["GET"])  
def health_check():
    """Verificar estado del servicio"""
    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "service": "Fraud Detection Prediction",
        "timestamp": datetime.now().isoformat()
    })


@app.route("/predict", methods=["POST"])
def predict_single():
    """
    Endpoint para predecir un solo registro.
    Recibe un JSON con los datos y devuelve la predicción.
    """
    try:
        if model is None:
            return jsonify({"error": "Modelo no cargado"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No se recibieron datos"}), 400

        df = pd.DataFrame([data])
        df = add_features(df)

        prediction = predict(model, df)
        prediction_label = int(prediction[:, 1] > 0.5)

        return jsonify({
            "prediction": "Si" if prediction_label == 1 else "No",
            "probabilities": {
                "no": float(prediction[0][0]),
                "si": float(prediction[0][1])
            },
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"❌ Error en predicción: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Error interno", "details": str(e)}), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Endpoint para predecir un batch de registros.
    Recibe una lista de objetos JSON.
    """
    try:
        if model is None:
            return jsonify({"error": "Modelo no cargado"}), 500

        data = request.get_json()
        if not data or "records" not in data:
            return jsonify({"error": "Se esperaba una clave 'records' con lista de datos"}), 400

        df = pd.DataFrame(data["records"])
        df = add_features(df)

        predictions = predict(model, df)
        results = []

        for idx, row in df.iterrows():
            label = "Si" if predictions[idx][1] > 0.5 else "No"
            results.append({
                "id": idx,
                "prediction": label,
                "probabilities": {
                    "no": float(predictions[idx][0]),
                    "si": float(predictions[idx][1])
                }
            })

        return jsonify({
            "results": results,
            "total_records": len(results),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"❌ Error en batch prediction: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Error interno", "details": str(e)}), 500


if __name__ == "__main__":
    logger.info("🚀 Iniciando servidor Flask en http://localhost:9696")
    app.run(debug=True, host="0.0.0.0", port=9696)
