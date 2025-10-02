import mlflow
from mlflow.tracking import MlflowClient

# 1. Configurar conexión a tu servidor MLflow local
mlflow.set_tracking_uri("http://127.0.0.1:5000")  

# 2. Nombre del experimento que usaste
experiment_name = "fraude-optuna"

# 3. Buscar el experimento
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    raise Exception(f"⚠️ No se encontró el experimento: {experiment_name}")

# 4. Obtener todos los runs del experimento
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_f1 DESC"]  # cambia la métrica si usaste otra (ej: roc_auc)
)

if runs.empty:
    raise Exception("⚠️ No hay runs en este experimento.")

# 5. Seleccionar el mejor run
best_run = runs.iloc[0]
best_run_id = best_run.run_id
best_metric = best_run["metrics.test_f1"]
print(f"✅ Mejor run encontrado: {best_run_id} con F1 = {best_metric:.4f}")

# 6. Registrar el modelo en el Model Registry
model_name = experiment_name  # "fraude-optuna"
model_uri = f"runs:/{best_run_id}/model"

client = MlflowClient()

# Crear nueva versión en el registry
model_version = client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=best_run_id
)

# 7. Mover la versión a "Production"
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production",
    archive_existing_versions=True  # mueve anteriores a Archived
)

print(f"🎉 Modelo {model_name} registrado en Production (versión {model_version.version})")