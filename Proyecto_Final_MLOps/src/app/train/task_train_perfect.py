"""
Prefect workflow for training fraud detection models with Optuna and MLflow.
"""

import json
from datetime import datetime

import mlflow
import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from etl import Dataloader
from feature_engineer import add_features
from train_with_mlflow_optuna import TrainWithMLflowOptuna


# ---------------------------
# TASK 1: Load data
# ---------------------------
@task(name="Load_Data", retries=2, retry_delay_seconds=10)
def task_load_data(path: str) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f"Loading dataset from {path}...")

    loader = Dataloader(path)
    df = loader.load_data()
    loader.drop_name_columns()
    df = loader.df

    summary_df = pd.DataFrame({
        "Metric": ["Total Samples", "Total Features", "Missing Values", "Target Distribution"],
        "Value": [
            len(df),
            len(df.columns),
            df.isnull().sum().sum(),
            df["isFraud"].value_counts().to_dict() if "isFraud" in df.columns else "N/A"
        ]
    })

    create_table_artifact(
        key="data-summary",
        table=summary_df.to_dict(orient="records"),
        description=f"Dataset Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    return df


# ---------------------------
# TASK 2: Feature Engineering
# ---------------------------
@task(name="Feature_Engineering", retries=2, retry_delay_seconds=10)
def task_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info("Applying feature engineering...")

    initial_columns = len(df.columns)
    df = add_features(df)

    feature_summary = pd.DataFrame({
        "Metric": ["Initial Features", "Final Features", "Features Added", "Dataset Size"],
        "Value": [
            initial_columns,
            len(df.columns),
            len(df.columns) - initial_columns,
            f"{len(df)} rows"
        ]
    })

    create_table_artifact(
        key="feature-engineering-summary",
        table=feature_summary.to_dict(orient="records"),
        description="Feature Engineering Summary"
    )

    return df


# ---------------------------
# TASK 3: Train Model with Optuna + MLflow
# ---------------------------
@task(name="Train_Model_Optuna", retries=1, retry_delay_seconds=30)
def task_train_with_optuna(
    df: pd.DataFrame,
    model_type: str = "LogisticRegression",
    n_trials: int = 20,
    optimization_metric: str = "accuracy"
):
    logger = get_run_logger()
    logger.info(f"Starting Optuna optimization for {model_type}...")

    # Features
    numeric_features = [
        "step", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "diff_old_new_orig", "diff_old_new_dest",
        "amount_to_orig_balance", "amount_to_dest_balance"
    ]
    categorical_features = ["type"]
    target_column = "isFraud"

    # Param distributions
    if model_type == "LogisticRegression":
        model_class = LogisticRegression
        param_distributions = {
            "C": ("float", 0.001, 100, True),
            "penalty": ("categorical", ["l1", "l2"]),
            "max_iter": ("int", 200, 2000),
            "solver": ("categorical", ["liblinear", "saga"])
        }
        fixed_params = {"class_weight": "balanced"}
    elif model_type == "RandomForest":
        model_class = RandomForestClassifier
        param_distributions = {
            "n_estimators": ("int", 50, 200),
            "max_depth": ("int", 5, 30),
            "min_samples_split": ("int", 2, 15),
            "min_samples_leaf": ("int", 1, 10),
            "max_features": ("categorical", ["sqrt", "log2"])
        }
        fixed_params = {"random_state": 42, "n_jobs": -1, "class_weight": "balanced"}
    elif model_type == "XGBoost":
        model_class = XGBClassifier
        param_distributions = {
            "n_estimators": ("int", 50, 200),
            "max_depth": ("int", 3, 10),
            "learning_rate": ("float", 0.01, 0.3, True),
            "subsample": ("float", 0.5, 1.0, False),
            "colsample_bytree": ("float", 0.5, 1.0, False)
        }
        fixed_params = {"random_state": 42, "scale_pos_weight": 10}
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    mlflow.set_experiment(f"prefect_{model_type.lower()}_training")
    mlflow.sklearn.autolog()

    trainer = TrainWithMLflowOptuna(
        df=df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_column=target_column,
        model_class=model_class,
        test_size=0.25,
        n_trials=n_trials,
        optimization_metric=optimization_metric,
        param_distributions=param_distributions,
        model_params=fixed_params
    )

    best_pipeline, best_run_id, study = trainer.train()

    # Validation (sample)
    X_test = df.drop(columns=[target_column]).head(500)
    y_test = df[target_column].head(500)
    y_pred = best_pipeline.predict(X_test)

    metrics_dict = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }

    logger.info(f"Best {optimization_metric}: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")

    return best_pipeline, best_run_id, study, metrics_dict


# ---------------------------
# FLOW: Train One Model
# ---------------------------
@flow(name="Train_Model_With_Optuna", log_prints=True)
def train_model_flow(
    path: str,
    model_type: str = "LogisticRegression",
    n_trials: int = 10,
    optimization_metric: str = "accuracy"
):
    df = task_load_data(path)
    df = task_feature_engineering(df)

    best_pipeline, best_run_id, study, metrics_dict = task_train_with_optuna(
        df,
        model_type,
        n_trials,
        optimization_metric
    )

    return best_pipeline, best_run_id


if __name__ == "__main__":
    # Ejemplo: entrenar un modelo con Logistic Regression
    pipeline, run_id = train_model_flow(
        path=r"C:\Users\Valentina Molina\Documents\repositorios\Proyecto_Final_MLOps\data\PS_20174392719_1491204439457_log.csv",
        model_type="LogisticRegression",
        n_trials=5,
        optimization_metric="accuracy"
    )

# ---------------------------
# FLOW: Compare Multiple Models
# ---------------------------
@flow(name="Compare_Models", log_prints=True)
def compare_models_flow(
    path: str,
    n_trials: int = 10
):
    """
    Flow to compare multiple models with Optuna optimization.
    
    Args:
        path: Path to CSV dataset
        n_trials: Number of Optuna trials per model
    """
    logger = get_run_logger()
    logger.info("Starting model comparison flow...")

    # Load + feature engineering once
    df = task_load_data(path)
    df = task_feature_engineering(df)

    results = {}
    models_to_compare = ["LogisticRegression", "RandomForest", "XGBoost"]
    metrics_to_try = ["accuracy", "f1"]

    comparison_data = []

    for model_type in models_to_compare:
        for metric in metrics_to_try:
            logger.info(f"Training {model_type} optimizing for {metric}...")

            best_pipeline, best_run_id, study, metrics_dict = task_train_with_optuna(
                df,
                model_type,
                n_trials,
                metric
            )

            comparison_data.append({
                "Model": model_type,
                "Optimization Metric": metric,
                "Best Score": f"{study.best_value:.4f}",
                "Accuracy": f"{metrics_dict['accuracy']:.4f}",
                "F1 Score": f"{metrics_dict['f1']:.4f}",
                "MLflow Run ID": best_run_id[:8] + "..."
            })

            results[f"{model_type}_{metric}"] = {
                "pipeline": best_pipeline,
                "run_id": best_run_id,
                "best_score": study.best_value
            }

    # Create comparison artifact
    comparison_df = pd.DataFrame(comparison_data)

    create_table_artifact(
        key="model-comparison-results",
        table=comparison_df.to_dict(orient="records"),
        description="Comparison of LogisticRegression, RandomForest, and XGBoost"
    )

    best_model_key = max(results.keys(), key=lambda k: results[k]['best_score'])
    logger.info(f"Best model overall: {best_model_key}")

    return results


if __name__ == "__main__":
    # Ejemplo 1: entrenar un solo modelo
    pipeline, run_id = train_model_flow(
        path=r"C:\Users\Valentina Molina\Documents\repositorios\Proyecto_Final_MLOps\data\PS_20174392719_1491204439457_log.csv",
        model_type="LogisticRegression",
        n_trials=5,
        optimization_metric="accuracy"
    )
    print(f"Modelo entrenado con run_id: {run_id}")

    # Ejemplo 2: comparar varios modelos
    results = compare_models_flow(
        path=r"C:\Users\Valentina Molina\Documents\repositorios\Proyecto_Final_MLOps\data\PS_20174392719_1491204439457_log.csv",
        n_trials=5
    )
    print("Resultados de comparación:")
    print(results)
