from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

from etl import Dataloader
from feature_engineer import add_features
from train import Train

class TrainWithMLflow:
    def __init__(self, df, numeric_features, categorical_features, target_column, model, test_size=0.2, model_params=None, mlflow_setup=None):
        self.df = df
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_column = target_column
        self.test_size = test_size
        self.model = model
        self.model_params = model_params if model_params else {}
        self.setup = mlflow_setup

    def train_test_split(self):
        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def create_pipeline_numeric(self):
        numeric_transformer = Pipeline(steps=[
            ('Scaler', StandardScaler())
        ])
        return numeric_transformer

    def create_pipeline_categorical(self):
        categorical_transformer = Pipeline(steps=[
            ('OneHotEncoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        return categorical_transformer
    
    def create_preprocessor(self):
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.create_pipeline_numeric(), self.numeric_features),
                ('cat', self.create_pipeline_categorical(), self.categorical_features)
            ]
        )
        return preprocessor
    
    def create_pipeline_train(self):
        pipeline = Pipeline(steps=[
            ('preprocessor', self.create_preprocessor()),
            ('classifier', self.model)
        ])
        return pipeline
    
    def train(self):
        with mlflow.start_run() as run:
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("model_type", type(self.model.__class__.__name__))
            mlflow.log_param("n_categorical_features", len(self.categorical_features))
            mlflow.log_param("n_numeric_features", len(self.numeric_features))
            mlflow.log_param("target_column", self.target_column)

            for param_name, param_value in self.model_params.items():
                mlflow.log_param(f"model_{param_name}", param_value)

            mlflow.log_param("numeric_features", ", ".join(self.numeric_features))
            mlflow.log_param("categorical_features", ", ".join(self.categorical_features))

            X_train, X_test, y_train, y_test = self.train_test_split()

            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_test_samples", len(X_test))

            pipeline = self.create_pipeline_train()
            pipeline.fit(X_train, y_train)

            y_train_pred = pipeline.predict(X_train)
            y_test_pred = pipeline.predict(X_test)

            train_metrics = self.calculate_metrics(y_train, y_train_pred, prefix="train_")
            test_metrics = self.calculate_metrics(y_test, y_test_pred, prefix="test_")

            all_metrics = {**train_metrics, **test_metrics}
            for metric_name, metric_value in all_metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            run_id = run.info.run_id

            print(f"MLflow Run ID: {run_id}")
            print(f"Tracking URI: {mlflow.get_tracking_uri()}")
            print(f"Train Accuracy: {train_metrics['train_accuracy']:.4f}")
            print(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")
            
            mlflow.sklearn.log_model(pipeline, "model")
            return pipeline, run_id

    def calculate_metrics(self, y_true, y_pred, prefix=""):
        return {
            f"{prefix}accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}precision": precision_score(y_true, y_pred, zero_division=0),
            f"{prefix}recall": recall_score(y_true, y_pred, zero_division=0),
            f"{prefix}f1": f1_score(y_true, y_pred, zero_division=0)
        }

    def train(self):
        with mlflow.start_run() as run:
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("model_type", self.model.__class__.__name__)
            mlflow.log_param("n_categorical_features", len(self.categorical_features))
            mlflow.log_param("n_numeric_features", len(self.numeric_features))
            mlflow.log_param("target_column", self.target_column)

            for param_name, param_value in self.model_params.items():
                mlflow.log_param(f"model_{param_name}", param_value)

            mlflow.log_param("numeric_features", ", ".join(self.numeric_features))
            mlflow.log_param("categorical_features", ", ".join(self.categorical_features))

            X_train, X_test, y_train, y_test = self.train_test_split()

            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_test_samples", len(X_test))

            pipeline = self.create_pipeline_train()
            pipeline.fit(X_train, y_train)

            y_train_pred = pipeline.predict(X_train)
            y_test_pred = pipeline.predict(X_test)

            train_metrics = self.calculate_metrics(y_train, y_train_pred, prefix="train_")
            test_metrics = self.calculate_metrics(y_test, y_test_pred, prefix="test_")

            all_metrics = {**train_metrics, **test_metrics}
            for metric_name, metric_value in all_metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            run_id = run.info.run_id

            print(f"MLflow Run ID: {run_id}")
            print(f"Tracking URI: {mlflow.get_tracking_uri()}")
            print(f"Train Accuracy: {train_metrics['train_accuracy']:.4f}")
            print(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")

            mlflow.sklearn.log_model(pipeline, "model")
            return pipeline, run_id