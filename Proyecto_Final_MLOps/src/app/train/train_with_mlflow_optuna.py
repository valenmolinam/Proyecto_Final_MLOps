import optuna
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TrainWithMLflowOptuna:
    def __init__(self, df, numeric_features, categorical_features, target_column,
                 model_class, test_size=0.2, n_trials=20, optimization_metric='f1',
                 param_distributions=None, model_params=None, mlflow_setup=None):
        
        self.df = df
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_column = target_column
        self.test_size = test_size
        self.model_class = model_class
        self.n_trials = n_trials
        self.optimization_metric = optimization_metric
        self.param_distributions = param_distributions
        self.model_params = model_params if model_params else {}
        self.mlflow_setup = mlflow_setup

    def train_test_split(self):
        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]
        return train_test_split(X, y, test_size=self.test_size, random_state=42)

    def create_pipeline_numeric(self):
        return Pipeline(steps=[("scaler", StandardScaler())])

    def create_pipeline_categorical(self):
        return Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    def create_preprocessor(self):
        return ColumnTransformer(transformers=[
            ("num", self.create_pipeline_numeric(), self.numeric_features),
            ("cat", self.create_pipeline_categorical(), self.categorical_features)
        ])

    def create_pipeline(self, model):
        return Pipeline(steps=[
            ("preprocessor", self.create_preprocessor()),
            ("classifier", model)
        ])

    def calculate_metrics(self, y_true, y_pred, prefix=""):
        return {
            f"{prefix}accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}precision": precision_score(y_true, y_pred, zero_division=0),
            f"{prefix}recall": recall_score(y_true, y_pred, zero_division=0),
            f"{prefix}f1": f1_score(y_true, y_pred, zero_division=0)
        }

    def objective(self, trial):
        # Crear espacio de hiperparámetros dinámico
        params = self.model_params.copy()
        if self.param_distributions:
            for param, (ptype, *values) in self.param_distributions.items():
                if ptype == 'int':
                    params[param] = trial.suggest_int(param, values[0], values[1])
                elif ptype == 'float':
                    params[param] = trial.suggest_float(param, values[0], values[1])
                elif ptype == 'categorical':
                    params[param] = trial.suggest_categorical(param, values[0])

        model = self.model_class(**params)
        pipeline = self.create_pipeline(model)

        X_train, X_test, y_train, y_test = self.train_test_split()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = self.calculate_metrics(y_test, y_pred, prefix="test_")
        return metrics[f"test_{self.optimization_metric}"]

    def train(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)

        best_params = self.model_params.copy()
        best_params.update(study.best_params)

        best_model = self.model_class(**best_params)
        pipeline = self.create_pipeline(best_model)

        X_train, X_test, y_train, y_test = self.train_test_split()
        pipeline.fit(X_train, y_train)

        with mlflow.start_run() as run:
            mlflow.log_params(best_params)
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("target_column", self.target_column)

            y_pred = pipeline.predict(X_test)
            metrics = self.calculate_metrics(y_test, y_pred, prefix="test_")
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            run_id = run.info.run_id
            print(f"Optuna best params: {best_params}")
            print(f"MLflow run_id: {run_id}")

        return pipeline, run_id, study
