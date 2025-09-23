from etl import Dataloader
from feature_engineer import FeatureEngineer
from train import Train
from train_with_mlflow import Train as TrainWithMlflow

def task_train():
    # Cargar y preprocesar los datos
    loader = Dataloader(r"C:/Users/Valentina Molina/Documents/Repositorios/Proyecto_Final_MLOps/data/PS_20174392719_1491204439457_log.csv")
    df = loader.load_data()
    loader.drop_name_columns()
    df = loader.df

    #feature engineering
    feature_engineer = FeatureEngineer(df)
    df = feature_engineer.add_features()

    #training with mlflow
    class TrainWithMLflow(TrainWithMlflow):
        def train(self, model_name=None):
            with mlflow.start_run() as run:
                if model_name:
                    mlflow.log_param("model_name", model_name)
                return super().train()
            
    
    numeric_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
                        'diff_old_new_orig', 'diff_old_new_dest', 'amount_to_orig_balance', 'amount_to_dest_balance']
    categorical_features = ['type']
    target_column = 'isFraud'

    # Definir modelos a entrenar
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import lightgbm as lgb

    modelos = [
        ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
        ('LightGBM', lgb.LGBMClassifier(random_state=42))
    ]

    resultados = {}

    for nombre, modelo in modelos:
        trainer = TrainWithMLflow(df, numeric_features, categorical_features, target_column, modelo, test_size=0.2)
        pipeline, run_id = trainer.train(model_name=nombre)
        resultados[nombre] = run_id

    return resultados