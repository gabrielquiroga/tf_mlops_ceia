import sys
from pathlib import Path

import mlflow
import mlflow.sklearn

from ml.config import load_model_config, get_ml_settings
from ml.preprocessing import run_preprocessing
from ml.evaluate import compute_metrics, print_report
from ml.trainers.random_forest import RandomForestTrainer
from ml.trainers.xgboost import XGBoostTrainer

TRAINERS = {
    "random_forest": RandomForestTrainer,
    "xgboost": XGBoostTrainer
    
}

mlsettings = get_ml_settings()

def train(model_type: str, source: str) -> str:
    """
    Entrena el modelo, loguea todo a MLflow y registra el pipeline.
    
    :param model_type: Tipo de modelo que se va a entrenar, entre los configurados en trainers/
    :type model_type: str
    :param source: Path al dataset de entrenamiento
    :type source: str

    :return: RunID de la ejecución en MLflow
    :rtype: str
    """
    # 1. Cargar configuración de modelo
    config_path = Path(f"ml/config/{model_type}.yaml")
    config = load_model_config(config_path)

    # 2. Cargar datos y preprocesar para obtener sets de entrenamiento y test
    print(f"[1/4] Cargando y preprocesando datos desde {source}...")
    X_train, X_test, y_train, y_test = run_preprocessing(source, config)
    print(f"    Train: {X_train.shape} | Test: {X_test.shape}")

    # 3. Construir y entrenar el pipeline
    print(f"[2/4] Entrenando {model_type}...")
    trainer = TRAINERS[model_type](config)
    pipeline = trainer.build_pipeline()
    pipeline.fit(X_train, y_train)

    # 4. Evaluar
    print(f"[3/4] Evaluando...")
    metrics = compute_metrics(pipeline, X_test, y_test)
    print_report(pipeline, X_test, y_test)

    # 5. Registrar en el MLflow
    print("[4/4] Logueando a MLflow...")
    mlflow.set_tracking_uri(mlsettings.mlflow_tracking_uri)
    mlflow.set_experiment("Clasificación Estelar")

    with mlflow.start_run(run_name=model_type) as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("source", source)
        mlflow.log_param("features", config.features)
        mlflow.log_param("test_size", config.test_size)
        mlflow.log_params(config.hyperparameters)

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_name="model",
            registered_model_name="stellar-classifier"
        )

        run_id = run.info.run_id
        print(f"\n Run logueado: {run_id}")
    
    return run_id

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python -m ml.train <model_type> <source>")
        sys.exit(1)

    train(
        model_type=sys.argv[1],
        source=sys.argv[2]
    )