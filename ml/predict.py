from dotenv import load_dotenv
load_dotenv()
import ast
import os

import pandas as pd
import mlflow.pyfunc
from mlflow import MlflowClient

from ml.preprocessing import preprocess_for_inference

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTERED_NAME="stellar-classifier"

def load_champion() -> mlflow.pyfunc.PyFuncModel:
    """Carga el modelo con el alias @champion del Model Registry."""
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.pyfunc.load_model(f"models:/{REGISTERED_NAME}@champion")

def load_model_by_version(version: int) -> mlflow.pyfunc.PyFuncModel:
    """Carga una versión específica. Útil para comparar runs."""
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.pyfunc.load_model(f"models:/{REGISTERED_NAME}/{version}")

def promote_to_champion(version: int) -> None:
    """
    Asigna el alias @champion a una versión específica.
    Lo que antes era 'copiar el modelo a models/latest/'.
    """
    client = MlflowClient(tracking_uri=TRACKING_URI)
    client.set_registered_model_alias(
        name=REGISTERED_NAME,
        alias="champion",
        version=str(version),
    )
    print(f"Versión {version} promovida a @champion")

def get_champion_features() -> list[str]:
    """Recupera las features usadas al enternar el modelo champion desde MLflow."""
    client = MlflowClient(tracking_uri=TRACKING_URI)
    version = client.get_model_version_by_alias(REGISTERED_NAME, "champion")
    run = client.get_run(version.run_id)
    return ast.literal_eval(run.data.params["features"])

def get_champion_metric(metric_name: str) -> str:
    """
    Recupera el valor de una métrica específica del modelo champion.
    Devuelve un dict vacío si no hay champion.
    """
    try:
        client = MlflowClient(tracking_uri=TRACKING_URI)
        version_info = client.get_model_version_by_alias(REGISTERED_NAME, "champion")
        run = client.get_run(version_info.run_id)
        metric_value = run.data.metrics.get(metric_name)
        return {
            "version": version_info.version,
            "run_id": version_info.run_id,
            metric_name: metric_value,
        }
    except Exception:
        return {}


def predict(input_df: pd.DataFrame) -> pd.Series:
    """
    Predice usando el modelo champion.
    """
    model = load_champion()
    features = get_champion_features()
    clean_df = preprocess_for_inference(input_df, features)
    return pd.Series(model.predict(clean_df))


if __name__ == "__main__":
    """
    Para ejecutar una prueba de inferencia localmente.
    > python -m ml.predict
    """
    predict_df = pd.read_csv("dataset/test_sc.csv")
    print(predict(predict_df))