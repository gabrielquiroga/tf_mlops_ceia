from pathlib import Path
import yaml
from dataclasses import dataclass

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

@dataclass
class ModelConfig:
    name: str
    model_type: str
    features: list[str]
    hyperparameters: dict
    test_size: float
    random_state: int

def load_model_config(config_path = str | Path) -> ModelConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return ModelConfig(
        name=raw["model"]["name"],
        model_type=raw["model"]["type"],
        features=raw["features"],
        hyperparameters=raw["hyperparameters"],
        test_size=raw["training"]["test_size"],
        random_state=raw["training"]["random_state"],
    )

class MLSettings(BaseSettings):
    """Configuración relacionada a ML cargada desde variables de entorno."""
    model_config = SettingsConfigDict(
        env_file=[".env"],
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    mlflow_tracking_uri: str = Field(
        default="",
        description="URL en la que sirve MLFlow"
    )

    log_level: str = Field(
        default="INFO",
        description="Nivel de logging"
    )

    data_source_url: str = Field(
        default="dataset\star_classification.csv",
        description="Ubicación del dataset."
    )

def get_ml_settings() -> MLSettings:
    return MLSettings()

