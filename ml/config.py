from pathlib import Path
import yaml
from dataclasses import dataclass

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