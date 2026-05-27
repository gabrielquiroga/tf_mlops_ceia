from pathlib import path
import yaml
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    hyperparamenters: dict

def load_model_config(config_path = str | Path) -> ModelConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return ModelConfig(
        name=raw["name"],
        hyperparamenters=raw["hyperparameters"]
    )