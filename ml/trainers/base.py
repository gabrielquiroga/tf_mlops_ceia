from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline

from ml.config import ModelConfig

class BaseTrainer(ABC):
    """Contrato que todo trainer debe cumplir.
    Permite que se pueda trbajar con ml/train.py independientemente
    del modelo que tenga dentro."""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        """Construye el Pipeline y lo devuelve listo para entrenar."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Nombre con el cual registrarlo"""
        ...