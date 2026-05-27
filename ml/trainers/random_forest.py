from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from ml.trainers.base import BaseTrainer

class RandomForestTrainer(BaseTrainer):
    
    @property
    def model_name(self) -> str:
        return "stellar-classifier"

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifer", RandomForestClassifier(**self.config.hyperparamenters)),
        ])