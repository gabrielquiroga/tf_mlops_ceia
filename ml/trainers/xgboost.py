import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from ml.trainers.base import BaseTrainer


class LabelEncodedXGBClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper de XGBClassifier que codifica internamente el target string -> int,
    y decodifica las predicciones de vuelta a las clases originales.
    Necesario porque multi:softmax requiere enteros [0, 1, 2, ...]."""

    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self._le = LabelEncoder()
        self._model = XGBClassifier(**xgb_params)

    def fit(self, X, y):
        # Codifica, entrena y guarda las clases igual que sklearn
        y_encoded = self._le.fit_transform(y)
        self._model.fit(X, y_encoded)
        self.classes_ = self._le.classes_
        return self

    def predict(self, X):
        # Verifica que esté entrenado, predice y descodifica la predicción
        check_is_fitted(self, "classes_")
        y_encoded = self._model.predict(X)
        return self._le.inverse_transform(y_encoded.astype(int))

    def predict_proba(self, X):
        check_is_fitted(self, "classes_")
        return self._model.predict_proba(X)

    def get_params(self, deep=True):
        return self.xgb_params

    def set_params(self, **params):
        self.xgb_params.update(params)
        self._model.set_params(**params)
        return self


class XGBoostTrainer(BaseTrainer):

    @property
    def model_name(self) -> str:
        return "stellar-classifier"

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LabelEncodedXGBClassifier(
                **self.config.hyperparameters
            )),
        ])
