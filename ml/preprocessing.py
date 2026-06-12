from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit

from ml.schema import (
    GROUP_COL,
    TARGET,
    OUTLIER_SENTINEL,
    OUTLIER_COLS,
)

from ml.config import ModelConfig

def load_data(source: str) -> pd.DataFrame:
    """Acepta URL o path local"""
    if not source:
        print("No data source specified, using default URL")
        # source = settings.data_source_url
    return pd.read_csv(source)

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame: 
    """Solo elimina outliers que son claro error, saturado a -9999."""
    mask = pd.Series(True, index=df.index)
    for col in OUTLIER_COLS:
        mask &= df[col] != OUTLIER_SENTINEL
    return df[mask].copy()

def resolve_label_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Para cada `obj_ID` con múltiples registros, conserva solo los de la clase
    mayoritaria. En caso de empate, conserva la primera alfabéticamente."""
    counts = (
        df.groupby([GROUP_COL, TARGET]) # Agrupa el dataframe por `obj_ID` y `class`.
        .size() # Cuenta las filas por grupo. Produce una salida 'Series' con un MultiIndex de `obj_ID` y `class`, y el conteo como valores.
        .rename('class_count') # Renombra la serie
        .reset_index() # Convierte el índice en columnas normales, así el resultado es un DataFrame manipulable
    )

    # Clase mayoritaria por obj_ID
    # Si hay empate, se conserva el primer max segun el orden resultante
    majority = (
        counts.sort_values([GROUP_COL, 'class_count'], ascending=[True, False]) # Ordena por `obj_ID` y luego por `class_count` en orden descendente, asegurando que la clase mayoritaria esté primero para cada `obj_ID`.
        .groupby('obj_ID') # Agrupa por `obj_ID` para identificar la clase mayoritaria.
        .head(1) # Toma la primera fila por cada grupo de `obj_ID`.
        .rename(columns={TARGET: 'majority_class'}) # Renombra la columna
        [[GROUP_COL, 'majority_class']] # Se queda solo con las columnas relevantes para el merge
    )

    out = df.merge(majority, on=GROUP_COL, how='left') # Combina los DataFrames
    # Filtra el DataFrame mergeado, donde `class` es igual a `majority_class`, y luego dropea la columna `majority_class`.
    out = out[out[TARGET] == out['majority_class']].drop(columns=['majority_class'])

    # El DataFrame resultante tiene el orden del DataFrame original.
    return out.reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    config: ModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split del dataset evitando separar los `obj_ID` para evitar data leakage."""

    groups = df[GROUP_COL].to_numpy()

    splitter = GroupShuffleSplit(
        test_size=config.test_size,
        n_splits=1,
        random_state=config.random_state
    )
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    X_train = _get_relevant_features(train_df, config.features)
    X_test = _get_relevant_features(test_df, config.features)
    y_train, y_test = _get_target(train_df), _get_target(test_df)

    return X_train, X_test, y_train, y_test

def run_preprocessing(
    source: str,
    config: ModelConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Función de entrada que ejecuta el preprocesamiento completo.
    Devuelve los sets de training y test.
    No incluye escalado ni codificación, ya que se integran el pipeline propio del modelo.
    """
    df = load_data(source)
    df = remove_outliers(df)
    df = resolve_label_conflicts(df)
    return split_dataset(df, config)

def preprocess_for_inference(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Preprocesamiento para inferencia: solo aplica los pasos válidos sobre datos nuevos.
    No incluye resolve_label_conflicts.
    """
    df = remove_outliers(df)
    return df[features].copy()

def _get_relevant_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Filtra el DataFrame para quedarse solo con las columnas relevantes."""
    return df[features].copy()

def _get_target(df: pd.DataFrame, target_col: str = TARGET) -> pd.Series:
    """Extrae la columna objetivo del DataFrame."""
    return df[TARGET].copy()
