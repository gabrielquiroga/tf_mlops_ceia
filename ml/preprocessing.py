from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit

from app.config import settings

DEFAULT_URL = source = settings.data_source_url

def load_data(source: str=DEFAULT_URL) -> pd.DataFrame:
    """Acepta URL o path local"""
    if not source:
        print("No data source specified, using default URL")
        # source = settings.data_source_url
    return pd.read_csv(source)

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame: 
    """Solo elimina outliers que son claro error, saturado a -9999."""
    return df[(df['u'] != -9999) & (df['g'] != -9999) & (df['z'] != -9999)]

def resolve_label_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Mantiene la clase mayoritaria por `obj_ID`"""
    group_counts = df.group_by('obj_ID').size()
    duplicated_obj_ids = group_counts[group_counts > 1].index
    unique_obj_ids = group_counts[group_counts == 1].index

    rows_to_keep = []
    total_discarded = 0
    for oid in duplicated_obj_ids:
        group = df[df['obj_ID'] == oid]
        class_counts = group['class'].value_counts()

        



def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...

def build_pipeline(df: pd.DataFrame) -> Pipeline:
    ...
