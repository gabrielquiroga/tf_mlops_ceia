import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


def compute_metrics(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Devuelve métricas escalares."""
    y_pred = pipeline.predict(X_test)

    return _compute_metrics(y_test, y_pred)

def _compute_metrics(
    y_test: pd.Series,
    y_pred: pd.Series,
) -> dict:
    """
    Devuelve métricas escalares. Es para uso interno.
    No vuelve a hacer la predicciones sino que recibe y_pred e y_test.
    """

    return {
        "accuracy":         accuracy_score(y_test, y_pred),
        "f1_macro":         f1_score(y_test, y_pred, average="macro"),
        "f1_weighted":      f1_score(y_test, y_pred, average="weighted"),
        "precision_macro":  precision_score(y_test, y_pred, average="macro"),
        "recall_macro":     recall_score(y_test, y_pred, average="macro"),        
    }

def print_report(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """
    Imprime por consola un reporte de métricas con la siguiente estructura:

    ─── Métricas ────────────────────────────────
    accuracy:           <accuracy_value>
    f1_macro:           <f1_macro_value>
    f1_weighted:        <f1_weighted_value>
    precision_macro:    <precision_macro_value>
    recall_macro:       <recall_macro_value>

    ─── Classification Report ───────────────────
    <classification_report>

    ─── Confusion Matrix ────────────────────────
    <confussion_matrix

    """
    y_pred = pipeline.predict(X_test)

    print("\n─── Métricas ────────────────────────────────")
    for name, value in _compute_metrics(y_test, y_pred).items():
        print(f"    {name}:         {value:.4f}")

    print("\n─── Classification Report ───────────────────")
    print(classification_report(y_test, y_pred))

    print("\n─── Confusion Matrix ────────────────────────")
    labels=sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    print(cm_df)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()

    return disp