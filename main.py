import mlflow
import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from mlflow.models import infer_signature

from ml.preprocessing import load_data, remove_outliers, resolve_label_conflicts, split_dataset


def calcular_metricas(y_true, y_pred):
    return {
        "accuracy":         accuracy_score(y_true, y_pred),
        "f1_macro":         f1_score(y_true, y_pred, average="macro"),
        "f1_weighted":      f1_score(y_true, y_pred, average="weighted"),
        "precision_macro":  precision_score(y_true, y_pred, average="macro"),
        "recall_macro":     recall_score(y_true, y_pred, average="macro"),
    }


def main():
    # ─────────────────────────────────────────────
    # Preparar los datos
    # ─────────────────────────────────────────────
    df = load_data()
    df = remove_outliers(df)
    df = resolve_label_conflicts(df)
    X_train, X_test, y_train, y_test = split_dataset(df)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ─────────────────────────────────────────────
    # MLflow
    # ─────────────────────────────────────────────
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Clasificacion_Cuerpos_Celeste")

    # ─────────────────────────────────────────────
    # Random Forest
    # ─────────────────────────────────────────────
    with mlflow.start_run(run_name="Random_Forest"):
        params_rf = {"n_estimators": 100, "random_state": 42}

        rf = RandomForestClassifier(**params_rf, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        metricas_rf = calcular_metricas(y_test, y_pred_rf)

        mlflow.log_params(params_rf)
        mlflow.log_metrics(metricas_rf)

        signature = infer_signature(X_train, y_pred_rf)
        mlflow.sklearn.log_model(rf, "modelo_rf", signature=signature)

        print("RF registrado en MLflow")
        print(metricas_rf)

    # ─────────────────────────────────────────────
    # XGBoost
    # ─────────────────────────────────────────────
    with mlflow.start_run(run_name="XGBoost"):
        params_xgb = {
            "n_estimators":     471,
            "learning_rate":    0.01885,
            "max_depth":        7,
            "min_child_weight": 4,
            "gamma":            0.5363,
            "subsample":        0.7057,
            "colsample_bytree": 0.6554,
            "reg_alpha":        0.0386,
            "reg_lambda":       1.0642,
            "objective":        "multi:softmax",
            "num_class":        3,
            "random_state":     42,
            "eval_metric":      "mlogloss",
        }

        xgb_model = xgb.XGBClassifier(**params_xgb, n_jobs=-1)
        xgb_model.fit(X_train_scaled, y_train)
        y_pred_xgb = xgb_model.predict(X_test_scaled)

        metricas_xgb = calcular_metricas(y_test, y_pred_xgb)

        mlflow.log_params(params_xgb)
        mlflow.log_metrics(metricas_xgb)

        signature = infer_signature(X_train_scaled, y_pred_xgb)
        mlflow.xgboost.log_model(xgb_model, "modelo_xgb", signature=signature)

        print("XGBoost registrado en MLflow")
        print(metricas_xgb)


if __name__ == "__main__":
    main()
