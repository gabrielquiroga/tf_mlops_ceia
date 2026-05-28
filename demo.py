from ml.preprocessing import (
    run_preprocessing
)
from ml.config import load_model_config
from ml.trainers.random_forest import RandomForestTrainer
from ml.trainers.xgboost import XGBoostTrainer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

def main():
    print("Hello from tf-mlops-ceia!")

    config = load_model_config(r"ml\config\xgboost.yaml")

    Xtr, Xte, ytr, yte = run_preprocessing(
        source=r"dataset\star_classification.csv",
        config=config
    )

    print(Xtr.shape, Xte.shape, ytr.shape, yte.shape)

    model = XGBoostTrainer(config)
    pipeline = model.build_pipeline()
    pipeline.fit(Xtr, ytr)

    y_pred = pipeline.predict(Xte)

    
    print(f"accuracy:         {accuracy_score(yte, y_pred)}")
    print(f"f1_macro:         {f1_score(yte, y_pred, average='macro')}")
    print(f"f1_weighted:      {f1_score(yte, y_pred, average='weighted')}")
    print(f"precision_macro:  {precision_score(yte, y_pred, average='macro')}")
    print(f"recall_macro:     {recall_score(yte, y_pred, average='macro')}")



if __name__ == "__main__":
    main()