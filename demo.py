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

from ml.evaluate import print_report

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

    print_report(pipeline, Xte, yte)



if __name__ == "__main__":
    main()