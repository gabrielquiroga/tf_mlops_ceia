from ml.preprocessing import (
    run_preprocessing
)
from ml.config import load_model_config

def main():
    print("Hello from tf-mlops-ceia!")

    config = load_model_config(r"ml\config\random_forest.yaml")

    Xtr, Xte, ytr, yte = run_preprocessing(
        source=r"dataset\star_classification.csv",
        config=config
    )

    print(Xtr.shape, Xte.shape, ytr.shape, yte.shape)


if __name__ == "__main__":
    main()
