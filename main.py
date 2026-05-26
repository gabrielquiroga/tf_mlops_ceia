from ml.preprocessing import (
    load_data,
    remove_outliers,
    resolve_label_conflicts,
    split_dataset,
)
def main():
    print("Hello from tf-mlops-ceia!")
    df = load_data()
    df = remove_outliers(df)
    print(df.shape)
    df = resolve_label_conflicts(df)
    print(df.shape)
    Xtr, Xte, ytr, yte = split_dataset(df)

    print(Xtr.shape, Xte.shape, ytr.shape, yte.shape)



if __name__ == "__main__":
    main()
