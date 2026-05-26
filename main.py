from ml.preprocessing import (
    load_data,
    remove_outliers
)
def main():
    print("Hello from tf-mlops-ceia!")
    df = load_data()
    df = remove_outliers(df)

    print(df.shape)



if __name__ == "__main__":
    main()
