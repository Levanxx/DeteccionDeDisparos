import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("data/processed/features_scaled.csv")

def split_dataset():

    # Cargar dataset
    df = pd.read_csv(INPUT_PATH)

    # Features y labels
    X = df.drop(columns=["label"])
    y = df["label"]

    # División
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nDataset dividido correctamente.")

    print(f"\nX_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")

    print(f"\ny_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

if __name__ == "__main__":
    split_dataset()