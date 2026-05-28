import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================
# RUTA DEL DATASET
# =========================

INPUT_PATH = Path("data/processed/features_scaled.csv")

# =========================
# FUNCIÓN PRINCIPAL
# =========================

def train_random_forest():

    # =========================
    # CARGAR DATASET
    # =========================

    df = pd.read_csv(INPUT_PATH)

    print("\nDataset cargado correctamente.")

    # =========================
    # FEATURES Y LABELS
    # =========================

    X = df.drop(columns=["label"])
    y = df["label"]

    # =========================
    # DIVISIÓN DEL DATASET
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nDataset dividido correctamente.")

    # =========================
    # MODELO RANDOM FOREST
    # =========================

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # =========================
    # ENTRENAMIENTO
    # =========================

    model.fit(X_train, y_train)

    print("\nModelo Random Forest entrenado correctamente.")

    # =========================
    # PREDICCIONES
    # =========================

    predictions = model.predict(X_test)

    # =========================
    # MÉTRICAS
    # =========================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_test,
        predictions
    ))


# =========================
# EJECUCIÓN
# =========================

if __name__ == "__main__":
    train_random_forest()