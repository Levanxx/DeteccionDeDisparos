import pandas as pd
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# RUTA DEL DATASET
# =========================

INPUT_PATH = Path(
    "data/processed/features_scaled.csv"
)

# =========================
# FUNCIÓN PRINCIPAL
# =========================

def optimize_random_forest():

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
    # MODELO BASE
    # =========================

    model = RandomForestClassifier(
        random_state=42
    )

    # =========================
    # HIPERPARÁMETROS
    # =========================

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5]
    }

    # =========================
    # GRID SEARCH
    # =========================

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    print("\nOptimizando modelo...")

    grid_search.fit(
        X_train,
        y_train
    )

    # =========================
    # MEJOR MODELO
    # =========================

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # =========================
    # RESULTADOS
    # =========================

    print("\n=== MEJORES PARÁMETROS ===")
    print(grid_search.best_params_)

    print(f"\nAccuracy optimizado: {accuracy:.4f}")


# =========================
# EJECUCIÓN
# =========================

if __name__ == "__main__":
    optimize_random_forest()