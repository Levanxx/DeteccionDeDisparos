import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# RUTAS
# =========================

INPUT_PATH = Path(
    "data/processed/features_scaled.csv"
)

MODEL_PATH = Path(
    "models/random_forest_model.pkl"
)

# =========================
# FUNCIÓN PRINCIPAL
# =========================

def save_model():

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
        max_depth=20,
        min_samples_split=2,
        random_state=42
    )

    # =========================
    # ENTRENAMIENTO
    # =========================

    model.fit(
        X_train,
        y_train
    )

    print("\nModelo entrenado correctamente.")

    # =========================
    # EVALUACIÓN
    # =========================

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"\nAccuracy: {accuracy:.4f}")

    # =========================
    # CREAR CARPETA MODELS
    # =========================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # =========================
    # GUARDAR MODELO
    # =========================

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\nModelo guardado correctamente.")

    print(f"\nRuta:")
    print(MODEL_PATH)


# =========================
# EJECUCIÓN
# =========================

if __name__ == "__main__":
    save_model()