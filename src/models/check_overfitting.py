import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

INPUT_PATH = Path("data/processed/features_scaled.csv")
MODEL_PATH = Path("models/random_forest_model.pkl")

def check_overfitting():
    df = pd.read_csv(INPUT_PATH)

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = joblib.load(MODEL_PATH)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    precision = precision_score(y_test, test_pred)
    recall = recall_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)

    difference = train_acc - test_acc

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    print("\n=== CHECK OVERFITTING ===")

    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Diferencia    : {difference:.4f}")

    print("\n=== MÉTRICAS EN TEST ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\n=== CROSS VALIDATION ===")
    print(f"Scores: {cv_scores}")
    print(f"Media : {cv_scores.mean():.4f}")
    print(f"Std   : {cv_scores.std():.4f}")

    print("\n=== DIAGNÓSTICO ===")

    if train_acc >= 0.98 and difference >= 0.08:
        print("Posible overfitting fuerte.")
    elif train_acc >= 0.95 and difference >= 0.05:
        print("Posible overfitting moderado.")
    elif train_acc >= 0.98 and test_acc >= 0.98:
        print("Métricas muy altas. Revisa posible fuga de datos.")
    else:
        print("No se observa overfitting fuerte.")

if __name__ == "__main__":
    check_overfitting()