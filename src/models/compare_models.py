import pandas as pd

# =========================
# RESULTADOS DE LOS MODELOS
# =========================

results = {
    "Modelo": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        0.8398,
        0.9702
    ],
    "Precision": [
        0.82,
        0.96
    ],
    "Recall": [
        0.84,
        0.97
    ],
    "F1-Score": [
        0.83,
        0.97
    ]
}

# =========================
# DATAFRAME
# =========================

df = pd.DataFrame(results)

print("\n=== COMPARACIÓN DE MODELOS ===\n")

print(df)

# =========================
# MEJOR MODELO
# =========================

best_model = df.loc[
    df["Accuracy"].idxmax()
]

print("\n=== MEJOR MODELO ===\n")

print(
    f"{best_model['Modelo']} "
    f"con accuracy de "
    f"{best_model['Accuracy']:.4f}"
)