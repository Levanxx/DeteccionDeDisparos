import os
import numpy as np
from src.data.load_data import cargar_dataset
from src.features.visualization import mostrar_waveforms_por_clase

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
carpeta = os.path.join(BASE_DIR, "data", "raw")

dataset = cargar_dataset(carpeta)

if not dataset:
    print("No se cargaron audios")
    exit()

print(f"\nAudios cargados: {len(dataset)}")

for item in dataset[:10]:
    amplitud_maxima = np.max(np.abs(item["audio"]))

    print("-" * 40)
    print(f"Archivo: {item['archivo']}")
    print(f"Clase: {item['clase']}")
    print(f"Label: {item['label']}")
    print(f"Sample rate: {item['sample_rate']}")
    print(f"Muestras: {len(item['audio'])}")
    print(f"Duración: {item['duracion']:.2f} segundos")
    print(f"Amplitud máxima: {amplitud_maxima:.4f}")

'''
# ================================
# ANÁLISIS DE DURACIÓN
# ================================

duraciones = [d["duracion"] for d in dataset]

print("\n--- Estadísticas de duración ---")
print(f"Duración mínima: {min(duraciones):.2f} s")
print(f"Duración máxima: {max(duraciones):.2f} s")
print(f"Duración promedio: {sum(duraciones)/len(duraciones):.2f} s")


print("\n--- Audios muy cortos (<1s) ---")
for d in dataset:
    if d["duracion"] < 1:
        print(d["archivo"], f"{d['duracion']:.2f}s")


print("\n--- Audios muy largos (>10s) ---")
for d in dataset:
    if d["duracion"] > 10:
        print(d["archivo"], f"{d['duracion']:.2f}s")
'''


# ================================
# WAVEFORMS POR CLASE
# ================================

mostrar_waveforms_por_clase(dataset, cantidad=5)