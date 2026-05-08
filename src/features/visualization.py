import os
import matplotlib.pyplot as plt
import librosa.display


def mostrar_waveform(audio, sample_rate, titulo="Waveform"):
    """
    Muestra la forma de onda de un audio.
    """
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sample_rate)
    plt.title(titulo)
    plt.xlabel("Tiempo (segundos)")
    plt.ylabel("Amplitud")
    plt.tight_layout()
    plt.show()


def mostrar_waveforms_por_clase(dataset, cantidad=5):
    """
    Muestra algunos waveforms por cada clase del dataset.
    """
    clases_mostradas = {}

    for item in dataset:
        label = item["label"]

        if label not in clases_mostradas:
            clases_mostradas[label] = 0

        if clases_mostradas[label] < cantidad:
            archivo = os.path.basename(item["archivo"])
            titulo = f"Clase: {label} | Archivo: {archivo}"

            mostrar_waveform(
                item["audio"],
                item["sample_rate"],
                titulo
            )

            clases_mostradas[label] += 1