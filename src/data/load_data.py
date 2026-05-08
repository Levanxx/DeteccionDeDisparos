import os
import numpy as np
import librosa


SAMPLE_RATE = 22050
DURACION_OBJETIVO = 3
FORMATOS_VALIDOS = (".wav", ".mp3", ".flac", ".ogg")


def cargar_audio(ruta, sample_rate=SAMPLE_RATE, duracion=DURACION_OBJETIVO):
    """
    Carga un audio de forma uniforme:
    - Convierte a mono
    - Convierte al mismo sample rate
    - Ajusta todos los audios a la misma duración
    """

    try:
        audio, sr = librosa.load(
            ruta,
            sr=sample_rate,
            mono=True
        )

        muestras_objetivo = sample_rate * duracion

        # Si el audio es más largo, se recorta
        if len(audio) > muestras_objetivo:
            audio = audio[:muestras_objetivo]

        # Si el audio es más corto, se rellena con ceros
        elif len(audio) < muestras_objetivo:
            faltante = muestras_objetivo - len(audio)
            audio = np.pad(audio, (0, faltante), mode="constant")

        return audio, sample_rate

    except Exception as e:
        print(f"Error cargando {ruta}: {e}")
        return None, None


def cargar_dataset(base_path):
    datos = []

    clases = {
        "disparos": 1,
        "no_disparos": 0
    }

    for clase, label in clases.items():
        carpeta = os.path.join(base_path, clase)

        if not os.path.exists(carpeta):
            print(f"No existe la carpeta: {carpeta}")
            continue

        for archivo in os.listdir(carpeta):
            if archivo.lower().endswith(FORMATOS_VALIDOS):
                ruta = os.path.join(carpeta, archivo)

                audio, sr = cargar_audio(ruta)

                if audio is not None:
                    datos.append({
                        "archivo": archivo,
                        "ruta": ruta,
                        "audio": audio,
                        "sample_rate": sr,
                        "label": label,
                        "clase": clase,
                        "duracion": len(audio) / sr
                    })

    return datos