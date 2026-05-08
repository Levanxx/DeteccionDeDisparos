import os
import soundfile as sf


def guardar_audio_procesado(audio, sample_rate, clase, nombre_archivo):
    """
    Guarda un audio procesado en la carpeta data/processed.

    Parámetros:
    - audio: señal de audio procesada
    - sample_rate: frecuencia de muestreo
    - clase: disparos o no_disparos
    - nombre_archivo: nombre original del archivo
    """

    # Convertir cualquier extensión a .wav
    nombre_salida = os.path.splitext(nombre_archivo)[0] + ".wav"

    # Ruta de salida
    ruta_salida = os.path.join(
        "data",
        "processed",
        clase,
        nombre_salida
    )

    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    # Guardar audio
    sf.write(ruta_salida, audio, sample_rate)

    print(f"Audio guardado: {ruta_salida}")