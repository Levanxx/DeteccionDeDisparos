import os
import librosa

carpeta = "data/raw/"
formatos_validos = (".wav", ".mp3", ".flac", ".ogg")

for archivo in os.listdir(carpeta):
    if archivo.lower().endswith(formatos_validos):
        ruta = os.path.join(carpeta, archivo)
        
        try:
            audio, sr = librosa.load(ruta, sr=None)
            print(f"{archivo} → OK")
        except Exception as e:
            print(f"{archivo} → ERROR")