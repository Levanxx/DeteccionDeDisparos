# Sistema Inteligente de Detección de Disparos mediante Análisis Acústico con Machine Learning

## Descripción del Proyecto

El presente proyecto tiene como finalidad desarrollar un sistema inteligente capaz de detectar sonidos de disparos mediante análisis acústico y técnicas de Machine Learning.

El sistema analizará archivos de audio para clasificarlos en dos categorías:

* Disparos
* No disparos

Para ello, se implementarán procesos de:

* carga y procesamiento de audio
* extracción de características acústicas
* entrenamiento de modelos de Machine Learning
* evaluación de rendimiento
* desarrollo de un prototipo funcional

Este proyecto aplica conceptos de Inteligencia Artificial, Procesamiento Digital de Señales y Ciencia de Datos en un problema real relacionado con seguridad y monitoreo acústico.

---

# Tecnologías Utilizadas

## Lenguaje Principal

* Python

## Librerías

* Librosa
* NumPy
* Matplotlib
* Scikit-learn
* Pandas
* SoundFile

## Herramientas

* Git
* GitHub
* Visual Studio Code
* Jira

---

# Arquitectura del Proyecto

```plaintext
DeteccionDeDisparos/
│
├── data/
│   ├── processed/
│   └── raw/
│       ├── disparos/
│       └── no_disparos/
│
├── src/
│   ├── app/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   └── visualization.py
│   │
│   ├── models/
│   └── tests/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Funcionalidades Implementadas

## Carga de Dataset

* Lectura automática de audios
* Clasificación automática por carpetas
* Soporte para múltiples formatos de audio
* Validación básica de errores

## Carga Uniforme de Audio

* Conversión de audios a mono
* Estandarización de sample rate
* Ajuste automático de duración
* Padding y trimming automático

## Visualización de Waveforms

* Representación gráfica de señales acústicas
* Visualización por clases
* Comparación entre disparos y no disparos

---

# Estado Actual del Proyecto

## Completado

* Estructura inicial del proyecto
* Configuración del entorno virtual
* Carga de dataset
* Carga uniforme de audios
* Visualización de waveforms
* Configuración de Git y GitHub

## En Desarrollo

* Extracción de características acústicas
* Mel Spectrograms
* MFCC
* Entrenamiento de modelos

## Futuro

* Streamlit / FastAPI
* Predicción en tiempo real
* Optimización de modelos
* Despliegue

---

# Roles del Equipo

| Rol                                   | Integrante | GitHub       |
| ------------------------------------- | ---------- | ------------ |
| Líder de Machine Learning             | Leonardo   | Levanxx      |
| Responsable de Datos (Dataset)        | Jesus      | Yumecry      |
| Responsable de Documentación Técnica  | Xiomara    | Xiomara306V  |
| Desarrollador de Interfaz / Prototipo | Miguel     | mofuel       |
| Responsable de Pruebas e Integración  | Jhoan      | JhoanAronith |

---

# Metodología

El proyecto sigue una metodología basada en:

* CRISP-DM
* Scrum
* desarrollo iterativo incremental

---

# Instalación

## Clonar repositorio

```bash
git clone https://github.com/Levanxx/DeteccionDeDisparos.git
```

## Crear entorno virtual

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# Ejecución

```bash
python main.py
```

En macOS:

```bash
python3 main.py
```

---

# Dataset

El dataset se encuentra organizado en:

```plaintext
data/raw/disparos
data/raw/no_disparos
```

Cada carpeta contiene audios correspondientes a su categoría.

---

# Próximas Implementaciones

* Extracción de MFCC
* Spectrogramas
* Chroma Features
* Zero Crossing Rate
* Random Forest
* SVM
* XGBoost
* Redes Neuronales
* Streamlit
* FastAPI
