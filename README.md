# Baseline de MLOps - UCI Wine Quality

Proyecto académico para la materia de MLOps, enfocado en el entrenamiento, evaluación y seguimiento de un modelo de clasificación mediante scikit-learn y MLflow.

## Estructura del proyecto

- `train.py`: script de entrenamiento, evaluación y registro del experimento.
- `requirements.txt`: dependencias necesarias para ejecutar el proyecto.
- `README.md`: documentación e instrucciones de ejecución.
- `mlflow.db`: base de datos SQLite generada localmente para MLflow Tracking.

## Requisitos

Se requiere Python instalado y conexión a internet para descargar el dataset Wine Quality Red desde UCI.

## Instalación

Clonar el repositorio:

```bash
git clone URL_DE_TU_REPOSITORIO
cd NOMBRE_DE_TU_REPOSITORIO
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución del entrenamiento

Ejecutar con los valores por defecto:

```bash
python train.py
```

Ejecutar con hiperparámetros personalizados:

```bash
python train.py --C 0.5 --max-iter 1000 --class-weight balanced --random-state 42
```

## Visualización de experimentos

Iniciar la interfaz de MLflow desde la carpeta del proyecto:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abrir en el navegador:

http://127.0.0.1:5000

En la interfaz se pueden consultar los parámetros, métricas, artefactos y modelos registrados para cada corrida.

## Dataset

Wine Quality Red, disponible en el repositorio de datos de UCI:

https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
