# Baseline de MLOps - UCI Wine Quality

Proyecto académico para la materia de MLOps enfocado en la producción y versionado de un modelo de Regresión Logística utilizando MLflow.

## Estructura del Proyecto
- `train.py`: Script principal de entrenamiento y registro en MLflow.
- `requirements.txt`: Dependencias del entorno y versiones exactas.
- `mlflow.db`: Base de datos SQLite para el seguimiento de experimentos (generada localmente).

## Instrucciones de Ejecución

1. Clonar el repositorio e instalar las dependencias:
   ```bash
   pip install -r requirements.txt

## Ejecutar el script de entrenamiento
   ```bash
  python train.py

## Ejecutar personalizando hiperparámetros
   ```bash
  python train.py --C 0.5 --max-iter 1000 --class-weight balanced --random-state 42

## Visualizar los experimentos en MLFlow UI
   ```bash
  mlflow ui --backend-store-uri sqlite:///mlflow.db
