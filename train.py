"""
train.py
Baseline de clasificación para el dataset Wine Quality (vino tinto).

Uso con valores por defecto:
    python train.py

Ejemplo personalizando hiperparámetros:
    python train.py --C 0.5 --max-iter 1000 --class-weight balanced --random-state 123

El script carga los datos, elimina duplicados, define la variable objetivo
binaria (quality >= 6), divide los datos, entrena una regresión logística,
evalúa el modelo y registra parámetros, métricas y modelo con MLflow.
"""

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
TARGET_COLUMN = "quality"
TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42
EXPERIMENT_NAME = "Diagnostico_MLOps_Baseline"


def parse_args():
    """Define los argumentos configurables desde la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrena y evalúa el baseline de Wine Quality con MLflow."
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inversa de la regularización de LogisticRegression (por defecto: 1.0).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Número máximo de iteraciones del optimizador (por defecto: 500).",
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Pesos de clase: 'none' o 'balanced' (por defecto: none).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Semilla aleatoria para la partición y el modelo (por defecto: 42).",
    )
    return parser.parse_args()


def load_data(url: str = DATA_URL) -> pd.DataFrame:
    """Carga el dataset Wine Quality Red desde la URL de UCI."""
    df = pd.read_csv(url, sep=";")
    print(f"Dataset cargado: {df.shape[0]} filas y {df.shape[1]} columnas.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina registros duplicados, como en el notebook original."""
    initial_rows = len(df)
    df_clean = df.drop_duplicates().copy()
    removed = initial_rows - len(df_clean)
    print(f"Registros duplicados eliminados: {removed}")
    print(f"Tamaño después de la limpieza: {df_clean.shape}")
    return df_clean


def prepare_data(df: pd.DataFrame, target_column: str = TARGET_COLUMN):
    """
    Separa X e y. La variable objetivo es binaria:
    1 = calidad alta (quality >= 6); 0 = calidad baja/regular (quality < 6).
    """
    if target_column not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{target_column}'.")

    X = df.drop(columns=[target_column])
    y = (df[target_column] >= 6).astype(int)
    return X, y


def split_data(
    X,
    y,
    test_size: float = TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """Divide los datos en entrenamiento y prueba conservando proporciones de clase."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_model(
    C: float = 1.0,
    max_iter: int = 500,
    class_weight: str = "none",
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Construye el pipeline de escalado y regresión logística."""
    sklearn_class_weight = None if class_weight == "none" else "balanced"

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="lbfgs",
                    C=C,
                    max_iter=max_iter,
                    class_weight=sklearn_class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate_model(model, X_test, y_test) -> dict:
    """Calcula e imprime las métricas principales del clasificador."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    print("\n=== Métricas de evaluación ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n=== Reporte de clasificación ===")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Baja/Regular (0)", "Alta Calidad (1)"],
            zero_division=0,
        )
    )

    print("=== Matriz de confusión ===")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(cm)

    return metrics, y_pred, cm


def run_experiment(args) -> None:
    """Ejecuta el flujo completo y registra la corrida en MLflow."""
    df = load_data()
    df_clean = clean_data(df)
    X, y = prepare_data(df_clean)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=args.random_state,
    )
    print(f"Registros de entrenamiento: {len(X_train)}")
    print(f"Registros de prueba: {len(X_test)}")

    model = build_model(
        C=args.C,
        max_iter=args.max_iter,
        class_weight=args.class_weight,
        random_state=args.random_state,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="wine_quality_logistic_regression"):
        model.fit(X_train, y_train)
        metrics, y_pred, cm = evaluate_model(model, X_test, y_test)

        # 1. Registrar los hiperparámetros y la configuración de la corrida.
        params = {
            "model": "LogisticRegression",
            "solver": "lbfgs",
            "C": args.C,
            "max_iter": args.max_iter,
            "class_weight": args.class_weight,
            "test_size": TEST_SIZE,
            "random_state": args.random_state,
            "target_definition": "quality >= 6",
            "duplicate_rows_removed": len(df) - len(df_clean),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        for name, value in params.items():
            mlflow.log_param(name, value)

        # 2. Registrar cada métrica de evaluación.
        for name, value in metrics.items():
            mlflow.log_metric(name, float(value))

        # 3. Crear y registrar artefactos: métricas en JSON y matriz de confusión en CSV.
        # Los archivos temporales se eliminan automáticamente al terminar la corrida.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            metrics_path = temp_path / "metrics.json"
            with metrics_path.open("w", encoding="utf-8") as file:
                json.dump(metrics, file, indent=4, ensure_ascii=False)

            confusion_path = temp_path / "confusion_matrix.csv"
            confusion_df = pd.DataFrame(
                cm,
                index=["real_0_baja_regular", "real_1_alta_calidad"],
                columns=["pred_0_baja_regular", "pred_1_alta_calidad"],
            )
            confusion_df.to_csv(confusion_path, index=True)

            mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
            mlflow.log_artifact(str(confusion_path), artifact_path="evaluation")

        # 4. Registrar el pipeline completo (escalador + regresión logística).
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="baseline_model",
        )

    print("\nExperimento registrado en MLflow.")
    print("Base de datos de tracking: mlflow.db")


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
