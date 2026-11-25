"""Configuración central de rutas y constantes del pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONTRACT_PATH = ROOT / "contracts" / "iris_data_contract.json"

RAW_PATH = ROOT / "data" / "raw" / "iris.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"

MODEL_PATH = ROOT / "models" / "model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"
LOGS_DIR = ROOT / "logs"

MLFLOW_URI = "sqlite:///" + (ROOT / "mlflow.db").as_posix()
EXPERIMENT_NAME = "Pipeline_Iris_Actividad2"

SEED = 42
TEST_SIZE = 0.2
MIN_ACCURACY = 0.90
