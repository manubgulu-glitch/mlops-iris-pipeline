"""Entrena el modelo Random Forest y registra el experimento en MLflow."""
from __future__ import annotations

import csv
import json
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src import config
from src.validation import load_contract, raise_if_invalid, validate_csv

LOGGER = logging.getLogger("pipeline.train")


def read_xy(path: Path, contract: Dict) -> Tuple[List[List[float]], List[int]]:
    """Lee un CSV validado y devuelve (X, y) según el orden del contrato."""
    names = [spec["name"] for spec in contract["fields"]]
    target = contract["target"]["name"]
    features, labels = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            features.append([float(record[name]) for name in names])
            labels.append(int(record[target]))
    return features, labels


def run(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    seed: int = config.SEED,
) -> Dict[str, object]:
    import joblib
    import mlflow
    import mlflow.sklearn
    import sklearn
    from sklearn.ensemble import RandomForestClassifier

    contract = load_contract(config.CONTRACT_PATH)

    train_report = raise_if_invalid(validate_csv(config.TRAIN_PATH, contract))
    test_report = raise_if_invalid(validate_csv(config.TEST_PATH, contract))

    X_train, y_train = read_xy(config.TRAIN_PATH, contract)
    X_test, y_test = read_xy(config.TEST_PATH, contract)

    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    run_name = f"pipeline_n{n_estimators}_depth{max_depth}"
    with mlflow.start_run(run_name=run_name) as active:
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "seed": seed,
            "n_train": len(X_train),
            "n_test": len(X_test),
        })
        mlflow.set_tags({
            "contract_name": contract["contract_name"],
            "contract_version": contract["contract_version"],
            "data_validation": "passed",
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "mlflow": mlflow.__version__,
        })

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
        model.fit(X_train, y_train)
        accuracy = float(model.score(X_test, y_test))
        mlflow.log_metric("accuracy", accuracy)

        config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, config.MODEL_PATH)

        metrics = {
            "run_id": active.info.run_id,
            "run_name": run_name,
            "accuracy": accuracy,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "contract_version": contract["contract_version"],
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }
        config.METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

        # MLflow 3 guarda con skops y exige declarar el tipo de árbol como confiable.
        mlflow.sklearn.log_model(
            model, name="iris_model", skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        mlflow.log_artifact(str(config.CONTRACT_PATH), artifact_path="contract")
        mlflow.log_artifact(str(config.METRICS_PATH), artifact_path="metrics")
        mlflow.log_dict(
            {"train": train_report.to_dict(), "test": test_report.to_dict()},
            "contract/validation_report.json",
        )

    LOGGER.info("Modelo entrenado: accuracy = %.3f (Run ID %s)", accuracy, metrics["run_id"])
    LOGGER.info("Modelo guardado en %s", config.MODEL_PATH.name)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(run())
