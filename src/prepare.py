"""Carga, valida y divide el dataset Iris en entrenamiento y prueba."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

from src import config
from src.validation import load_contract, raise_if_invalid, validate_csv

LOGGER = logging.getLogger("pipeline.prepare")


def _write_csv(path: Path, header: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def run(seed: int = config.SEED, test_size: float = config.TEST_SIZE) -> Dict[str, int]:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    contract = load_contract(config.CONTRACT_PATH)
    feature_names = [spec["name"] for spec in contract["fields"]]
    target_name = contract["target"]["name"]
    header = feature_names + [target_name]

    iris = load_iris()
    rows = []
    for measures, species in zip(iris.data, iris.target):
        record = {name: float(value) for name, value in zip(feature_names, measures)}
        record[target_name] = int(species)
        rows.append(record)

    _write_csv(config.RAW_PATH, header, rows)
    LOGGER.info("Datos crudos escritos en %s (%d filas)", config.RAW_PATH.name, len(rows))

    raise_if_invalid(validate_csv(config.RAW_PATH, contract))

    train_rows, test_rows = train_test_split(rows, test_size=test_size, random_state=seed)
    _write_csv(config.TRAIN_PATH, header, train_rows)
    _write_csv(config.TEST_PATH, header, test_rows)
    LOGGER.info(
        "Partición (random_state=%d, test_size=%s): %d train / %d test",
        seed, test_size, len(train_rows), len(test_rows),
    )

    raise_if_invalid(validate_csv(config.TRAIN_PATH, contract))
    raise_if_invalid(validate_csv(config.TEST_PATH, contract))
    return {"raw": len(rows), "train": len(train_rows), "test": len(test_rows)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(run())
