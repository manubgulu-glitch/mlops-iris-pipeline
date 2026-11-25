"""Inferencia con validación previa contra el contrato de datos.

Uso: python -m src.predict --example valid | --example invalid | --json "{...}"
Retorna 0 si predice, 2 si la entrada es inválida y 3 si falta el modelo entrenado.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict

from src import config
from src.validation import DataValidationError, ValidationReport, load_contract, validate_record


def predict(payload: Dict[str, object]) -> Dict[str, object]:
    contract = load_contract(config.CONTRACT_PATH)

    issues = validate_record(payload, contract, strict=True, include_target=False)
    if issues:
        report = ValidationReport(source="solicitud de inferencia", rows_checked=1, issues=issues)
        raise DataValidationError(report)

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe {config.MODEL_PATH}. Ejecute primero: python pipeline.py")

    import joblib

    model = joblib.load(config.MODEL_PATH)
    row = [[payload[spec["name"]] for spec in contract["fields"]]]
    species = int(model.predict(row)[0])
    return {"species": species, "label": contract["target"]["labels"][str(species)]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Predicción validada con el contrato de datos")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", choices=["valid", "invalid"], help="usa un ejemplo del contrato")
    source.add_argument("--json", help="entrada en formato JSON")
    source.add_argument("--file", help="ruta a un archivo .json con la entrada")
    args = parser.parse_args(argv)

    if args.example:
        payload = load_contract(config.CONTRACT_PATH)["examples"][args.example]["input"]
    elif args.json:
        payload = json.loads(args.json)
    else:
        with open(args.file, encoding="utf-8") as handle:
            payload = json.load(handle)

    print("Entrada:", json.dumps(payload, ensure_ascii=False))
    try:
        result = predict(payload)
    except DataValidationError as exc:
        print("ENTRADA INVALIDA - rechazada antes de inferencia:")
        for issue in exc.report.issues:
            print("  -", issue)
        return 2
    except FileNotFoundError as exc:
        print("ERROR:", exc)
        return 3

    print(f"Predicción: {result['label']} (clase {result['species']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
