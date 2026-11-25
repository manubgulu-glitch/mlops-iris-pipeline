#!/usr/bin/env python
"""Script maestro: ejecuta TODO el pipeline con un solo comando.

    python pipeline.py

Etapas (en este orden; si una falla, las siguientes se omiten):

    1. prepare   Carga Iris, lo guarda y lo divide en train/test (80/20, semilla fija).
    2. validate  Verifica el contrato y sus ejemplos, y valida los datos (tipos, nulos, rangos).
    3. train     Revalida los datos, entrena el modelo y registra el experimento en MLflow.
    4. tests     Valida el modelo y la lógica de validación (pruebas automáticas).

Cada ejecución deja un registro en logs/ con las validaciones que pasaron y las que fallaron.

Uso avanzado:
    python pipeline.py --only train                  # solo una etapa
    python pipeline.py --from-step validate          # desde una etapa hasta el final
    python pipeline.py --n-estimators 200 --max-depth 5
"""
from __future__ import annotations

import argparse
import io
import logging
import shutil
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

LOGGER = logging.getLogger("pipeline")
STEPS = ["prepare", "validate", "train", "tests"]


def setup_logging() -> Path:
    config.LOGS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOGS_DIR / f"pipeline_{stamp}.log"

    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    return log_file


# --------------------------------------------------------------------------- pasos
def step_prepare(_args) -> str:
    from src import prepare

    sizes = prepare.run(seed=_args.seed)
    return f"{sizes['train']} train / {sizes['test']} test"


def step_validate(_args) -> str:
    from src.validation import load_contract, raise_if_invalid, validate_csv, validate_record

    contract = load_contract(config.CONTRACT_PATH)
    LOGGER.info("Contrato '%s' versión %s cargado", contract["contract_name"], contract["contract_version"])

    # 1) Coherencia del contrato: el ejemplo válido se acepta y el inválido se rechaza.
    valid = contract["examples"]["valid"]["input"]
    issues = validate_record(valid, contract)
    if issues:
        raise RuntimeError(f"El ejemplo VÁLIDO del contrato no pasa la validación: {issues}")
    LOGGER.info("Ejemplo válido: aceptado, sin problemas")

    invalid = contract["examples"]["invalid"]
    found = {issue.code for issue in validate_record(invalid["input"], contract)}
    expected = set(invalid["expected_errors"])
    if found != expected:
        raise RuntimeError(f"El ejemplo INVÁLIDO no produce los errores esperados. Esperado={expected} Obtenido={found}")
    LOGGER.info("Ejemplo inválido: rechazado con los errores esperados %s", sorted(expected))

    # 2) Datos: los tres archivos deben cumplir el contrato (tipos, nulos y rangos).
    for path in (config.RAW_PATH, config.TRAIN_PATH, config.TEST_PATH):
        raise_if_invalid(validate_csv(path, contract))
    return f"contrato v{contract['contract_version']}, datos válidos"


def step_train(args) -> str:
    from src import train

    metrics = train.run(n_estimators=args.n_estimators, max_depth=args.max_depth, seed=args.seed)
    return f"accuracy={metrics['accuracy']:.3f}"


def step_tests(_args) -> str:
    suite = unittest.TestLoader().discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    stream = io.StringIO()
    # Las pruebas provocan errores de validación a propósito: se silencian en el log para que
    # no se confundan con fallos reales de los datos.
    validation_logger = logging.getLogger("pipeline.validation")
    previous_level = validation_logger.level
    validation_logger.setLevel(logging.CRITICAL)
    try:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    finally:
        validation_logger.setLevel(previous_level)
    for line in stream.getvalue().splitlines():
        if line.strip():
            LOGGER.info("  %s", line)
    if not result.wasSuccessful():
        raise RuntimeError(f"{len(result.failures)} prueba(s) fallida(s) y {len(result.errors)} con error")
    skipped = f", {len(result.skipped)} omitida(s)" if result.skipped else ""
    return f"{result.testsRun} pruebas aprobadas{skipped}"


RUNNERS = {"prepare": step_prepare, "validate": step_validate, "train": step_train, "tests": step_tests}
TITLES = {
    "prepare": "Preparación de datos",
    "validate": "Validación de datos (contrato)",
    "train": "Entrenamiento + MLflow",
    "tests": "Validación del modelo y de la lógica",
}


# --------------------------------------------------------------------------- orquestación
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento y validación con un solo comando")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--only", choices=STEPS, help="ejecuta solo una etapa")
    scope.add_argument("--from-step", choices=STEPS, help="ejecuta desde una etapa hasta el final")
    parser.add_argument("--n-estimators", type=int, default=100, help="número de árboles (default 100)")
    parser.add_argument("--max-depth", type=int, default=None, help="profundidad máxima (default: sin límite)")
    parser.add_argument("--seed", type=int, default=config.SEED, help="semilla de reproducibilidad (default 42)")
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(errors="replace")
    log_file = setup_logging()
    if args.only:
        selected = [args.only]
    elif args.from_step:
        selected = STEPS[STEPS.index(args.from_step):]
    else:
        selected = STEPS

    LOGGER.info("=" * 62)
    LOGGER.info("PIPELINE IRIS - inicio (%s)", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    LOGGER.info("Etapas: %s | semilla=%d", ", ".join(selected), args.seed)
    LOGGER.info("=" * 62)

    results = []
    failed = False
    for index, name in enumerate(selected, start=1):
        if failed:
            results.append((name, "OMITIDO", 0.0, "una etapa anterior falló"))
            continue
        LOGGER.info("")
        LOGGER.info("[%d/%d] %s", index, len(selected), TITLES[name])
        started = time.perf_counter()
        try:
            detail = RUNNERS[name](args)
            status = "OK"
        except Exception as exc:  # noqa: BLE001 - se informa y se corta el pipeline
            LOGGER.error("FALLO en '%s': %s", name, exc)
            detail, status, failed = str(exc).splitlines()[0][:70], "FALLO", True
        results.append((name, status, time.perf_counter() - started, detail))

    LOGGER.info("")
    LOGGER.info("=" * 62)
    LOGGER.info("RESUMEN")
    for name, status, seconds, detail in results:
        LOGGER.info("  %-9s %-8s %6.1fs  %s", name, status, seconds, detail)
    LOGGER.info("Registro de esta ejecución: logs/%s", log_file.name)
    LOGGER.info("RESULTADO FINAL: %s", "FALLO" if failed else "PIPELINE COMPLETADO CON EXITO")
    LOGGER.info("=" * 62)

    for handler in LOGGER.handlers:
        handler.flush()
    shutil.copyfile(log_file, config.LOGS_DIR / "latest.log")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
