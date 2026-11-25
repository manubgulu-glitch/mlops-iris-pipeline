"""Validación de datos contra el contrato usando solo la biblioteca estándar.

Verifica tipos, nulos y rangos, además de campos faltantes e inesperados.
"""
from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

LOGGER = logging.getLogger("pipeline.validation")

CSV_NULL_TOKENS = ("", "nan", "null", "none", "na", "n/a")
SUPPORTED_TYPES = ("float", "int")


class ContractError(Exception):
    """Error de estructura en el archivo de contrato."""


class DataValidationError(Exception):
    """Error cuando los datos incumplen el contrato."""

    def __init__(self, report: "ValidationReport"):
        self.report = report
        super().__init__(report.summary())


@dataclass(frozen=True)
class ValidationIssue:
    """Detalle de un incumplimiento detectado durante la validación."""

    row: Optional[int]
    field: str
    rule: str
    message: str
    value: Any = None

    @property
    def code(self) -> str:
        return f"{self.field}:{self.rule}"

    def __str__(self) -> str:
        donde = f"fila {self.row}, " if self.row is not None else ""
        return f"[{self.rule}] {donde}campo '{self.field}': {self.message}"


@dataclass
class ValidationReport:
    source: str
    rows_checked: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def counts_by_rule(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for issue in self.issues:
            counts[issue.rule] = counts.get(issue.rule, 0) + 1
        return counts

    def summary(self) -> str:
        if self.ok:
            return f"{self.source}: {self.rows_checked} registro(s) validado(s), sin problemas"
        detalle = ", ".join(f"{regla}={n}" for regla, n in sorted(self.counts_by_rule().items()))
        return f"{self.source}: {len(self.issues)} problema(s) en {self.rows_checked} registro(s) ({detalle})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "ok": self.ok,
            "rows_checked": self.rows_checked,
            "issues_by_rule": self.counts_by_rule(),
            "issues": [
                {"row": i.row, "field": i.field, "rule": i.rule, "message": i.message}
                for i in self.issues[:200]
            ],
        }


def load_contract(path: Path | str) -> Dict[str, Any]:
    """Carga y verifica la estructura del contrato JSON."""
    try:
        contract = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"No existe el contrato de datos: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"El contrato no es un JSON válido: {exc}") from exc

    fields = contract.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ContractError("El contrato debe tener una lista 'fields' no vacía")
    names = set()
    for spec in fields + ([contract["target"]] if "target" in contract else []):
        for key in ("name", "type"):
            if key not in spec:
                raise ContractError(f"Falta la clave '{key}' en una definición de campo: {spec}")
        if spec["type"] not in SUPPORTED_TYPES:
            raise ContractError(f"Tipo no soportado '{spec['type']}' en el campo '{spec['name']}'")
        if spec["name"] in names:
            raise ContractError(f"Campo duplicado en el contrato: '{spec['name']}'")
        names.add(spec["name"])
        low, high = spec.get("min"), spec.get("max")
        if low is not None and high is not None and low > high:
            raise ContractError(f"Rango inválido (min > max) en el campo '{spec['name']}'")
    return contract


def _is_null(value: Any, strict: bool) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if not strict and isinstance(value, str) and value.strip().lower() in CSV_NULL_TOKENS:
        return True
    return False


def _coerce(type_name: str, value: Any, strict: bool):
    """Convierte y valida el tipo de dato. Devuelve (valor, error)."""
    if isinstance(value, bool):
        return None, "se recibió un booleano en lugar de un número"

    if type_name == "float":
        if isinstance(value, (int, float)):
            return float(value), None
        if not strict and isinstance(value, str):
            try:
                return float(value.strip()), None
            except ValueError:
                return None, f"'{value}' no se puede convertir a número decimal"
        return None, f"se esperaba un número decimal y se recibió {type(value).__name__}"

    if type_name == "int":
        if isinstance(value, int):
            return value, None
        if not strict and isinstance(value, str):
            text = value.strip()
            try:
                return int(text), None
            except ValueError:
                try:
                    number = float(text)
                except ValueError:
                    return None, f"'{value}' no se puede convertir a entero"
                if number.is_integer():
                    return int(number), None
                return None, f"'{value}' no es un entero"
        return None, f"se esperaba un entero y se recibió {type(value).__name__}"

    return None, f"tipo no soportado: {type_name}"


def _check_field(spec: Dict[str, Any], value: Any, row: Optional[int], strict: bool) -> List[ValidationIssue]:
    name = spec["name"]

    # 1) Nulos
    if _is_null(value, strict):
        if spec.get("nullable", False):
            return []
        return [ValidationIssue(row, name, "null", "el valor es nulo o está vacío", value)]

    # 2) Tipo
    number, error = _coerce(spec["type"], value, strict)
    if error:
        return [ValidationIssue(row, name, "type", error, value)]

    # 3) Rango lógico
    if not math.isfinite(number):
        return [ValidationIssue(row, name, "range", "el valor no es finito", value)]
    issues: List[ValidationIssue] = []
    low, high = spec.get("min"), spec.get("max")
    if (low is not None and number < low) or (high is not None and number > high):
        issues.append(
            ValidationIssue(row, name, "range", f"{number} fuera del rango permitido [{low}, {high}]", value)
        )
    allowed = spec.get("allowed_values")
    if allowed is not None and number not in allowed:
        issues.append(ValidationIssue(row, name, "allowed_values", f"{number} no está en {allowed}", value))
    return issues


def validate_record(
    record: Dict[str, Any],
    contract: Dict[str, Any],
    row: Optional[int] = None,
    strict: bool = True,
    include_target: bool = False,
) -> List[ValidationIssue]:
    """Valida un registro individual contra el contrato."""
    specs = list(contract["fields"])
    if include_target and "target" in contract:
        specs.append(contract["target"])

    issues: List[ValidationIssue] = []
    known = {spec["name"] for spec in specs}

    for spec in specs:
        if spec["name"] not in record:
            if spec.get("required", True):
                issues.append(ValidationIssue(row, spec["name"], "missing", "el campo es obligatorio y no vino"))
            continue
        issues.extend(_check_field(spec, record[spec["name"]], row, strict))

    if not contract.get("rules", {}).get("allow_extra_fields", False):
        for extra in record:
            if extra not in known:
                issues.append(ValidationIssue(row, str(extra), "unexpected", "campo no definido en el contrato"))
    return issues


def validate_rows(
    rows: Iterable[Dict[str, Any]],
    contract: Dict[str, Any],
    source: str = "registros",
    strict: bool = True,
    include_target: bool = False,
) -> ValidationReport:
    report = ValidationReport(source=source)
    for number, record in enumerate(rows, start=1):
        report.rows_checked += 1
        report.issues.extend(
            validate_record(record, contract, row=number, strict=strict, include_target=include_target)
        )
    return report


def validate_csv(path: Path | str, contract: Dict[str, Any], include_target: bool = True) -> ValidationReport:
    """Valida un archivo CSV completo contra el contrato."""
    path = Path(path)
    report = ValidationReport(source=path.name)
    if not path.exists():
        report.issues.append(ValidationIssue(None, str(path), "missing", "el archivo no existe"))
        LOGGER.error(report.summary())
        return report

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = [spec["name"] for spec in contract["fields"]]
        if include_target and "target" in contract:
            expected.append(contract["target"]["name"])
        columns = reader.fieldnames or []
        for name in expected:
            if name not in columns:
                report.issues.append(ValidationIssue(None, name, "missing", "falta la columna en el encabezado"))
        for name in columns:
            if name not in expected:
                report.issues.append(ValidationIssue(None, name, "unexpected", "columna no definida en el contrato"))

        for number, record in enumerate(reader, start=1):
            report.rows_checked += 1
            report.issues.extend(
                validate_record(record, contract, row=number, strict=False, include_target=include_target)
            )

    if report.rows_checked == 0:
        report.issues.append(ValidationIssue(None, path.name, "empty", "el archivo no tiene filas de datos"))

    (LOGGER.info if report.ok else LOGGER.error)(report.summary())
    return report


def raise_if_invalid(report: ValidationReport) -> ValidationReport:
    """Lanza DataValidationError si el reporte contiene incumplimientos."""
    if not report.ok:
        for issue in report.issues[:10]:
            LOGGER.error("  %s", issue)
        if len(report.issues) > 10:
            LOGGER.error("  ... y %d problema(s) más", len(report.issues) - 10)
        raise DataValidationError(report)
    return report
