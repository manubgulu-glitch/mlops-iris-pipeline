"""Pruebas de la validación de datos (solo biblioteca estándar: unittest).

Se ejecutan con:  python -m unittest discover -s tests -t . -v
"""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from src import config
from src.validation import (
    ContractError,
    DataValidationError,
    load_contract,
    raise_if_invalid,
    validate_csv,
    validate_record,
    validate_rows,
)

VALID = {
    "sepal_length_cm": 5.1,
    "sepal_width_cm": 3.5,
    "petal_length_cm": 1.4,
    "petal_width_cm": 0.2,
}


def codes(issues):
    return {issue.code for issue in issues}


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract(config.CONTRACT_PATH)

    def test_contract_loads_and_has_examples(self):
        self.assertEqual(len(self.contract["fields"]), 4)
        self.assertIn("valid", self.contract["examples"])
        self.assertIn("invalid", self.contract["examples"])

    def test_valid_example_of_contract_passes(self):
        example = self.contract["examples"]["valid"]["input"]
        self.assertEqual(validate_record(example, self.contract), [])

    def test_invalid_example_of_contract_fails_with_expected_errors(self):
        invalid = self.contract["examples"]["invalid"]
        found = codes(validate_record(invalid["input"], self.contract))
        self.assertEqual(found, set(invalid["expected_errors"]))

    def test_bad_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"fields": []}), encoding="utf-8")
            with self.assertRaises(ContractError):
                load_contract(path)
            path.write_text("{no es json", encoding="utf-8")
            with self.assertRaises(ContractError):
                load_contract(path)
        with self.assertRaises(ContractError):
            load_contract(Path("no_existe.json"))


class RecordValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract(config.CONTRACT_PATH)

    def check(self, **changes):
        record = copy.deepcopy(VALID)
        record.update(changes)
        return validate_record(record, self.contract)

    # --- tipos
    def test_integer_is_accepted_for_float_field(self):
        self.assertEqual(self.check(sepal_length_cm=5), [])

    def test_numeric_string_is_a_type_error(self):
        self.assertEqual(codes(self.check(sepal_length_cm="5.1")), {"sepal_length_cm:type"})

    def test_boolean_is_a_type_error(self):
        self.assertEqual(codes(self.check(sepal_width_cm=True)), {"sepal_width_cm:type"})

    def test_list_is_a_type_error(self):
        self.assertEqual(codes(self.check(petal_width_cm=[0.2])), {"petal_width_cm:type"})

    # --- nulos
    def test_none_is_null(self):
        self.assertEqual(codes(self.check(petal_length_cm=None)), {"petal_length_cm:null"})

    def test_nan_is_null(self):
        self.assertEqual(codes(self.check(petal_length_cm=math.nan)), {"petal_length_cm:null"})

    # --- rangos
    def test_range_limits_are_inclusive(self):
        self.assertEqual(self.check(sepal_length_cm=4.0), [])
        self.assertEqual(self.check(sepal_length_cm=8.0), [])

    def test_values_just_outside_range_fail(self):
        self.assertEqual(codes(self.check(sepal_length_cm=3.99)), {"sepal_length_cm:range"})
        self.assertEqual(codes(self.check(sepal_length_cm=8.01)), {"sepal_length_cm:range"})

    def test_negative_and_infinite_values_fail(self):
        self.assertEqual(codes(self.check(petal_width_cm=-0.5)), {"petal_width_cm:range"})
        self.assertEqual(codes(self.check(petal_width_cm=math.inf)), {"petal_width_cm:range"})

    # --- estructura
    def test_missing_field(self):
        record = copy.deepcopy(VALID)
        del record["petal_width_cm"]
        self.assertEqual(codes(validate_record(record, self.contract)), {"petal_width_cm:missing"})

    def test_unexpected_field(self):
        self.assertEqual(codes(self.check(color="rojo")), {"color:unexpected"})

    def test_target_is_not_accepted_in_inference(self):
        self.assertEqual(codes(self.check(species=1)), {"species:unexpected"})

    def test_all_problems_are_reported_not_only_the_first(self):
        found = codes(self.check(sepal_length_cm="x", sepal_width_cm=None, petal_length_cm=99))
        self.assertEqual(found, {"sepal_length_cm:type", "sepal_width_cm:null", "petal_length_cm:range"})

    def test_target_allowed_values_in_training_data(self):
        record = dict(VALID, species=3)
        issues = validate_record(record, self.contract, include_target=True)
        self.assertIn("species:allowed_values", codes(issues))
        self.assertEqual(validate_record(dict(VALID, species=2), self.contract, include_target=True), [])

    def test_validate_rows_counts_rows_and_raise_if_invalid(self):
        report = validate_rows([VALID, dict(VALID, sepal_length_cm=None)], self.contract)
        self.assertEqual(report.rows_checked, 2)
        self.assertFalse(report.ok)
        with self.assertRaises(DataValidationError):
            raise_if_invalid(report)


class CsvValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract(config.CONTRACT_PATH)

    def run_csv(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "datos.csv"
            path.write_text(text, encoding="utf-8")
            return validate_csv(path, self.contract)

    HEADER = "sepal_length_cm,sepal_width_cm,petal_length_cm,petal_width_cm,species\n"

    def test_valid_csv(self):
        report = self.run_csv(self.HEADER + "5.1,3.5,1.4,0.2,0\n6.3,3.3,6.0,2.5,2\n")
        self.assertTrue(report.ok)
        self.assertEqual(report.rows_checked, 2)

    def test_csv_detects_null_type_and_range_with_row_numbers(self):
        report = self.run_csv(self.HEADER + "5.1,,1.4,0.2,0\nabc,3.5,1.4,0.2,0\n5.1,3.5,99,0.2,0\n")
        self.assertEqual(
            {(issue.row, issue.rule) for issue in report.issues},
            {(1, "null"), (2, "type"), (3, "range")},
        )

    def test_csv_missing_and_unexpected_columns(self):
        report = self.run_csv("sepal_length_cm,color\n5.1,rojo\n")
        self.assertTrue({"missing", "unexpected"} <= {issue.rule for issue in report.issues})

    def test_csv_without_rows_is_invalid(self):
        report = self.run_csv(self.HEADER)
        self.assertEqual([issue.rule for issue in report.issues], ["empty"])

    def test_csv_missing_file(self):
        report = validate_csv(Path("no_existe.csv"), self.contract)
        self.assertFalse(report.ok)

    def test_csv_text_null_tokens(self):
        report = self.run_csv(self.HEADER + "NaN,3.5,1.4,0.2,0\n5.1,null,1.4,0.2,0\n")
        self.assertEqual({issue.rule for issue in report.issues}, {"null"})
        self.assertEqual(len(report.issues), 2)


class RealDataTests(unittest.TestCase):
    def test_real_iris_dataset_complies_with_the_contract(self):
        try:
            from sklearn.datasets import load_iris
        except ImportError:  # la validación no depende de sklearn; esta prueba sí
            self.skipTest("scikit-learn no está instalado")
        contract = load_contract(config.CONTRACT_PATH)
        names = [spec["name"] for spec in contract["fields"]]
        iris = load_iris()
        rows = [dict(zip(names, map(float, measures))) for measures in iris.data]
        report = validate_rows(rows, contract, source="iris (sklearn)")
        self.assertTrue(report.ok, report.summary())
        self.assertEqual(report.rows_checked, 150)


if __name__ == "__main__":
    unittest.main()
