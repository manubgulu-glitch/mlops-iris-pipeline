"""Pruebas de validación del modelo entrenado (unittest).

Requieren que el pipeline ya haya entrenado el modelo:  python pipeline.py
"""
import json
import unittest

from src import config
from src.train import read_xy
from src.validation import load_contract, validate_record


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not config.MODEL_PATH.exists():
            raise AssertionError(
                f"No existe {config.MODEL_PATH}. Ejecute primero el entrenamiento: python pipeline.py"
            )
        import joblib

        cls.contract = load_contract(config.CONTRACT_PATH)
        cls.model = joblib.load(config.MODEL_PATH)
        cls.X_test, cls.y_test = read_xy(config.TEST_PATH, cls.contract)

    def test_metrics_file_is_consistent_with_contract(self):
        metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(metrics["contract_version"], self.contract["contract_version"])
        self.assertEqual(metrics["n_test"], len(self.y_test))

    def test_model_accuracy_meets_minimum_threshold(self):
        accuracy = self.model.score(self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, config.MIN_ACCURACY, f"accuracy {accuracy:.3f} < {config.MIN_ACCURACY}")

    def test_model_predicts_valid_contract_example(self):
        example = self.contract["examples"]["valid"]["input"]
        self.assertEqual(validate_record(example, self.contract), [])
        row = [[example[spec["name"]] for spec in self.contract["fields"]]]
        prediction = int(self.model.predict(row)[0])
        self.assertIn(prediction, self.contract["target"]["allowed_values"])
        self.assertEqual(prediction, 0)

    def test_invalid_contract_example_is_rejected_before_inference(self):
        invalid = self.contract["examples"]["invalid"]["input"]
        self.assertNotEqual(validate_record(invalid, self.contract), [])

    def test_model_predicts_only_known_classes(self):
        predictions = set(int(p) for p in self.model.predict(self.X_test))
        self.assertTrue(predictions <= set(self.contract["target"]["allowed_values"]))


if __name__ == "__main__":
    unittest.main()
