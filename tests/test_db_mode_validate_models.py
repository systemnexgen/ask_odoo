import types
import unittest

from tests.test_support import load_module


load_module("ask_odoo.models.prompts", "prompts.py")
db_mode = load_module("ask_odoo.models.db_mode", "db_mode.py")


class FakeEnv(dict):
    @property
    def registry(self):
        return list(self.keys())


class ValidateCodeModelsTests(unittest.TestCase):
    def _build_model(self, env):
        model = db_mode.AskOdooModel()
        model.env = env
        return model

    def test_validate_code_models_accepts_known_models(self):
        env = FakeEnv(
            {
                "res.partner": types.SimpleNamespace(_abstract=False),
                "sale.order": types.SimpleNamespace(_abstract=False),
            }
        )
        model = self._build_model(env)
        code = "result = self.env['res.partner'].search([])\nself.env[\"sale.order\"].search_count([])"

        is_valid, err = model._validate_code_models(code)

        self.assertTrue(is_valid)
        self.assertIsNone(err)

    def test_validate_code_models_rejects_unknown_model_with_suggestions(self):
        env = FakeEnv(
            {
                "sale.order": types.SimpleNamespace(_abstract=False),
                "sale.report": types.SimpleNamespace(_abstract=False),
                "sale.abstract": types.SimpleNamespace(_abstract=True),
                "res.partner": types.SimpleNamespace(_abstract=False),
            }
        )
        model = self._build_model(env)

        is_valid, err = model._validate_code_models("result = self.env['sale.invoice'].search([])")

        self.assertFalse(is_valid)
        self.assertIn("Model 'sale.invoice' does not exist", err)
        self.assertIn("sale.order", err)
        self.assertIn("sale.report", err)
        self.assertNotIn("sale.abstract", err)

    def test_validate_code_models_returns_success_when_no_model_references(self):
        env = FakeEnv({"res.partner": types.SimpleNamespace(_abstract=False)})
        model = self._build_model(env)

        is_valid, err = model._validate_code_models("result = 1 + 1")

        self.assertTrue(is_valid)
        self.assertIsNone(err)


if __name__ == "__main__":
    unittest.main()
