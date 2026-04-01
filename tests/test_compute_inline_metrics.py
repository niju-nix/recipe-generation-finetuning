import unittest
from scripts.inline_metrics import compute_inline_metrics


class TestComputeInlineMetrics(unittest.TestCase):
    def test_token_count_repetition_and_coverage_from_explicit_ingredients(self):
        row = {
            "prediction": "Salt pepper salt",
            "ingredients_normalized": ["salt", "pepper", "garlic"],
        }
        out = compute_inline_metrics(row)
        self.assertEqual(out["output_tokens"], 3)
        self.assertAlmostEqual(out["repetition_ratio"], 1.0 - (2 / 3), places=8)
        self.assertAlmostEqual(out["ingredient_coverage"], 2 / 3, places=8)

    def test_uses_priority_order_for_ingredient_sources(self):
        row = {
            "prediction": "contains only apple",
            "ner_ingredients": ["apple"],
            "ingredients_normalized": ["apple", "banana"],
        }
        out = compute_inline_metrics(row)
        self.assertAlmostEqual(out["ingredient_coverage"], 1.0, places=8)

    def test_fallback_extracts_ingredients_from_input_raw(self):
        row = {
            "input_raw": "Ingredients: onion, garlic; pepper\nDirections: mix and cook",
            "prediction": "Saute onion and garlic first.",
        }
        out = compute_inline_metrics(row)
        self.assertAlmostEqual(out["ingredient_coverage"], 2 / 3, places=8)

    def test_empty_prediction_and_no_candidates_are_stable(self):
        row = {"prediction": "", "input_raw": "No ingredients here."}
        out = compute_inline_metrics(row)
        self.assertEqual(out["output_tokens"], 0)
        self.assertAlmostEqual(out["ingredient_coverage"], 0.0, places=8)
        self.assertAlmostEqual(out["repetition_ratio"], 1.0, places=8)

    def test_parses_numpy_style_list_string_in_ner_ingredients(self):
        row = {
            "prediction": "Use ginger ale, flavor gelatin, and boiling water.",
            "ner_ingredients": "['dark sweet pitted cherries' 'ginger ale' 'flavor gelatin'\\n 'boiling water' 'almond extract' 'marshmallows']",
        }
        out = compute_inline_metrics(row)
        self.assertAlmostEqual(out["ingredient_coverage"], 3 / 6, places=8)


if __name__ == "__main__":
    unittest.main()
