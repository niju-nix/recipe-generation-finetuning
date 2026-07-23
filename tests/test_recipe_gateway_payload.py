import unittest

from scripts.recipe_prompt import build_chat_completion_payload


class TestRecipeGatewayPayload(unittest.TestCase):
    def test_build_chat_completion_payload_uses_expected_defaults(self):
        payload = build_chat_completion_payload(
            "test-model",
            "Tomato Pasta",
            ["pasta", "tomatoes"],
        )
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["top_p"], 0.95)
        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(payload["stop"], ["<|im_end|>", "<|endoftext|>"])
        self.assertEqual(payload["messages"][1]["content"], "Title: Tomato Pasta\n\nIngredients:\n- pasta\n- tomatoes")


if __name__ == "__main__":
    unittest.main()
