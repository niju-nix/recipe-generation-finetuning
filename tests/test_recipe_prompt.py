import unittest

from scripts.recipe_prompt import SYSTEM_PROMPT, build_chat_messages, format_title_and_ingredients


class TestRecipePrompt(unittest.TestCase):
    def test_format_title_and_ingredients_matches_expected_layout(self):
        rendered = format_title_and_ingredients(
            "Tomato Pasta",
            ["pasta", " tomatoes ", "", "olive oil"],
        )
        self.assertEqual(
            rendered,
            "Title: Tomato Pasta\n\nIngredients:\n- pasta\n- tomatoes\n- olive oil",
        )

    def test_build_chat_messages_uses_system_prompt_and_formatted_user_input(self):
        messages = build_chat_messages("Soup", ["water", "salt"])
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(
            messages[1],
            {
                "role": "user",
                "content": "Title: Soup\n\nIngredients:\n- water\n- salt",
            },
        )


if __name__ == "__main__":
    unittest.main()
