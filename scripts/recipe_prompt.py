"""Shared recipe prompt helpers for deployment and validation."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a culinary assistant. Write step-by-step cooking directions "
    "using the given title and ingredients. Use all relevant ingredients. "
    "Do NOT repeat the ingredient list. Use complete sentences. "
    "Use numbered steps with action verbs."
)


def format_title_and_ingredients(title: str, ingredients: list[str]) -> str:
    """Format recipe inputs in the same style used during notebook inference."""
    normalized_title = title.strip()
    normalized_ingredients = [item.strip() for item in ingredients if item.strip()]
    lines = [f"Title: {normalized_title}", "", "Ingredients:"]
    lines.extend(f"- {item}" for item in normalized_ingredients)
    return "\n".join(lines)


def build_chat_messages(title: str, ingredients: list[str]) -> list[dict[str, str]]:
    """Build OpenAI-style chat messages for recipe generation."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": format_title_and_ingredients(title, ingredients),
        },
    ]


def build_chat_completion_payload(
    model: str,
    title: str,
    ingredients: list[str],
    *,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_tokens: int = 512,
    stop: list[str] | None = None,
) -> dict[str, object]:
    """Build a stable chat-completions payload for recipe generation."""
    return {
        "model": model,
        "messages": build_chat_messages(title, ingredients),
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stop": stop or ["<|im_end|>", "<|endoftext|>"],
    }
