"""Smoke-test a deployed vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from typing import Any

from scripts.recipe_prompt import build_chat_messages

NUMBERED_STEP_RE = re.compile(r"(?m)^\s*\d+[.)]\s+\S+")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a deployed recipe model endpoint."
    )
    parser.add_argument("--base-url", required=True, help="Base URL of the deployed API")
    parser.add_argument(
        "--model",
        default="recipe-model",
        help="Model field to send in the OpenAI-compatible request payload",
    )
    parser.add_argument(
        "--title",
        default="Tomato Pasta",
        help="Recipe title used for the smoke test",
    )
    parser.add_argument(
        "--ingredients",
        nargs="+",
        default=["pasta", "tomatoes", "garlic", "olive oil", "salt"],
        help="Ingredient list used for the smoke test",
    )
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout in seconds")
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Generation limit for validation"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=0.95, help="Sampling top_p")
    return parser.parse_args()


def _post_chat_completion(base_url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_text(response_json: dict[str, Any]) -> str:
    try:
        return response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AssertionError("Response did not contain choices[0].message.content") from exc


def _assert_recipe_shape(recipe_text: str) -> None:
    if not recipe_text.strip():
        raise AssertionError("Model returned empty text")
    if "ingredients:" in recipe_text.lower():
        raise AssertionError("Model repeated an ingredient-list heading in the output")
    if not NUMBERED_STEP_RE.search(recipe_text):
        raise AssertionError("Model output did not contain numbered steps")


def main() -> int:
    args = _parse_args()
    payload = {
        "model": args.model,
        "messages": build_chat_messages(args.title, args.ingredients),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }

    try:
        response_json = _post_chat_completion(args.base_url, payload, args.timeout)
        recipe_text = _extract_text(response_json)
        _assert_recipe_shape(recipe_text)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP error {exc.code}: {body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Connection error: {exc}", file=sys.stderr)
        return 1
    except AssertionError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    print("Smoke test passed.")
    print(f"Endpoint: {args.base_url.rstrip('/')}/v1/chat/completions")
    print(f"Title: {args.title}")
    print(f"Ingredients: {', '.join(args.ingredients)}")
    print("Recipe preview:")
    print(recipe_text.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
