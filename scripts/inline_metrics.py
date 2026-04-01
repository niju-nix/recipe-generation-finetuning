"""Shared low-cost inline metrics used by inference and tests."""

from __future__ import annotations

import ast
import re
from typing import Any, Mapping


def _normalize_lines(value: Any) -> list[str]:
    """Normalize heterogeneous ingredient inputs into lowercase string tokens."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(x).strip().lower() for x in value if str(x).strip()]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(x).strip().lower() for x in parsed if str(x).strip()]
            except Exception:
                pass

            quoted = [a or b for a, b in re.findall(r"'([^']+)'|\"([^\"]+)\"", text)]
            if quoted:
                return [q.strip().lower() for q in quoted if q and q.strip()]

        lines = re.split(r"[\n,;]", text)
        return [x.strip(" []'\"\t\r").lower() for x in lines if x.strip(" []'\"\t\r")]

    scalar = str(value).strip()
    return [scalar.lower()] if scalar else []


def format_ingredient_field(value: Any) -> str:
    """Flatten mixed ingredient values to a readable string for exports."""
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and isinstance(value[0], str):
            serialized = value[0].strip()
            quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", serialized)
            parts = [a or b for a, b in quoted if (a or b)]
            if parts:
                return ", ".join(parts)
        return ", ".join(str(i).strip() for i in value if str(i).strip())

    serialized = str(value).strip()
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", serialized)
    parts = [a or b for a, b in quoted if (a or b)]
    if parts:
        return ", ".join(parts)

    return serialized


def _extract_ingredient_candidates(row: Mapping[str, Any]) -> list[str]:
    """Extract candidate ingredient strings from the best available source."""
    for key in ["ner_ingredients", "ingredients_normalized", "ingredients_bullets"]:
        if key in row and row.get(key) is not None:
            vals = _normalize_lines(row.get(key))
            if vals:
                return vals

    input_text = str(row.get("input_raw", ""))
    marker = "ingredients:"
    lower = input_text.lower()
    if marker in lower:
        start = lower.index(marker) + len(marker)
        tail = input_text[start:]
        if "directions:" in tail.lower():
            idx = tail.lower().index("directions:")
            tail = tail[:idx]
        vals = _normalize_lines(tail)
        if vals:
            return vals
    return []


def compute_inline_metrics(row: Mapping[str, Any]) -> dict[str, float | int]:
    """Compute low-cost inline metrics per prediction row."""
    prediction = str(row.get("prediction", "") or "")
    tokens = prediction.lower().split()
    repetition = 1.0 - (len(set(tokens)) / max(len(tokens), 1))
    ingredient_candidates = _extract_ingredient_candidates(row)
    pred_lower = prediction.lower()
    used = sum(1 for ing in ingredient_candidates if ing and ing in pred_lower)
    coverage = used / max(len(ingredient_candidates), 1)
    return {
        "output_tokens": int(len(tokens)),
        "ingredient_coverage": float(coverage),
        "repetition_ratio": float(repetition),
    }
