"""Thin FastAPI wrapper for recipe generation over a local vLLM server."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scripts.recipe_prompt import build_chat_completion_payload

BACKEND_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001")
MODEL_PATH = os.getenv("MODEL_PATH", "")
API_KEY = os.getenv("API_KEY", "")
CORS_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]

app = FastAPI(title="Recipe Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class GenerateRecipeRequest(BaseModel):
    title: str = Field(min_length=1)
    ingredients: list[str] = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)


def _require_api_key(provided_key: str | None) -> None:
    if API_KEY and provided_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _backend_request(
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
) -> tuple[int, bytes, str]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url=f"{BACKEND_BASE_URL.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, response.read(), response.headers.get_content_type()
    except urllib.error.HTTPError as exc:
        raise HTTPException(
            status_code=exc.code,
            detail=exc.read().decode("utf-8", errors="replace"),
        ) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail=f"Backend unavailable: {exc}") from exc


@app.get("/health")
def health() -> dict[str, object]:
    try:
        status, _, _ = _backend_request("GET", "/health")
        backend_ready = status == 200
    except HTTPException:
        backend_ready = False
    return {"status": "ok" if backend_ready else "degraded", "backend_ready": backend_ready}


@app.get("/v1/models")
def list_models() -> Response:
    status, content, content_type = _backend_request("GET", "/v1/models")
    return Response(content=content, media_type=content_type, status_code=status)


@app.post("/v1/chat/completions")
def proxy_chat_completions(
    payload: dict[str, Any],
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Response:
    _require_api_key(x_api_key)
    status, content, content_type = _backend_request("POST", "/v1/chat/completions", body=payload)
    return Response(content=content, media_type=content_type, status_code=status)


@app.post("/generate-recipe")
def generate_recipe(
    request: GenerateRecipeRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    _require_api_key(x_api_key)
    payload = build_chat_completion_payload(
        MODEL_PATH,
        request.title,
        request.ingredients,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )
    _, content, _ = _backend_request("POST", "/v1/chat/completions", body=payload)
    raw_response = json.loads(content.decode("utf-8"))
    recipe_text = raw_response["choices"][0]["message"]["content"]
    return {
        "recipe_text": recipe_text,
        "model": raw_response.get("model", MODEL_PATH),
        "usage": raw_response.get("usage"),
        "raw_response": raw_response,
    }
