# RunPod Deployment

This deployment package serves the merged RecipeNLG fine-tuned model through the
vLLM OpenAI-compatible API plus a thin app-facing wrapper.

## Default model options

Set `MODEL_PATH` to either published model:

- `nijumich/Qwen2.5-7B-Instruct-recipieNLG_rank8_alpha16`
- `nijumich/Qwen2.5-7B-Instruct-recipieNLG_rank16_alpha32`

No code changes are required to switch models.

## Files

- `deploy/Dockerfile` - container image for RunPod
- `deploy/start.sh` - env-configured startup command
- `deploy/recipe_gateway.py` - FastAPI wrapper for app-facing endpoints
- `deploy/.env.example` - required runtime variables
- `scripts/smoke_test.py` - post-deploy validation script

## RunPod setup

1. Build and publish the container from the repo root:

```bash
docker build -f deploy/Dockerfile -t your-registry/recipe-vllm:latest .
docker push your-registry/recipe-vllm:latest
```

2. In RunPod, create a GPU pod or endpoint using that image.

3. Set these environment variables in RunPod:

```text
MODEL_PATH=nijumich/Qwen2.5-7B-Instruct-recipieNLG_rank8_alpha16
HF_TOKEN=hf_...
PORT=8000
VLLM_PORT=8001
MAX_MODEL_LEN=1024
GPU_MEMORY_UTILIZATION=0.9
TENSOR_PARALLEL_SIZE=1
DTYPE=auto
ENFORCE_EAGER=1
VLLM_USE_FLASHINFER_SAMPLER=0
API_KEY=replace_with_a_secret_key
CORS_ALLOW_ORIGINS=*
```

4. Expose port `8000`.

## Recommended hardware

- Recommended first deployment: A100 80GB or similar
- Lower-memory cards may require smaller limits or may fail during model load

## Validate the deployment

After RunPod reports the service as healthy, run:

```bash
python -m scripts.smoke_test --base-url https://YOUR-ENDPOINT
```

The smoke test fails if:

- the API is unreachable
- the server does not return chat-completions JSON
- the output is empty
- the output does not contain numbered steps
- the model repeats an `Ingredients:` heading

Reusable test payloads are included in:

- `tests/recipe_api_payloads.json`

They include a balanced mix of dataset-like prompts, realistic app prompts, and
edge cases for manual regression testing after redeploys.

## Public endpoints

The wrapper exposes:

- `GET /health` - wrapper and backend readiness
- `GET /v1/models` - proxy to local vLLM
- `POST /v1/chat/completions` - proxied OpenAI-compatible endpoint
- `POST /generate-recipe` - simplified app-facing endpoint

If `API_KEY` is set, send it as:

```text
X-API-Key: your-secret-key
```

`CORS_ALLOW_ORIGINS` controls which browser origins may call the wrapper. For
quick testing you can leave it as `*`. For a real frontend, replace it with a
comma-separated list such as:

```text
CORS_ALLOW_ORIGINS=https://your-frontend.example,https://hoppscotch.io
```

`/generate-recipe` request body:

```json
{
  "title": "Tomato Pasta",
  "ingredients": ["pasta", "tomatoes", "garlic", "olive oil", "salt"],
  "temperature": 0.0,
  "top_p": 0.95,
  "max_tokens": 512
}
```

Response shape:

```json
{
  "recipe_text": "1. ...",
  "model": "nijumich/Qwen2.5-7B-Instruct-recipieNLG_rank8_alpha16",
  "usage": {
    "prompt_tokens": 77,
    "total_tokens": 148,
    "completion_tokens": 71
  },
  "raw_response": {}
}
```

## Notes on common startup failures

- If logs mention `Failed to find C compiler`, rebuild and redeploy with the current
  `deploy/Dockerfile`, which now installs `build-essential`.
- If logs mention `Could not find nvcc` from `flashinfer`, set
  `VLLM_USE_FLASHINFER_SAMPLER=0` so vLLM falls back to its non-FlashInfer sampler.
- This deployment defaults `ENFORCE_EAGER=1` and `VLLM_USE_FLASHINFER_SAMPLER=0`
  for a more stable container-only setup.

## Example raw API request

```bash
curl https://YOUR-ENDPOINT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "recipe-model",
    "messages": [
      {
        "role": "system",
        "content": "You are a culinary assistant. Write step-by-step cooking directions using the given title and ingredients. Use all relevant ingredients. Do NOT repeat the ingredient list. Use complete sentences. Use numbered steps with action verbs."
      },
      {
        "role": "user",
        "content": "Title: Tomato Pasta\n\nIngredients:\n- pasta\n- tomatoes\n- garlic\n- olive oil\n- salt"
      }
    ],
    "temperature": 0.0,
    "top_p": 0.95,
    "max_tokens": 512,
    "stop": ["<|im_end|>", "<|endoftext|>"]
  }'
```
