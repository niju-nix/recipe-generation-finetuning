#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
VLLM_PORT="${VLLM_PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-auto}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

export VLLM_USE_FLASHINFER_SAMPLER

echo "Starting vLLM server"
echo "MODEL_PATH=${MODEL_PATH}"
echo "HOST=${HOST}"
echo "PORT=${PORT}"
echo "VLLM_PORT=${VLLM_PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
echo "DTYPE=${DTYPE}"
echo "ENFORCE_EAGER=${ENFORCE_EAGER}"
echo "VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER}"
if [[ -n "${API_KEY:-}" ]]; then
  echo "API_KEY is set"
else
  echo "API_KEY is not set"
fi

ARGS=(
  --host "127.0.0.1"
  --port "${VLLM_PORT}"
  --model "${MODEL_PATH}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --dtype "${DTYPE}"
  --trust-remote-code
)

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi

python -m vllm.entrypoints.openai.api_server "${ARGS[@]}" &
VLLM_PID=$!

cleanup() {
  kill "${VLLM_PID}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}"

exec python -m uvicorn deploy.recipe_gateway:app \
  --host "${HOST}" \
  --port "${PORT}"
