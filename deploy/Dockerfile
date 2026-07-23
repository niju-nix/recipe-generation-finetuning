FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "vllm>=0.6.0" "huggingface_hub>=0.30.0" "fastapi>=0.116.0" "uvicorn>=0.35.0"

COPY scripts /app/scripts
COPY deploy/recipe_gateway.py /app/deploy/recipe_gateway.py
COPY deploy/start.sh /app/deploy/start.sh

RUN chmod +x /app/deploy/start.sh

EXPOSE 8000

CMD ["/app/deploy/start.sh"]
