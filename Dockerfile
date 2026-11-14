# syntax=docker/dockerfile:1
# Runtime image for the API and the worker (same image, different command).
# Weights are NOT baked in: mount them read-only at /models (see docs/DEPLOYMENT.md).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DOPPEL_DATA_DIR=/data \
    DOPPEL_MODELS_DIR=/models \
    DOPPEL_HOST=0.0.0.0 \
    DOPPEL_PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 doppel \
    && useradd --system --uid 10001 --gid doppel --no-create-home --shell /usr/sbin/nologin doppel \
    && mkdir -p /data /models \
    && chown doppel:doppel /data

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app
COPY migrations ./migrations
COPY models/manifest.json ./models/manifest.json
COPY run.py ./

USER doppel
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["python", "run.py"]
