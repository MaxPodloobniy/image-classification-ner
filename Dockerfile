# ── builder ───────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# ── runtime ───────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# onnxruntime requires libgomp1 at import time on Debian slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# App source code.
COPY src/ /app/src/

# Models pre-downloaded by CI (classifier ONNX + classes JSON, NER spaCy dir).
COPY models/ /app/models/

EXPOSE 8000

ENTRYPOINT ["uvicorn", "src.web.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
CMD ["--workers", "1"]
